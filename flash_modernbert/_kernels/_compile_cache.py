"""Skip CuteDSL's ~10 ms per-call dispatch by caching compiled functions.

Calling a `@cute.jit`-decorated launcher pays ~10 ms of host-side dispatch
on every invocation (arg canonicalization, name mangling, in-memory cache
lookup, MLIR call-site generation, capi marshaling), even when the cubin is
already cached. See `project_cutedsl_launch_overhead` memory for the
profile that established this.

`launcher(*args, compile_only=True)` returns a `JitCompiledFunction` whose
`__call__` skips that pipeline — measured at ~7 µs per call vs ~10 ms.

This module caches those precompiled functions, keyed by whatever the
caller declares as the compilation signature (typically the shape dims
that get baked into the kernel grid + the input dtypes).

Cache miss = one slow `compile_only=True` call (~hundreds of ms of compile
on top of the 10 ms dispatch). Cache hit = ~7 µs.

`current_cute_stream()` returns the current PyTorch CUDA stream wrapped as
a `cuda_driver.CUstream` so kernels can launch on the captured stream
during `torch.cuda.graph(...)` capture. Without this, CuteDSL kernels
default to `CU_STREAM_DEFAULT` (the legacy default stream), which is NOT
captured by stream-bound graph capture — producing silently-wrong replay
output (kernels never actually re-execute on replay; surrounding torch ops
read stale buffer state). This is documented in CuteDSL's own
`cute/testing.py` benchmark helper.

------------------------------------------------------------------------
Persistent disk cache (FLASH_MODERNBERT_DSL_CACHE=1) — cross-process compile reuse
------------------------------------------------------------------------
The fast `compile_only=True` path has one cost: it forces CuteDSL's
``no_cache=True`` (dsl.py: "Cache is disabled as user wants to compile
only"), which disables BOTH the in-memory jit cache AND CuteDSL's built-in
**disk file cache**. So every fresh `python …` process recompiles every
kernel from scratch — the dominant credit sink when iterating on the B200
(ptxas on a heavy tcgen05 kernel is tens of seconds *each*; a bench + smoke
+ A/B across separate processes pays it 3×).

CuteDSL's disk file cache (at ``$CUTE_DSL_CACHE_DIR``, default
``$TMPDIR/$USER/cutlass_python_cache``) persists the *compiled* module —
lowered IR with the cubin embedded — and on reload just re-JIT-links it,
skipping ptxas entirely. Measured locally: 48.5 s cold compile → 0.36 s on
a fresh process = 134×. But it only engages on a NORMAL (non-compile_only)
launcher call, because that's the only path that leaves ``no_cache=False``.

So when ``FLASH_MODERNBERT_DSL_CACHE=1`` we route the FIRST invocation of each kernel
through the normal call path (engages the disk cache and executes once),
then capture the resulting compiled ``JitCompiledFunction`` and hand it back
for all subsequent calls — recovering the ~15 µs fast dispatch. Net effect:
cross-process ptxas reuse *and* fast dispatch, so it's safe for benches too.

Default OFF: the production fast-dispatch path (`compile_only=True`) is
completely untouched unless the env var is set.
"""

from __future__ import annotations

import os
import threading
import warnings
from typing import Any

import cuda.bindings.driver as _cuda_driver
import torch

# Outer key: id(launcher) (stable per-shape-specialization). Inner key: caller-defined signature.
_caches: dict[int, dict[Any, Any]] = {}

# Opt-in persistent disk cache. When set, the first call of each kernel goes
# through CuteDSL's normal path so its built-in $CUTE_DSL_CACHE_DIR file cache
# persists the compiled cubin across processes (skips ptxas on reload).
_DSL_CACHE = os.environ.get("FLASH_MODERNBERT_DSL_CACHE", "0") == "1"


def get_compiled(launcher, args: tuple, key) -> Any:
    """Return a precompiled callable for `launcher(*args)` with the given signature key.

    First call with a new (launcher, key) compiles + caches. Subsequent calls return the
    cached callable. The caller calls the result with the same args shape they passed here.

    Default path returns a `JitCompiledFunction` (fast ~7 µs dispatch, no disk cache).
    With `FLASH_MODERNBERT_DSL_CACHE=1`, returns a `_FileCacheDispatch` wrapper that engages
    CuteDSL's persistent disk cache on first use (see module docstring).
    """
    inner = _caches.setdefault(id(launcher), {})
    compiled = inner.get(key)
    if compiled is not None:
        return compiled

    if _DSL_CACHE:
        # Lazy: defer compilation to the first compiled(*args) so the kernel
        # executes exactly once per logical call (correct for in-place kernels).
        compiled = _FileCacheDispatch(launcher)
    else:
        with warnings.catch_warnings():
            # CuteDSL warns "Cache is disabled as user wants to compile only" on every
            # compile_only call. We're caching the result ourselves, so the warning is
            # noise — silence it.
            warnings.simplefilter("ignore", UserWarning)
            compiled = launcher(*args, compile_only=True)
    inner[key] = compiled
    return compiled


# Guards the brief global monkeypatch window in _normal_call_capturing. Our
# kernel launches are single-threaded per process, but be defensive.
_capture_lock = threading.Lock()


def _normal_call_capturing(launcher, args):
    """Invoke `launcher(*args)` via CuteDSL's NORMAL path (engages the disk file
    cache and executes the kernel once), capturing the compiled JitCompiledFunction
    that the DSL stores in its in-memory cache so callers get fast dispatch after.

    Returns a fast-dispatch callable. Falls back to a plain re-dispatch wrapper if
    the compiled function can't be captured (so correctness never depends on the
    capture working).
    """
    import cutlass.base_dsl.cache_helpers as _ch

    # Hook BOTH set and get. set() fires when this call compiles (or loads from
    # disk) a new module. But shape-generic kernels (e.g. the persistent tcgen05
    # GEMM, whose grid/dims are runtime args → one module_hash for all M/N/K)
    # only compile on the FIRST shape; later shapes hit the DSL's in-memory jit
    # cache via get() and never call set(), so a set-only hook captures nothing
    # and we'd fall back to a per-call re-dispatch (~0.7 s on a big kernel). The
    # get() hook captures the cache-hit fn so every wrapper gets fast dispatch.
    captured: list[Any] = []
    with _capture_lock:
        orig_set = _ch.JitCacheDict.set
        orig_get = _ch.JitCacheDict.get

        def _set_hook(self, key, value, funcBody=None):  # noqa: ANN001
            captured.append(value)
            return orig_set(self, key, value, funcBody=funcBody)

        def _get_hook(self, key):  # noqa: ANN001
            value = orig_get(self, key)
            if value is not None:
                captured.append(value)
            return value

        _ch.JitCacheDict.set = _set_hook
        _ch.JitCacheDict.get = _get_hook
        try:
            launcher(*args)  # compiles-or-loads-from-disk-or-hits-memory, runs once
        finally:
            _ch.JitCacheDict.set = orig_set
            _ch.JitCacheDict.get = orig_get

    if os.environ.get("FLASH_MODERNBERT_DSL_CACHE_DEBUG") == "1":
        info = [
            (type(f).__name__, getattr(f, "engine", "NOATTR") is not None)
            for f in captured
        ]
        print(f"  [capture] {len(captured)} fn(s): {info}", flush=True)

    # Prefer the last fn carrying a live engine (the executable compiled fn). On a
    # disk-cache hit the loaded engine=None fn is seen first, then the JIT-linked one.
    for fn in reversed(captured):
        if getattr(fn, "engine", None) is not None:
            return fn
    # Nothing capturable (shouldn't happen) — get a real fast-dispatch handle via
    # compile_only rather than re-dispatching the slow normal path on every call.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return launcher(*args, compile_only=True)


class _FileCacheDispatch:
    """Callable wrapper used only under `FLASH_MODERNBERT_DSL_CACHE=1`.

    First call routes through CuteDSL's normal path (engages the persistent disk
    file cache + executes once + captures the compiled fn); subsequent calls use
    the captured fn's fast ~15 µs dispatch.
    """

    __slots__ = ("_launcher", "_fn")

    def __init__(self, launcher):
        self._launcher = launcher
        self._fn = None

    def __call__(self, *args):
        if self._fn is not None:
            return self._fn(*args)
        # First call: normal invocation engages the disk cache AND runs the kernel.
        self._fn = _normal_call_capturing(self._launcher, args)
        return None  # kernel already executed inside the capturing call


def current_cute_stream() -> _cuda_driver.CUstream:
    """Wrap the current PyTorch CUDA stream as a CuteDSL `CUstream`.

    Pass the result as the `stream` argument of every kernel launch so
    captures by `torch.cuda.graph(...)` actually record the kernel. Calling
    this from within a graph-capture region returns the captured stream.
    """
    return _cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
