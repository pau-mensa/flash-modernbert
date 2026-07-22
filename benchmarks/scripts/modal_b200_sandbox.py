"""Reusable Modal B200 sandbox for packed-encoders development and benchmarks.

Workflow:
    python benchmarks/scripts/modal_b200_sandbox.py up
    python benchmarks/scripts/modal_b200_sandbox.py sync
    python benchmarks/scripts/modal_b200_sandbox.py run -- \
        python -c "import torch; print(torch.cuda.get_device_name(0))"
    python benchmarks/scripts/modal_b200_sandbox.py tests
    python benchmarks/scripts/modal_b200_sandbox.py pull REMOTE LOCAL
    python benchmarks/scripts/modal_b200_sandbox.py shell
    python benchmarks/scripts/modal_b200_sandbox.py status
    python benchmarks/scripts/modal_b200_sandbox.py down

Iteration loop is `edit locally → sync → run`. No image rebuild per edit.

Install and authenticate the Modal CLI before the first run. The Modal workspace must
have access to the GPU selected by ``PACKED_ENCODERS_MODAL_GPU`` (B200 by default).
"""

from __future__ import annotations

import base64
import io
import os
import queue
import shlex
import shutil
import sys
import tarfile
import threading
from pathlib import Path

import modal

APP_NAME = os.environ.get(
    "PACKED_ENCODERS_MODAL_APP", "packed-encoders-b200-sandbox"
)
REMOTE_WORKDIR = "/workspace/packed-encoders"
GPU = os.environ.get("PACKED_ENCODERS_MODAL_GPU", "B200")
TIMEOUT_S = int(os.environ.get("PACKED_ENCODERS_MODAL_TIMEOUT", "10800"))

LOCAL_ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = LOCAL_ROOT / ".cache" / "packed_encoders_modal_b200_sandbox_id"

# Persistent CuteDSL compile cache. The Modal Volume survives sandbox teardown,
# avoiding repeated ptxas compilation across benchmark sessions.
CACHE_VOLUME_NAME = os.environ.get(
    "PACKED_ENCODERS_DSL_CACHE_VOLUME", "packed-encoders-dsl-cache"
)
CACHE_MOUNT = "/cache"


def _cache_volume() -> "modal.Volume":
    return modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)


# Repository paths synchronized into the sandbox. Generated data and results are
# intentionally excluded; use ``pull`` for artifacts that should be retained.
SYNC_PATHS = [
    "packed_encoders",
    "benchmarks",
    "tests",
    "pyproject.toml",
    "README.md",
]
EXCLUDE_NAMES = {
    ".git",
    ".venv",
    ".cache",
    "__pycache__",
    ".pytest_cache",
    "data_cache",
    "output",
    "results",
}
EXCLUDE_SUFFIXES = {".pyc", ".so", ".pyd"}


# ---------------------------------------------------------------------------
# Reference B200 environment used by the public training showcase.
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("git", "build-essential", "curl", "ca-certificates")
    .pip_install(
        "torch==2.11.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "nvidia-cutlass-dsl==4.5.2",
        "numpy",
        "einops",
        "pytest",
        "transformers>=4.45",
        "sentence-transformers==5.3.0",
        "scipy",
        "datasets",
        "safetensors",
    )
    .env({"PYTHONUNBUFFERED": "1", "CUTLASS_LOG_LEVEL": "WARN"})
)


def _app():
    return modal.App.lookup(APP_NAME, create_if_missing=True)


def _get_sandbox() -> modal.Sandbox | None:
    if not STATE_FILE.exists():
        return None
    sb_id = STATE_FILE.read_text().strip()
    try:
        sb = modal.Sandbox.from_id(sb_id)
    except Exception:
        STATE_FILE.unlink(missing_ok=True)
        return None
    if sb.poll() is not None:
        STATE_FILE.unlink(missing_ok=True)
        return None
    return sb


def _require_sandbox() -> modal.Sandbox:
    sb = _get_sandbox()
    if sb is None:
        print(
            "No running sandbox. Run "
            "`python benchmarks/scripts/modal_b200_sandbox.py up` first.",
            file=sys.stderr,
        )
        sys.exit(2)
    return sb


def _stream(proc) -> int:
    """Stream a ContainerProcess's stdout+stderr to our terminal, return exit code."""
    q: queue.Queue[tuple[object, object]] = queue.Queue()

    def pump(src, dst) -> None:
        try:
            for line in src:
                q.put((dst, line))
        finally:
            q.put((dst, None))

    threads = [
        threading.Thread(target=pump, args=(proc.stdout, sys.stdout), daemon=True),
        threading.Thread(target=pump, args=(proc.stderr, sys.stderr), daemon=True),
    ]
    for thread in threads:
        thread.start()

    done = 0
    while done < len(threads):
        dst, line = q.get()
        if line is None:
            done += 1
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        dst.write(line)
        dst.flush()

    return proc.wait()


def cmd_pull(argv: list[str]) -> None:
    """Download a file or directory tree from the sandbox to local.

    Usage: pull <remote_path> <local_path>
    """
    if len(argv) != 2:
        print("Usage: ... pull <remote_path> <local_path>", file=sys.stderr)
        sys.exit(2)
    remote, local = argv
    sb = _require_sandbox()
    # Tar remote into stdout, write tar.gz to a local file, then extract.
    local_path = Path(local).resolve()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tar_local = local_path.with_suffix(local_path.suffix + ".tar.gz")
    print(f"Pulling {remote} -> {local_path} (via gzipped tar) ...")
    # `-C parent_of_remote` and a single basename → stable archive layout
    remote_path = Path(remote)
    parent = str(remote_path.parent) or "/"
    base = remote_path.name
    # Modal decodes stdout as utf-8 text; binary gzipped tar trips that.
    # Wrap via base64 to round-trip cleanly.
    proc = sb.exec(
        "bash",
        "-lc",
        f"tar -czf - -C {shlex.quote(parent)} {shlex.quote(base)} | base64",
    )
    b64_buf = []
    for chunk in proc.stdout:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("ascii")
        b64_buf.append(chunk)
    rc = proc.wait()
    for line in proc.stderr:
        sys.stderr.write(line)
    if rc != 0:
        print(f"remote tar|base64 failed with exit {rc}", file=sys.stderr)
        sys.exit(rc)
    b64_text = "".join(b64_buf).replace("\n", "").replace("\r", "")
    binary = base64.b64decode(b64_text)
    n_bytes = len(binary)
    with open(tar_local, "wb") as f:
        f.write(binary)
    print(f"  wrote {n_bytes / 1024:.1f} KB to {tar_local}")
    # Extract under local_path's parent so archive's basename = local target.
    with tarfile.open(tar_local, "r:gz") as t:
        t.extractall(local_path.parent)
    extracted = local_path.parent / base
    if extracted != local_path:
        if local_path.exists():
            if local_path.is_dir():
                shutil.rmtree(local_path)
            else:
                local_path.unlink()
        extracted.rename(local_path)
    tar_local.unlink(missing_ok=True)
    print(f"  extracted to {local_path}")


def cmd_up() -> None:
    sb = _get_sandbox()
    if sb is not None:
        print(f"Sandbox already running: {sb.object_id}")
        return
    print(
        f"Creating {GPU} sandbox (image build is one-time, ~3-5 min; subsequent boots ~30 s)..."
    )
    sb = modal.Sandbox.create(
        "sleep",
        "infinity",
        image=image,
        gpu=GPU,
        timeout=TIMEOUT_S,
        app=_app(),
        workdir="/workspace",
        volumes={CACHE_MOUNT: _cache_volume()},
    )
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(sb.object_id)
    print(f"Sandbox up: {sb.object_id}  (timeout {TIMEOUT_S}s, GPU {GPU})")

    # Verify the env. Anything wrong here points at image drift before you waste time syncing.
    proc = sb.exec(
        "python",
        "-c",
        "import torch, cutlass; "
        "p = torch.cuda.get_device_properties(0); "
        "print(f'{p.name} | cc={p.major}.{p.minor} | SMs={p.multi_processor_count} | "
        "mem={p.total_memory/1e9:.1f} GB | torch={torch.__version__} cuda={torch.version.cuda} | "
        "cutlass={cutlass.__version__}', flush=True)",
    )
    _stream(proc)


def cmd_sync() -> None:
    sb = _require_sandbox()

    # Pack into an in-memory tar.gz, upload as a single file via sb.open(),
    # extract in a second exec. Avoids stdin-piping into `tar -x -` which
    # has been observed to hang waiting on stream EOF semantics.
    buf = io.BytesIO()
    n_files = 0
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=1) as tar:
        for rel in SYNC_PATHS:
            src = LOCAL_ROOT / rel
            if not src.exists():
                raise FileNotFoundError(f"configured sync path does not exist: {src}")
            if src.is_file():
                tar.add(src, arcname=rel)
                n_files += 1
                continue
            for path in src.rglob("*"):
                if not path.is_file():
                    continue
                if any(part in EXCLUDE_NAMES for part in path.parts):
                    continue
                if path.suffix in EXCLUDE_SUFFIXES:
                    continue
                arcname = path.relative_to(LOCAL_ROOT).as_posix()
                tar.add(path, arcname=arcname)
                n_files += 1
    payload = buf.getvalue()
    print(
        f"Syncing {n_files} files ({len(payload) / 1024:.0f} KB gzipped) -> {REMOTE_WORKDIR} ..."
    )

    sb.exec("mkdir", "-p", REMOTE_WORKDIR).wait()
    remote_tar = f"{REMOTE_WORKDIR}/.packed_encoders_sync.tar.gz"
    with sb.open(remote_tar, "wb") as f:
        f.write(payload)
    rc = sb.exec("tar", "-xzf", remote_tar, "-C", REMOTE_WORKDIR).wait()
    sb.exec("rm", "-f", remote_tar).wait()
    if rc != 0:
        print(f"tar extract failed with exit {rc}", file=sys.stderr)
        sys.exit(rc)
    print("Sync done.")


def cmd_run(argv: list[str]) -> None:
    sb = _require_sandbox()
    if not argv:
        print("Usage: ... run -- <cmd> [args...]", file=sys.stderr)
        sys.exit(2)
    # Wrap in bash -lc and prepend the workdir to PYTHONPATH so the synced
    # source tree is importable without `pip install -e .` (which we skip
    # to keep iteration fast — sync is rsync-cheap, install is not).
    #
    # Enable the repository's opt-in persistent CuteDSL cache by default. Both
    # variables remain overridable in the command environment.
    full = " ".join(shlex.quote(a) for a in argv)
    print(f"$ {full}", flush=True)
    workdir_q = shlex.quote(REMOTE_WORKDIR)
    proc = sb.exec(
        "bash",
        "-lc",
        f"cd {workdir_q} && "
        f"export PYTHONPATH={workdir_q}:${{PYTHONPATH:-}} && "
        f"export PACKED_ENCODERS_DSL_CACHE=${{PACKED_ENCODERS_DSL_CACHE:-1}} && "
        f"export CUTE_DSL_CACHE_DIR=${{CUTE_DSL_CACHE_DIR:-{CACHE_MOUNT}/packed_encoders_dsl_cache}} && "
        f"{full}",
    )
    sys.exit(_stream(proc))


def cmd_tests() -> None:
    cmd_run(["python", "-m", "pytest", "tests/", "-x", "-q"])


def cmd_shell() -> None:
    sb = _require_sandbox()
    print(f"Attaching to {sb.object_id} ...  (exit with `exit` or Ctrl-D)")
    os.execvp("modal", ["modal", "container", "exec", sb.object_id, "bash"])


def cmd_down() -> None:
    sb = _get_sandbox()
    if sb is None:
        print("No running sandbox.")
        STATE_FILE.unlink(missing_ok=True)
        return
    print(f"Terminating {sb.object_id} ...")
    sb.terminate()
    STATE_FILE.unlink(missing_ok=True)


def cmd_status() -> None:
    sb = _get_sandbox()
    if sb is None:
        print("No running sandbox.")
        return
    print(
        f"Running: {sb.object_id}  (GPU={GPU}, timeout={TIMEOUT_S}s, workdir={REMOTE_WORKDIR})"
    )


COMMANDS = {
    "up": cmd_up,
    "sync": cmd_sync,
    "tests": cmd_tests,
    "shell": cmd_shell,
    "down": cmd_down,
    "status": cmd_status,
    # pull is variadic; handled separately in main() below
}


def main(argv: list[str]) -> None:
    if len(argv) < 2 or argv[1] in {"-h", "--help"}:
        print(__doc__)
        return
    action = argv[1]
    if action == "run":
        rest = argv[2:]
        if rest and rest[0] == "--":
            rest = rest[1:]
        cmd_run(rest)
        return
    if action == "pull":
        rest = argv[2:]
        if rest and rest[0] == "--":
            rest = rest[1:]
        cmd_pull(rest)
        return
    fn = COMMANDS.get(action)
    if fn is None:
        print(f"Unknown command: {action}\n", file=sys.stderr)
        print(__doc__)
        sys.exit(2)
    fn()


if __name__ == "__main__":
    main(sys.argv)
