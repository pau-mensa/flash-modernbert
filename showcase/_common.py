"""Shared helpers for the flash-modernbert showcase scripts (Workstream F).

Kept dependency-light: yaml config loading, the AgentIR passage-corpus loader
(F2's real long-document corpus), a token-length profiler, and small plotting
utilities. The heavy measurement cores are reused from `benchmarks/` rather than
duplicated here.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
SHOWCASE_MODEL = "lightonai/GTE-ModernColBERT-v1"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def device_banner() -> dict:
    import torch

    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"device: {name}  sm_{cap[0]}{cap[1]}  torch {torch.__version__}")
    return {"device": name, "capability": list(cap), "torch": torch.__version__}


# ---------------------------------------------------------------------------
# AgentIR passage corpus (F2) — the model's own long-document workload
# ---------------------------------------------------------------------------


def load_agentir_passages(n_docs: int, seed: int = 0, streaming: bool = True) -> list[str]:
    """Return ~n_docs unique passage texts from Tevatron/AgentIR-data.

    Each row carries one positive + several negative passages; we pool them and
    de-duplicate by docid so the corpus is a realistic mix of long web documents
    (the workload Agent-ModernColBERT was built to index). Streaming avoids
    pulling the whole dataset for a downsampled smoke run.
    """
    import itertools

    from datasets import load_dataset

    ds = load_dataset("Tevatron/AgentIR-data", split="train", streaming=streaming)
    seen: dict[str, str] = {}
    # pull enough rows to cover n_docs unique passages (≈8 passages/row)
    for row in itertools.islice(ds, max(1, n_docs // 6 + 16)):
        for key in ("positive_passages", "negative_passages"):
            for p in row.get(key) or []:
                docid = p.get("docid") or p.get("text", "")[:64]
                if docid not in seen and p.get("text"):
                    seen[docid] = p["text"]
        if len(seen) >= n_docs:
            break
    docs = list(seen.values())[:n_docs]
    if not docs:
        raise RuntimeError("AgentIR passage corpus came back empty")
    return docs


def load_agentir_queries(
    n_queries: int,
    *,
    instruction: str = "",
    streaming: bool = True,
) -> list[str]:
    """Return the first unique non-empty AgentIR queries.

    Kept separate from ``load_agentir_passages`` because inference configs often
    use hundreds of documents but only a small serving batch of queries.
    """
    from datasets import load_dataset

    ds = load_dataset("Tevatron/AgentIR-data", split="train", streaming=streaming)
    queries: list[str] = []
    seen: set[str] = set()
    for row in ds:
        query = row.get("query", "").strip()
        if query and query not in seen:
            queries.append(instruction + query)
            seen.add(query)
        if len(queries) >= n_queries:
            break
    if not queries:
        raise RuntimeError("AgentIR query corpus came back empty")
    return queries


def load_agentir_groups(
    n_groups: int,
    *,
    num_negatives: int,
    instruction: str = "",
    streaming: bool = True,
) -> list[dict]:
    """Load query/positive/negative groups for the packed training showcase."""
    from datasets import load_dataset

    ds = load_dataset("Tevatron/AgentIR-data", split="train", streaming=streaming)
    groups = []
    for row in ds:
        positives = row.get("positive_passages") or []
        negatives = row.get("negative_passages") or []
        if not row.get("query") or not positives or (num_negatives and not negatives):
            continue
        groups.append(
            {
                "query": instruction + row["query"],
                "positive": positives[0]["text"],
                "negatives": [
                    negatives[i % len(negatives)]["text"] for i in range(num_negatives)
                ],
            }
        )
        if len(groups) >= n_groups:
            break
    if len(groups) < n_groups:
        raise RuntimeError(f"AgentIR returned only {len(groups)} usable groups")
    return groups


def profile_token_lengths(texts: list[str], tokenizer, max_length: int) -> dict:
    """Tokenize (no padding) and summarize the real token-length distribution —
    the data behind F2's padding-waste visual."""
    lengths = []
    for t in texts:
        ids = tokenizer(t, truncation=True, max_length=max_length,
                        add_special_tokens=True)["input_ids"]
        lengths.append(len(ids))
    lengths.sort()
    return {
        "lengths": lengths,
        "n": len(lengths),
        "min": lengths[0],
        "max": lengths[-1],
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def ema(values, span: int = 10):
    if not values:
        return values
    alpha = 2.0 / (span + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def new_figure(*args, **kwargs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt, plt.subplots(*args, **kwargs)
