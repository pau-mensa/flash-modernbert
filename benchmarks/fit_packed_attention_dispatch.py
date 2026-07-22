"""Find compact monotonic packed Triton/Flash score candidates.

The runtime-visible feature set intentionally excludes values read from CUDA
``cu_seqlens``: eager and captured paths both know N, M, and Smax from host-visible
shapes/arguments. Products such as ``N*Smax`` and its difference from M are allowed.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


FEATURES = (
    "live_tokens",
    "n_sequences",
    "max_seqlen",
    "rectangle_pair_work",
    "mean_square_work",
    "live_max_work",
    "query_ctas",
)


def feature(row: dict, name: str) -> int:
    n = int(row["n_sequences"])
    m = int(row["live_tokens"])
    s = int(row["max_seqlen"])
    if name in row:
        return int(row[name])
    if name == "rectangle_pair_work":
        return n * s * s
    if name == "mean_square_work":
        return m * m // n
    if name == "live_max_work":
        return m * s
    if name == "query_ctas":
        block_m = 16 if s <= 64 else 64
        return n * ((s + block_m - 1) // block_m)
    raise KeyError(name)


def _key(row: dict) -> tuple[object, ...]:
    return row["profile"], row["n_sequences"], row["max_seqlen"]


def load_rows(coarse: Path, refinements: list[Path]) -> list[dict]:
    rows = {_key(row): row for row in json.loads(coarse.read_text())["rows"]}
    for refine in refinements:
        if refine.exists():
            rows.update({_key(row): row for row in json.loads(refine.read_text())["rows"]})
    return list(rows.values())


def fit_subset(rows: list[dict], fields: tuple[str, ...]) -> dict | None:
    raw = np.asarray(
        [[feature(row, field) for field in fields] for row in rows], dtype=float
    )
    center = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale[scale == 0] = 1
    x = (raw - center) / scale
    y = np.asarray(
        [1.0 if row["triton_time_over_flash_time"] >= 1.03 else -1.0 for row in rows]
    )
    # Variables: standardized weights, intercept, margin. Bound weights to fix scale
    # and maximize the geometric separation available to this feature subset.
    constraints = []
    for point, label in zip(x, y):
        constraints.append([*(-label * point), -label, 1.0])
    objective = np.zeros(len(fields) + 2)
    objective[-1] = -1.0
    result = linprog(
        objective,
        A_ub=np.asarray(constraints),
        b_ub=np.zeros(len(rows)),
        bounds=[(-1.0, 1.0)] * len(fields) + [(None, None), (0.0, None)],
        method="highs",
    )
    if not result.success or result.x[-1] <= 1e-9:
        return None
    standardized_weights = result.x[: len(fields)]
    weights = standardized_weights / scale
    intercept = result.x[-2] - float(np.dot(weights, center))
    scores = raw @ weights
    wins = scores[y > 0]
    losses = scores[y < 0]
    min_win, max_loss = float(wins.min()), float(losses.max())
    return {
        "fields": fields,
        "weights": weights.tolist(),
        "intercept": intercept,
        "threshold": (min_win + max_loss) / 2,
        "gap": min_win - max_loss,
        "standardized_margin": float(result.x[-1]),
        "min_win_score": min_win,
        "max_loss_score": max_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("--refine", type=Path, action="append", default=[])
    parser.add_argument("--max-features", type=int, default=4)
    args = parser.parse_args()
    rows = load_rows(args.coarse, args.refine)
    candidates = []
    for size in range(1, args.max_features + 1):
        for fields in itertools.combinations(FEATURES, size):
            candidate = fit_subset(rows, fields)
            if candidate is not None:
                candidates.append(candidate)
    candidates.sort(
        key=lambda value: (len(value["fields"]), -value["standardized_margin"])
    )
    print(json.dumps({
        "rows": len(rows),
        "production_flash_wins": sum(
            row["triton_time_over_flash_time"] >= 1.03 for row in rows
        ),
        "clean_candidates": len(candidates),
        "candidates": candidates[:20],
    }, indent=2))


if __name__ == "__main__":
    main()
