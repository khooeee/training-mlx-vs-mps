"""Perplexity of the base model and every converted checkpoint, all in mlx-lm (bf16).

Same harness, same tokens and same dtype for every model, so differences come from training only.
"""

import json
import math

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
import numpy as np

import common as C

MODELS = {"base": C.BASE_MODEL} | {
    run: C.CKPT / f"{run}-mlx" for run in C.ALL_RUNS
}
BATCH = 8


def mean_nll(model, rows):
    total, count = 0.0, 0
    for i in range(0, len(rows), BATCH):
        b = np.asarray(rows[i : i + BATCH])
        x, y = mx.array(b[:, :-1].astype(np.int32)), mx.array(b[:, 1:].astype(np.int32))
        nll = nn.losses.cross_entropy(model(x).astype(mx.float32), y, reduction="sum")
        total += nll.item()
        count += y.size
    return total / count


def main():
    splits = {s: C.load_split(s) for s in ("val_arxiv", "val_wikitext")}
    results = {}
    for name, path in MODELS.items():
        if not path.exists():
            continue
        model, _ = mlx_lm.load(str(path))
        results[name] = {}
        for split, rows in splits.items():
            nll = mean_nll(model, rows)
            results[name][split] = {"nll": nll, "ppl": math.exp(nll)}
        print(name, json.dumps(results[name]), flush=True)
    (C.RUNS / "eval.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
