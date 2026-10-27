"""Tokenize the arXiv shard and wikitext-103 test once, pack into fixed-length rows.

Both trainers read the resulting .npy files, so tokenization, packing and data
order are identical across frameworks. Self-contained (no project imports) so it
can run with `python -I`.

Outputs (data/):
  train.npy, val_arxiv.npy, val_wikitext.npy  uint32 [rows, SEQ_LEN + 1]
  meta.json                                    counts, sha256 of each array, paper ids per split

Compare the sha256 values in meta.json across machines to confirm identical training tokens.
"""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"
MODEL = ROOT / "models" / "Qwen3-0.6B-Base"

SEQ_LEN = 1024
ROW = SEQ_LEN + 1  # inputs + one extra token for the shifted targets
VAL_ROWS = 512  # ~525k tokens of held-out papers
TRAIN_TOKENS = 24_000_000  # headroom; the run length is chosen after the throughput probe
SEED = 0


def pack(token_lists, n_rows=None):
    flat = np.fromiter((t for toks in token_lists for t in toks), dtype=np.uint32)
    rows = len(flat) // ROW if n_rows is None else n_rows
    return flat[: rows * ROW].reshape(rows, ROW)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    eos = tok.eos_token_id

    with gzip.open(RAW / "arxiv-papers-0007.json.gz", "rt") as f:
        docs = [json.loads(line) for line in f]
    order = np.random.default_rng(SEED).permutation(len(docs))

    # Split by paper so no paper contributes to both train and val.
    val_toks, train_toks, val_ids, train_ids = [], [], [], []
    n_val = n_train = 0
    for start in range(0, len(order), 64):
        chunk = [docs[i] for i in order[start : start + 64]]
        ids = tok([d["text"] for d in chunk])["input_ids"]
        for d, t in zip(chunk, ids):
            t.append(eos)
            if n_val < VAL_ROWS * ROW:
                val_toks.append(t), val_ids.append(d["id"])
                n_val += len(t)
            elif n_train < TRAIN_TOKENS:
                train_toks.append(t), train_ids.append(d["id"])
                n_train += len(t)
        if n_train >= TRAIN_TOKENS:
            break

    val = pack(val_toks, VAL_ROWS)
    train = pack(train_toks)

    wiki = pq.read_table(RAW / "wikitext-103-test.parquet").column("text").to_pylist()
    wiki_toks = tok("".join(wiki))["input_ids"]
    val_wiki = pack([wiki_toks])

    arrays = {"train": train, "val_arxiv": val, "val_wikitext": val_wiki}
    for name, arr in arrays.items():
        np.save(OUT / f"{name}.npy", arr)
    meta = {
        "tokenizer": str(MODEL.name),
        "seq_len": SEQ_LEN,
        "row_len": ROW,
        "eos_token_id": eos,
        "rows": {k: len(a) for k, a in arrays.items()},
        "tokens": {k: int(a.size) for k, a in arrays.items()},
        "sha256": {k: sha256(a) for k, a in arrays.items()},
        "papers": {"train": train_ids, "val_arxiv": val_ids},
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("rows", "tokens", "sha256")}, indent=1))


def sha256(arr):
    """Checksum of the token values themselves (uint32, C order), independent of the .npy header."""
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.uint32).tobytes()).hexdigest()


if __name__ == "__main__":
    main()
