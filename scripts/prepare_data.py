"""Tokenize the arXiv shard and wikitext-103 test once, pack into fixed-length rows.

Both trainers read the resulting .npy files, so tokenization, packing and data
order are identical across frameworks. Self-contained (no project imports) so it
can run with `python -I`.

Outputs (data/):
  train.npy, val_arxiv.npy, val_wikitext.npy  uint32 [rows, SEQ_LEN + 1]
  meta.json                                    counts and the paper ids in each split
"""

import gzip
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

    np.save(OUT / "train.npy", train)
    np.save(OUT / "val_arxiv.npy", val)
    np.save(OUT / "val_wikitext.npy", val_wiki)
    meta = {
        "tokenizer": str(MODEL.name),
        "seq_len": SEQ_LEN,
        "row_len": ROW,
        "eos_token_id": eos,
        "rows": {"train": len(train), "val_arxiv": len(val), "val_wikitext": len(val_wiki)},
        "tokens": {"train": int(train.size), "val_arxiv": int(val.size), "val_wikitext": int(val_wiki.size)},
        "papers": {"train": train_ids, "val_arxiv": val_ids},
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("rows", "tokens")}))


if __name__ == "__main__":
    main()
