"""Download the model and data at pinned revisions (idempotent).

  models/Qwen3-0.6B-Base/               Qwen/Qwen3-0.6B-Base
  data/raw/arxiv-papers-0007.json.gz    common-pile/arxiv_papers_filtered, smallest shard (~211 MB, ~10k papers)
  data/raw/wikitext-103-test.parquet    Salesforce/wikitext, wikitext-103-raw-v1 test split
"""

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

MODEL = ("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
FILES = {  # local name -> (dataset repo, path in repo, revision)
    "arxiv-papers-0007.json.gz": (
        "common-pile/arxiv_papers_filtered", "arxiv-papers-0007.json.gz", "033cf7f53f9b348deec868c1a5a48484f3ee9e52"),
    "wikitext-103-test.parquet": (
        "Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet", "b08601e04326c79dfdd32d625aee71d232d685c3"),
}


def main():
    repo, rev = MODEL
    snapshot_download(repo, revision=rev, local_dir=ROOT / "models" / repo.split("/")[1])
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (repo, path, rev) in FILES.items():
        if not (RAW / name).exists():
            shutil.copy(hf_hub_download(repo, path, revision=rev, repo_type="dataset"), RAW / name)
        print(f"{name}: {(RAW / name).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
