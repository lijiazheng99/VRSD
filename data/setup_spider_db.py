#!/usr/bin/env python
"""Fetch the Spider sqlite databases the text-to-SQL execution reward needs, and lay them out at a repo-local
path so the sql task runs without a manual download.

Source: HuggingFace dataset `HAL-9001/spider-databases` (spider_data.zip, 196 MB compressed → ~839 MB of
sqlite). Verified to cover every db_id in datasets/text2sql/spider/{train,test}.parquet (train 140/140,
test 19/19). The extracted databases go under datasets/text2sql/spider/database/ (git-ignored); the sql
reward reads SPIDER_DB_DIR, which scripts/train.sh defaults to exactly that path.

    python data/setup_spider_db.py [target_dir]   # default: <repo>/datasets/text2sql/spider/database
Idempotent: if the target already has the databases it exits immediately. Set SKIP_SPIDER_DB=1 to no-op.
"""
import os, sys, zipfile, glob, shutil
if {"-h", "--help"} & set(sys.argv[1:]): print(__doc__); sys.exit(0)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(REPO, "datasets/text2sql/spider/database")
REPO_ID = "HAL-9001/spider-databases"
ZIP = "spider_data.zip"


def n_dbs(d):
    return len(glob.glob(os.path.join(d, "*", "*.sqlite")))


def main():
    if os.environ.get("SKIP_SPIDER_DB"):
        print("[spider-db] SKIP_SPIDER_DB set — skipping.")
        return
    if n_dbs(TARGET) > 100:
        print(f"[spider-db] already present ({n_dbs(TARGET)} databases at {TARGET}) — skipping.")
        return
    print(f"[spider-db] downloading {REPO_ID}/{ZIP} (~196 MB) via HuggingFace ...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("[spider-db] huggingface_hub not installed — run the install step first, or `pip install huggingface_hub`.")
    zpath = hf_hub_download(REPO_ID, ZIP, repo_type="dataset")

    os.makedirs(TARGET, exist_ok=True)
    print(f"[spider-db] extracting the database/ folder to {TARGET} ...")
    with zipfile.ZipFile(zpath) as zf:
        for name in zf.namelist():
            if "__MACOSX" in name or name.endswith("/"):
                continue
            # keep only spider_data/database/<db_id>/... , strip the "spider_data/database/" prefix
            key = "spider_data/database/"
            i = name.find(key)
            if i == -1:
                continue
            rel = name[i + len(key):]
            if not rel:
                continue
            dst = os.path.join(TARGET, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with zf.open(name) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)
    got = n_dbs(TARGET)
    if got < 100:
        sys.exit(f"[spider-db] ERROR: only {got} databases extracted to {TARGET} — expected ~166.")
    print(f"[spider-db] done — {got} databases at {TARGET}")


if __name__ == "__main__":
    main()
