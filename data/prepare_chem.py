#!/usr/bin/env python
"""Build datasets/sciknoweval/chemistry_all/{train,test}.parquet: SciKnowEval chemistry, levels L1-L4, multiple
choice only (6263 items), split 90/10 with seed 42 (5636 train / 627 test).

    python data/prepare_chem.py
"""
import os, sys
if {"-h", "--help"} & set(sys.argv[1:]): print(__doc__); sys.exit(0)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from data.format.sciknoweval import load_sciknoweval
from data.preprocess import make_map_fn, write_rowgrouped_large

OUT = os.path.join(ROOT, "datasets/sciknoweval/chemistry_all")

if __name__ == "__main__":
    ds = load_sciknoweval(domains=["Chemistry"], levels=["L1", "L2", "L3", "L4"],
                          types=["mcq-4-choices", "mcq-2-choices"])
    ds = ds.add_column("idx", list(range(len(ds))))
    split = ds.train_test_split(test_size=0.1, seed=42)
    os.makedirs(OUT, exist_ok=True)
    for name in ("train", "test"):
        rows = split[name].map(make_map_fn(name), with_indices=True)
        write_rowgrouped_large(rows, os.path.join(OUT, f"{name}.parquet"))
        print(name, len(rows))
