#!/usr/bin/env python
"""Build datasets/code/lcb_v6/{train,test}.parquet, the LiveCodeBench v6 setup of the SDPO paper.

Reproduces the SDPO data pipeline in one pass:
  1. data/format/code.py :: load_code("livecodebench/code_generation_lite-v6")
        = load_livecodebench(dataset_split="test", until=2025-05-01)
        -> problems with contest_date in [2025-02-01, 2025-05-01)
  2. test-case split (PERCENTAGE_TO_KEEP=0.5, np.random.seed(0))
        test.json  = ALL problems, 100% of the private test cases
        train.json = THE SAME problems, a random 50% of each problem's test cases
  3. data/preprocess.py  -> verl parquet rows (incl. the livecodebench-only
        "(self, " -> "(" signature fix)

Train and test hold the same problems: train keeps a random 50% of each problem's test cases, test keeps all of them,
as in the SDPO setup.

Usage: python data/prepare_lcb_v6.py
"""
import os, sys, json, copy
if {"-h", "--help"} & set(sys.argv[1:]): print(__doc__); sys.exit(0)
from datetime import datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, datasets
from data.utils.livecodebench import load_livecodebench
from data.preprocess import write_rowgrouped_large
from data.format.prompts import CODE_PROMPT

OUT = os.path.join(ROOT, "datasets", "code", "lcb_v6")
PERCENTAGE_TO_KEEP = 0.5          # test-case split
UNTIL = datetime(2025, 5, 1)      # data/format/code.py :: load_code

def sample_tests(tests_json):
    """Byte-for-byte the SDPO test-case split."""
    tests = json.loads(tests_json)
    inputs, outputs = tests["inputs"], tests["outputs"]
    num_tests = len(inputs)
    keep_count = max(1, int(num_tests * PERCENTAGE_TO_KEEP))
    keep_indices = np.sort(np.random.choice(num_tests, size=keep_count, replace=False))
    reduced = copy.deepcopy(tests)
    reduced["inputs"]  = [inputs[i] for i in keep_indices]
    reduced["outputs"] = [outputs[i] for i in keep_indices]
    return json.dumps(reduced, ensure_ascii=False)

def to_rows(ds, split, reduce_tests):
    rows = []
    for i, ex in enumerate(ds):
        problem = ex["problem"]
        description = ex.get("description", "")
        # data/preprocess.py: livecodebench-only method-signature fix
        problem = problem.replace("(self, ", "(")
        description = description.replace("(self, ", "(")
        tests = sample_tests(ex["tests"]) if reduce_tests else ex["tests"]
        rows.append({
            "data_source": "livecodebench",
            "prompt": [{"role": "user", "content": CODE_PROMPT.format(problem=problem)}],
            "ability": "code",
            "reward_model": {"style": "code", "ground_truth": tests},
            "extra_info": {"split": split, "index": i, "description": description, "problem": problem},
        })
    return rows

def write(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_rowgrouped_large(datasets.Dataset.from_list(rows), path)
    print(f"  {len(rows):5d} rows -> {os.path.relpath(path, ROOT)}")

def main():
    print(f"== lcb_v6 replica: contest_date in [2025-02-01, {UNTIL:%Y-%m-%d}), tests split {PERCENTAGE_TO_KEEP}")
    ds = load_livecodebench("test", until=UNTIL)
    print(f"   problems: {ds.num_rows}  (SAME set in train and test, by design)")
    n_all = [len(json.loads(e["tests"])["inputs"]) for e in ds]
    # test = full tests (written first, as in the SDPO pipeline)
    write(to_rows(ds, "test", reduce_tests=False), os.path.join(OUT, "test.parquet"))
    # train = same problems, 50% of tests; seed 0 immediately before the reduction, as in the SDPO pipeline
    np.random.seed(0)
    tr = to_rows(ds, "train", reduce_tests=True)
    write(tr, os.path.join(OUT, "train.parquet"))
    n_red = [len(json.loads(r["reward_model"]["ground_truth"])["inputs"]) for r in tr]
    print(f"== mean tests per item: {np.mean(n_all):.2f} -> {np.mean(n_red):.2f}   (train={len(tr)}, test={ds.num_rows})")

if __name__ == "__main__":
    main()
