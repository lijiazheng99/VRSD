#!/usr/bin/env python
"""Build datasets/ordinal_scoring/asap/{train,test}.parquet from ASAP-SAS essay sets 1, 2, 5 and 6 (score range 0-3).

Input: the ASAP-SAS files from Kaggle (https://www.kaggle.com/c/asap-sas, accept the competition rules and download
the data). Train and validation essays come from train.tsv (score: Score1), test essays from the public leaderboard
essays (public_leaderboard*.tsv) with their scores in public_leaderboard_solution.csv. The data are not redistributed
here. Alternatively, pass one jsonl file with one row per essay: {"Id": int, "Mark": int, "Response": str}.
The split of each set is given by the essay Ids in data/asap/split_ids.json; before writing, the essay Ids and scores
(and, as a warning only, the essay texts) are checked against data/asap/checksums.json, so a successful run
reproduces the data of the paper. Question and rubric texts: data/asap/prompts.json.
Train: all train.jsonl rows (5229). Test: 500 rows drawn with random.Random(42).sample from the pooled
validation + test rows of the four sets.

    python data/prepare_asap.py --kaggle_dir <dir with the Kaggle files>
    python data/prepare_asap.py --essays asap_essays.jsonl
"""
import argparse, glob, hashlib, json, os, random, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from data.format.ordinal_scoring import SYSTEM_PROMPT, _clean_rubric, _build_user_prompt
from data.preprocess import write_rowgrouped_large
import datasets

SETS = ["asap_1", "asap_2", "asap_5", "asap_6"]


def rows_of(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def read_table(path):
    import pandas as pd
    sep = "," if path.endswith(".csv") else "\t"
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, sep=sep, quoting=3 if sep == "\t" else 0, encoding=enc, dtype=str)
            return df.rename(columns={c: c.strip().lower() for c in df.columns})
        except UnicodeDecodeError:
            continue


def col(df, path, *names):
    for n in names:
        if n in df.columns: return n
    sys.exit(f"{path}: none of the columns {names} found (columns: {list(df.columns)})")


def rows_of_kaggle(d):
    def one(pattern):
        hits = sorted(glob.glob(os.path.join(d, pattern)))
        if not hits: sys.exit(f"no file matching {pattern} in {d}")
        return hits[0]
    rows = []
    tr = one("train*.tsv"); df = read_table(tr)
    for _, r in df.iterrows():
        rows.append({"Id": int(r[col(df, tr, "id")]), "Mark": int(r[col(df, tr, "score1")]),
                     "Response": r[col(df, tr, "essaytext")]})
    ess, sol = one("public_leaderboard*.tsv"), one("public_leaderboard_solution*.csv")
    de, ds = read_table(ess), read_table(sol)
    score = dict(zip(ds[col(ds, sol, "id")].astype(int), ds[col(ds, sol, "essay_score", "score", "score1")].astype(int)))
    for _, r in de.iterrows():
        i = int(r[col(de, ess, "id")])
        if i in score: rows.append({"Id": i, "Mark": score[i], "Response": r[col(de, ess, "essaytext")]})
    return rows


def verify(by_id, split_ids):
    sums = json.load(open(os.path.join(ROOT, "data/asap/checksums.json")))
    h = lambda lines: hashlib.sha256("\n".join(lines).encode()).hexdigest()
    for s in SETS:
        ids = sorted(i for p in ("train", "validation", "test") for i in split_ids[s][p])
        miss = [i for i in ids if i not in by_id]
        if miss: sys.exit(f"{s}: {len(miss)} essay Ids of the split are missing from the input (e.g. {miss[:5]})")
        if h(f"{i}\t{int(by_id[i]['Mark'])}" for i in ids) != sums[s]["id_mark_sha256"]:
            sys.exit(f"{s}: essay scores differ from the data of the paper (checksum mismatch)")
        if h(str(by_id[i]["Response"]).replace("\n", " ").strip() for i in ids) != sums[s]["response_sha256"]:
            print(f"warning: {s}: essay texts differ from the data of the paper (e.g. encoding); scores match")


def record(src, set_id, question, rubric, split, index):
    user = _build_user_prompt(question, rubric, src["Response"])
    return {"system": SYSTEM_PROMPT,
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "data_source": "asap_binary", "ability": "ordinal_score",
            "reward_model": {"style": "ordinal_score", "ground_truth": str(int(src["Mark"]))},
            "extra_info": {"split": split, "index": str(index), "description": question, "problem": user,
                           "elo": 1500, "achievement_prior": 0, "prompt_id": set_id,
                           "score_max": 3, "source_id": str(src["Id"])}}


def main(essays, kaggle_dir, out):
    by_id = {int(r["Id"]): r for r in (rows_of(essays) if essays else rows_of_kaggle(kaggle_dir))}
    split_ids = json.load(open(os.path.join(ROOT, "data/asap/split_ids.json")))
    verify(by_id, split_ids)
    train, pool = [], []
    for s in SETS:
        meta = json.load(open(os.path.join(ROOT, "data/asap/prompts.json")))[s]
        q, rub = meta["question"], _clean_rubric(meta["rubric"], s)
        tr = [by_id[i] for i in split_ids[s]["train"]]
        ev = [by_id[i] for i in split_ids[s]["validation"] + split_ids[s]["test"]]
        train += [record(r, s, q, rub, "train", i) for i, r in enumerate(tr)]
        pool += [record(r, s, q, rub, "test", len(tr) + i) for i, r in enumerate(ev)]
    test = random.Random(42).sample(pool, 500)
    os.makedirs(out, exist_ok=True)
    for name, rows in (("train", train), ("test", test)):
        write_rowgrouped_large(datasets.Dataset.from_list(rows), os.path.join(out, f"{name}.parquet"))
        print(name, len(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--kaggle_dir", help="directory with the Kaggle ASAP-SAS files")
    g.add_argument("--essays", help="jsonl with Id, Mark, Response per essay")
    p.add_argument("--out", default=os.path.join(ROOT, "datasets/ordinal_scoring/asap"))
    a = p.parse_args(); main(a.essays, a.kaggle_dir, a.out)
