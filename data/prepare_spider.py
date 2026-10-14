#!/usr/bin/env python
# Prepare Spider (xlangai/spider) for text-to-SQL RL. Reward = execution accuracy (run predicted SQL vs gold on the
# question's sqlite DB, compare result sets) via feedback/sql.py.
#   data_source="spider"; prompt = schema (CREATE TABLE) + NL question; reward_model.ground_truth = gold SQL;
#   extra_info.db_id -> $SPIDER_DB_DIR/<db_id>/<db_id>.sqlite (resolved in the reward).
# Schema is extracted from each sqlite (CREATE TABLE statements) since the HF text fields have no schema.
#
# Usage: python data/prepare_spider.py --n_train 3000 --n_test 200 --seed 42
import argparse, os, random, sqlite3
import pyarrow as pa, pyarrow.parquet as pq
from datasets import load_dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.environ.get("SPIDER_OUT") or f"{ROOT}/datasets/text2sql/spider"
DB_DIR = os.environ.get("SPIDER_DB_DIR", f"{ROOT}/datasets/text2sql/spider/database")

PROMPT = """You are an expert data analyst. Given a database schema and a question, write a single \
SQLite query that answers it.

Database schema:
{schema}

Question: {question}

Reason step by step, then give ONLY the final query inside a ```sql code block."""

_schema_cache = {}


def get_schema(db_id: str) -> str:
    if db_id in _schema_cache:
        return _schema_cache[db_id]
    path = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")
    con = sqlite3.connect(path)
    con.text_factory = lambda b: b.decode(errors="ignore")
    rows = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
    ).fetchall()
    con.close()
    schema = "\n".join(r[0].strip() for r in rows)
    _schema_cache[db_id] = schema
    return schema


def build_rows(examples, split):
    out = []
    for i, ex in enumerate(examples):
        db_id = ex["db_id"]
        try:
            schema = get_schema(db_id)
        except Exception:
            continue
        if not schema:
            continue
        out.append({
            "data_source": "spider",
            "prompt": [{"role": "user", "content": PROMPT.format(schema=schema, question=ex["question"])}],
            "ability": "text2sql",
            "reward_model": {"ground_truth": ex["query"], "style": "spider", "gold_solution": f"```sql\n{ex['query']}\n```"},
            "extra_info": {"index": int(i), "split": split, "db_id": db_id},
        })
    return out


def main(n_train, n_test, seed):
    train = list(load_dataset("xlangai/spider", split="train"))
    val = list(load_dataset("xlangai/spider", split="validation"))
    random.Random(seed).shuffle(train)
    # test comes from the held-out dev (validation) set — DBs differ from train, the real generalization test
    random.Random(seed).shuffle(val)
    test_ex = val[:n_test] if n_test else val
    train_ex = train[:n_train] if n_train else train
    tr = build_rows(train_ex, "train")
    te = build_rows(test_ex, "test")
    os.makedirs(OUT, exist_ok=True)
    for name, rows in (("train", tr), ("test", te)):
        pq.write_table(pa.Table.from_pylist(rows), f"{OUT}/{name}.parquet")
        print(f"wrote {OUT}/{name}.parquet  rows={len(rows)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=3000)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.n_train, a.n_test, a.seed)
