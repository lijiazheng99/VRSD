"""Text-to-SQL execution reward (Spider / BIRD).

Open-ended, execution-verifiable — the SQL analog of feedback/code.py: run the model's
predicted SQL against the question's sqlite DB and compare its RESULT SET to the gold
query's result set. Reward = 1 iff the result sets match, 0 on mismatch / SQL error /
timeout / malformed output. This is Spider's "execution accuracy (EX)".

Result-set comparison semantics mirror the official test-suite-sql-eval (taoyds) exec_eval:
  - bag/multiset equality (duplicate rows count),
  - column order is permutation-invariant,
  - row order matters ONLY if the gold query contains `order by`.

DB lookup: extra_info["db_id"] -> $SPIDER_DB_DIR/<db_id>/<db_id>.sqlite (BIRD: point
SPIDER_DB_DIR at dev_databases). The gold SQL is passed as ground_truth (it also becomes
the teacher's verified-answer privilege under SDPO gold-teacher mode).
"""
import os
import re
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

DB_DIR = os.environ.get("SPIDER_DB_DIR", "datasets/text2sql/spider/database")
SQL_TIMEOUT = float(os.environ.get("SQL_REWARD_TIMEOUT", "15"))
# Hard row cap on any result set. A legitimate Spider/BIRD answer is orders of magnitude
# smaller; an unbounded cross join is not, and fetchall() would materialise it inside the
# training process. Anything above the cap is treated as a failed query, not a match.
SQL_MAX_ROWS = int(os.environ.get("SQL_REWARD_MAX_ROWS", "100000"))
# sqlite calls the progress handler every N VM instructions; that callback is the ONLY way
# to abort a running query from outside the C call.
_PROGRESS_EVERY = 10000


def _extract_sql(text: str) -> str:
    """Pull the SQL out of the model output: prefer a ```sql fenced block, then any ```
    block, then an <answer> tag, else the last SELECT/WITH statement in the text."""
    if not text:
        return ""
    m = re.findall(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m[-1].strip()
    m = re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m[-1].strip()
    m = re.findall(r"<answer>\s*(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m[-1].strip()
    # fallback: last SELECT/WITH ... statement
    m = re.findall(r"((?:SELECT|WITH)\b.*)", text, re.DOTALL | re.IGNORECASE)
    return (m[-1].strip() if m else text.strip())


def _db_path(db_id: str) -> str:
    return os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")


def _run(db_path: str, sql: str):
    """Execute one query read-only with a REAL wall-clock timeout. Returns (ok, rows_or_errmsg).

    A generated query can be an unbounded cross join, and this runs inside the main Ray/vLLM
    process, so two things have to hold or one bad rollout takes the whole training run down:

      * the timeout must actually interrupt sqlite. `fut.result(timeout=...)` does NOT --
        it only abandons the future, and `with ThreadPoolExecutor(...)` then blocks in
        shutdown(wait=True) until the runaway query finishes on its own. The progress
        handler aborts it from inside instead, so the worker thread really does end.
      * the result must be bounded. fetchall() materialises the full cartesian product.
    """
    box = {}

    def _q():
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        con.text_factory = lambda b: b.decode(errors="ignore")
        deadline = time.monotonic() + SQL_TIMEOUT
        con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, _PROGRESS_EVERY)
        try:
            rows = con.execute(sql).fetchmany(SQL_MAX_ROWS + 1)
            if len(rows) > SQL_MAX_ROWS:
                raise RuntimeError(f"result exceeds {SQL_MAX_ROWS} rows")
            box["rows"] = rows
        finally:
            con.close()

    ex = ThreadPoolExecutor(max_workers=1)
    try:
        ex.submit(_q).result(timeout=SQL_TIMEOUT + 5)   # backstop; the handler should fire first
        return True, box["rows"]
    except Exception as e:  # timeout, sql error, row cap, decode error, ...
        return False, f"{type(e).__name__}: {e}"[:300]
    finally:
        ex.shutdown(wait=False)   # never block the training process on a runaway thread


def _multiset(rows):
    return Counter(tuple(r) for r in rows)


def _cols_permutation_eq(pred, gold):
    """Column-order-invariant multiset equality: try to match gold's columns to a
    permutation of pred's. Cheap check: same #cols and same per-column value multiset."""
    if not pred and not gold:
        return True
    if not pred or not gold:
        return False
    if len(pred[0]) != len(gold[0]):
        return False
    # transpose to per-column value multisets and compare as multisets of columns
    pred_cols = Counter(tuple(sorted((str(r[i]) for r in pred))) for i in range(len(pred[0])))
    gold_cols = Counter(tuple(sorted((str(r[i]) for r in gold))) for i in range(len(gold[0])))
    return pred_cols == gold_cols


def _result_eq(pred, gold, order_matters):
    if order_matters:
        return list(pred) == list(gold)
    if _multiset(pred) == _multiset(gold):
        return True
    # column-permutation-invariant fallback (matches exec_eval spirit)
    return _cols_permutation_eq(pred, gold)


def compute_score(solution: str, ground_truth: str, extra_info: dict = None) -> dict:
    extra_info = extra_info or {}
    db_id = extra_info.get("db_id", "")
    pred_sql = _extract_sql(solution)
    db_path = _db_path(db_id)

    fmt_ok = bool(re.search(r"```sql", solution or "", re.IGNORECASE))

    if not db_id or not os.path.exists(db_path):
        return {"score": 0.0, "acc": 0.0, "pred": pred_sql[:200],
                "incorrect_format": 0 if fmt_ok else 1,
                "feedback": f"DB not found for db_id={db_id!r}"}
    if not pred_sql:
        return {"score": 0.0, "acc": 0.0, "pred": "",
                "incorrect_format": 1, "feedback": "No SQL found in the output."}

    gold_ok, gold_rows = _run(db_path, ground_truth)
    if not gold_ok:  # gold must run; if not, we can't score -> treat as 0 but flag
        return {"score": 0.0, "acc": 0.0, "pred": pred_sql[:200],
                "incorrect_format": 0 if fmt_ok else 1,
                "feedback": f"GOLD query failed: {gold_rows}"}

    pred_ok, pred_rows = _run(db_path, pred_sql)
    if not pred_ok:
        return {"score": 0.0, "acc": 0.0, "pred": pred_sql[:200],
                "incorrect_format": 0 if fmt_ok else 1,
                "feedback": f"Execution error: {pred_rows}"}

    order_matters = "order by" in (ground_truth or "").lower()
    correct = _result_eq(pred_rows, gold_rows, order_matters)
    fb = "" if correct else (
        f"Wrong result set. Predicted {len(pred_rows)} row(s), gold has {len(gold_rows)}."
    )
    return {"score": float(correct), "acc": float(correct), "pred": pred_sql[:200],
            "incorrect_format": 0 if fmt_ok else 1, "feedback": fb}
