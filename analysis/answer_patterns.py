#!/usr/bin/env python
# Form of the students' TEST answers, over the whole test set (every problem x every sample of a val dump).
# Table 2 and Appendix I of the paper.
#
# For every run it reads ONE val dump (<step>.jsonl: input / output / acc / pred / gts, 16 samples per problem) and
# reports, per task, the mean answer length and the share of answers (%) that show each pattern. A pattern is a
# case-insensitive regex (or a small structural test) on the answer text; it counts ANSWERS, not occurrences.
# "Prose" is the reasoning: the <reasoning> block for chem/grading, the text before the last OPENING code fence for
# sql/lcb (an explanation written after the final program is not counted). Longer answers hit more patterns, so
# read every share next to the words column.
# The task-specific patterns mirror the aspect prompts d_1..d_K the multi-prompt teacher uses, so the table reads as
# "which aspects did the student pick up":
#   chem    : all_options (names all of A-D), rules_out (says why an option fails)
#   grading : checklist (walks numbered rubric elements), evidence (quotes the student answer), other_scores (says why
#             another mark does not apply)
#   sql     : schema (names >= 2 tables of the schema in the prose), join_keys, shape (GROUP BY / HAVING / ORDER BY /
#             LIMIT / aggregation planned in the prose), draft (more than one code block)
#   lcb     : no_prose (code with no text before it), io_contract, mentions_samples, boundary, multi_code (> 1 block)
#   all     : words, contrast (however/but/although...), verify (verify/double-check/re-examine...)
# It also counts FLIPS against the OPSD arm: problems OPSD gets right on <= 25 % of its samples and the other arm on
# >= 75 % (and the reverse).
#
#   usage (from the repo root):
#     python analysis/answer_patterns.py --outputs outputs --model Qwen3-4B [--step best|last|<int>] [--out patterns_Qwen3-4B]
#     explicit runs instead of discovery:  --runs chem:opsd=<dump dir> chem:multi=<dump dir> ...
#   writes <out>.json (all numbers) and <out>.md (one table per task); prints the tables.
# Runs are discovered as  <outputs>/<model>-<task>-<method>/  (the VAL_DIR of scripts/train.sh) with task in
# {grading, chem, sql, python} and method in {opsd, antisd, single, multi}; the step is best_step.txt if present, else the
# highest-numbered dump (override with --step).
import argparse, collections, glob, json, os, re, sys

TASKS = ["grading", "chem", "sql", "lcb"]
ARMS = [("sdpo", "OPSD"), ("antisd", "AntiSD"), ("single", "single"), ("multi", "multi")]
I = re.I
COMMON = {
    "contrast": re.compile(r"\b(however|but|although|even though|despite)\b", I),
    "verify": re.compile(r"\b(verif\w*|double[- ]check\w*|re-?examin\w*|re-?evaluat\w*|re-?check\w*|confirm\w*)\b", I),
}
RULES_OUT = re.compile(r"\b(rule[sd]? out|eliminat\w*|incorrect because|is wrong because|not correct because|does not (fit|match|apply))\b", I)
ALL_OPT = re.compile(r"\b([A-D])\)|option ([A-D])\b|\b([A-D])\s*(?:is|:)")
OTHER_SCORES = re.compile(r"\b(not worth|does not (meet|earn|qualify|warrant)|would (earn|be worth|receive)|cannot (be awarded|receive)|not (enough|sufficient) for)\b", I)
CHECKLIST = re.compile(r"(?m)^\s*(?:[-*]\s*)?(?:\*\*)?(?:\d+[.)]|element \d|point \d|criterion \d)", I)
QUOTED = re.compile(r"[\"“]([^\"”]{8,})[\"”]")
norm = lambda t: re.sub(r"\W+", " ", t.lower()).strip()


def student_answer(inp):
    """The [Student Answer] section of a grading prompt, normalised; a quote is evidence only if it comes from here."""
    m = re.search(r"\[Student Answer\]:\s*(.*?)(?:\nassistant\b|\Z)", inp, re.S)
    return norm(m.group(1)) if m else ""

JOIN = re.compile(r"\bforeign key\b|\bjoin(?:ed|ing|s)?\b[^.\n]{0,80}\b(on|via|through|using)\b|\blink(?:s|ed|ing)?\b[^.\n]{0,60}\b(table|key|column)", I)
SHAPE = re.compile(r"\b(group by|having|order by|limit \d+|aggregat\w*)\b", I)   # clause-level planning; bare NULL/COUNT/DISTINCT also match schema descriptions
IO = re.compile(r"\b(stdin|stdout|standard (input|output)|sys\.stdin|reads? (the )?input|prints? (the )?(output|result|answer))\b|\binput\(\)", I)
SAMPLE = re.compile(r"\b(sample (input|output|case)s?|example \d|walk(ing)? through|trac(e|ing) (the|through|each)|expected output)\b", I)
BOUNDARY = re.compile(r"\b(edge cases?|corner cases?|boundar(y|ies)|off[- ]by[- ]one|empty (input|list|array|string)|n\s*=\s*[01]\b|single[- ]element)\b", I)

def open_fences(out):
    """Start offsets of the OPENING code fences. Fences alternate open/close, so the even-numbered ``` open a block;
    a closing fence followed by text (39 % of the LiveCodeBench answers explain the code afterwards) must not be
    taken for a new block, nor for the start of the final answer."""
    return [m.start() for k, m in enumerate(re.finditer(r"```", out)) if k % 2 == 0]


def prose(task, out):
    """The reasoning part of an answer: inside <reasoning> for chem/grading, before the LAST code fence for sql/lcb."""
    if task in ("chem", "grading"):
        m = re.search(r"<reasoning>(.*?)</reasoning>", out, re.S)
        return (m.group(1) if m else out.split("<answer>")[0]).strip()
    f = open_fences(out)
    return out[: f[-1]].strip() if f else out.strip()


def schema_tables(inp):
    return {t.lower() for t in re.findall(r"CREATE TABLE\s+[`\"\[]?(\w+)", inp, I)}


def patterns(task, row):
    out = row["output"]; pr = prose(task, out); d = {k: bool(p.search(out)) for k, p in COMMON.items()}
    if task == "chem":
        d["all_options"] = len({x for g in ALL_OPT.findall(out) for x in g if x}) >= 4
        d["rules_out"] = bool(RULES_OUT.search(out))
    elif task == "grading":
        d["checklist"] = len(CHECKLIST.findall(pr)) >= 2
        sa = student_answer(row.get("input", ""))
        d["evidence"] = any(norm(q) and norm(q) in sa for q in QUOTED.findall(pr))
        d["other_scores"] = bool(OTHER_SCORES.search(out))
    elif task == "sql":
        tabs = schema_tables(row.get("input", "")); low = pr.lower()
        d["schema"] = sum(1 for t in tabs if re.search(r"\b" + re.escape(t) + r"\b", low)) >= 2
        d["join_keys"] = bool(JOIN.search(pr)); d["shape"] = bool(SHAPE.search(pr))
        d["draft"] = len(open_fences(out)) > 1
    elif task == "lcb":
        f = open_fences(out)
        d["no_prose"] = bool(f) and not out[: f[0]].strip()
        d["io_contract"] = bool(IO.search(pr)); d["mentions_samples"] = bool(SAMPLE.search(pr)); d["boundary"] = bool(BOUNDARY.search(pr))
        d["multi_code"] = len(f) > 1
    return len(pr.split()), len(out.split()), d


def pick_dump(d, step):
    js = sorted(int(os.path.basename(f)[:-6]) for f in glob.glob(f"{d}/*.jsonl") if os.path.basename(f)[:-6].isdigit())
    if not js: return None, None
    if step == "last": st = js[-1]
    elif step == "best":
        st = js[-1]
        if os.path.exists(f"{d}/best_step.txt"):
            try:
                b = int(open(f"{d}/best_step.txt").read().split()[0]); st = b if b in js and b > 0 else js[-1]
                if b == 0: print(f"[note] {d}: best step is 0 (no checkpoint beat the untrained policy); using the last dump, step {st}", file=sys.stderr)
            except ValueError: pass
    else: st = int(step)
    return (f"{d}/{st}.jsonl", st) if st in js else (None, None)


def analyse(task, path):
    by = collections.defaultdict(list)
    for line in open(path):
        if line.strip():
            r = json.loads(line); by[r.get("input", "")].append(r)
    rows = [r for v in by.values() for r in v]; n = len(rows); agg = collections.Counter(); wp = wo = 0
    for r in rows:
        a, b, d = patterns(task, r); wp += a; wo += b
        for k, v in d.items(): agg[k] += int(v)
    keys = list(patterns(task, rows[0])[2].keys())
    accof = lambda r: float(r.get("acc", r.get("score", r.get("reward", 0))) or 0)
    stats = dict(n_answers=n, n_problems=len(by), acc=100 * sum(accof(r) for r in rows) / n, words=wo / n, reasoning_words=wp / n,
                 **{k: 100 * agg[k] / n for k in keys})
    per_problem = {k: sum(accof(r) for r in v) / len(v) for k, v in by.items()}
    return stats, per_problem, keys


def main(a):
    runs = collections.defaultdict(dict)                              # task -> arm -> dump dir
    for spec in a.runs or []:
        ta, p = spec.split("=", 1); t, arm = ta.split(":")
        runs["lcb" if t == "python" else t]["sdpo" if arm == "opsd" else arm] = p
    if not a.runs:
        for t in TASKS:
            for arm, _ in ARMS:
                d = f"{a.outputs}/{a.model}-{'python' if t == 'lcb' else t}-{'opsd' if arm == 'sdpo' else arm}"
                if os.path.isdir(d): runs[t][arm] = d
    res = {}; md = [f"# Answer patterns, {a.model}: share of test answers (%) showing each pattern\n"]
    for t in TASKS:
        if t not in runs: continue
        per = {}; table = []; keys = None
        for arm, lab in ARMS:
            d = runs[t].get(arm)
            if not d: continue
            path, st = pick_dump(d, a.step)
            if not path: print(f"[skip] no dump for {t}/{arm} in {d}", file=sys.stderr); continue
            stats, pp, keys = analyse(t, path); stats["step"] = st; stats["dump"] = path
            per[arm] = pp; res[f"{t}|{arm}"] = stats; table.append((lab, stats))
        if not table: continue
        cols = ["acc", "words", "reasoning_words"] + keys
        md.append(f"\n## {t}  ({table[0][1]['n_problems']} problems x {table[0][1]['n_answers'] // max(1, table[0][1]['n_problems'])} samples)\n")
        md.append("| student | step | " + " | ".join(cols) + " |"); md.append("|---|---|" + "---|" * len(cols))
        for lab, s in table:
            md.append(f"| {lab} | {s['step']} | " + " | ".join(f"{s[c]:.0f}" if c.endswith("words") else f"{s[c]:.1f}" for c in cols) + " |")
        if "sdpo" in per:                                             # flips against OPSD, per problem
            for arm in ("single", "multi"):
                if arm not in per: continue
                common = set(per["sdpo"]) & set(per[arm])
                up = sum(1 for k in common if per["sdpo"][k] <= 0.25 and per[arm][k] >= 0.75)
                down = sum(1 for k in common if per[arm][k] <= 0.25 and per["sdpo"][k] >= 0.75)
                res[f"{t}|{arm}"]["flips_vs_opsd"] = dict(opsd_wrong_ours_right=up, ours_wrong_opsd_right=down, problems=len(common))
                md.append(f"\nflips vs OPSD, {arm}: OPSD wrong (<=25%) and {arm} right (>=75%) on {up} of {len(common)} problems; reverse {down}.")
    text = "\n".join(md); print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out + ".json", "w"), indent=1); open(a.out + ".md", "w").write(text + "\n")
        print(f"\nwrote {a.out}.json and {a.out}.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outputs", default="outputs"); p.add_argument("--model", default="Qwen3-4B")
    p.add_argument("--step", default="best", help="best (best_step.txt, else last) | last | <int>")
    p.add_argument("--runs", nargs="*", help="explicit <task>:<arm>=<dump dir> pairs instead of discovery")
    p.add_argument("--out", default="")
    main(p.parse_args())
