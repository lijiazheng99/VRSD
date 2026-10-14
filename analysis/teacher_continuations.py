#!/usr/bin/env python
"""Qualitative teacher comparison for the paper: the SAME base model continues the SAME student prefix under
  bare      : question only                                   (what the student sees)
  c         : question + privileged solution c                (the OPSD teacher)
  c+d       : question + c + the single prompt d               (VRSD single teacher)
  c+d_k     : question + c + one aspect prompt d_k             (VRSD multi teachers)
Prefixes are the student's own rollouts (training temperature) cut after a fraction f of their reasoning, so every
condition continues from an identical partial answer. Everything is saved verbatim (continuations per problem,
prefix and condition) plus pattern statistics per condition and prefix fraction:
  continuation length (tokens), reasoning tokens before the answer tag, share of continuations that mention
  >= 3 options / rule an option out (chem) or state why other scores do not apply (grading), share containing a
  contrastive marker, share whose answer is correct, share whose FIRST sentence already names the final answer.
Teacher construction is teacher_probe.py's (privileged sibling from the student's own successful rollouts, the
production solution template, REFLECT_POSITION=after_privileged).
  python analysis/teacher_continuations.py --task chem --model Qwen/Qwen3-4B --limit 120 --out out.json
"""
import argparse, json, os, re, sys
ROOT = os.environ.get("SDPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "analysis"))
from teacher_probe import TASKS, SOLUTION_TEMPLATE, THINK, score_fn
RP = "prompts"   # relative to ROOT; chem d3 / grading d4 use the first versions of the adversarial prompts (c3_counter, a4_counter), as in the paper
CONDS = {"chem":    [("bare", None), ("c", ""), ("c+d", f"{RP}/single/chem.txt"), ("c+d1", f"{RP}/aspects/chem/c1_eliminate.txt"),
                     ("c+d2", f"{RP}/aspects/chem/c2_mechanism.txt"), ("c+d3", f"{RP}/aspects/chem/c3_counter.txt")],
         "grading": [("bare", None), ("c", ""), ("c+d", f"{RP}/single/grading.txt"), ("c+d1", f"{RP}/aspects/grading/a1_rubric.txt"),
                     ("c+d2", f"{RP}/aspects/grading/a2_calibration.txt"), ("c+d3", f"{RP}/aspects/grading/a3_evidence.txt"),
                     ("c+d4", f"{RP}/aspects/grading/a4_counter.txt")],
         "sql":     [("bare", None), ("c", ""), ("c+d", f"{RP}/single/sql.txt"), ("c+d1", f"{RP}/aspects/sql/s1_schema.txt"),
                     ("c+d2", f"{RP}/aspects/sql/s2_join.txt"), ("c+d3", f"{RP}/aspects/sql/s3_shape.txt"), ("c+d4", f"{RP}/aspects/sql/s4_verify.txt")],
         "lcb":     [("bare", None), ("c", ""), ("c+d", f"{RP}/single/python.txt"), ("c+d1", f"{RP}/aspects/python/k_contract.txt"),
                     ("c+d2", f"{RP}/aspects/python/k_minimal.txt"), ("c+d3", f"{RP}/aspects/python/k_boundary.txt"), ("c+d4", f"{RP}/aspects/python/k_trace.txt")]}
TASKS = dict(TASKS, lcb=("datasets/code/lcb_v6", 8192))
CONTRAST = re.compile(r"\b(but|however|although|though|yet|instead|rather|whereas|nonetheless|actually|wait)\b", re.I)
RULEOUT = re.compile(r"\b(rule[sd]? out|eliminat|incorrect because|is wrong because|not correct because|does not (fit|match|apply|hold))\b", re.I)
CALIB = re.compile(r"\b(not worth|does not (meet|earn|qualify|warrant)|would (earn|be worth|receive)|cannot (be awarded|receive)|not (enough|sufficient) for|other scores?)\b", re.I)
SENT = re.compile(r"[.!?]+(?:\s|$)|\n")
def reasoning_part(text, task="chem"):
    if task in ("sql", "lcb"):
        m = list(re.finditer(r"```[a-zA-Z]*\n", text))
        return text[: m[-1].start()] if m else text
    m = re.search(r"<reasoning>(.*?)</reasoning>", text, re.S)
    return m.group(1) if m else text.split("<answer>")[0]
def scorer_for(task):
    if task == "lcb":
        from verl.utils.reward_score.feedback.code import compute_score
        return lambda text, gt, extra: compute_score(text, gt, dict(extra or {}, split=(extra or {}).get("split", "test")))
    if task == "sql":
        from verl.utils.reward_score.feedback.sql import compute_score
        return lambda text, gt, extra: compute_score(text, gt, extra)
    return score_fn(task)
def cut(tok, ids, frac):
    if frac <= 0: return []
    c = int(round(frac * len(ids)))
    if c >= len(ids): c = len(ids)
    txt = tok.decode(ids[:c]); k = max(txt.rfind("\n"), txt.rfind(". "))
    if k > len(txt) // 2: txt = txt[:k + 1]
    return tok(txt, add_special_tokens=False)["input_ids"]
def stats(task, text, gt, scorer, extra):
    r = scorer(text, gt, extra); acc = float(r.get("acc", r.get("score", 0.0))) if isinstance(r, dict) else float(r)
    pred = str(r.get("pred", "")) if isinstance(r, dict) else ""
    reas = reasoning_part(text, task); first = SENT.split(reas.strip(), 1)[0] if reas.strip() else ""
    commit_first = bool(pred) and (re.search(rf"\b{re.escape(pred)}\b", first) is not None) if task == "chem" else \
                   bool(pred) and (re.search(rf"\b{re.escape(pred)}\s*(points?|/|out of)", first, re.I) is not None)
    d = dict(acc=acc, pred=pred, reasoning_words=len(reas.split()), contrast=bool(CONTRAST.search(reas)), commit_first_sentence=commit_first)
    if task == "chem":
        d["options_mentioned"] = len(set(re.findall(r"(?:option|\()\s*([A-D])\b|\b([A-D])\)\s", reas)) or set(re.findall(r"\b(?:Option|option)\s+([A-D])", reas)))
        d["rules_out"] = bool(RULEOUT.search(reas))
    elif task == "grading":
        d["other_scores"] = bool(CALIB.search(reas))
    else:
        d["verifies"] = bool(re.search(r"\b(verify|verif|check(ing)?|double[- ]check|edge case|boundary|sample (input|case)|test(ing)? (it|this|the)|trace|walk through)\b", reas, re.I))
        d["mentions_schema"] = bool(re.search(r"\b(schema|table|column|join|foreign key|primary key|constraint|input format|stdin)\b", reas, re.I))
    return d
def main(a):
    import pyarrow.parquet as pq
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    datarel, cap = TASKS[a.task]
    rows = pq.read_table(f"{ROOT}/{datarel}/test.parquet").to_pylist()[: a.limit]
    tok = AutoTokenizer.from_pretrained(a.model); scorer = scorer_for(a.task)
    conds = [(n, (open(os.path.join(ROOT, p)).read().strip() if p else p)) for n, p in CONDS[a.task]]
    llm = LLM(model=a.model, gpu_memory_utilization=a.gmem, max_model_len=cap + 8192, enable_prefix_caching=True, disable_log_stats=True)
    if a.task == "lcb": os.environ.setdefault("CODE_REWARD_POOL_WORKERS", "16")
    msgs_all, stu_prompts = [], []
    for r in rows:
        m = [dict(x) for x in r["prompt"]] if isinstance(r.get("prompt"), list) else [{"role": "user", "content": str(r["prompt"])}]
        msgs_all.append(m); stu_prompts.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    stu = llm.generate(stu_prompts, SamplingParams(n=a.n_roll, temperature=1.0, top_p=1.0, max_tokens=cap, seed=a.seed))
    items = []   # problems with a sibling AND a prefix rollout distinct from the sibling
    for i, (r, g) in enumerate(zip(rows, stu)):
        gt = r["reward_model"]["ground_truth"]; extra = r.get("extra_info"); scored = []
        for o in g.outputs:
            s = scorer(o.text, gt, extra); scored.append((float(s.get("acc", s.get("score", 0.0))) if isinstance(s, dict) else float(s), o))
        sib = next((o for acc, o in scored if acc >= 0.5), None)
        if sib is None: continue
        pre = next((o for acc, o in scored if o is not sib), None)   # prefix source: another rollout (any score)
        if pre is None: continue
        items.append(dict(i=i, gt=gt, extra=extra, sibling=THINK.sub("", sib.text).strip(), prefix_full=pre.text, prefix_acc=[acc for acc, o in scored if o is pre][0]))
    print(f"[stage 1] {len(rows)} problems, {len(items)} with sibling + prefix", flush=True)
    fracs = [float(x) for x in a.fracs.split(",")]
    out = dict(model=a.model, task=a.task, n_items=len(items), fracs=fracs, conditions=[n for n, _ in conds], items=[], summary={})
    reqs, meta = [], []
    for it in items:
        base = [dict(x) for x in msgs_all[it["i"]]]; sys_msgs, user = base[:-1], base[-1]["content"]
        reas_ids = tok(reasoning_part(it["prefix_full"], a.task), add_special_tokens=False)["input_ids"]
        for f in fracs:
            pre_ids = cut(tok, reas_ids, f)
            open_ids = tok("<reasoning>\n", add_special_tokens=False)["input_ids"] if (pre_ids and a.task in ("chem", "grading")) else []
            for name, disp in conds:
                if disp is None: msg = user
                else:
                    msg = user + SOLUTION_TEMPLATE.format(successful_previous_attempt=it["sibling"])
                    if disp: msg = msg.rstrip() + "\n\n" + disp
                p_ids = tok(tok.apply_chat_template(sys_msgs + [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]
                reqs.append(dict(prompt_token_ids=p_ids + open_ids + pre_ids)); meta.append((it, f, name, tok.decode(open_ids + pre_ids)))
    print(f"[stage 2] {len(reqs)} teacher continuations", flush=True)
    gens = llm.generate(reqs, SamplingParams(n=1, temperature=a.teacher_temp, top_p=1.0, max_tokens=cap, seed=a.seed))
    by = {}
    for (it, f, name, pretxt), g in zip(meta, gens):
        cont = g.outputs[0].text; full = pretxt + cont
        st = stats(a.task, full, it["gt"], scorer, it["extra"]); st["cont_tokens"] = len(g.outputs[0].token_ids)
        by.setdefault(it["i"], dict(i=it["i"], gt=it["gt"], question=msgs_all[it["i"]][-1]["content"], sibling=it["sibling"], prefix_acc=it["prefix_acc"], runs=[]))
        by[it["i"]]["runs"].append(dict(frac=f, cond=name, prefix=pretxt, continuation=cont, **st))
    out["items"] = list(by.values())
    keys = ["acc", "cont_tokens", "reasoning_words", "contrast", "commit_first_sentence"] + ({"chem": ["options_mentioned", "rules_out"], "grading": ["other_scores"]}.get(a.task, ["verifies", "mentions_schema"]))
    for f in fracs:
        for name, _ in conds:
            rs = [r for it in out["items"] for r in it["runs"] if r["frac"] == f and r["cond"] == name]
            if not rs: continue
            out["summary"][f"f={f}|{name}"] = {k: round(sum(float(r[k]) for r in rs) / len(rs), 3) for k in keys} | {"n": len(rs)}
    print(f"{'f':>4} {'cond':6} " + " ".join(f"{k[:14]:>14}" for k in keys))
    for k2, v in out["summary"].items():
        f, name = k2.split("|"); print(f"{f[2:]:>4} {name:6} " + " ".join(f"{v[k]:14.3f}" for k in keys), flush=True)
    json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out)
if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-4B"); p.add_argument("--task", required=True, type=lambda t: "lcb" if t == "python" else t, choices=list(CONDS))
    p.add_argument("--limit", type=int, default=120); p.add_argument("--n_roll", type=int, default=8); p.add_argument("--fracs", default="0,0.5")
    p.add_argument("--teacher_temp", type=float, default=1.0); p.add_argument("--gmem", type=float, default=0.85); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True); main(p.parse_args())
