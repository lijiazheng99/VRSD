#!/usr/bin/env python
"""TEACHER-side probe: reproduce the EXACT self-distillation teacher condition and measure what the
disposition prompt does to it -- accuracy AND per-token entropy -- without training anything.

In SDPO the disposition never conditions the student. It conditions
the TEACHER, which additionally sees the privileged context (a verified sibling solution). The
multi-teacher claim is about the shape of that teacher distribution, so it has to be measured there.

Reproduces ray_trainer._maybe_build_self_distillation_batch exactly:
  * privileged source = sibling: a rollout of the SAME problem that scored >= success_reward_threshold,
    excluding the sample itself (dont_reprompt_on_self_success=True), thinking trace removed;
  * reprompt = "{prompt}{solution}{feedback}" with
        solution_template = "\n\nCorrect solution:\n\n{successful_previous_attempt}\n"
        feedback_template = "\n\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n"
  * REFLECT_POSITION=after_privileged (the training default): the disposition is appended to the END of
    the user reprompt, right before the teacher generates; "system" prepends it to the system message;
  * same sampling as training rollouts (temperature 1.0, top_p 1.0) unless overridden.

Procedure: (1) sample n_roll student rollouts per problem at the training temperature to obtain real
siblings and the real success mask, (2) rebuild each teacher message under each disposition condition,
(3) sample the teacher once per message and score it, logging per-token entropy from top-K logprobs.

  python analysis/teacher_probe.py --model Qwen/Qwen3-4B --task chem --limit 120 --n_roll 8 \
     --conds bare: single:prompts/single/chem.txt
"""
import argparse, json, math, os, re, sys
ROOT = os.environ.get("SDPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

TASKS = {"chem": ("datasets/sciknoweval/chemistry_all", 2048),
         "grading": ("datasets/ordinal_scoring/asap", 4096),
         "sql": ("datasets/text2sql/spider", 2048),
         "python": ("datasets/code/lcb_v6", 8192)}

SOLUTION_TEMPLATE = "\n\nCorrect solution:\n\n{successful_previous_attempt}\n"
FEEDBACK_TEMPLATE = "\n\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n"

THINK = re.compile(r"<think>.*?</think>\s*", re.S)

def score_fn(task):
    if task == "chem":
        from verl.utils.reward_score.feedback.mcq import compute_score
        return lambda text, gt, extra: compute_score(text, gt)
    if task == "grading":
        from verl.utils.reward_score.feedback.ordinal_score import compute_score
        return lambda text, gt, extra: compute_score(text, gt, extra)
    if task == "sql":
        from verl.utils.reward_score.feedback.sql import compute_score
        return lambda text, gt, extra: compute_score(text, gt, extra)
    if task == "python":
        from verl.utils.reward_score.feedback.code import compute_score
        return lambda text, gt, extra: compute_score(text, gt, dict(extra or {}, split=(extra or {}).get("split", "test")))
    if task == "code":
        from verl.utils.reward_score.feedback.code import compute_score
        return lambda text, gt, extra: compute_score(text, gt, extra)
    if task == "math":
        from verl.utils.reward_score.feedback.math import compute_score
        return lambda text, gt, extra: compute_score(text, gt, extra or {"split": "train"})
    raise SystemExit(f"no scorer wired for {task}")

def trunc_entropy(logprob_dict):
    """Shannon entropy (nats) from vLLM's top-K logprobs, unresolved tail as one bucket."""
    if not logprob_dict:
        return None
    lps = [v.logprob for v in logprob_dict.values()]
    ps = [math.exp(x) for x in lps]
    s = sum(ps)
    h = -sum(p * math.log(max(p, 1e-12)) for p in ps)
    tail = max(0.0, 1.0 - s)
    if tail > 1e-9:
        h += -tail * math.log(tail)
    return h

def main(a):
    import pyarrow.parquet as pq
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    datarel, cap = TASKS[a.task]
    rows = pq.read_table(f"{ROOT}/{datarel}/test.parquet").to_pylist()[: a.limit]
    tok = AutoTokenizer.from_pretrained(a.model)
    scorer = score_fn(a.task)
    conds = []
    for c in a.conds:
        name, _, path = c.partition(":")
        txt = ""
        if path:
            p = path if os.path.isabs(path) else f"{ROOT}/{path}"
            txt = open(p).read().strip()
        conds.append((name, txt))

    llm = LLM(model=a.model, gpu_memory_utilization=a.gmem, max_model_len=cap + 8192,
              enable_prefix_caching=True, disable_log_stats=True)

    # ---- stage 1: student rollouts at the TRAINING temperature -> real siblings + success mask ----
    stu_prompts, msgs_all = [], []
    for r in rows:
        m = [dict(x) for x in r["prompt"]] if isinstance(r.get("prompt"), list) else [{"role": "user", "content": str(r["prompt"])}]
        msgs_all.append(m)
        stu_prompts.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    sp_stu = SamplingParams(n=a.n_roll, temperature=a.train_temp, top_p=a.train_top_p, max_tokens=cap)
    print(f"[stage 1] {len(rows)} problems x {a.n_roll} student rollouts @ T={a.train_temp}", flush=True)
    stu = llm.generate(stu_prompts, sp_stu)

    siblings, stu_acc = [], []
    for r, g in zip(rows, stu):
        gt = r["reward_model"]["ground_truth"]
        best, accs = None, []
        for o in g.outputs:
            s = scorer(o.text, gt, r.get("extra_info"))
            acc = float(s.get("acc", s.get("score", 0.0))) if isinstance(s, dict) else float(s)
            accs.append(acc)
            if best is None and acc >= a.success_threshold:
                best = THINK.sub("", o.text).strip()   # remove_thinking_from_demonstration=True
        siblings.append(best)
        stu_acc.append(sum(accs) / len(accs))
    n_priv = sum(1 for s in siblings if s)
    print(f"[stage 1] student acc {sum(stu_acc)/len(stu_acc):.4f} | problems with a usable sibling: "
          f"{n_priv}/{len(rows)} ({100*n_priv/len(rows):.0f}%)", flush=True)

    # ---- stage 2+3: teacher under each disposition ----
    out = {"model": a.model, "task": a.task, "n_problems": len(rows), "n_roll": a.n_roll,
           "reflect_position": a.reflect_position, "teacher_temp": a.teacher_temp,
           "student_acc_mean": round(sum(stu_acc)/len(stu_acc), 4),
           "problems_with_sibling": n_priv, "conditions": {}}
    idxs = [i for i, s in enumerate(siblings) if s]     # teachers only exist where a sibling exists
    sp_tea = SamplingParams(n=1, temperature=a.teacher_temp, top_p=a.teacher_top_p,
                            max_tokens=cap, logprobs=a.topk)
    for name, disp in conds:
        prompts = []
        for i in idxs:
            base_msgs = [dict(x) for x in msgs_all[i]]
            sys_msgs, user = base_msgs[:-1], base_msgs[-1]["content"]
            reprompt = user + SOLUTION_TEMPLATE.format(successful_previous_attempt=siblings[i])
            if disp:
                if a.reflect_position == "after_privileged":
                    reprompt = reprompt.rstrip() + "\n\n" + disp
                else:
                    if sys_msgs and sys_msgs[0].get("role") == "system":
                        sys_msgs = [{"role": "system", "content": disp + "\n\n" + sys_msgs[0]["content"]}] + sys_msgs[1:]
                    else:
                        sys_msgs = [{"role": "system", "content": disp}] + sys_msgs
            prompts.append(tok.apply_chat_template(sys_msgs + [{"role": "user", "content": reprompt}],
                                                   tokenize=False, add_generation_prompt=True, enable_thinking=False))
        gens = llm.generate(prompts, sp_tea)
        accs, ents, lens, trunc = [], [], [], 0
        for i, g in zip(idxs, gens):
            gt = rows[i]["reward_model"]["ground_truth"]
            o = g.outputs[0]
            s = scorer(o.text, gt, rows[i].get("extra_info"))
            accs.append(float(s.get("acc", s.get("score", 0.0))) if isinstance(s, dict) else float(s))
            lens.append(len(o.token_ids))
            if len(o.token_ids) >= cap:
                trunc += 1
            if o.logprobs:
                per = [trunc_entropy(d) for d in o.logprobs if d]
                per = [x for x in per if x is not None]
                if per:
                    ents.append(sum(per) / len(per))
        out["conditions"][name] = {
            "prompt_words": len(disp.split()),
            "teacher_acc": round(sum(accs)/len(accs), 4),
            "teacher_entropy_nats": round(sum(ents)/len(ents), 4) if ents else None,
            "mean_len_tokens": round(sum(lens)/len(lens), 1),
            "truncated_pct": round(100*trunc/len(idxs), 2),
            "n_scored": len(idxs),
        }
        c = out["conditions"][name]
        print(f"[{name:9s}] teacher_acc {c['teacher_acc']:.4f} | entropy {c['teacher_entropy_nats']} nats "
              f"| len {c['mean_len_tokens']:.0f} tok | trunc {c['truncated_pct']:.1f}%", flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("wrote", a.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", default="chem", choices=list(TASKS))
    p.add_argument("--conds", nargs="+", required=True)
    p.add_argument("--limit", type=int, default=120)
    p.add_argument("--n_roll", type=int, default=8, help="student rollouts per problem (training default 8)")
    p.add_argument("--train_temp", type=float, default=1.0, help="training rollout temperature")
    p.add_argument("--train_top_p", type=float, default=1.0)
    p.add_argument("--teacher_temp", type=float, default=1.0, help="teacher forward runs at the training temperature")
    p.add_argument("--teacher_top_p", type=float, default=1.0)
    p.add_argument("--success_threshold", type=float, default=0.5, help="self_distillation.success_reward_threshold")
    p.add_argument("--reflect_position", default="after_privileged", choices=["after_privileged", "system"])
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--gmem", type=float, default=0.85)
    p.add_argument("--out", default="")
    main(p.parse_args())
