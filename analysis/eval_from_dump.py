#!/usr/bin/env python
# Reproduce verl's FULL validation metric suite from a saved val-prediction dump, for ANY task.
#
# WHY: verl computes avg@N / maj@N / best@N(=pass@N) / worst@N during validation and logs them to
# WANDB ONLY. The per-sample predictions ARE persisted (trainer.validation_data_dir -> <step>.jsonl,
# all N samples per problem), but the aggregate metrics are not — so once wandb is gone (or you never
# had it), pass@k / averages are unrecoverable *unless* you recompute from the dump. This does exactly
# that, by feeding the dump back through verl's own `process_validation_metrics`, so the numbers are
# byte-identical to what training logged (same bootstrap, same seed=42).
#
#   usage: python analysis/eval_from_dump.py --dump <dir_or_file> [--task code] [--out x.json]
#     --dump a directory -> uses the step named in best_step.txt (early stopping) else its
#                           highest-numbered <step>.jsonl (the final/trained eval)
import argparse, json, os, sys, hashlib, glob
# repo root: env override, else derive from this file's location (analysis/eval_from_dump.py)
ROOT = os.environ.get("SDPO_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def resolve(dump):
    if os.path.isdir(dump):
        # only <step>.jsonl files (numeric stem); ignore any stray non-numeric .jsonl in the dir
        js = sorted((int(os.path.basename(f)[:-6]), f) for f in glob.glob(f"{dump}/*.jsonl")
                    if os.path.basename(f)[:-6].isdigit())
        if not js:
            sys.exit(f"no <step>.jsonl in {dump}")
        # With early stopping the LAST validation is <patience> validations past the peak, while the
        # checkpoint kept on disk is the BEST one. best_step.txt (written by ray_trainer) says which
        # step that was -> score THAT dump so metrics and checkpoint describe the same model.
        best = os.path.join(dump, "best_step.txt")
        if os.path.exists(best):
            try:
                bs = int(open(best).read().split()[0])
                hit = [f for st, f in js if st == bs]
                if hit:
                    if bs != js[-1][0]:
                        print(f"[eval_from_dump] using BEST step {bs} (early stopping; last logged step is {js[-1][0]})")
                    return hit[0], js[0][1] if len(js) > 1 else None
                print(f"[eval_from_dump] WARNING: best_step.txt says {bs} but {bs}.jsonl is missing; using last step {js[-1][0]}")
            except Exception:
                pass
        return js[-1][1], js[0][1] if len(js) > 1 else None  # (final, base)
    return dump, None


def load(f):
    return [json.loads(l) for l in open(f) if l.strip()]


def uid_of(r):
    # Group by the FULL input = one problem. NOTE: a dump's `prompt_id` is often the SUBSET id
    # (e.g. asap_1/2/5/6 for grading), NOT per-problem, so grouping by it would collapse 500 essays
    # into 4 groups and corrupt every @k. Full-input hash is per-problem for every task we have.
    return hashlib.md5(str(r.get("input", "")).encode()).hexdigest()


def metrics(rows, task):
    from verl.trainer.ppo.metric_utils import process_validation_metrics
    ds = [task] * len(rows)
    uids = [uid_of(r) for r in rows]
    def num(r, *keys):
        for k in keys:
            if k in r and r[k] is not None:
                try: return float(r[k])
                except Exception: pass
        return 0.0
    infos = {
        "acc": [num(r, "acc", "score", "reward") for r in rows],
        "reward": [num(r, "reward", "score", "acc") for r in rows],
        "pred": [str(r.get("pred", "")) for r in rows],
    }
    res = process_validation_metrics(ds, uids, infos, seed=42)
    return res[task]["acc"], len(set(uids))


def looks_ordinal(rows):
    # ordinal grading rows carry integer gold in gts/ground_truth and a small score range
    r = rows[0]
    return ("prompt_id" in r) or ("gts" in r and str(r.get("gts", "")).lstrip("-").isdigit())


def summary(acc, task, n_problems):
    def g(k):
        return acc.get(k)
    print(f"  problems={n_problems}")
    print(f"  avg@16 (pass@1 mean) : {g('mean@16'):.4f}" if g('mean@16') is not None else "  avg@16: n/a")
    for n in (2, 4, 8, 16):
        b = g(f"best@{n}/mean"); m = g(f"maj@{n}/mean")
        line = f"  @{n:<2}  pass@{n}(best)={b:.4f}" if b is not None else f"  @{n}: n/a"
        if m is not None:
            line += f"   maj@{n}={m:.4f}"
        print(line)


def main(a):
    final, base = resolve(a.dump)
    rows = load(final)
    task = a.task or ("grading" if looks_ordinal(rows) else "code")
    acc, npb = metrics(rows, task)
    print(f"=== {os.path.basename(final)}  task={task} ===")
    summary(acc, task, npb)

    out = {"dump": final, "task": task, "n_problems": npb, "acc": acc}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True, help="val dump dir (uses best_step.txt if present, else the final step) or a single <step>.jsonl")
    p.add_argument("--task", default=None, help="data-source label (auto: grading if ordinal else code)")
    p.add_argument("--out", default=None)
    main(p.parse_args())
