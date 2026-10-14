# Self-Distillation with Verbal Regularization (VRSD)

Jiazheng Li, Yawei Li, Junru Lu, Zhenyi Shen, Di Yin, Lin Gui, Xing Sun, David Rügamer, Xinpeng Wang, Yulan He

[[Paper]](https://arxiv.org/abs/2610.16354)

On-policy self-distillation (OPSD) trains a language model on the next-token distributions of a teacher built from
the same model and given privileged information (PI), such as a verified solution. PI can push the teacher toward
shortcut reasoning, with little verification or reconsideration and a sharp drop in entropy. VRSD adds a
teacher-only system prompt that regulates how the teacher uses PI, encouraging it to re-check and explore before
committing. The teacher sees the question, a correct solution and the prompt; the student sees only the question.
The model, the distillation objective and the student input are unchanged.

Across Qwen3-4B, Qwen3-8B and OLMo-3-7B-Instruct on four tasks (short-answer grading on ASAP-SAS, chemistry on
SciKnowEval, text-to-SQL on Spider and Python on LiveCodeBench), VRSD improves Pass@1 over OPSD in 11 of 12
model-task settings, by 3.9 percentage points on average, while generally keeping higher token entropy.

Two prompt designs are provided: `single` (one task-level reflection prompt) and `multi` (the rollouts of each
question are split across aspect-specific prompts). The implementation extends [verl](https://github.com/volcengine/verl)
and [SDPO](https://arxiv.org/abs/2601.20802); everything runs through `scripts/train.sh`.

## Setup

```bash
conda create -n vrsd python=3.12 -y && conda activate vrsd
pip install -r install/requirements.txt
pip install -r install/requirements-cuda.txt   # flash-attn
```

Tested with torch 2.7.1 (CUDA 12.6), vLLM 0.10.0, transformers 4.57.1, Ray 2.55.1. Each run uses two GPUs; the text tasks fit
on 80 GB cards, the Python runs were made on 141 GB cards.

## Data

Each script writes `datasets/<...>/{train,test}.parquet` with the same prompts, answers and splits as in the paper.

```bash
python data/prepare_asap.py --kaggle_dir <dir>      # ASAP-SAS grading, sets 1, 2, 5 and 6 (Kaggle files, see below)
python data/prepare_chem.py                         # SciKnowEval chemistry, L1-L4 multiple choice (Hugging Face)
python data/setup_spider_db.py                      # Spider SQLite databases (needed first: schemas and reward) ...
python data/prepare_spider.py                       # ... then the Spider questions (Hugging Face)
python data/prepare_lcb_v6.py                       # LiveCodeBench v6 (Hugging Face)
```

The ASAP-SAS essays are not redistributed here. Download them from https://www.kaggle.com/c/asap-sas (after accepting
the competition rules) and pass the directory holding `train.tsv`, the public leaderboard essays
(`public_leaderboard*.tsv`) and `public_leaderboard_solution.csv`. The split is fixed by essay Id in
`data/asap/split_ids.json`, and the script checks the essay Ids and scores against `data/asap/checksums.json`
before writing anything, so a successful run reproduces the grading data of the paper.

## Training

```bash
scripts/train.sh <task> <method> [gpus]
#   task   : grading | chem | sql | python
#   method : opsd | antisd | single | multi | prompt_only
scripts/train.sh chem multi 0,1
```

`single` uses `prompts/single/<task>.txt`; `multi` spreads the rollouts of each question over the aspect prompts in
`prompts/aspects/<task>/`; `prompt_only` is the control in which the teacher gets the aspect prompts but no
solution. The recipe of each task (Appendix D) is set inside `scripts/train.sh`. Runs stop after four
validations without improvement and keep the best checkpoint under `checkpoints/<dataset dir>/<run>/` (a full FSDP
checkpoint, about 47 GB for Qwen3-4B); re-running the same cell resumes from it. Every validation writes the
per-sample predictions to `outputs/<model>-<task>-<method>/<step>.jsonl`; `metrics.json` in the same directory
holds Pass@1 (`mean@16`), Maj@16 and Worst@16 of the best step.

## Analysis

```bash
# Pass@1 / Maj@16 / Worst@16 from a prediction directory
python analysis/eval_from_dump.py --dump outputs/Qwen3-4B-chem-multi --out metrics.json

# response patterns of the students (Table 2, Appendix I)
python analysis/answer_patterns.py --outputs outputs --model Qwen3-4B --out patterns_Qwen3-4B

# truncated-reasoning test (Table 3); checkpoints are first converted to Hugging Face format
python -m verl.model_merger merge --backend fsdp \
    --local_dir checkpoints/<dataset dir>/<run>/global_step_<best>/actor --target_dir merged/<run>
python analysis/trace_truncation.py --task chem --data datasets/sciknoweval/chemistry_all/test.parquet \
    --models base=Qwen/Qwen3-4B multi=merged/<run> --out trace_chem.json

# teacher continuations with and without the prompt (Appendix I, Table 9)
python analysis/teacher_continuations.py --task chem --model Qwen/Qwen3-4B --out teacher_chem.json

# teacher token entropy with and without the prompt (task: grading | chem | sql | python)
python analysis/teacher_probe.py --model Qwen/Qwen3-4B --task chem --limit 120 --n_roll 8 \
    --conds bare: single:prompts/single/chem.txt --out teacher_probe_chem.json
```

## Citation

```bibtex
@misc{li2026vrsd,
  title         = {Self-Distillation with Verbal Regularization},
  author        = {Li, Jiazheng and Li, Yawei and Lu, Junru and Shen, Zhenyi and Yin, Di and Gui, Lin and Sun, Xing and
                   R{\"u}gamer, David and Wang, Xinpeng and He, Yulan},
  year          = {2026},
  eprint        = {2610.16354},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2610.16354}
}
```

## Acknowledgements and license

The training code is a fork of [verl](https://github.com/volcengine/verl) (Apache 2.0) with the
self-distillation trainer of [SDPO](https://arxiv.org/abs/2601.20802); this repository is released under the same
Apache 2.0 license (`LICENSE`). The datasets keep their original licenses: ASAP-SAS (Kaggle), SciKnowEval,
Spider and LiveCodeBench.
