"""Loader for ordinal student-answer scoring (ASAP-SAS + AQA).

Each prompt directory contains:
  - {train,validation,test}.jsonl  — rows with Id, EssaySet/Question Id, Mark, Response
  - prompt.json — with `question` (task description) and `rubric` (marking criteria)

This loader builds CoT-formatted prompts:
    system  : forces <reasoning>...</reasoning><answer>k</answer>
    user    : question + cleaned rubric + student answer

The rubric is taken verbatim from prompt.json with the legacy few-shot trailer
stripped. Q4.3 is the only special case: its stored rubric is stale (0-4) but
gold labels span 0-6, so we regenerate it from the standard template.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


# Observed gold-label maxima per prompt, cross-checked against train+val+test
# JSONLs. Q4.3's stored rubric is wrong (says K=4); the question header
# ("[6 marks]") and the actual labels confirm K=6.
K_TABLE: Dict[str, int] = {
    "asap_1": 3, "asap_2": 3, "asap_5": 3, "asap_6": 3,
    "Q1.5": 4, "Q4.3": 6, "Q7.6": 3,
}

FAMILY_PROMPTS: Dict[str, List[str]] = {
    "asap": ["asap_1", "asap_2", "asap_5", "asap_6"],
    "aqa":  ["Q1.5", "Q4.3", "Q7.6"],
}


# Everything from this sentence on is the legacy few-shot template
# (`{{demonstation}}`, `[Student Answer]: "{{student_answer}}"`, `[Mark]:`).
# Strip it before injecting the student answer in our CoT prompt.
_RUBRIC_TRAILER_RE = re.compile(
    r"\s*Please assess the student answer with a score based on the above marking rubric.*",
    re.DOTALL,
)


SYSTEM_PROMPT = """You are an expert science teacher marking student answers.

You will be given a question, a marking rubric, and a student answer.
First, reason through your assessment step by step inside <reasoning>...</reasoning>:
  - Identify the criteria the rubric requires (e.g. key elements, valid conclusions).
  - Check each one against the student answer.
  - Decide on the final mark.
Then output ONLY the integer mark inside <answer>...</answer>.

Respond in EXACTLY this format and nothing else:
<reasoning>
your step-by-step reasoning
</reasoning>
<answer>
k
</answer>

where k is an integer in [0, K] and K is the maximum mark defined by the rubric."""


# Thinking-aware variant. When the model's chat template wraps the assistant
# turn in <think>...</think> (e.g. Qwen3 with enable_thinking=true), the
# rationale lives inside the implicit think block — asking for an extra
# <reasoning> block on top is pure token waste. The parser only reads
# <answer>...</answer> regardless.
SYSTEM_PROMPT_THINKING = """You are an expert science teacher marking student answers.

You will be given a question, a marking rubric, and a student answer.
Think through your assessment carefully:
  - Identify the criteria the rubric requires (e.g. key elements, valid conclusions).
  - Check each one against the student answer.
  - Decide on the final mark.
Then output ONLY the integer mark inside <answer>...</answer>.

Respond in EXACTLY this format and nothing else:
<answer>
k
</answer>

where k is an integer in [0, K] and K is the maximum mark defined by the rubric."""


_Q43_REPLACEMENT_RUBRIC = (
    "[Marking Rubric]:\n"
    "6 points: The student answer correctly answered six key answer elements.\n"
    "5 points: The student answer correctly answered five key answer elements.\n"
    "4 points: The student answer correctly answered four key answer elements.\n"
    "3 points: The student answer correctly answered three key answer elements.\n"
    "2 points: The student answer correctly answered two key answer elements.\n"
    "1 point:  The student answer correctly answered one key answer element.\n"
    "0 points: The student answer does not answer any key answer elements."
)


def _clean_rubric(raw_rubric: str, prompt_id: str) -> str:
    if prompt_id == "Q4.3":
        return _Q43_REPLACEMENT_RUBRIC
    return _RUBRIC_TRAILER_RE.sub("", raw_rubric).strip()


def _build_user_prompt(question: str, rubric: str, response: str) -> str:
    return f"{question}\n\n{rubric}\n\n[Student Answer]:\n{response}"


def _format_row(row: dict, family: str, prompt_id: str,
                rubric: str, question: str, thinking: bool) -> dict:
    """Shape one source row into the verl-compatible record.

    `dataset` is set to `prompt_id` (not `family`), so when preprocess
    appends "_{shape}" each row gets a per-prompt data_source like
    "asap_1_binary" or "Q4.3_mae". This (a) lets verl emit per-prompt
    validation metrics automatically, and (b) is necessary for combined
    training across prompts with different K (binary works regardless of K
    but the per-prompt metric breakdown is what reveals K-sensitivity).
    """
    K = K_TABLE[prompt_id]
    system_prompt = SYSTEM_PROMPT_THINKING if thinking else SYSTEM_PROMPT
    return {
        "dataset": prompt_id,              # → data_source = "<prompt_id>_<shape>"
        "system": system_prompt,
        "prompt": _build_user_prompt(question, rubric, row["Response"]),
        "answer": str(int(row["Mark"])),   # ground_truth for the reward fn
        "tests": None,
        "description": question,
        "kind": "ordinal_score",
        "elo": 1500,
        # Extras (preprocess pushes these into extra_info):
        "prompt_id": prompt_id,
        "family": family,
        "score_max": K,
        "source_id": str(row["Id"]),
    }


def _load_split(jsonl_path: Path, family: str, prompt_id: str,
                rubric: str, question: str, thinking: bool) -> List[dict]:
    K = K_TABLE[prompt_id]
    rows: List[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src = json.loads(line)
            mark = int(src["Mark"])
            if mark < 0 or mark > K:
                # Out-of-range gold (shouldn't happen given observed K) — skip
                continue
            rows.append(_format_row(src, family, prompt_id, rubric, question, thinking))
    return rows


def load_ordinal_scoring(family: str, prompt_id: str,
                         data_root: Path,
                         thinking: bool = False) -> Dict[str, List[dict]]:
    """Return {'train': [...], 'test': [...]} for one (family, prompt_id).

    `validation.jsonl` is merged into the held-out 'test' set. The source
    already provides splits so we honour them directly.

    `thinking=True` swaps in the minimal system prompt for chat templates
    that wrap the assistant turn in <think>...</think> (e.g. Qwen3 with
    enable_thinking=true).
    """
    prompt_dir = Path(data_root) / family / prompt_id
    with (prompt_dir / "prompt.json").open() as f:
        meta = json.load(f)
    rubric = _clean_rubric(meta["rubric"], prompt_id)
    question = meta["question"]

    out: Dict[str, List[dict]] = {"train": [], "test": []}
    out["train"] = _load_split(prompt_dir / "train.jsonl",
                               family, prompt_id, rubric, question, thinking)
    for split_file in ("validation.jsonl", "test.jsonl"):
        path = prompt_dir / split_file
        if path.exists():
            out["test"].extend(
                _load_split(path, family, prompt_id, rubric, question, thinking)
            )
    return out


def _peek_one(family: str, prompt_id: str, data_root: Path, thinking: bool) -> None:
    """Debug helper: print the formatted prompt for the first training row."""
    splits = load_ordinal_scoring(family, prompt_id, data_root, thinking=thinking)
    first = splits["train"][0]
    print(f"--- {family}/{prompt_id} (K={first['score_max']}, thinking={thinking}) ---")
    print("[SYSTEM]")
    print(first["system"])
    print("[USER]")
    print(first["prompt"])
    print("[GOLD]", first["answer"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect the formatted prompt for one row of a prompt_id."
    )
    parser.add_argument("--data_root", type=Path,
                        default=Path("datasets/ordinal_scoring"))
    parser.add_argument("--family", type=str, required=True,
                        choices=list(FAMILY_PROMPTS))
    parser.add_argument("--prompt_id", type=str, required=True)
    parser.add_argument("--thinking", action="store_true",
                        help="Use the minimal system prompt (no <reasoning>) "
                             "for chat templates with <think> wrapping.")
    args = parser.parse_args()
    _peek_one(args.family, args.prompt_id, args.data_root, args.thinking)
