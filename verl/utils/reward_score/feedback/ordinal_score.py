"""Reward function for ordinal student-answer scoring (ASAP-SAS + AQA).

Five reward shapes per ideas/grpo_graded_reward_idea_doc.md §5 (all normalised
to [0, 1]; score range is 0..K where K is read from `extra_info["score_max"]`):

  binary : 1[pred == gold]
  mae    : 1 - |pred - gold| / K
  mse    : 1 - ((pred - gold) / K)^2
  exp    : exp(-|pred - gold| / tau)               (tau via SDPO_EXP_TAU, default 1.0)
  hybrid : 0.5 * 1[pred == gold] + 0.5 * (1 - |pred - gold| / K)

Parse failures (no <answer> tag, non-integer, or out of [0, K]) yield reward 0
with `incorrect_format = 1`. The model is also expected to emit a <reasoning>
block before the answer (system prompt enforces this), but the reasoning text
itself is not graded — only the integer inside <answer>...</answer>.
"""
from __future__ import annotations

import math
import os
import re
from typing import Optional


SUPPORTED_SHAPES = ("binary", "mae", "mse", "exp", "hybrid")


_ANSWER_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>", re.IGNORECASE)
# Fallback: an UNCLOSED tag whose integer is the very last thing in the response, i.e. the model
# emitted EOS right after the digit. OLMo-3-7B-Instruct does this on 87% of its answers ("<answer>\n0<eos>"),
# which the strict regex scored as format-error (base acc 0.065 vs 0.514 with this fallback). Only fires when
# nothing follows the digit, so a closed tag or any trailing text is still handled by the strict pattern.
_ANSWER_UNCLOSED_TAIL_RE = re.compile(r"<answer>\s*(-?\d+)\s*$", re.IGNORECASE)


def _extract_int_answer(text: str) -> Optional[int]:
    """Return the integer inside the LAST <answer>...</answer> tag (or an unclosed <answer>k at end-of-text), or None."""
    matches = _ANSWER_RE.findall(text)
    if not matches:
        m = _ANSWER_UNCLOSED_TAIL_RE.search(text.rstrip())
        matches = [m.group(1)] if m else []
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def _shape_reward(shape: str, pred: int, gold: int, K: int) -> float:
    if K <= 0:
        # Degenerate K=0 — only the binary case is meaningful.
        return float(pred == gold)
    if shape == "binary":
        return float(pred == gold)
    if shape == "mae":
        return 1.0 - abs(pred - gold) / K
    if shape == "mse":
        return 1.0 - (abs(pred - gold) / K) ** 2
    if shape == "exp":
        tau = float(os.environ.get("SDPO_EXP_TAU", "1.0"))
        return math.exp(-abs(pred - gold) / tau)
    if shape == "hybrid":
        return 0.5 * float(pred == gold) + 0.5 * (1.0 - abs(pred - gold) / K)
    raise ValueError(f"Unknown reward shape: {shape!r}")


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    shape: str = "binary",
) -> dict:
    """Compute the reward for one (response, gold) pair.

    Args:
        solution_str: full model response, including <reasoning> and <answer> tags.
        ground_truth: gold integer score as a string (set in reward_model.ground_truth).
        extra_info:   must contain `score_max` (K). Optional: `prompt_id`, `source_id`.
        shape:        one of SUPPORTED_SHAPES.

    Returns the standard reward dict — at minimum `score`, `acc` (binary
    agreement), `pred`, `abs_err`, `incorrect_format`, `feedback`. Diagnostics
    further down the pipeline pick `pred` and `acc` out of this dict.
    """
    if shape not in SUPPORTED_SHAPES:
        raise ValueError(
            f"shape={shape!r} not in {SUPPORTED_SHAPES}. "
            f"Add it to ordinal_score.py or fix the data_source dispatch."
        )

    K = int(extra_info.get("score_max", 0))
    gold = int(ground_truth)
    pred = _extract_int_answer(solution_str)

    # Format gating: parse failure or out-of-range -> reward 0, max abs_err.
    # Use -1 (not None) as the sentinel for parse failure so downstream metric
    # aggregation (np.mean over per-sample preds) doesn't crash on
    # `int + None`. -1 is unambiguous because valid preds are in [0, K].
    if pred is None or pred < 0 or pred > K:
        return {
            "score": 0.0,
            "acc": 0.0,
            "acc_within_1": 0.0,
            "pred": -1,
            "gold": gold,
            "prompt_id": extra_info.get("prompt_id", "?"),   # for per-prompt QWK from val dump
            "abs_err": K,
            "incorrect_format": 1,
            "feedback": (
                f"Output must contain <answer>k</answer> with an integer "
                f"k in [0, {K}]."
            ),
        }

    reward = _shape_reward(shape, pred, gold, K)
    # Clip to [0, 1] in case of float drift.
    reward = max(0.0, min(1.0, reward))
    abs_err = abs(pred - gold)

    return {
        "score": reward,
        "acc": float(pred == gold),
        "acc_within_1": float(abs_err <= 1),    # ±1 ordinal accuracy
        "pred": pred,
        "gold": gold,                            # needed for QWK + macro-F1
        "prompt_id": extra_info.get("prompt_id", "?"),   # for per-prompt QWK from val dump
        "abs_err": abs_err,
        "incorrect_format": 0,
        "feedback": "",
    }
