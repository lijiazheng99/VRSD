from verl.utils.reward_score.feedback import math
from verl.utils.reward_score.feedback import code
from verl.utils.reward_score.feedback import gpqa
from verl.utils.reward_score.feedback import mcq
from verl.utils.reward_score.feedback import tooluse
from verl.utils.reward_score.feedback import ordinal_score
from verl.utils.reward_score.feedback import sql


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> dict:
    if data_source in ["code", "livecodebench", "humanevalplus"]:
        # max_test_cases=None: no cap, matching the SDPO paper's original reward call (v1.0.0). The lcb_v6
        # dataset already halves each problem's tests on the TRAIN side (the SDPO test-case split, 37 -> 18 mean),
        # which is the paper's own way of making training cheaper; a second cap here would diverge from it.
        results = code.compute_score(solution_str, ground_truth, extra_info, sparse_rewards=True, max_test_cases=None)
    elif data_source in ["math", "math500", "dapo_math", "gsm8k", "aime24", "aime25", "amc23"]:   # math task: train=math, val=aime24/aime25/math500
        results = math.compute_score(solution_str, ground_truth, extra_info)
    elif data_source in ["gpqa"]:
        results = gpqa.compute_score(solution_str, ground_truth)
    elif data_source in ["sciknoweval"]:
        results = mcq.compute_score(solution_str, ground_truth)
    elif data_source in ["spider", "bird"]:
        results = sql.compute_score(solution_str, ground_truth, extra_info)
    elif data_source in ["tooluse"]:
        results = tooluse.compute_score(solution_str, ground_truth)
    elif "_" in data_source and data_source.rsplit("_", 1)[0] in {"asap","aqa","asap_1","asap_2","asap_5","asap_6","Q1.5","Q4.3","Q7.6"}:
        _, shape = data_source.rsplit("_", 1)
        results = ordinal_score.compute_score(solution_str, ground_truth, extra_info, shape=shape)
    else:
        raise ValueError(f"Reward style {data_source} not found.")
    return results
