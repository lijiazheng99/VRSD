# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import re
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from string import Template
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.model import compute_position_id_with_mask
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.torch_functional import postprocess_data
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

try:
    from math_verify import parse as mv_parse, verify as mv_verify
    _HAS_MATH_VERIFY = True
except ImportError:
    _HAS_MATH_VERIFY = False


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = config.actor_rollout_ref.actor.get("self_distillation", {}).get("reprompt_truncation", "error")
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

        # Print the first few samples to stdout (-> training log) so each eval's outputs can be eyeballed
        # without opening the jsonl. SDPO_PRINT_SAMPLES=0 disables; default 3. (In these runs _dump_generations
        # is only called for validation, so this fires once per eval point, not per train step.)
        _ns = int(os.environ.get("SDPO_PRINT_SAMPLES", "3"))
        if _ns > 0 and n > 0:
            print(f"===== first {min(_ns, n)} eval samples @ step {self.global_steps} ({os.path.basename(dump_path)}) =====", flush=True)
            for i in range(min(_ns, n)):
                row = {k: v[i] for k, v in base_data.items()}
                q = str(row.get("input", "")).replace("\n", " ")[-350:]
                a = str(row.get("output", "")).replace("\n", " ")[:600]
                print(f"[{i}] gold={row.get('gts')}  pred={row.get('pred', 'n/a')}  score={row.get('score')}  reward={row.get('reward', 'n/a')}")
                print(f"    Q(tail): ...{q}")
                print(f"    A(head): {a}")
            print("===== end eval samples =====", flush=True)

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        return_dict: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor | dict[str, Any]:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            return_dict: Whether to return dict format with reward_extra_info (for validation)
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
            If return_dict=False and sum_reward=True: summed reward_tensor (1D tensor)
            If return_dict=False and sum_reward=False: reward_tensor (2D tensor)
        """
        # When rm_scores already exists, extract it directly (format conversion only)
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if return_dict:
                # Extract reward_extra_info if available
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                # If sum_reward=True, only return tensor (for REMAX baseline)
                if sum_reward:
                    return reward_tensor
                # Otherwise, return tuple with reward_extra_info (for training loop)
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_infos_dict = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return reward_tensor, reward_extra_infos_dict

        # Otherwise, compute reward using reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if return_dict:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_info = result.get("reward_extra_info", {})
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    @staticmethod
    def _collect_feedback(
        include_environment_feedback: bool,
        reward_extra_infos_dict: Optional[dict[str, Any]],
        batch_size: int
    ) -> list[Any]:
        """
        Collect environment feedback from reward_extra_infos_dict.

        Args:
            include_environment_feedback: Whether to include environment feedback
            reward_extra_infos_dict: Dictionary containing reward extra information
            batch_size: Size of the batch

        Returns:
            List of feedback strings (or None for entries without feedback)
        """
        feedback_list: list[Any] = [None] * batch_size
        if include_environment_feedback and reward_extra_infos_dict is not None:
            raw_feedback = reward_extra_infos_dict.get("feedback", [])
            for i in range(min(len(raw_feedback), batch_size)):
                # Only include non-empty feedback strings
                if raw_feedback[i] and isinstance(raw_feedback[i], str) and raw_feedback[i].strip():
                    feedback_list[i] = raw_feedback[i]
        return feedback_list

    def _collect_solutions_by_uid(self, batch: DataProto, reward_tensor: torch.Tensor, success_reward_threshold: float) -> dict[Any, list[int]]:
        seq_scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()
        uids = batch.non_tensor_batch["uid"]
        success_by_uid: dict[Any, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            if seq_scores[idx] >= success_reward_threshold:
                success_by_uid[uid].append(idx)
        return success_by_uid

    def _head_tail_truncate(self, tokenized_batch: dict, max_len: int, head_ratio: float = 1/3) -> dict:
        """Truncate middle of sequences, keeping head (prompt) and tail (instruction).

        Inserts a "..." marker at the truncation point so the model knows content
        was omitted. With left-padded batches, shorter samples have leading pad
        tokens (attn=0) which are naturally preserved and ignored.

        Args:
            tokenized_batch: dict with "input_ids" and "attention_mask" tensors [B, L].
            max_len: maximum sequence length after truncation.
            head_ratio: fraction of budget allocated to the head (prompt).
        """
        input_ids = tokenized_batch["input_ids"]
        attention_mask = tokenized_batch["attention_mask"]
        seq_len = input_ids.shape[1]

        if seq_len <= max_len:
            return tokenized_batch

        # Encode the truncation marker
        marker_ids = self.tokenizer.encode(" ... ", add_special_tokens=False)
        marker_len = len(marker_ids)
        marker_tensor = torch.tensor(marker_ids, dtype=input_ids.dtype, device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        marker_mask = torch.ones(input_ids.shape[0], marker_len, dtype=attention_mask.dtype, device=attention_mask.device)

        budget = max_len - marker_len
        head_len = int(budget * head_ratio)
        tail_len = budget - head_len

        tokenized_batch["input_ids"] = torch.cat(
            [input_ids[:, :head_len], marker_tensor, input_ids[:, -tail_len:]], dim=1
        )
        tokenized_batch["attention_mask"] = torch.cat(
            [attention_mask[:, :head_len], marker_mask, attention_mask[:, -tail_len:]], dim=1
        )
        return tokenized_batch

    @staticmethod
    def _remove_thinking_trace(text: str) -> str:
        """Remove <think>...</think> tags and their content from text."""
        return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)

    @staticmethod
    def _remove_boxed_answer(text: str) -> str:
        r"""Remove the last \boxed{...} and any trailing whitespace from text.

        Handles nested braces correctly. If no \boxed is found, returns text unchanged.
        """
        last_boxed_start = text.rfind(r"\boxed{")
        if last_boxed_start == -1:
            return text
        # Find matching closing brace
        depth = 0
        i = last_boxed_start + len(r"\boxed{")
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                if depth == 0:
                    # Remove \boxed{...} and any trailing whitespace/punctuation
                    return (text[:last_boxed_start].rstrip() + text[i + 1:].lstrip()).strip()
                depth -= 1
            i += 1
        # Malformed \boxed, remove from start of \boxed to end
        return text[:last_boxed_start].rstrip()

    @staticmethod
    def _find_all_boxed_positions(text: str) -> list[tuple[int, int, str]]:
        r"""Find all \boxed{...} occurrences in text, handling nested braces.

        Returns list of (start_pos, end_pos, content) tuples where:
        - start_pos: index of the '\' in '\boxed{'
        - end_pos: index just after the closing '}'
        - content: the string inside the outermost braces
        """
        results = []
        search_start = 0
        prefix = r"\boxed{"
        while True:
            pos = text.find(prefix, search_start)
            if pos == -1:
                break
            # Scan for matching closing brace
            depth = 0
            i = pos + len(prefix)
            content_start = i
            while i < len(text):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    if depth == 0:
                        content = text[content_start:i]
                        results.append((pos, i + 1, content))
                        break
                    depth -= 1
                i += 1
            # Move past this occurrence regardless of whether we found a match
            search_start = pos + len(prefix)
        return results

    @staticmethod
    def _truncate_at_last_correct_boxed(text: str, ground_truth: str) -> str:
        r"""Truncate text right after the last \boxed{} whose content matches ground_truth.

        Scans from the end so we keep the full reasoning chain up to the final answer,
        avoiding truncation at intermediate calculations that coincidentally match.
        Returns text unchanged if no matching \boxed{} is found.
        """
        positions = RayPPOTrainer._find_all_boxed_positions(text)
        if not positions:
            return text

        gt_stripped = ground_truth.strip()

        for start_pos, end_pos, content in reversed(positions):
            content_stripped = content.strip()
            # Simple string match
            if content_stripped == gt_stripped:
                return text[:end_pos]
            # math_verify fallback
            if _HAS_MATH_VERIFY:
                try:
                    parsed_content = mv_parse(content_stripped)
                    parsed_gt = mv_parse(gt_stripped)
                    if mv_verify(parsed_gt, parsed_content):
                        return text[:end_pos]
                except Exception:
                    pass
        return text

    def _get_solution(
        self,
        idx: int,
        success_by_uid: dict[Any, list[int]],
        uids: list[Any],
        response_texts: list[str],
        dont_reprompt_on_self_success: bool = False,
        remove_thinking_from_demonstration: bool = False,
        response_mask=None,
        ground_truth: Optional[str] = None,
        solution_selection: str = "random",
        truncate_at_correct_answer: bool = False,
    ) -> Optional[str]:
        uid = uids[idx]
        solution_idxs = success_by_uid[uid]
        if dont_reprompt_on_self_success:
            solution_idxs = [j for j in solution_idxs if j != idx]
        if len(solution_idxs) == 0:
            return None

        # Length-aware solution selection
        if solution_selection == "prefer_short" and response_mask is not None and len(solution_idxs) > 1:
            lengths = [(j, response_mask[j].sum().item()) for j in solution_idxs]
            lengths.sort(key=lambda x: x[1])
            k = max(1, len(lengths) // 2)
            shorter_half = [j for j, _ in lengths[:k]]
            solution_idx = shorter_half[np.random.randint(len(shorter_half))]
        else:
            solution_idx = solution_idxs[0]

        solution_str = response_texts[solution_idx]
        if remove_thinking_from_demonstration:
            solution_str = self._remove_thinking_trace(solution_str)
        # Truncate after last correct \boxed{answer} to remove post-answer verbosity
        if truncate_at_correct_answer and ground_truth:
            solution_str = self._truncate_at_last_correct_boxed(solution_str, ground_truth)
        return solution_str

    def _truncate_solution_to_budget(
        self,
        solution_str: str,
        prompt_text: str,
        feedback_text: Optional[str],
        max_reprompt_len: int,
        max_solution_tokens: Optional[int],
        chat_overhead: int = 300,
    ) -> str:
        """Truncate solution to fit within the teacher prompt token budget.

        Preserves prompt and instruction fully, only compresses the solution.
        Keeps the head (problem setup) and tail (final answer) of the solution,
        removing the middle with a '...' marker.
        """
        prompt_len = len(self.tokenizer.encode(prompt_text, add_special_tokens=False))
        feedback_len = len(self.tokenizer.encode(feedback_text, add_special_tokens=False)) if feedback_text else 0
        available = max_reprompt_len - prompt_len - feedback_len - chat_overhead
        available = max(available, 128)  # minimum budget to keep some answer context
        if max_solution_tokens is not None:
            available = min(available, max_solution_tokens)

        tokens = self.tokenizer.encode(solution_str, add_special_tokens=False)
        if len(tokens) > available:
            # Keep head (25%) + tail (75%), drop middle
            head_budget = available // 4
            tail_budget = available - head_budget
            head_tokens = tokens[:head_budget]
            tail_tokens = tokens[-tail_budget:]
            head_text = self.tokenizer.decode(head_tokens, skip_special_tokens=False)
            tail_text = self.tokenizer.decode(tail_tokens, skip_special_tokens=False)
            return head_text + "\n...\n" + tail_text
        return solution_str

    # ---- privileged-context source selection ---------------------------------
    # Lets the teacher's
    # privileged block be drawn from an offline GT-derived abstraction (cot / summary /
    # rubric) instead of the self-generated sibling, looked up by question.

    @staticmethod
    def _norm_question(s: str) -> str:
        """Whitespace-insensitive, lowercased key. The bare problem is a substring
        of the wrapped training prompt, so this matches a prompt to a record."""
        return re.sub(r"\s+", "", s or "").lower()

    def _load_privileged_map(self, self_distillation_cfg) -> dict:
        """Load + cache {norm_question: {cot, summary, rubric}} from the offline
        privileged-context JSONs (built offline; not part of this release)."""
        if getattr(self, "_privileged_map", None) is not None:
            return self._privileged_map
        import glob
        import json as _json
        d = self_distillation_cfg.get("privileged_context_dir", None) or os.path.join(
            os.environ.get("SDPO_ROOT", "."), "datasets", "privileged_context")
        pmap: dict[str, dict] = {}
        for fp in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                for rec in _json.load(open(fp)):
                    q = rec.get("question")
                    if q:
                        pmap[self._norm_question(q)] = {
                            "cot": rec.get("cot"),
                            "summary": rec.get("summary"),
                            "rubric": rec.get("rubric"),
                        }
            except Exception as e:  # never break training on a bad/missing file
                print(f"[privileged-context] failed to load {fp}: {e}", flush=True)
        self._privileged_map = pmap
        self._privileged_keys = list(pmap.keys())
        print(f"[privileged-context] loaded {len(pmap)} records from {d}", flush=True)
        return pmap

    def _lookup_privileged(self, prompt_text: str) -> Optional[dict]:
        nprompt = self._norm_question(prompt_text)
        rec = self._privileged_map.get(nprompt)
        if rec is not None:
            return rec
        for k in self._privileged_keys:  # bare problem is a substring of the wrapped prompt
            if k and k in nprompt:
                return self._privileged_map[k]
        return None


    def _maybe_build_self_distillation_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, list]] = None,
    ) -> Optional[tuple[DataProto, dict[str, float]]]:
        self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if self_distillation_cfg is None or loss_mode not in ("sdpo", "grpo_ccir", "grpo_st", "grpo_ca"):
            return None

        device = batch.batch["input_ids"].device
        response_mask = batch.batch["response_mask"]
        responses = batch.batch["responses"]
        response_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]
        prompt_texts = [msgs[-1]["content"] for msgs in batch.non_tensor_batch["raw_prompt"]]
        batch_size = batch.batch.batch_size[0]
        seq_scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()

        # Extract feedback if available and include_environment_feedback is enabled
        feedback_list = self._collect_feedback(
            include_environment_feedback=self_distillation_cfg.include_environment_feedback,
            reward_extra_infos_dict=reward_extra_infos_dict,
            batch_size=batch_size,
        )

        success_by_uid = self._collect_solutions_by_uid(batch, reward_tensor, success_reward_threshold=self_distillation_cfg.success_reward_threshold)
        reward_models = batch.non_tensor_batch.get("reward_model", np.array([{}] * batch_size))
        solution_selection = self_distillation_cfg.get("solution_selection", "random")
        truncate_at_correct = self_distillation_cfg.get("truncate_solution_at_correct_answer", False)
        # Solution source selection: controls whether to use group rollout solutions,
        # external (dataset) solutions, or a combination with fallback.
        solution_source = self_distillation_cfg.get("solution_source", "group_first")
        extra_infos = batch.non_tensor_batch.get("extra_info", np.array([None] * batch_size))

        def _get_external_solution(i):
            ei = extra_infos[i] if i < len(extra_infos) else None
            sol = ei.get("solution") if isinstance(ei, dict) else None
            return sol if sol else None

        def _get_group_solutions():
            return [
                self._get_solution(
                    i,
                    success_by_uid,
                    batch.non_tensor_batch["uid"],
                    response_texts,
                    self_distillation_cfg.dont_reprompt_on_self_success,
                    self_distillation_cfg.get("remove_thinking_from_demonstration", False),
                    response_mask=response_mask,
                    ground_truth=(reward_models[i].get("ground_truth") if isinstance(reward_models[i], dict) else None),
                    solution_selection=solution_selection,
                    truncate_at_correct_answer=truncate_at_correct,
                )
                for i in range(batch_size)
            ]

        num_source_external = 0
        num_source_group = 0

        if solution_source in ("group_first", "group_only"):
            solution_strs = _get_group_solutions()
            num_source_group = sum(1 for s in solution_strs if s is not None)
            if solution_source == "group_first":
                for i in range(batch_size):
                    if solution_strs[i] is None:
                        ext = _get_external_solution(i)
                        if ext is not None:
                            solution_strs[i] = ext
                            num_source_external += 1
        elif solution_source in ("external_first", "external_only"):
            solution_strs = [_get_external_solution(i) for i in range(batch_size)]
            num_source_external = sum(1 for s in solution_strs if s is not None)
            if solution_source == "external_first":
                group_solutions = _get_group_solutions()
                for i in range(batch_size):
                    if solution_strs[i] is None:
                        if group_solutions[i] is not None:
                            solution_strs[i] = group_solutions[i]
                            num_source_group += 1

        # Optionally strip final \boxed{...} from solutions so teacher only sees reasoning process
        if self_distillation_cfg.get("remove_answer_from_solution", False):
            for i in range(batch_size):
                if solution_strs[i] is not None:
                    solution_strs[i] = self._remove_boxed_answer(solution_strs[i])

        # Budget-aware solution truncation: preserve prompt + feedback fully, compress solution to fit
        max_reprompt_len = self_distillation_cfg.max_reprompt_len
        max_solution_tokens = self_distillation_cfg.get("max_solution_tokens", None)
        _head_tail_reprompt = self_distillation_cfg.get("reprompt_head_tail_truncation", False)
        feedback_only_without_sol = self_distillation_cfg.get("environment_feedback_only_without_solution", False)
        # AntiSD compresses over-budget solutions head/tail BEFORE templating; SDPO (base) does not touch the
        # solution and truncates the tokenized reprompt instead (see below). Off by default = base behaviour.
        for i in (range(batch_size) if self_distillation_cfg.get("truncate_solution_to_budget", False) else ()):
            if solution_strs[i] is not None:
                # Match _build_teacher_message logic: feedback is excluded when solution exists
                # and environment_feedback_only_without_solution is True
                effective_feedback = None
                if not feedback_only_without_sol and feedback_list[i] is not None:
                    effective_feedback = feedback_list[i]
                solution_strs[i] = self._truncate_solution_to_budget(
                    solution_strs[i], prompt_texts[i], effective_feedback,
                    max_reprompt_len, max_solution_tokens,
                )

        # When solution_content="feedback_only", suppress all solutions so the teacher
        # prompt contains only correctness feedback (e.g. "Your answer is correct/incorrect").
        if self_distillation_cfg.get("solution_content", "full") == "feedback_only":
            for i in range(batch_size):
                solution_strs[i] = None

        # Apply solution_mode transforms (structural hypothesis experiments)
        solution_mode = self_distillation_cfg.get("solution_mode", "normal")
        if solution_mode != "normal":
            import random as _random
            import re as _re

            if solution_mode == "cross_problem":
                # Shuffle solutions across batch: each sample gets a different problem's solution
                valid = [(i, solution_strs[i]) for i in range(batch_size) if solution_strs[i] is not None]
                if len(valid) > 1:
                    indices, sols = zip(*valid)
                    sols = list(sols)
                    _random.shuffle(sols)
                    for idx, sol in zip(indices, sols):
                        solution_strs[idx] = sol

            elif solution_mode == "none":
                for i in range(batch_size):
                    solution_strs[i] = None

            elif solution_mode == "shuffle_sentences":
                for i in range(batch_size):
                    if solution_strs[i] is not None:
                        lines = [s for s in solution_strs[i].split('\n') if s.strip()]
                        _random.shuffle(lines)
                        solution_strs[i] = '\n'.join(lines)

            elif solution_mode == "answer_only":
                for i in range(batch_size):
                    if solution_strs[i] is not None:
                        matches = _re.findall(r'\\boxed\{[^}]*\}', solution_strs[i])
                        solution_strs[i] = matches[-1] if matches else "\\boxed{?}"

            elif solution_mode == "fixed_detailed":
                _FIXED = (
                    "Let me work through this step by step.\n\n"
                    "Step 1: I'll carefully read the problem and identify the key quantities and relationships.\n\n"
                    "Step 2: I'll set up the appropriate equations or inequalities based on the given conditions.\n\n"
                    "Step 3: I'll solve the equations systematically, showing each algebraic manipulation.\n\n"
                    "Step 4: I'll verify my solution by substituting back into the original conditions.\n\n"
                    "Step 5: After confirming the answer is correct, I'll present the final result.\n\n"
                    "The answer is \\boxed{42}."
                )
                for i in range(batch_size):
                    if solution_strs[i] is not None:
                        solution_strs[i] = _FIXED

            elif solution_mode == "fixed_generic":
                _FIXED = (
                    "First, understand the problem. "
                    "Then, apply relevant techniques. "
                    "Finally, compute the answer. "
                    "\\boxed{42}."
                )
                for i in range(batch_size):
                    if solution_strs[i] is not None:
                        solution_strs[i] = _FIXED

            elif solution_mode == "fixed_unrelated":
                _FIXED = (
                    "The history of mathematics spans thousands of years. "
                    "Ancient civilizations in Mesopotamia developed early number systems. "
                    "The Greeks contributed geometry and logic. "
                    "In the 17th century, Newton and Leibniz independently developed calculus. "
                    "Modern mathematics encompasses algebra, analysis, topology, and many other fields."
                )
                for i in range(batch_size):
                    if solution_strs[i] is not None:
                        solution_strs[i] = _FIXED

        reprompt_style = self_distillation_cfg.get("reprompt_style", "suffix")

        # --- GOLD-TEACHER on UNSOLVED (code experiment "A"): the default sibling teacher only exists
        # for problems the model ALREADY solved (a successful rollout), so problems it NEVER solves get
        # solution_strs[i]=None and no distillation signal at all. When GOLD_TEACHER_ON_UNSOLVED=1, inject
        # the offline reference solution (reward_model["gold_solution"], set by the prepare script) as the
        # teacher target for exactly those unsolved samples — giving signal where there was none, without
        # touching the problems that already have a self-sibling. Default off => unchanged SDPO. ---
        if os.environ.get("GOLD_TEACHER_ON_UNSOLVED"):
            rm = batch.non_tensor_batch.get("reward_model")
            n_gold = n_miss = 0
            for i in range(batch_size):
                if solution_strs[i] is not None:
                    continue  # already has a self-sibling — leave it
                g = (rm[i].get("gold_solution") if (rm is not None and isinstance(rm[i], dict)) else None)
                if g:
                    solution_strs[i] = g
                    n_gold += 1
                else:
                    n_miss += 1
            print(f"[SDPO gold-teacher] injected gold on {n_gold} unsolved samples "
                  f"({n_miss} unsolved had no gold)", flush=True)

        # --- Privileged-context SOURCE selection. Default "sibling"
        # = current SDPO. Other sources replace the sibling text with an offline
        # GT-derived abstraction (cot / summary / rubric) looked up by question,
        # WITHOUT changing which samples are privileged: we substitute only where a
        # successful sibling already exists, so self_distillation_mask is unchanged and
        # the conditions differ only in privileged *content*. A lookup miss keeps the
        # sibling (never drops a sample). Env PRIVILEGED_SOURCE overrides the config. ---
        privileged_source = os.environ.get("PRIVILEGED_SOURCE") or self_distillation_cfg.get("privileged_source", "sibling")
        sd_source_replaced = sd_source_fallback = 0
        if privileged_source == "none":
            # Teacher-only-prompt control: the teacher sees the question and its
            # system prompt d but NOT the correct sibling solution c nor the feedback, so it knows nothing the
            # student does not. The samples keep their distillation mask (they would otherwise drop out as
            # "unprivileged"), so the loss covers the same rollouts as the sibling-conditioned run.
            self._sd_mask_override = [solution_strs[i] is not None or feedback_list[i] is not None
                                      for i in range(batch_size)]
            solution_strs = [None] * batch_size
            feedback_list = [None] * batch_size
        elif privileged_source != "sibling":
            field = {"cot": "cot", "summary": "summary", "rubric": "rubric"}.get(privileged_source)
            if field is None:
                raise ValueError(
                    "self_distillation.privileged_source must be one of "
                    f"sibling/cot/summary/rubric, got {privileged_source!r}")
            self._load_privileged_map(self_distillation_cfg)
            for i in range(batch_size):
                if solution_strs[i] is None:
                    continue  # no successful sibling -> stays unprivileged (mask unchanged)
                rec = self._lookup_privileged(prompt_texts[i])
                val = rec.get(field) if rec else None
                if isinstance(val, list):  # rubric -> bullet list
                    val = "\n".join(f"- {x}" for x in val)
                if val:
                    solution_strs[i] = val
                    sd_source_replaced += 1
                else:
                    sd_source_fallback += 1  # lookup miss -> keep sibling

        # Resolve an optional TEACHER-ONLY system prompt (reflection / workflow disposition).
        # The student never sees this — only the teacher conditions on it — so we can study whether
        # a system-prompt disposition transfers to the student via self-distillation. Read once.
        teacher_system_prompt = self_distillation_cfg.get("teacher_system_prompt", None)
        teacher_system_prompt_file = self_distillation_cfg.get("teacher_system_prompt_file", None)
        if teacher_system_prompt_file:
            if not hasattr(self, "_teacher_sys_prompt_cache"):
                with open(teacher_system_prompt_file) as _f:
                    self._teacher_sys_prompt_cache = _f.read().strip()
            teacher_system_prompt = self._teacher_sys_prompt_cache

        # Position of the teacher-only disposition
        # is placed. "system" (default) = prepended to the system message, far from the answer.
        # "after_privileged" = appended to the END of the user reprompt, right after the privileged
        # solution/feedback and just before the teacher generates — i.e. as close to the answer as
        # possible. Env override wins.
        # "before_privileged" = inserted between the problem statement and the privileged solution section
        # of the user reprompt (so the teacher reads d before it reads c).
        reflect_position = os.environ.get("REFLECT_POSITION") or self_distillation_cfg.get("reflect_position", "system")
        if reflect_position not in ("system", "after_privileged", "before_privileged"):
            raise ValueError(f"reflect_position must be system|after_privileged|before_privileged, got {reflect_position!r}")

        # MULTI-TEACHER: diversify the distillation TARGET across a prompt's rollouts.
        # MULTI_TEACHER=half => of the n rollouts per prompt, the FIRST
        # half distill toward the BARE teacher (no disposition) and the SECOND half toward the
        # teacher_system_prompt disposition. Rollouts are grouped consecutively (repeat interleave=True),
        # so (i % n_roll) is the rollout index within a prompt's group. Env override wins.
        multi_teacher = os.environ.get("MULTI_TEACHER") or self_distillation_cfg.get("multi_teacher", None)
        n_roll = self.config.actor_rollout_ref.rollout.n

        # MULTI_TEACHER=aspects: split each prompt's n rollouts evenly across K *different* teacher
        # dispositions (TEACHER_PROMPT_FILES=a1.txt,a2.txt,...). Unlike "half" — which spends half the
        # rollouts on an uninformative BARE teacher to buy diversity — every aspect still teaches, just a
        # different facet, so we get diversity WITHOUT losing signal. An entry of "bare" means no prompt.
        aspect_prompts = None
        if multi_teacher == "aspects":
            if not hasattr(self, "_aspect_prompt_cache"):
                _files = (os.environ.get("TEACHER_PROMPT_FILES")
                          or self_distillation_cfg.get("teacher_prompt_files", "") or "")
                _files = [f.strip() for f in _files.split(",") if f.strip()]
                cache = []
                for _f in _files:
                    if _f.lower() == "bare":
                        cache.append(None)
                    else:
                        with open(_f) as _fh:
                            cache.append(_fh.read().strip())
                self._aspect_prompt_cache = cache
                print(f"[SDPO multi-teacher] aspects mode: {len(cache)} dispositions from {_files}", flush=True)
            aspect_prompts = self._aspect_prompt_cache or None

        def _build_teacher_message(i: int) -> list[dict]:
            system_messages = list(batch.non_tensor_batch["raw_prompt"][i][:-1])
            # per-rollout teacher disposition (multi-teacher): first half of each prompt's rollouts use the
            # bare teacher (no disposition), the rest use the configured teacher_system_prompt.
            _tsp = teacher_system_prompt
            if multi_teacher == "half" and _tsp is not None and (i % n_roll) < (n_roll // 2):
                _tsp = None
            elif multi_teacher == "aspects" and aspect_prompts:
                # rollout idx -> aspect bucket: n rollouts split evenly over K dispositions
                _k = len(aspect_prompts)
                _tsp = aspect_prompts[min((i % n_roll) * _k // n_roll, _k - 1)]
            has_solution = solution_strs[i] is not None
            has_feedback = feedback_list[i] is not None
            feedback_only_without_solution = self_distillation_cfg.get("environment_feedback_only_without_solution", False)

            # If feedback_only_without_solution is True, only use feedback when no solution exists
            use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)

            # build solution section
            solution_section = ""
            if has_solution:
                solution_section = self_distillation_cfg.solution_template.format(
                    successful_previous_attempt=solution_strs[i]
                )

            # build feedback section
            feedback_section = ""
            if use_feedback:
                feedback_raw = feedback_list[i]
                # Optionally append ground truth answer for feedback-only cases
                if not has_solution and self_distillation_cfg.get("provide_ground_truth_in_feedback", False):
                    gt = reward_models[i].get("ground_truth") if isinstance(reward_models[i], dict) else None
                    if gt:
                        feedback_raw = f"{feedback_raw}\nThe correct answer is \\boxed{{{gt}}}."
                feedback_section = self_distillation_cfg.feedback_template.format(
                    feedback_raw=feedback_raw
                )

            if reprompt_style == "multi_turn" and (has_solution or use_feedback):
                # multi_turn mode: solution appears as a prior assistant turn,
                # feedback as a follow-up user turn. Simulates conversation history.
                # The final user turn asks model to try again → response follows.
                prior_assistant = solution_strs[i] if has_solution else ""
                feedback_text = feedback_list[i] if use_feedback else ""
                retry_prompt = "Please try again." if not feedback_text else f"{feedback_text} Please try again."

                return system_messages + [
                    {"role": "user", "content": prompt_texts[i]},
                    {"role": "assistant", "content": prior_assistant},
                    {"role": "user", "content": retry_prompt},
                ]

            elif reprompt_style == "system_prefix" and (has_solution or use_feedback):
                # system_prefix mode: solution/feedback injected into system message,
                # user message is bare prompt (identical to non-reprompted version).
                # This eliminates template-induced distribution shift in the user turn.
                context_text = self_distillation_cfg.get(
                    "reprompt_system_prefix_template",
                    "Here is a previous attempt at the problem that follows:"
                    "{solution}{feedback}",
                ).format(solution=solution_section, feedback=feedback_section)

                # Append context to last system message (or create new one)
                modified_system = list(system_messages)
                if modified_system and modified_system[-1]["role"] == "system":
                    modified_system[-1] = dict(modified_system[-1])
                    modified_system[-1]["content"] += "\n\n" + context_text
                else:
                    modified_system.append({"role": "system", "content": context_text})

                # User message is bare prompt — identical to non-reprompted
                return modified_system + [
                    {"role": "user", "content": prompt_texts[i]},
                ]

            # Default "suffix" mode: solution appended to prompt in user message
            # "before_privileged": d goes between the problem statement and the solution section.
            _before = bool(_tsp) and reflect_position == "before_privileged" and (has_solution or use_feedback)
            _prompt_for_tmpl = prompt_texts[i].rstrip() + "\n\n" + _tsp if _before else prompt_texts[i]
            if has_solution:
                reprompt_text = self_distillation_cfg.reprompt_template.format(
                    prompt=_prompt_for_tmpl,
                    solution=solution_section,
                    feedback=feedback_section,
                )
            elif use_feedback:
                # SDPO (base) builds the feedback-only reprompt from reprompt_template with an empty solution
                # section; AntiSD uses a dedicated template. null (our default) keeps the base behaviour.
                _fb_tmpl = self_distillation_cfg.get("reprompt_template_feedback_only", None)
                if _fb_tmpl:
                    reprompt_text = _fb_tmpl.format(prompt=_prompt_for_tmpl, feedback=feedback_section)
                else:
                    reprompt_text = self_distillation_cfg.reprompt_template.format(
                        prompt=_prompt_for_tmpl, solution="", feedback=feedback_section,
                    )
            else:
                suffix = self_distillation_cfg.get("teacher_prompt_suffix", "")
                reprompt_text = prompt_texts[i] + suffix if suffix else prompt_texts[i]

            # Place the teacher-only disposition. "after_privileged" appends it to the end of the user
            # reprompt (closest to the answer); "system" (default) prepends it to the system message.
            if _before:
                pass  # already inserted before the privileged section
            elif _tsp and reflect_position in ("after_privileged", "before_privileged"):
                # after_privileged, or before_privileged on a reprompt with no privileged section
                reprompt_text = reprompt_text.rstrip() + "\n\n" + _tsp
            elif _tsp:  # "system"
                if system_messages and system_messages[0].get("role") == "system":
                    system_messages = [
                        {"role": "system", "content": _tsp + "\n\n" + system_messages[0]["content"]}
                    ] + system_messages[1:]
                else:
                    system_messages = [{"role": "system", "content": _tsp}] + system_messages

            return system_messages + [
                {"role": "user", "content": reprompt_text},
            ]


        messages = [_build_teacher_message(i) for i in range(batch_size)]

        # --- debug: dump the first 3 teacher re-prompts so we can inspect the prior
        # knowledge injected into the self-teacher (correct sibling solution / feedback)
        # that the student never sees. Limited to the first 3 calls to avoid log spam. ---
        if getattr(self, "_sd_debug_dumps_remaining", 3) > 0:
            self._sd_debug_dumps_remaining = getattr(self, "_sd_debug_dumps_remaining", 3) - 1
            for i in range(min(3, batch_size)):
                teacher_sys = messages[i][0]["content"] if messages[i] and messages[i][0].get("role") == "system" else "(none)"
                block = (
                    f"\n===== [SDPO teacher re-prompt] step={self.global_steps} sample={i} "
                    f"has_solution={solution_strs[i] is not None} "
                    f"has_feedback={feedback_list[i] is not None} =====\n"
                    f"--- TEACHER system prompt (student does NOT see) ---\n{teacher_sys}\n"
                    f"--- STUDENT sees (bare prompt) ---\n{prompt_texts[i]}\n"
                    f"--- TEACHER sees (re-prompt user turn) ---\n{messages[i][-1]['content']}\n"
                    f"{'=' * 88}"
                )
                print(block, flush=True)
                # Ray only forwards a subset of actor stdout, so also persist to a stable
                # file for easy inspection (best-effort; must never break training).
                try:
                    exp = os.environ.get("EXPERIMENT", "run")
                    dump_dir = os.path.join(os.environ.get("SDPO_ROOT", "."), "output", "teacher_reprompts")
                    os.makedirs(dump_dir, exist_ok=True)
                    with open(os.path.join(dump_dir, f"{exp}.log"), "a") as fh:
                        fh.write(block + "\n")
                except Exception:
                    pass

        enable_thinking = self.config.data.apply_chat_template_kwargs.get("enable_thinking", True) if self.config.data.apply_chat_template_kwargs else True
        teacher_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            continue_final_message=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            padding=True,
            # SDPO (base): tokenizer truncation to max_reprompt_len (side = reprompt_truncation). AntiSD: no
            # tokenizer truncation, then head+tail truncation below. Off by default = base behaviour.
            **({"truncation": False} if _head_tail_reprompt else {"truncation": True, "max_length": max_reprompt_len}),
        )
        if _head_tail_reprompt:
            # Safety-net: head+tail truncation catches any overflows from budget estimation
            teacher_prompt = self._head_tail_truncate(teacher_prompt, max_reprompt_len)
        teacher_input_ids = torch.cat([teacher_prompt["input_ids"].to(device), responses], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt["attention_mask"].to(device), response_mask], dim=1)
        teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

        # SI (Solution Influence): Build nosol teacher batch (reference context without correct solution)
        # SI_t = t_sol - t_nosol measures how much knowing the correct solution helps predict each token.
        # si_reference controls what the nosol teacher sees:
        #   "bare": question only (default)
        #   "wrong_sibling": question + wrong sibling solution from same group (contrastive)
        ccir_cfg = self.config.actor_rollout_ref.actor.get("ccir", None)
        si_mode = ccir_cfg.get("si_mode", "none") if ccir_cfg else "none"
        si_reference = ccir_cfg.get("si_reference", "bare") if ccir_cfg else "bare"
        teacher_nosol_input_ids = None
        teacher_nosol_attention_mask = None
        teacher_nosol_position_ids = None
        si_wrong_sibling_fraction = None
        # Also need nosol for prm_construction modes that use t(nosol), or contrastive brake
        prm_construction = ccir_cfg.get("prm_construction", "raw") if ccir_cfg else "raw"
        contrastive_brake_beta = ccir_cfg.get("contrastive_brake_beta", 0.0) if ccir_cfg else 0.0
        contrastive_brake_adaptive = ccir_cfg.get("contrastive_brake_adaptive", False) if ccir_cfg else False
        kl_ref_beta = ccir_cfg.get("kl_ref_beta", 0.0) if ccir_cfg else 0.0
        needs_nosol = (
            (si_mode != "none" and si_mode != "teacher_only")
            or prm_construction in ("teacher_contrastive", "teacher_contrastive_reversed", "s_minus_t_wrong", "t_wrong_minus_s", "reverse_combined")
            or contrastive_brake_beta > 0
            or contrastive_brake_adaptive
            or kl_ref_beta > 0
        )
        if needs_nosol or si_mode != "none":   # their debug line; silent unless the nosol teacher is in play
            print(
                f"[nosol] si_mode={si_mode}, si_reference={si_reference}, "
                f"prm_construction={prm_construction}, needs_nosol={needs_nosol}"
            )
        if needs_nosol:
            if si_reference == "wrong_sibling":
                # Build failure_by_uid: collect wrong sibling indices per uid
                uids_array = batch.non_tensor_batch["uid"]
                failure_by_uid = defaultdict(list)
                for idx, uid in enumerate(uids_array):
                    if seq_scores[idx] < self_distillation_cfg.success_reward_threshold:
                        failure_by_uid[uid].append(idx)

                has_wrong_count = 0

                def _build_wrong_sibling_message(i: int) -> list[dict]:
                    nonlocal has_wrong_count
                    uid = uids_array[i]
                    wrong_idxs = [j for j in failure_by_uid[uid] if j != i]
                    if not wrong_idxs:
                        # Fallback to bare question
                        system_messages = batch.non_tensor_batch["raw_prompt"][i][:-1]
                        return system_messages + [{"role": "user", "content": prompt_texts[i]}]
                    has_wrong_count += 1
                    # Random wrong sibling
                    wrong_idx = wrong_idxs[np.random.randint(len(wrong_idxs))]
                    wrong_solution = response_texts[wrong_idx]
                    # Truncate wrong solution same way as correct
                    wrong_solution = self._truncate_solution_to_budget(
                        wrong_solution, prompt_texts[i], None,
                        max_reprompt_len, max_solution_tokens,
                    )
                    # Use same template structure as teacher message (solution only, no feedback)
                    system_messages = batch.non_tensor_batch["raw_prompt"][i][:-1]
                    si_wrong_label = ccir_cfg.get("si_wrong_label", "same") if ccir_cfg else "same"
                    if si_wrong_label == "honest":
                        template = self_distillation_cfg.get(
                            "wrong_solution_template",
                            "\nHere is an incorrect attempt:\n\n{successful_previous_attempt}\n\n")
                    else:
                        template = self_distillation_cfg.solution_template
                    solution_section = template.format(
                        successful_previous_attempt=wrong_solution
                    )
                    reprompt_text = self_distillation_cfg.reprompt_template.format(
                        prompt=prompt_texts[i],
                        solution=solution_section,
                        feedback="",
                    )
                    return system_messages + [{"role": "user", "content": reprompt_text}]

                nosol_messages = [_build_wrong_sibling_message(i) for i in range(batch_size)]
                si_wrong_sibling_fraction = has_wrong_count / batch_size
            elif si_reference == "feedback_contrastive":
                # Build nosol context with counterfactual (flipped) feedback.
                # Sol side has honest feedback (from existing teacher prompt).
                # Nosol side has opposite feedback.
                # SI_t = π(y|honest) - π(y|counterfactual) encodes correctness direction.
                success_threshold = self_distillation_cfg.success_reward_threshold

                def _build_counterfactual_feedback_message(i: int) -> list[dict]:
                    is_correct = seq_scores[i] >= success_threshold
                    # Counterfactual = opposite of reality
                    cf_raw = "Your answer is incorrect." if is_correct else "Your answer is correct."
                    # Append ground truth answer (symmetric with sol side)
                    if self_distillation_cfg.get("provide_ground_truth_in_feedback", False):
                        gt = reward_models[i].get("ground_truth") if isinstance(reward_models[i], dict) else None
                        if gt:
                            cf_raw = f"{cf_raw}\nThe correct answer is \\boxed{{{gt}}}."
                    feedback_section = self_distillation_cfg.feedback_template.format(
                        feedback_raw=cf_raw
                    )
                    reprompt_text = self_distillation_cfg.get(
                        "reprompt_template_feedback_only",
                        "{prompt}{feedback}\n\nBased on the feedback above, "
                        "please rethink the problem carefully and try to solve it again.\n",
                    ).format(
                        prompt=prompt_texts[i],
                        feedback=feedback_section,
                    )
                    system_messages = batch.non_tensor_batch["raw_prompt"][i][:-1]
                    return system_messages + [{"role": "user", "content": reprompt_text}]

                nosol_messages = [_build_counterfactual_feedback_message(i) for i in range(batch_size)]
            elif si_reference == "solution_contrastive":
                # Contrastive Context PRM: nosol uses a WRONG solution in the same template.
                # SI_t = log π(y|correct_sol) - log π(y|wrong_sol) = Δ_correct - Δ_wrong.
                # s_t(bare) exactly cancels since both sides share the same model and template.
                # Fallback: when no wrong solution exists, mirror sol side → SI_t = 0 → pure ORM.
                uids_array = batch.non_tensor_batch["uid"]
                failure_by_uid = defaultdict(list)
                for idx, uid in enumerate(uids_array):
                    if seq_scores[idx] < self_distillation_cfg.success_reward_threshold:
                        failure_by_uid[uid].append(idx)

                has_wrong_count = 0

                def _build_solution_contrastive_nosol(i: int) -> list[dict]:
                    nonlocal has_wrong_count
                    uid = uids_array[i]
                    wrong_idxs = [j for j in failure_by_uid[uid] if j != i]
                    if not wrong_idxs or solution_strs[i] is None:
                        # No wrong solution (all-correct) OR no correct solution (all-fail)
                        # → mirror sol side → SI_t ≈ 0 → pure ORM.
                        # All-fail case: sol side = bare prompt, so nosol must also = bare.
                        # Without this check, nosol = reprompt+wrong_sol → template mismatch.
                        return _build_teacher_message(i)

                    has_wrong_count += 1
                    wrong_idx = wrong_idxs[np.random.randint(len(wrong_idxs))]
                    wrong_solution = response_texts[wrong_idx]
                    wrong_solution = self._truncate_solution_to_budget(
                        wrong_solution, prompt_texts[i], None,
                        max_reprompt_len, max_solution_tokens,
                    )
                    # Template: use same as sol side for perfect symmetry (si_wrong_label="same")
                    # or honest framing (si_wrong_label="honest")
                    si_wrong_label = ccir_cfg.get("si_wrong_label", "same") if ccir_cfg else "same"
                    if si_wrong_label == "honest":
                        template = self_distillation_cfg.get(
                            "wrong_solution_template",
                            "\nHere is an incorrect attempt:\n\n{successful_previous_attempt}\n\n")
                    else:
                        template = self_distillation_cfg.solution_template
                    solution_section = template.format(
                        successful_previous_attempt=wrong_solution
                    )
                    # Include feedback (symmetric with sol side)
                    system_messages = batch.non_tensor_batch["raw_prompt"][i][:-1]
                    feedback_section = ""
                    feedback_only_without_solution = self_distillation_cfg.get(
                        "environment_feedback_only_without_solution", False)
                    if feedback_list[i] is not None and not feedback_only_without_solution:
                        feedback_section = self_distillation_cfg.feedback_template.format(
                            feedback_raw=feedback_list[i]
                        )

                    # Match reprompt_style with sol side for perfect template cancellation
                    if reprompt_style == "system_prefix":
                        context_text = self_distillation_cfg.get(
                            "reprompt_system_prefix_template",
                            "Here is a previous attempt at the problem that follows:"
                            "{solution}{feedback}",
                        ).format(solution=solution_section, feedback=feedback_section)
                        modified_system = list(system_messages)
                        if modified_system and modified_system[-1]["role"] == "system":
                            modified_system[-1] = dict(modified_system[-1])
                            modified_system[-1]["content"] += "\n\n" + context_text
                        else:
                            modified_system.append({"role": "system", "content": context_text})
                        return modified_system + [{"role": "user", "content": prompt_texts[i]}]

                    reprompt_text = self_distillation_cfg.reprompt_template.format(
                        prompt=prompt_texts[i],
                        solution=solution_section,
                        feedback=feedback_section,
                    )
                    return system_messages + [{"role": "user", "content": reprompt_text}]

                nosol_messages = [_build_solution_contrastive_nosol(i) for i in range(batch_size)]
                si_wrong_sibling_fraction = has_wrong_count / batch_size
            else:
                # bare: question only (existing behavior)
                def _build_bare_teacher_message(i: int) -> list[dict]:
                    """Teacher message with just the question — no solution, no feedback."""
                    system_messages = batch.non_tensor_batch["raw_prompt"][i][:-1]
                    return system_messages + [{"role": "user", "content": prompt_texts[i]}]

                nosol_messages = [_build_bare_teacher_message(i) for i in range(batch_size)]

            # Tokenize (shared path for both si_reference modes)
            bare_prompt = self.tokenizer.apply_chat_template(
                nosol_messages,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                continue_final_message=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                padding=True,
                truncation=False,
            )
            bare_prompt = self._head_tail_truncate(bare_prompt, max_reprompt_len)
            teacher_nosol_input_ids = torch.cat([bare_prompt["input_ids"].to(device), responses], dim=1)
            teacher_nosol_attention_mask = torch.cat([bare_prompt["attention_mask"].to(device), response_mask], dim=1)
            teacher_nosol_position_ids = compute_position_id_with_mask(teacher_nosol_attention_mask)

        # Compute which samples actually use feedback (accounting for environment_feedback_only_without_solution)
        feedback_only_without_solution = self_distillation_cfg.get("environment_feedback_only_without_solution", False)
        feedback_used = [
            feedback_list[i] is not None and (not feedback_only_without_solution or solution_strs[i] is None)
            for i in range(batch_size)
        ]

        # self_distillation_mask: controls which samples participate in KL loss
        require_solution = self_distillation_cfg.get("require_solution_for_distillation", False)
        _sd_mask_override = getattr(self, "_sd_mask_override", None)
        if _sd_mask_override is not None:  # PRIVILEGED_SOURCE=none: mask of the sibling-conditioned run
            self._sd_mask_override = None
            self_distillation_mask = torch.tensor(_sd_mask_override, dtype=torch.float32, device=device)
        elif require_solution:
            # Only samples with a solution (rollout or dataset) get KL loss;
            # feedback-only samples are masked out to avoid biased gradients.
            self_distillation_mask = torch.tensor(
                [solution_strs[i] is not None for i in range(batch_size)],
                dtype=torch.float32,
                device=device
            )
        else:
            # Original behavior: solution OR feedback triggers KL loss
            self_distillation_mask = torch.tensor(
                [solution_strs[i] is not None or feedback_used[i] for i in range(batch_size)],
                dtype=torch.float32,
                device=device
            )

        # SD_SKIP_TRUNCATED=1: drop TRUNCATED rollouts (hit max_response_length, no EOS) from the distillation
        # loss. With token-mean aggregation a few degenerate rollouts that run to the cap can dominate the token
        # count, and the teacher-forced target on such a prefix rarely emits EOS. Env-gated, default off.
        sd_truncated_masked = 0.0
        if os.environ.get("SD_SKIP_TRUNCATED", "").strip().lower() not in ("", "0", "false", "no", "off"):
            max_resp = int(self.config.data.max_response_length)
            truncated = response_mask.sum(-1) >= max_resp
            sd_truncated_masked = (truncated.float() * self_distillation_mask).mean().item()
            self_distillation_mask = self_distillation_mask * (~truncated).float()

        uids = set(batch.non_tensor_batch["uid"])
        num_with_feedback_available = sum(1 for f in feedback_list if f is not None)
        num_with_feedback_used = sum(1 for f in feedback_used if f)
        num_with_solution = sum(1 for s in solution_strs if s is not None)  # after fallback (rollout + dataset)
        metrics = {
            "self_distillation/success_group_fraction": len([uid for uid in uids if len(success_by_uid[uid]) > 0]) / len(uids),
            "self_distillation/success_sample_fraction": num_with_solution / batch_size,   # base key (our collectors read it)
            "self_distillation/solution_source_group": num_source_group / batch_size,
            "self_distillation/solution_source_external": num_source_external / batch_size,
            "self_distillation/total_solution_fraction": num_with_solution / batch_size,
            "self_distillation/feedback_available_fraction": num_with_feedback_available / batch_size,
            "self_distillation/feedback_used_fraction": num_with_feedback_used / batch_size,
            "self_distillation/reprompt_sample_fraction": self_distillation_mask.float().mean().item(),
            "self_distillation/source_replaced_fraction": sd_source_replaced / batch_size,
            "self_distillation/source_fallback_fraction": sd_source_fallback / batch_size,
            "self_distillation/truncated_masked_fraction": sd_truncated_masked,
        }
        if si_wrong_sibling_fraction is not None:
            metrics["self_distillation/si_wrong_sibling_fraction"] = si_wrong_sibling_fraction
        if si_mode != "none":
            metrics["self_distillation/si_reference"] = {"bare": 0, "wrong_sibling": 1, "feedback_contrastive": 2}.get(si_reference, -1)
            metrics["self_distillation/si_model"] = {"ema": 0, "student": 1}.get(
                ccir_cfg.get("si_model", "ema") if ccir_cfg else "ema", -1)
        if si_reference == "feedback_contrastive" and si_mode != "none":
            success_thr = self_distillation_cfg.success_reward_threshold
            n_correct = sum(1 for s in seq_scores if s >= success_thr)
            metrics["self_distillation/si_correct_fraction"] = n_correct / batch_size
            has_gt = self_distillation_cfg.get("provide_ground_truth_in_feedback", False)
            if has_gt:
                n_has_gt = sum(
                    1 for i in range(batch_size)
                    if isinstance(reward_models[i], dict) and reward_models[i].get("ground_truth")
                )
                metrics["self_distillation/si_has_ground_truth_fraction"] = n_has_gt / batch_size

        # Compute solution length and truncation metrics
        solution_lengths = []
        num_truncated = 0
        for i in range(batch_size):
            if solution_strs[i] is not None:
                sol_tokens = self.tokenizer.encode(solution_strs[i], add_special_tokens=False)
                solution_lengths.append(len(sol_tokens))
                if truncate_at_correct:
                    # Detect truncation: if solution ends exactly at a \boxed{} boundary,
                    # truncation removed post-answer content.
                    positions = self._find_all_boxed_positions(solution_strs[i])
                    if positions and any(end_pos == len(solution_strs[i]) for _, end_pos, _ in positions):
                        num_truncated += 1
        if solution_lengths:
            metrics["self_distillation/avg_solution_length_tokens"] = sum(solution_lengths) / len(solution_lengths)
        if truncate_at_correct:
            metrics["self_distillation/solution_truncated_fraction"] = num_truncated / max(len(solution_lengths), 1)

        # Collect debug samples for wandb table logging (up to 3 samples)
        debug_samples = []
        # Pick up to 3 diverse samples: 1 with solution (correct), 1 with solution (wrong), 1 feedback-only
        sample_indices = {"solution_correct": None, "solution_wrong": None, "feedback_only": None}
        for i in range(batch_size):
            has_sol = solution_strs[i] is not None
            is_correct = seq_scores[i] >= self_distillation_cfg.success_reward_threshold
            if has_sol and is_correct and sample_indices["solution_correct"] is None:
                sample_indices["solution_correct"] = i
            elif has_sol and not is_correct and sample_indices["solution_wrong"] is None:
                sample_indices["solution_wrong"] = i
            elif not has_sol and sample_indices["feedback_only"] is None:
                sample_indices["feedback_only"] = i
        for label, idx in sample_indices.items():
            if idx is not None:
                teacher_text = self.tokenizer.decode(teacher_input_ids[idx], skip_special_tokens=True)
                student_text = prompt_texts[idx]
                response_text = response_texts[idx]
                sample = {
                    "type": label,
                    "score": float(seq_scores[idx]),
                    "mask": float(self_distillation_mask[idx].item()),
                    "solution_len": len(solution_strs[idx]) if solution_strs[idx] else 0,
                    "student_prompt": student_text,
                    "teacher_context": teacher_text,
                    "response": response_text,
                }
                if teacher_nosol_input_ids is not None:
                    sample["nosol_context"] = self.tokenizer.decode(
                        teacher_nosol_input_ids[idx], skip_special_tokens=True)
                debug_samples.append(sample)
        if loss_mode in ("grpo_ccir", "grpo_st", "grpo_ca"):   # their W&B teacher-context table; not for our arms
            metrics["_debug_teacher_samples"] = debug_samples

        # CCIR cross-problem: build batches with swapped problem context
        # s(x'): student evaluates response y_i under different problem x'_j
        # t(x',y'_j): teacher evaluates response y_i under x'_j with x'_j's correct solution
        cross_student_input_ids = None
        cross_teacher_input_ids = None
        ccir_cross_problem_enabled = ccir_cfg.get("ccir_cross_problem", False) if ccir_cfg else False
        if ccir_cross_problem_enabled:
            uids_list = list(batch.non_tensor_batch["uid"])
            # Create cross-problem mapping: i → j where uid[i] != uid[j]
            rng = np.random.RandomState(self.global_steps * 1000 + 42)
            uid_to_indices: dict = {}
            for idx, uid in enumerate(uids_list):
                uid_to_indices.setdefault(uid, []).append(idx)
            # Try random shuffle with cross-UID constraint
            cross_indices = list(range(batch_size))
            found = False
            for _ in range(200):
                rng.shuffle(cross_indices)
                if all(uids_list[cross_indices[i]] != uids_list[i] for i in range(batch_size)):
                    found = True
                    break
            if not found:
                # Fallback: cyclic shift by first UID group size
                uid_order = list(uid_to_indices.keys())
                rng.shuffle(uid_order)
                flat_target = []
                for uid in uid_order:
                    flat_target.extend(uid_to_indices[uid])
                shift = len(uid_to_indices[uid_order[0]])
                cross_indices = [0] * batch_size
                for i in range(batch_size):
                    cross_indices[flat_target[i]] = flat_target[(i + shift) % batch_size]

            # Build cross-problem student batch: bare prompt of x'_j + response y_i
            cross_student_messages = []
            for i in range(batch_size):
                j = cross_indices[i]
                sys_msgs = batch.non_tensor_batch["raw_prompt"][j][:-1]
                user_msg = batch.non_tensor_batch["raw_prompt"][j][-1]
                cross_student_messages.append(sys_msgs + [user_msg])

            cross_student_prompt = self.tokenizer.apply_chat_template(
                cross_student_messages,
                tokenize=True, return_tensors="pt", return_dict=True,
                continue_final_message=False, add_generation_prompt=True,
                enable_thinking=enable_thinking, padding=True, truncation=False,
            )
            cross_student_prompt = self._head_tail_truncate(cross_student_prompt, max_reprompt_len)
            cross_student_input_ids = torch.cat([cross_student_prompt["input_ids"].to(device), responses], dim=1)
            cross_student_attention_mask = torch.cat([cross_student_prompt["attention_mask"].to(device), response_mask], dim=1)
            cross_student_position_ids = compute_position_id_with_mask(cross_student_attention_mask)

            # Build cross-problem teacher batch: x'_j prompt + x'_j's correct solution + response y_i
            # Reuse _build_teacher_message but with swapped prompt and solution.
            # Skip entirely when the selected mode does not need t':
            #   - full + beta=0                → PRM = s_specific only
            #   - blend_current_teacher        → uses current-problem teacher t(x), not t'(x')
            _ccir_cp_mode = ccir_cfg.get("ccir_cross_problem_mode", "full")
            _ccir_cp_beta = float(ccir_cfg.get("ccir_cross_problem_beta", 1.0))
            _skip_cross_teacher = (
                (_ccir_cp_mode == "full" and _ccir_cp_beta == 0.0)
                or (_ccir_cp_mode == "blend_current_teacher")
            )
            cross_solution_strs = [None] * batch_size
            if not _skip_cross_teacher:
                for i in range(batch_size):
                    j = cross_indices[i]
                    cross_solution_strs[i] = solution_strs[j]  # j's solution (may be None)

                def _build_cross_teacher_message(i):
                    j = cross_indices[i]
                    system_messages = batch.non_tensor_batch["raw_prompt"][j][:-1]
                    prompt_text_j = prompt_texts[j]
                    has_sol = cross_solution_strs[i] is not None
                    solution_section = ""
                    if has_sol:
                        solution_section = self_distillation_cfg.solution_template.format(
                            successful_previous_attempt=cross_solution_strs[i]
                        )
                    feedback_section = ""
                    has_fb = feedback_list[j] is not None
                    fb_only_no_sol = self_distillation_cfg.get("environment_feedback_only_without_solution", False)
                    if has_fb and (not fb_only_no_sol or not has_sol):
                        feedback_section = self_distillation_cfg.feedback_template.format(
                            feedback_raw=feedback_list[j]
                        )
                    content = prompt_text_j + solution_section + feedback_section
                    return system_messages + [{"role": "user", "content": content}]

                cross_teacher_messages = [_build_cross_teacher_message(i) for i in range(batch_size)]
                cross_teacher_prompt = self.tokenizer.apply_chat_template(
                    cross_teacher_messages,
                    tokenize=True, return_tensors="pt", return_dict=True,
                    continue_final_message=False, add_generation_prompt=True,
                    enable_thinking=enable_thinking, padding=True, truncation=False,
                )
                cross_teacher_prompt = self._head_tail_truncate(cross_teacher_prompt, max_reprompt_len)
                cross_teacher_input_ids = torch.cat([cross_teacher_prompt["input_ids"].to(device), responses], dim=1)
                cross_teacher_attention_mask = torch.cat([cross_teacher_prompt["attention_mask"].to(device), response_mask], dim=1)
                cross_teacher_position_ids = compute_position_id_with_mask(cross_teacher_attention_mask)

            metrics["self_distillation/cross_problem_enabled"] = 1.0
            metrics["self_distillation/cross_problem_skip_cross_teacher"] = 1.0 if _skip_cross_teacher else 0.0
            metrics["self_distillation/cross_problem_has_solution_fraction"] = (
                (sum(1 for s in cross_solution_strs if s is not None) / batch_size)
                if not _skip_cross_teacher else 0.0
            )

        tensors = {
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "teacher_position_ids": teacher_position_ids,
            "self_distillation_mask": self_distillation_mask,
        }
        if teacher_nosol_input_ids is not None:
            tensors["teacher_nosol_input_ids"] = teacher_nosol_input_ids
            tensors["teacher_nosol_attention_mask"] = teacher_nosol_attention_mask
            tensors["teacher_nosol_position_ids"] = teacher_nosol_position_ids
        if cross_student_input_ids is not None:
            tensors["cross_student_input_ids"] = cross_student_input_ids
            tensors["cross_student_attention_mask"] = cross_student_attention_mask
            tensors["cross_student_position_ids"] = cross_student_position_ids
        if cross_teacher_input_ids is not None:
            tensors["cross_teacher_input_ids"] = cross_teacher_input_ids
            tensors["cross_teacher_attention_mask"] = cross_teacher_attention_mask
            tensors["cross_teacher_position_ids"] = cross_teacher_position_ids
        if si_mode != "none":   # their debug line; only meaningful when the SI nosol teacher is built
            print(f"[nosol-build] teacher_nosol_input_ids is None: {teacher_nosol_input_ids is None}, "
                  f"cross_problem: {cross_student_input_ids is not None}, tensors keys: {list(tensors.keys())}")
        return DataProto.from_dict(tensors=tensors), metrics

    def _build_ccir_contrastive_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, list]],
        self_distillation_data: Optional[tuple],
    ) -> Optional[tuple[DataProto, dict[str, float]]]:
        """Build contrastive teacher batch for CCIR weighting.

        Creates shuffled-prompt teacher inputs: same feedback/solution but different prompt x'.
        This allows measuring how much of the teacher's signal is x-specific vs generic.
        """
        ccir_cfg = self.config.actor_rollout_ref.actor.get("ccir", None)
        if ccir_cfg is None or not ccir_cfg.enabled:
            return None
        if self_distillation_data is None:
            return None

        self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        device = batch.batch["input_ids"].device
        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        batch_size = batch.batch.batch_size[0]
        prompt_texts = [msgs[-1]["content"] for msgs in batch.non_tensor_batch["raw_prompt"]]
        uids = list(batch.non_tensor_batch["uid"])

        # Need at least 2 distinct UIDs to form a meaningful contrastive pair
        unique_uids = set(uids)
        if len(unique_uids) < 2:
            return None

        response_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]

        # Re-collect feedback and solutions (same logic as _maybe_build_self_distillation_batch)
        feedback_list = self._collect_feedback(
            include_environment_feedback=self_distillation_cfg.include_environment_feedback,
            reward_extra_infos_dict=reward_extra_infos_dict,
            batch_size=batch_size,
        )
        success_by_uid = self._collect_solutions_by_uid(
            batch, reward_tensor,
            success_reward_threshold=self_distillation_cfg.success_reward_threshold,
        )
        reward_models = batch.non_tensor_batch.get("reward_model", np.array([{}] * batch_size))
        solution_selection = self_distillation_cfg.get("solution_selection", "random")
        truncate_at_correct = self_distillation_cfg.get("truncate_solution_at_correct_answer", False)
        solution_source = self_distillation_cfg.get("solution_source", "group_first")
        extra_infos = batch.non_tensor_batch.get("extra_info", np.array([None] * batch_size))

        def _get_external_solution_ccir(i):
            ei = extra_infos[i] if i < len(extra_infos) else None
            sol = ei.get("solution") if isinstance(ei, dict) else None
            return sol if sol else None

        def _get_group_solutions_ccir():
            return [
                self._get_solution(
                    i, success_by_uid, uids, response_texts,
                    self_distillation_cfg.dont_reprompt_on_self_success,
                    self_distillation_cfg.get("remove_thinking_from_demonstration", False),
                    response_mask=response_mask,
                    ground_truth=(reward_models[i].get("ground_truth") if isinstance(reward_models[i], dict) else None),
                    solution_selection=solution_selection,
                    truncate_at_correct_answer=truncate_at_correct,
                )
                for i in range(batch_size)
            ]

        if solution_source in ("group_first", "group_only"):
            solution_strs = _get_group_solutions_ccir()
            if solution_source == "group_first":
                for i in range(batch_size):
                    if solution_strs[i] is None:
                        solution_strs[i] = _get_external_solution_ccir(i)
        elif solution_source in ("external_first", "external_only"):
            solution_strs = [_get_external_solution_ccir(i) for i in range(batch_size)]
            if solution_source == "external_first":
                group_solutions = _get_group_solutions_ccir()
                for i in range(batch_size):
                    if solution_strs[i] is None:
                        solution_strs[i] = group_solutions[i]

        # Budget-aware solution truncation (same as main distillation batch)
        max_reprompt_len = self_distillation_cfg.max_reprompt_len
        max_solution_tokens = self_distillation_cfg.get("max_solution_tokens", None)
        feedback_only_without_sol_ccir = self_distillation_cfg.get("environment_feedback_only_without_solution", False)
        for i in range(batch_size):
            if solution_strs[i] is not None:
                effective_feedback = None
                if not feedback_only_without_sol_ccir and feedback_list[i] is not None:
                    effective_feedback = feedback_list[i]
                solution_strs[i] = self._truncate_solution_to_budget(
                    solution_strs[i], prompt_texts[i], effective_feedback,
                    max_reprompt_len, max_solution_tokens,
                )

        feedback_only_without_solution = self_distillation_cfg.get("environment_feedback_only_without_solution", False)

        def _uid_aware_derangement(n: int, uids_list: list, seed: int) -> list[int]:
            """Create a derangement that avoids mapping to same UID.

            Ensures indices[i] != i AND uids_list[indices[i]] != uids_list[i]
            whenever possible (different prompt, not just different rollout).
            Falls back to no-self-mapping only if same-UID avoidance is impossible.
            """
            rng = np.random.RandomState(seed)

            # Build uid -> list of indices mapping
            uid_to_indices: dict = {}
            for idx, uid in enumerate(uids_list):
                uid_to_indices.setdefault(uid, []).append(idx)

            # Try random shuffle with cross-UID constraint
            indices = list(range(n))
            for _ in range(200):
                rng.shuffle(indices)
                if all(uids_list[indices[i]] != uids_list[i] for i in range(n)):
                    return indices

            # Greedy fallback: cycle across UID groups
            # Sort indices by UID, then shift by one UID group
            uid_order = list(uid_to_indices.keys())
            rng.shuffle(uid_order)
            result = [0] * n
            # Collect all indices ordered by shuffled UID groups
            flat_target = []
            for uid in uid_order:
                flat_target.extend(uid_to_indices[uid])
            # Shift by the size of the first UID group to guarantee cross-UID
            shift = len(uid_to_indices[uid_order[0]])
            for i in range(n):
                result[flat_target[i]] = flat_target[(i + shift) % n]

            # Verify no self-mapping (should hold if >= 2 distinct UIDs)
            if all(result[i] != i for i in range(n)):
                return result

            # Ultimate fallback: simple cyclic shift
            return [(i + 1) % n for i in range(n)]

        all_contrastive_tensors = {}

        for k in range(ccir_cfg.num_contrastive):
            # Use global_steps in seed so different steps get different shuffles
            shuffled_idx = _uid_aware_derangement(
                batch_size, uids, seed=self.global_steps * ccir_cfg.num_contrastive + k
            )

            # Build contrastive teacher messages: swapped prompt x' but same feedback/solution
            contrastive_messages = []
            for i in range(batch_size):
                j = shuffled_idx[i]  # contrastive index - use prompt from j
                system_messages = batch.non_tensor_batch["raw_prompt"][i][:-1]

                has_solution = solution_strs[i] is not None
                has_feedback = feedback_list[i] is not None
                use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)

                solution_section = ""
                if has_solution:
                    solution_section = self_distillation_cfg.solution_template.format(
                        successful_previous_attempt=solution_strs[i]
                    )

                feedback_section = ""
                if use_feedback:
                    feedback_raw = feedback_list[i]
                    if not has_solution and self_distillation_cfg.get("provide_ground_truth_in_feedback", False):
                        gt = reward_models[i].get("ground_truth") if isinstance(reward_models[i], dict) else None
                        if gt:
                            feedback_raw = f"{feedback_raw}\nThe correct answer is \\boxed{{{gt}}}."
                    feedback_section = self_distillation_cfg.feedback_template.format(
                        feedback_raw=feedback_raw
                    )

                # Use prompt from j (contrastive) but keep solution/feedback from i
                if has_solution:
                    reprompt_text = self_distillation_cfg.reprompt_template.format(
                        prompt=prompt_texts[j],  # SWAPPED prompt
                        solution=solution_section,
                        feedback=feedback_section,
                    )
                elif use_feedback:
                    reprompt_text = self_distillation_cfg.get(
                        "reprompt_template_feedback_only",
                        "{prompt}{feedback}\n\nBased on the feedback above, "
                        "please rethink the problem carefully and try to solve it again.\n",
                    ).format(
                        prompt=prompt_texts[j],  # SWAPPED prompt
                        feedback=feedback_section,
                    )
                else:
                    reprompt_text = prompt_texts[j]  # SWAPPED prompt

                contrastive_messages.append(
                    system_messages + [{"role": "user", "content": reprompt_text}]
                )

            # Tokenize
            enable_thinking = (
                self.config.data.apply_chat_template_kwargs.get("enable_thinking", True)
                if self.config.data.apply_chat_template_kwargs
                else True
            )
            contrastive_prompt = self.tokenizer.apply_chat_template(
                contrastive_messages,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                continue_final_message=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                padding=True,
                truncation=False,
            )
            contrastive_prompt = self._head_tail_truncate(contrastive_prompt, self_distillation_cfg.max_reprompt_len)
            contrastive_input_ids = torch.cat(
                [contrastive_prompt["input_ids"].to(device), responses], dim=1
            )
            contrastive_attention_mask = torch.cat(
                [contrastive_prompt["attention_mask"].to(device), response_mask], dim=1
            )
            contrastive_position_ids = compute_position_id_with_mask(contrastive_attention_mask)

            all_contrastive_tensors[f"ccir_teacher_input_ids_{k}"] = contrastive_input_ids
            all_contrastive_tensors[f"ccir_teacher_attention_mask_{k}"] = contrastive_attention_mask
            all_contrastive_tensors[f"ccir_teacher_position_ids_{k}"] = contrastive_position_ids

        ccir_metrics = {
            "ccir/num_contrastive": ccir_cfg.num_contrastive,
            # Track how many samples got a truly different prompt (cross-UID)
            "ccir/cross_uid_rate": sum(
                1 for i in range(batch_size) if uids[shuffled_idx[i]] != uids[i]
            ) / batch_size,
        }
        return DataProto.from_dict(tensors=all_contrastive_tensors), ccir_metrics

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid", "raw_prompt"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # Compute response lengths (non-padding tokens)
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            response_lengths = (output_ids != pad_id).sum(dim=-1).cpu().tolist()
            reward_extra_infos_dict["response_length"].extend(response_lengths)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            result = self._compute_or_extract_reward(test_batch, reward_fn=self.val_reward_fn, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_info = result.get("reward_extra_info", {})
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        # Response length statistics (per data-source and overall)
        response_lengths = reward_extra_infos_dict.get("response_length", [])
        if len(response_lengths) > 0:
            all_lens = np.array(response_lengths, dtype=np.float64)
            metric_dict["val-aux/response_length/mean"] = all_lens.mean()
            metric_dict["val-aux/response_length/median"] = float(np.median(all_lens))
            metric_dict["val-aux/response_length/min"] = all_lens.min()
            metric_dict["val-aux/response_length/max"] = all_lens.max()
            metric_dict["val-aux/response_length/std"] = all_lens.std()
            metric_dict["val-aux/response_length/p90"] = float(np.percentile(all_lens, 90))
            metric_dict["val-aux/response_length/p95"] = float(np.percentile(all_lens, 95))
            # Per data-source breakdown
            for ds in np.unique(data_sources):
                ds_mask = data_sources == ds
                ds_lens = all_lens[ds_mask]
                if len(ds_lens) > 0:
                    metric_dict[f"val-aux/{ds}/response_length/mean"] = ds_lens.mean()
                    metric_dict[f"val-aux/{ds}/response_length/median"] = float(np.median(ds_lens))
                    metric_dict[f"val-aux/{ds}/response_length/min"] = ds_lens.min()
                    metric_dict[f"val-aux/{ds}/response_length/max"] = ds_lens.max()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            # Note: mode is always "async" since sync mode is deprecated
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rm_resource_pool=rm_resource_pool,
        )

    def _write_best_step(self):
        """Record the best-so-far validation step in BOTH the checkpoint dir and the val-dump dir.
        eval_from_dump.py reads the val-dump copy so that the reported metrics come from the SAME
        step as the surviving (best) checkpoint, instead of from the last (post-peak) validation."""
        line = f"{getattr(self, '_es_best_step', 0)} {getattr(self, '_es_best', float('nan')):.6f}\n"
        for _d in (self.config.trainer.get("default_local_dir", None),
                   self.config.trainer.get("validation_data_dir", None)):
            if not _d:
                continue
            try:
                os.makedirs(_d, exist_ok=True)
                with open(os.path.join(_d, "best_step.txt"), "w") as _f:
                    _f.write(line)
            except Exception:
                pass

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)
        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
            group_name=self.config.trainer.get("group_name", None),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            # ---- early stopping (SDPO_EARLY_STOP=<patience in validations>) ----
            if int(os.environ.get("SDPO_EARLY_STOP", "0") or 0) > 0:
                _core = [v for k, v in val_metrics.items() if k.startswith("val-core/") and "/acc/mean@" in k]
                if _core:
                    # mean over val data sources (one source -> unchanged). On a RESUMED run global_steps is the resumed
                    # step, not 0: record that step, and carry over the best of the earlier attempt from best_step.txt
                    # The patience counter restarts at 0.
                    self._es_best, self._es_bad, self._es_best_step = float(sum(_core) / len(_core)), 0, int(self.global_steps)
                    if self.global_steps > 0:
                        try:
                            _vd = self.config.trainer.get("validation_data_dir", None)
                            _ps, _pv = open(os.path.join(_vd, "best_step.txt")).read().split()[:2]
                            if int(_ps) > 0 and float(_pv) > self._es_best:
                                self._es_best, self._es_best_step = float(_pv), int(_ps)
                                print(f"[early-stop] resumed: keeping earlier best {self._es_best:.4f} @ step {self._es_best_step}", flush=True)
                        except Exception:
                            pass
                    self._write_best_step()
                    print(f"[early-stop] baseline val {float(sum(_core) / len(_core)):.4f} @ step {int(self.global_steps)}", flush=True)
                else:
                    print("[early-stop] WARNING: no val-core/*/acc/mean@N metric found -- early stop is INACTIVE", flush=True)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # Truncate training to train_response_length if configured.
                    # Only response_mask is modified — attention_mask retains full length
                    # so metrics (response_length, clip_ratio) reflect actual generation.
                    train_response_length = self.config.data.get("train_response_length", None)
                    if train_response_length is not None:
                        full_response_length = batch.batch["responses"].size(1)
                        if train_response_length < full_response_length:
                            batch.batch["response_mask"][:, train_response_length:] = 0

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Compute or extract reward for training
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, return_dict=False
                            )

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        self_distillation_data = self._maybe_build_self_distillation_batch(batch, reward_tensor, reward_extra_infos_dict)
                        if self_distillation_data is not None:
                            self_distillation_batch, self_distillation_metrics = self_distillation_data
                            batch = batch.union(self_distillation_batch)
                            metrics.update(self_distillation_metrics)

                        # CCIR: Build contrastive teacher batch (only if self-distillation is active)
                        ccir_data = self._build_ccir_contrastive_batch(
                            batch, reward_tensor, reward_extra_infos_dict, self_distillation_data
                        )
                        if ccir_data is not None:
                            ccir_batch, ccir_metrics = ccir_data
                            batch = batch.union(ccir_batch)
                            metrics.update(ccir_metrics)

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                    # ---- SDPO early-stop + best-checkpoint (env SDPO_EARLY_STOP=<patience>, 0/unset = off) ----
                    # Tracks val-core/*/acc/mean@N. On improvement: save a checkpoint (run with save_freq=0 and
                    # max_actor_ckpt_to_keep=1 so the latest saved checkpoint IS the best one) + write best_step.txt.
                    # After <patience> consecutive validations without improvement, finish training early.
                    _es_pat = int(os.environ.get("SDPO_EARLY_STOP", "0") or 0)
                    if _es_pat > 0:
                        _core = [v for k, v in val_metrics.items() if k.startswith("val-core/") and "/acc/mean@" in k]
                        if _core:
                            _v = float(sum(_core) / len(_core))   # mean over val data sources (math: aime24/aime25/math500 equally weighted)
                            if _v > getattr(self, "_es_best", float("-inf")) + 1e-9:
                                self._es_best, self._es_bad, self._es_best_step = _v, 0, self.global_steps
                                # SDPO_ES_NO_CKPT=1: track the best step without writing the FSDP checkpoint.
                                # Early stopping and best_step.txt are unaffected; only the weights are not kept.
                                if os.environ.get("SDPO_ES_NO_CKPT", "") == "1":
                                    self._write_best_step()
                                    print(f"[early-stop] new best val {_v:.4f} @ step {self.global_steps} (checkpoint SKIPPED: SDPO_ES_NO_CKPT=1)", flush=True)
                                else:
                                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                                        self._save_checkpoint()
                                    self._write_best_step()
                                    print(f"[early-stop] new best val {_v:.4f} @ step {self.global_steps} (checkpoint saved)", flush=True)
                            else:
                                self._es_bad = getattr(self, "_es_bad", 0) + 1
                                print(f"[early-stop] no improvement ({self._es_bad}/{_es_pat}) -- best "
                                      f"{getattr(self, '_es_best', float('nan')):.4f} @ step {getattr(self, '_es_best_step', 0)}", flush=True)
                                if self._es_bad >= _es_pat:
                                    print(f"[early-stop] STOP at step {self.global_steps}; best val "
                                          f"{self._es_best:.4f} @ step {self._es_best_step}", flush=True)
                                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                                    progress_bar.close()
                                    return

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                # Log self-distillation teacher context debug samples to wandb (at test_freq to save quota)
                debug_samples = metrics.pop("_debug_teacher_samples", None)
                test_freq = self.config.trainer.get("test_freq", 10)
                if debug_samples and "wandb" in self.config.trainer.logger and self.global_steps % test_freq == 0:
                    try:
                        import wandb
                        if wandb.run is not None:
                            columns = ["step", "type", "score", "mask", "solution_len",
                                       "student_prompt", "teacher_context", "response"]
                            if not hasattr(self, "_teacher_debug_table"):
                                self._teacher_debug_table = wandb.Table(columns=columns)
                            new_table = wandb.Table(columns=columns, data=self._teacher_debug_table.data)
                            for s in debug_samples:
                                new_table.add_data(
                                    self.global_steps, s["type"], s["score"], s["mask"],
                                    s["solution_len"], s["student_prompt"],
                                    s["teacher_context"], s["response"],
                                )
                            self._teacher_debug_table = new_table
                            wandb.log({"self_distillation/teacher_context_debug": new_table}, step=self.global_steps)
                    except Exception as e:
                        print(f"[DEBUG] Failed to log teacher context table: {e}")

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
