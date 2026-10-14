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
Single Process Actor
"""

import logging
import math
import os
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_self_distillation_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, slice_input_tensor, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TrustRegionTeacher(nn.Module):
    def __init__(self, ref_module: nn.Module, student_module: nn.Module, mix_coef: float) -> None:
        super().__init__()
        self.ref_module = ref_module
        self.student_module = student_module
        self.mix_coef = float(mix_coef)

    def forward(self, *args, **kwargs):
        ref_out = self.ref_module(*args, **kwargs)
        student_out = self.student_module(*args, **kwargs)
        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out[0]
        student_logits = student_out.logits if hasattr(student_out, "logits") else student_out[0]
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.teacher_module: Optional[nn.Module] = None
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        # Step counter for features like ca_lambda_step_cutoff
        self._global_training_steps = 0

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

        # MaxEnt RL state (lazy-initialized from config on first use)
        self._maxent_alpha = None
        self._maxent_h_target = None

    def _update_teacher(self) -> None:
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if not self_distillation_cfg or loss_mode not in ("sdpo", "grpo_ccir", "grpo_st", "grpo_ca"):
            return
        teacher_regularization = getattr(self_distillation_cfg, "teacher_regularization", "ema")
        if teacher_regularization != "ema":
            return
        update_rate = getattr(self_distillation_cfg, "teacher_update_rate", 0.0)
        if update_rate == 0.0:
            return
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise ValueError("EMA teacher requires a separate teacher_module in the actor worker.")
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_module.parameters(),
                self.actor_module.parameters(),
            ):
                student_data = student_param.data.to(device=teacher_param.device)
                teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)

    @staticmethod
    def _has_non_empty_multi_modal_inputs(multi_modal_inputs) -> bool:
        if multi_modal_inputs is None:
            return False
        for inputs in multi_modal_inputs:
            if inputs is None:
                continue
            inputs = getattr(inputs, "data", inputs)
            if isinstance(inputs, dict):
                if not inputs:
                    continue
                for value in inputs.values():
                    if value is None:
                        continue
                    if isinstance(value, torch.Tensor) and value.numel() == 0:
                        continue
                    return True
            else:
                return True
        return False

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        return_all_logps: bool = False,
        distill_topk: Optional[int] = None,
        topk_indices: Optional[torch.Tensor] = None,
        module: Optional[nn.Module] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
                if distill_topk or topk_indices is set:
                    topk_logps: (bs, response_len, k)
                    topk_indices: (bs, response_len, k)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        use_topk = distill_topk is not None or topk_indices is not None
        compute_all_logps = return_all_logps and not use_topk
        return_topk_indices = use_topk and topk_indices is None
        if (return_all_logps or use_topk) and self.use_fused_kernels:
            raise ValueError("Logit distillation requires disabling fused kernels.")

        model = module or self.actor_module

        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
                and not return_all_logps
                and not use_topk
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=model,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(model, "module", model).config,
                        "vision_config",
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    all_logps_rmpad = torch.log_softmax(logits_rmpad, dim=-1) if compute_all_logps else None

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits_rmpad.shape[-1])
                            topk_logits_rmpad, topk_indices_rmpad = torch.topk(logits_rmpad, topk, dim=-1)
                        else:
                            topk = topk_indices.size(-1)
                            full_topk_indices = torch.zeros(
                                batch_size,
                                seqlen,
                                topk,
                                device=topk_indices.device,
                                dtype=topk_indices.dtype,
                            )
                            full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
                            topk_indices_rmpad = index_first_axis(
                                rearrange(full_topk_indices, "b s k -> (b s) k"), indices
                            )
                            if self.use_ulysses_sp:
                                topk_indices_rmpad = slice_input_tensor(
                                    topk_indices_rmpad.unsqueeze(0), dim=1, padding=True
                                ).squeeze(0)
                            topk_logits_rmpad = torch.gather(logits_rmpad, dim=-1, index=topk_indices_rmpad)
                        logsumexp_rmpad = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                        topk_logps_rmpad = topk_logits_rmpad - logsumexp_rmpad

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if use_topk:
                        topk_logps_rmpad = gather_outputs_and_unpad(
                            topk_logps_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        if return_topk_indices:
                            topk_indices_rmpad = gather_outputs_and_unpad(
                                topk_indices_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    if compute_all_logps:
                        all_logps_rmpad = all_logps_rmpad[:0]
                    if use_topk:
                        topk_logps_rmpad = topk_logps_rmpad[:0]
                        if return_topk_indices:
                            topk_indices_rmpad = topk_indices_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if compute_all_logps:
                    full_all_logps = pad_input(
                        hidden_states=all_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if use_topk:
                    full_topk_logps = pad_input(
                        hidden_states=topk_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    if return_topk_indices:
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if compute_all_logps:
                    all_logps = full_all_logps[:, -response_length - 1 : -1, :]
                if use_topk:
                    topk_logps = full_topk_logps[:, -response_length - 1 : -1, :]
                    if return_topk_indices:
                        topk_indices = full_topk_indices[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if compute_all_logps:
                        all_logps = torch.log_softmax(logits, dim=-1)
                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits.size(-1))
                            topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)
                        else:
                            topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
                        logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
                        topk_logps = topk_logits - logsumexp
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if compute_all_logps:
                outputs["all_logps"] = all_logps
            if use_topk:
                outputs["topk_logps"] = topk_logps
                if return_topk_indices:
                    outputs["topk_indices"] = topk_indices
            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
            return grad_norm

        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

        self_distillation_enabled = loss_mode in ("sdpo", "grpo_ccir", "grpo_st", "grpo_ca")
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        if self_distillation_enabled:
            self_distillation_required_keys = {
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_position_ids",
                "self_distillation_mask",
            }
            assert self_distillation_required_keys.issubset(set(data.batch.keys())), f"Missing required keys: {self_distillation_required_keys - set(data.batch.keys())}"

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self_distillation_enabled:
            select_keys.extend(list(self_distillation_required_keys))
        # CCIR: include contrastive teacher keys if present
        ccir_cfg = getattr(self.config, "ccir", None)
        if ccir_cfg and ccir_cfg.enabled and self_distillation_enabled:
            for k in range(ccir_cfg.num_contrastive):
                for suffix in ["input_ids", "attention_mask", "position_ids"]:
                    key = f"ccir_teacher_{suffix}_{k}"
                    if key in data.batch.keys():
                        select_keys.append(key)
        # SI / nosol: include nosol teacher keys if SI is enabled or prm_construction needs them
        si_mode = ccir_cfg.get("si_mode", "none") if ccir_cfg else "none"
        prm_construction_select = ccir_cfg.get("prm_construction", "raw") if ccir_cfg else "raw"
        contrastive_brake_beta_select = ccir_cfg.get("contrastive_brake_beta", 0.0) if ccir_cfg else 0.0
        contrastive_brake_adaptive_select = ccir_cfg.get("contrastive_brake_adaptive", False) if ccir_cfg else False
        kl_ref_beta_select = ccir_cfg.get("kl_ref_beta", 0.0) if ccir_cfg else 0.0
        needs_nosol_keys = (
            si_mode != "none"
            or prm_construction_select in ("teacher_contrastive", "teacher_contrastive_reversed", "s_minus_t_wrong", "t_wrong_minus_s", "reverse_combined")
            or contrastive_brake_beta_select > 0
            or contrastive_brake_adaptive_select
            or kl_ref_beta_select > 0
        )
        if needs_nosol_keys and self_distillation_enabled:
            for suffix in ["input_ids", "attention_mask", "position_ids"]:
                key = f"teacher_nosol_{suffix}"
                if key in data.batch.keys():
                    select_keys.append(key)
        # CCIR cross-problem keys
        ccir_cross_problem_select = ccir_cfg.get("ccir_cross_problem", False) if ccir_cfg else False
        if ccir_cross_problem_select and self_distillation_enabled:
            for prefix in ["cross_student", "cross_teacher"]:
                for suffix in ["input_ids", "attention_mask", "position_ids"]:
                    key = f"{prefix}_{suffix}"
                    if key in data.batch.keys():
                        select_keys.append(key)
        if loss_mode in ("grpo_ccir", "grpo_st", "grpo_ca"):   # their debug line; silent for our loss modes
            _batch_keys = list(data.batch.keys()) if data.batch is not None else []
            print(f"[select_keys] needs_nosol={needs_nosol_keys}, prm={prm_construction_select}, "
                  f"nosol_in_batch={'teacher_nosol_input_ids' in _batch_keys}, "
                  f"nosol_in_select={'teacher_nosol_input_ids' in select_keys}, "
                  f"all_batch_keys={_batch_keys}")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        did_update = False
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                # Some diagnostics need multiple sequences to be meaningful. With the default
                # ppo_micro_batch_size_per_gpu=1, per-micro-batch correlation/std stats collapse
                # to zero, so accumulate per-seq summaries across the whole mini-batch and emit
                # the diagnostics once after gradient accumulation.
                mini_batch_prm_seq_means = []
                mini_batch_orm_seq_means = []
                # Token-level PRM-ORM correlation accumulators (all-reduced across DP ranks
                # at mini-batch boundary). Stores [N, Σx, Σy, Σxy, Σx², Σy²] over valid tokens.
                mini_batch_corr_stats = None

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    _ccir = self.config.ccir
                    _maxent_active = (_ccir.get("maxent_coeff", "none") != "none") or _ccir.get("maxent_entropy_gate", False)
                    _entropy_gate_active = _ccir.get("entropy_gate_mode", "none") != "none"
                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0) or _maxent_active or _entropy_gate_active
                    self_distillation_mask = model_inputs.get("self_distillation_mask") if self_distillation_enabled else None
                    if self_distillation_enabled:
                        assert not has_multi_modal_inputs, "Multi-modal inputs are not supported for distillation"

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    teacher_regularization = self_distillation_cfg.get("teacher_regularization", "ema")
                    if teacher_regularization == "trust-region" and self.use_fused_kernels:
                        raise ValueError("trust-region teacher requires disabling fused kernels to access logits.")
                    # all return: (bsz, response_length)
                    return_all_logps = self_distillation_cfg.full_logit_distillation and not self_distillation_cfg.distillation_topk
                    distill_topk = self_distillation_cfg.distillation_topk if self_distillation_cfg.full_logit_distillation else None
                    # grpo_st only needs sampled-token log probs, skip expensive topk extraction
                    if loss_mode in ("grpo_st", "grpo_ca"):
                        return_all_logps = False
                        distill_topk = None
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_all_logps=return_all_logps,
                        distill_topk=distill_topk,
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None
                    student_all_logps = outputs.get("all_logps") if return_all_logps else None
                    student_topk_logps = outputs.get("topk_logps") if distill_topk else None
                    student_topk_indices = outputs.get("topk_indices") if distill_topk else None

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    if self_distillation_enabled:
                        teacher_inputs = {
                            "responses": model_inputs["responses"],
                            "input_ids": model_inputs["teacher_input_ids"],
                            "attention_mask": model_inputs["teacher_attention_mask"],
                            "position_ids": model_inputs["teacher_position_ids"],
                        }
                        teacher_model = self.teacher_module or self.actor_module
                        if teacher_regularization == "trust-region" and (
                            self.teacher_module is None or self.teacher_module is self.actor_module
                        ):
                            raise ValueError("trust-region teacher requires a separate teacher_module in the actor worker.")
                        with torch.no_grad():
                            teacher_outputs = self._forward_micro_batch(
                                teacher_inputs,
                                temperature=temperature,
                                calculate_entropy=False,
                                return_all_logps=return_all_logps,
                                distill_topk=distill_topk,
                                topk_indices=student_topk_indices,
                                module=teacher_model,
                            )
                        teacher_log_prob = teacher_outputs["log_probs"]
                        teacher_all_logps = teacher_outputs.get("all_logps") if return_all_logps else None
                        teacher_topk_logps = teacher_outputs.get("topk_logps") if distill_topk else None

                        # CCIR: Contrastive teacher forward
                        ccir_contrastive_teacher_log_probs = None
                        ccir_contrastive_teacher_topk_log_probs = None
                        if ccir_cfg and ccir_cfg.enabled:
                            contrastive_log_probs_list = []
                            contrastive_topk_log_probs_list = []
                            for ck in range(ccir_cfg.num_contrastive):
                                contrastive_inputs = {
                                    "responses": model_inputs["responses"],
                                    "input_ids": model_inputs[f"ccir_teacher_input_ids_{ck}"],
                                    "attention_mask": model_inputs[f"ccir_teacher_attention_mask_{ck}"],
                                    "position_ids": model_inputs[f"ccir_teacher_position_ids_{ck}"],
                                }
                                with torch.no_grad():
                                    contrastive_outputs = self._forward_micro_batch(
                                        contrastive_inputs,
                                        temperature=temperature,
                                        calculate_entropy=False,
                                        return_all_logps=False,
                                        distill_topk=distill_topk,
                                        topk_indices=student_topk_indices,
                                        module=teacher_model,
                                    )
                                contrastive_log_probs_list.append(contrastive_outputs["log_probs"])
                                if distill_topk is not None:
                                    contrastive_topk_log_probs_list.append(contrastive_outputs["topk_logps"])
                            # Average across K shuffles: [batch, seq_len]
                            ccir_contrastive_teacher_log_probs = torch.stack(contrastive_log_probs_list).mean(dim=0)
                            if contrastive_topk_log_probs_list:
                                # Average across K shuffles: [batch, seq_len, k]
                                ccir_contrastive_teacher_topk_log_probs = torch.stack(contrastive_topk_log_probs_list).mean(dim=0)

                        # SI: Forward WITHOUT solution (bare/wrong_sibling context)
                        teacher_nosol_log_prob = None
                        si_mode = ccir_cfg.get("si_mode", "none") if ccir_cfg else "none"
                        si_model_source = ccir_cfg.get("si_model", "ema") if ccir_cfg else "ema"
                        prm_construction = ccir_cfg.get("prm_construction", "raw") if ccir_cfg else "raw"
                        # Need nosol forward for SI modes OR prm_construction modes that use nosol
                        contrastive_brake_beta_fwd = ccir_cfg.get("contrastive_brake_beta", 0.0)
                        contrastive_brake_adaptive_fwd = ccir_cfg.get("contrastive_brake_adaptive", False)
                        needs_nosol = (si_mode != "none") or (prm_construction in (
                            "teacher_contrastive", "teacher_contrastive_reversed",
                            "s_minus_t_wrong", "t_wrong_minus_s", "reverse_combined",
                        )) or contrastive_brake_beta_fwd > 0 or contrastive_brake_adaptive_fwd
                        if needs_nosol and "teacher_nosol_input_ids" in model_inputs:
                            if si_model_source == "student" and ccir_cfg.get("si_reference", "bare") == "bare":
                                # Optimization: π_student(y|x) = already-computed log_prob. Zero extra forwards.
                                teacher_nosol_log_prob = log_prob.detach()
                            else:
                                # si_model="ema" → teacher_model; si_model="student" → None (→ self.actor_module)
                                si_nosol_module = teacher_model if si_model_source == "ema" else None
                                nosol_inputs = {
                                    "responses": model_inputs["responses"],
                                    "input_ids": model_inputs["teacher_nosol_input_ids"],
                                    "attention_mask": model_inputs["teacher_nosol_attention_mask"],
                                    "position_ids": model_inputs["teacher_nosol_position_ids"],
                                }
                                with torch.no_grad():
                                    nosol_outputs = self._forward_micro_batch(
                                        nosol_inputs,
                                        temperature=temperature,
                                        calculate_entropy=False,
                                        return_all_logps=False,
                                        module=si_nosol_module,
                                    )
                                teacher_nosol_log_prob = nosol_outputs["log_probs"]

                        # CCIR cross-problem forwards: s(x') and t(x', y')
                        cross_student_log_prob = None
                        cross_teacher_log_prob = None
                        if "cross_student_input_ids" in model_inputs:
                            cross_student_inputs = {
                                "responses": model_inputs["responses"],
                                "input_ids": model_inputs["cross_student_input_ids"],
                                "attention_mask": model_inputs["cross_student_attention_mask"],
                                "position_ids": model_inputs["cross_student_position_ids"],
                            }
                            with torch.no_grad():
                                cross_student_outputs = self._forward_micro_batch(
                                    cross_student_inputs,
                                    temperature=temperature,
                                    calculate_entropy=False,
                                    return_all_logps=False,
                                    module=None,  # student model
                                )
                            cross_student_log_prob = cross_student_outputs["log_probs"]
                        if "cross_teacher_input_ids" in model_inputs:
                            cross_teacher_inputs = {
                                "responses": model_inputs["responses"],
                                "input_ids": model_inputs["cross_teacher_input_ids"],
                                "attention_mask": model_inputs["cross_teacher_attention_mask"],
                                "position_ids": model_inputs["cross_teacher_position_ids"],
                            }
                            with torch.no_grad():
                                cross_teacher_outputs = self._forward_micro_batch(
                                    cross_teacher_inputs,
                                    temperature=temperature,
                                    calculate_entropy=False,
                                    return_all_logps=False,
                                    module=teacher_model,  # EMA model
                                )
                            cross_teacher_log_prob = cross_teacher_outputs["log_probs"]

                        # SI: sol-side forward when si_model="student"
                        si_sol_log_prob = teacher_log_prob  # default: reuse EMA teacher forward
                        if si_mode != "none" and si_model_source == "student":
                            sol_inputs = {
                                "responses": model_inputs["responses"],
                                "input_ids": model_inputs["teacher_input_ids"],
                                "attention_mask": model_inputs["teacher_attention_mask"],
                                "position_ids": model_inputs["teacher_position_ids"],
                            }
                            with torch.no_grad():
                                sol_outputs = self._forward_micro_batch(
                                    sol_inputs,
                                    temperature=temperature,
                                    calculate_entropy=False,
                                    return_all_logps=False,
                                    module=None,  # → self.actor_module (student)
                                )
                            si_sol_log_prob = sol_outputs["log_probs"]

                        if loss_mode == "grpo_ccir":
                            # ---- Part 1: L_GRPO with original advantages (no S_t modification) ----
                            policy_loss_fn = get_policy_loss_fn("vanilla")
                            pg_loss, pg_metrics = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                            )

                            # ---- Part 2: L_CCIR_KL (distributional topk KL with CCIR weighting) ----
                            kl_loss_val = torch.tensor(0.0, device=log_prob.device)
                            kl_metrics = {}
                            if ccir_cfg and ccir_cfg.enabled and ccir_cfg.kl_coeff > 0:
                                kl_loss_val, kl_metrics = compute_self_distillation_loss(
                                    student_log_probs=log_prob,
                                    teacher_log_probs=teacher_log_prob,
                                    response_mask=response_mask,
                                    self_distillation_config=self_distillation_cfg,
                                    old_log_probs=old_log_prob,
                                    student_topk_log_probs=student_topk_logps,
                                    teacher_topk_log_probs=teacher_topk_logps,
                                    self_distillation_mask=self_distillation_mask,
                                    loss_agg_mode=loss_agg_mode,
                                    rollout_is_weights=rollout_is_weights,
                                    contrastive_teacher_log_probs=ccir_contrastive_teacher_log_probs,
                                    contrastive_teacher_topk_log_probs=ccir_contrastive_teacher_topk_log_probs,
                                    ccir_config=ccir_cfg,
                                )

                            # ---- Combine: L = L_GRPO + γ · L_CCIR_KL ----
                            pg_loss = pg_loss + ccir_cfg.kl_coeff * kl_loss_val

                            # ---- Metrics ----
                            pg_metrics["grpo_ccir/pg_loss"] = (pg_loss - ccir_cfg.kl_coeff * kl_loss_val).detach().item()
                            pg_metrics["grpo_ccir/kl_loss"] = kl_loss_val.detach().item()
                            pg_metrics["grpo_ccir/kl_coeff"] = ccir_cfg.kl_coeff
                            pg_metrics.update(kl_metrics)
                            pg_metrics["self_distillation/empty_target_batch"] = self_distillation_mask.sum().item() == 0
                            micro_batch_metrics.update(pg_metrics)
                        elif loss_mode == "grpo_st":
                            # ---- grpo_st: S_t-modulated GRPO advantages (token-level, no vocab KL) ----
                            # A_refined(t) = A_seq * (1 + alpha * S_t_norm(t))
                            # When A_seq=0 (all rollouts same outcome), refined advantage is also 0,
                            # consistent with GRPO's behavior of producing no gradient for uniform groups.

                            # Step 1: Compute S_t for sampled token
                            ccir_S_t = (teacher_log_prob - ccir_contrastive_teacher_log_probs).detach()  # (bs, seq)

                            # Step 2: Normalize S_t per micro-batch
                            valid_mask = response_mask.bool()
                            if self_distillation_mask is not None:
                                valid_mask = valid_mask & self_distillation_mask.unsqueeze(-1).bool()
                            valid_st = ccir_S_t[valid_mask]
                            if valid_st.numel() > 0:
                                st_mean = valid_st.mean()
                                st_std = valid_st.std().clamp(min=1e-6)
                                S_t_norm = (ccir_S_t - st_mean) / st_std
                            else:
                                S_t_norm = torch.zeros_like(ccir_S_t)

                            # Step 3: Modulate advantages
                            alpha = ccir_cfg.st_advantage_alpha
                            refined_advantages = advantages * (1.0 + alpha * S_t_norm)

                            # Step 4: Vanilla PPO loss with refined advantages
                            policy_loss_fn = get_policy_loss_fn("vanilla")
                            pg_loss, pg_metrics = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=refined_advantages,
                                response_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                            )

                            # Step 5: Metrics
                            pg_metrics["grpo_st/S_t_mean"] = ccir_S_t[response_mask.bool()].mean().item() if response_mask.any() else 0.0
                            pg_metrics["grpo_st/S_t_std"] = ccir_S_t[response_mask.bool()].std().item() if response_mask.any() else 0.0
                            pg_metrics["grpo_st/modulation_mean"] = (alpha * S_t_norm[response_mask.bool()]).abs().mean().item() if response_mask.any() else 0.0
                            pg_metrics["grpo_st/pg_loss"] = pg_loss.detach().item()
                            pg_metrics["self_distillation/empty_target_batch"] = self_distillation_mask.sum().item() == 0
                            micro_batch_metrics.update(pg_metrics)
                        elif loss_mode == "grpo_ca":
                            # ---- grpo_ca: Full PRM credit assignment ----
                            # A_t = orm_weight * A_seq + ca_lambda * PRM_processed
                            # PRM_t = (t - s) - ccir_weight * (t - c)
                            #   t = log π̄(y|x,z), s = log π_s(y|x), c = log π̄(y|x',z)
                            #
                            # Reset per-micro-batch diagnostics/gating state explicitly.
                            # These locals are read later by gates/metrics; without reset they
                            # can leak values from the previous micro-batch iteration.
                            _ccir_s_specific = None
                            _ccir_generic_gap = None
                            _ccir_generic_gap_weight = 0.0
                            _ca_lambda_broadcast = None
                            ca_lambda_seq = None
                            seq_teacher_perp = None
                            seq_student_perp = None
                            seq_ts_ratio = None
                            teacher_perp = 0.0
                            _mean_shift_amount = 0.0
                            _prm_c_std = None
                            _orm_c_std = None
                            _forward_token_weights = None
                            _forward_token_log_weights = None
                            _forward_token_clip_frac = 0.0
                            _forward_token_input_mean = 0.0
                            _forward_token_input_abs_mean = 0.0
                            _forward_token_output_mean = 0.0
                            _forward_token_output_abs_mean = 0.0
                            _forward_token_gap = None
                            _renyi_active = False
                            _renyi_clip_frac = 0.0
                            _renyi_prm_mean = 0.0
                            _renyi_prm_abs_mean = 0.0
                            _renyi_u_clipped_frac = 0.0
                            _jsd_active = False
                            _jsd_clip_frac = 0.0
                            _jsd_prm_mean = 0.0
                            _jsd_prm_abs_mean = 0.0
                            _jsd_u_clipped_frac = 0.0
                            _jsd_gap_mean = 0.0
                            _jsd_gap_abs_mean = 0.0
                            _u_clip_active = False
                            _u_clip_sigma_ref = 0.0
                            _u_clip_range = 0.0
                            _u_clip_frac = 0.0
                            _u_std_ratio = 1.0

                            # Step 1: Implicit PRM = t - s (teacher minus student)
                            # Positive when teacher (seeing feedback z) assigns higher prob → token is a key reasoning step.
                            # Sign: t - s gives negative feedback on s_t (∂PRM/∂s = -1 → self-correcting).
                            # Note: the original SDPO loss uses (s - t) · s which is correct for reverse KL minimization,
                            # but as a GRPO advantage signal (positive = reinforce), the sign must be flipped.
                            # prm_construction already read above (for nosol forward condition)
                            if prm_construction == "self_reward":
                                # Pure student confidence as reward (no teacher).
                                # Tests the self-reward hypothesis: is s-t ≈ s after normalization?
                                base_kl = log_prob.detach()
                            elif prm_construction == "reverse":
                                # s - gamma * t(sol): generalized baseline subtraction.
                                # gamma=1.0: full s-t (v3-v10 sign). gamma=0.0: pure s (self-reward).
                                # Intermediate gamma balances task structure vs template bias.
                                gamma = ccir_cfg.get("prm_gamma", 1.0)
                                base_kl = (log_prob - gamma * teacher_log_prob).detach()
                            elif prm_construction == "teacher_contrastive":
                                # t(sol) - t(nosol): pure teacher-internal contrastive, no student
                                # Template cancels if both use same format. Only solution content differs.
                                assert teacher_nosol_log_prob is not None, (
                                    "prm_construction='teacher_contrastive' requires nosol forward "
                                    "(set si_reference to 'solution_contrastive' or 'wrong_sibling')"
                                )
                                base_kl = (teacher_log_prob - teacher_nosol_log_prob).detach()
                            elif prm_construction == "teacher_contrastive_reversed":
                                # t(nosol) - t(sol): reversed contrastive. Flips the length bias.
                                # If teacher_contrastive causes length collapse, this should cause length growth.
                                assert teacher_nosol_log_prob is not None, (
                                    "prm_construction='teacher_contrastive_reversed' requires nosol forward"
                                )
                                base_kl = (teacher_nosol_log_prob - teacher_log_prob).detach()
                            elif prm_construction == "s_minus_t_wrong":
                                # s - t(nosol): student minus teacher-with-wrong-solution
                                # Positive on tokens where wrong solution confuses the model.
                                assert teacher_nosol_log_prob is not None, (
                                    "prm_construction='s_minus_t_wrong' requires nosol forward "
                                    "(set si_reference to 'solution_contrastive')"
                                )
                                base_kl = (log_prob - teacher_nosol_log_prob).detach()
                            elif prm_construction == "t_wrong_minus_s":
                                # t(nosol) - s: teacher-with-wrong-solution minus student
                                assert teacher_nosol_log_prob is not None, (
                                    "prm_construction='t_wrong_minus_s' requires nosol forward "
                                    "(set si_reference to 'solution_contrastive')"
                                )
                                base_kl = (teacher_nosol_log_prob - log_prob).detach()
                            elif prm_construction == "s_minus_s_shuffled":
                                # Shuffle s across positions within each sequence as perturbation.
                                # Tests H1: is any decorrelated perturbation sufficient?
                                s_det = log_prob.detach()
                                # Shuffle within each sequence independently
                                shuffled = s_det.clone()
                                for b in range(s_det.shape[0]):
                                    seq_len = int(response_mask[b].sum().item())
                                    if seq_len > 1:
                                        perm = torch.randperm(seq_len, device=s_det.device)
                                        shuffled[b, :seq_len] = s_det[b, perm]
                                base_kl = (s_det - shuffled) * response_mask
                            elif prm_construction == "s_minus_s_other":
                                # Subtract s from a different response (rolled by 1 in batch dim).
                                # Provides a structured but content-independent perturbation.
                                s_det = log_prob.detach()
                                s_other = torch.roll(s_det, shifts=1, dims=0)
                                base_kl = (s_det - s_other) * response_mask
                            elif prm_construction == "random_credit":
                                # Pure Gaussian noise as token credit. Extreme control.
                                # Tests: does ANY non-uniform signal accelerate learning?
                                base_kl = torch.randn_like(log_prob) * response_mask
                            elif prm_construction == "reverse_combined":
                                # (s - t) + contrastive_lambda * (t_correct - t_wrong)
                                # Combines on-policy decorrelation (s-t) with solution credit (contrastive).
                                # contrastive_lambda controls amplification of the solution signal.
                                assert teacher_nosol_log_prob is not None, (
                                    "prm_construction='reverse_combined' requires nosol forward "
                                    "(set si_reference to 'solution_contrastive')"
                                )
                                contrastive_lambda = ccir_cfg.get("contrastive_lambda", 5.0)
                                s_minus_t = (log_prob - teacher_log_prob).detach()
                                t_contrastive = (teacher_log_prob - teacher_nosol_log_prob).detach()
                                base_kl = s_minus_t + contrastive_lambda * t_contrastive
                            elif prm_construction == "indep_norm":
                                # Independent standardization: normalize t and s per-sequence before subtraction.
                                # Removes the log-prob ceiling effect: Δ = t - s is bounded by -s (saturation),
                                # creating Cov(Δ, s) < 0. By standardizing independently, both become
                                # "relative token rankings" and the ceiling-induced correlation vanishes.
                                t_det = teacher_log_prob.detach()
                                s_det = log_prob.detach()
                                resp_lens_pc = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)

                                t_mean_pc = (t_det * response_mask).sum(dim=-1, keepdim=True) / resp_lens_pc
                                t_c = (t_det - t_mean_pc) * response_mask
                                t_std_pc = (t_c ** 2 * response_mask).sum(dim=-1, keepdim=True).div(resp_lens_pc).sqrt().clamp(min=1e-6)
                                t_normed = (t_c / t_std_pc) * response_mask

                                s_mean_pc = (s_det * response_mask).sum(dim=-1, keepdim=True) / resp_lens_pc
                                s_c = (s_det - s_mean_pc) * response_mask
                                s_std_pc = (s_c ** 2 * response_mask).sum(dim=-1, keepdim=True).div(resp_lens_pc).sqrt().clamp(min=1e-6)
                                s_normed = (s_c / s_std_pc) * response_mask

                                base_kl = (t_normed - s_normed)  # already detached via t_det, s_det
                            elif prm_construction == "ccir_cross_problem":
                                # CCIR cross-problem: use x' context to separate x-specific from x-generic signal.
                                assert cross_student_log_prob is not None, (
                                    "prm_construction='ccir_cross_problem' requires ccir_cross_problem=True"
                                )
                                ccir_cp_mode = ccir_cfg.get("ccir_cross_problem_mode", "full")
                                if ccir_cp_mode == "full":
                                    # [s(x) - s(x')] - beta * [t(x,y') - t(x',y')]
                                    ccir_cp_beta = ccir_cfg.get("ccir_cross_problem_beta", 1.0)
                                    s_specific = (log_prob - cross_student_log_prob).detach()
                                    _ccir_s_specific = s_specific  # store for gate & diagnostics
                                    if cross_teacher_log_prob is not None and ccir_cp_beta != 0:
                                        t_specific = (teacher_log_prob - cross_teacher_log_prob).detach()
                                        base_kl = s_specific - ccir_cp_beta * t_specific
                                    else:
                                        base_kl = s_specific
                                elif ccir_cp_mode == "blend":
                                    # s(x) - 0.5 * [s(x') + t(x', y')]
                                    # Decomposition: base_kl = s_specific + 0.5 * generic_gap
                                    #   s_specific = s(x) - s(x')  [problem-specific student signal]
                                    #   generic_gap = s(x') - t(x',y')  [generic student-teacher gap]
                                    if cross_teacher_log_prob is not None:
                                        cross_blend = 0.5 * (cross_student_log_prob + cross_teacher_log_prob)
                                    else:
                                        cross_blend = cross_student_log_prob
                                    base_kl = (log_prob - cross_blend).detach()
                                    # Store decomposition for diagnostics
                                    _ccir_s_specific = (log_prob - cross_student_log_prob).detach()
                                    _ccir_generic_gap_weight = 0.5
                                    _ccir_generic_gap = (cross_student_log_prob - cross_teacher_log_prob).detach() if cross_teacher_log_prob is not None else None
                                elif ccir_cp_mode == "blend_current_teacher":
                                    # v27 family: s(x) - [(1-alpha) * s(x') + alpha * t(x)]
                                    # Decomposition: base_kl = s_specific + alpha * generic_gap
                                    #   s_specific = s(x) - s(x')        [problem-specific student signal]
                                    #   generic_gap = s(x') - t(x)       [generic student-teacher gap]
                                    # This keeps the v25 blend skeleton, but replaces cross-problem t'(x', y')
                                    # with the current-problem teacher t(x).
                                    ccir_cp_alpha = float(ccir_cfg.get("ccir_cross_problem_alpha", 0.5))
                                    ccir_cp_alpha = max(0.0, min(1.0, ccir_cp_alpha))
                                    if ccir_cp_alpha > 0.0 and teacher_log_prob is None:
                                        raise ValueError(
                                            "ccir_cross_problem_mode='blend_current_teacher' requires teacher_log_prob when alpha > 0"
                                        )
                                    teacher_term = teacher_log_prob if teacher_log_prob is not None else torch.zeros_like(cross_student_log_prob)
                                    cross_blend = (1.0 - ccir_cp_alpha) * cross_student_log_prob + ccir_cp_alpha * teacher_term
                                    base_kl = (log_prob - cross_blend).detach()
                                    _ccir_s_specific = (log_prob - cross_student_log_prob).detach()
                                    _ccir_generic_gap_weight = ccir_cp_alpha
                                    _ccir_generic_gap = (cross_student_log_prob - teacher_term).detach()
                                else:
                                    raise ValueError(f"Unknown ccir_cross_problem_mode: {ccir_cp_mode}")
                            else:
                                base_kl = (teacher_log_prob - log_prob).detach()  # t - s

                            # Step 1.5: Adaptive u clip (warmup-anchored).
                            # Collect σ_ref during warmup, then clamp base_kl to ±k·σ_ref.
                            # Works on any prm_construction (uses the computed base_kl directly).
                            u_clip_mode = ccir_cfg.get("prm_u_clip_mode", "none")
                            if u_clip_mode == "adaptive":
                                u_clip_k = float(ccir_cfg.get("prm_u_clip_k_sigma", 2.0))
                                u_clip_warmup = int(ccir_cfg.get("ca_lambda_warmup_steps", 10))
                                u_clip_sigma_fixed = float(ccir_cfg.get("prm_u_clip_sigma_ref_fixed", 0.0))
                                current_step = self._global_training_steps
                                valid_u = response_mask.bool()
                                # If fixed σ_ref provided, skip warmup and use it directly
                                if u_clip_sigma_fixed > 0 and not hasattr(self, '_u_clip_sigma_ref'):
                                    self._u_clip_sigma_ref = u_clip_sigma_fixed
                                if valid_u.any():
                                    base_kl_valid = base_kl[valid_u]
                                    batch_std = base_kl_valid.std().item() if base_kl_valid.numel() > 1 else 0.0
                                    if u_clip_sigma_fixed <= 0 and current_step < u_clip_warmup:
                                        # Warmup: collect σ batch-stds, do NOT clip
                                        if not hasattr(self, '_u_clip_warmup_stds'):
                                            self._u_clip_warmup_stds = []
                                        self._u_clip_warmup_stds.append(batch_std)
                                    else:
                                        # Post-warmup: calibrate σ_ref once (all-reduced), then clamp
                                        if not hasattr(self, '_u_clip_sigma_ref'):
                                            if hasattr(self, '_u_clip_warmup_stds') and self._u_clip_warmup_stds:
                                                sigma_ref_local = sum(self._u_clip_warmup_stds) / len(self._u_clip_warmup_stds)
                                            else:
                                                sigma_ref_local = batch_std if batch_std > 0 else 1.0
                                            sigma_tensor = torch.tensor([sigma_ref_local], device=base_kl.device)
                                            if torch.distributed.is_initialized():
                                                torch.distributed.all_reduce(sigma_tensor, op=torch.distributed.ReduceOp.AVG)
                                            self._u_clip_sigma_ref = max(sigma_tensor.item(), 1e-6)
                                        _u_clip_sigma_ref = self._u_clip_sigma_ref
                                        _u_clip_range = u_clip_k * _u_clip_sigma_ref
                                        _u_clip_active = True
                                        # Zero-out outliers (hard gate): tokens with |base_kl| > c_range
                                        # contribute 0 PRM (not saturated at ±c_range).
                                        # Rationale: extreme u is likely spurious outlier signal;
                                        # better to ignore than to push in any direction.
                                        safe_mask = (base_kl.abs() <= _u_clip_range)
                                        base_kl_clipped = torch.where(safe_mask, base_kl, torch.zeros_like(base_kl))
                                        # Metrics: fraction zeroed (was outside safe range)
                                        zeroed_mask = ~safe_mask[valid_u]
                                        _u_clip_frac = zeroed_mask.float().mean().item()
                                        _u_std_ratio = batch_std / _u_clip_sigma_ref
                                        base_kl = base_kl_clipped

                            # Step 2: CCIR S_t (x-specificity)
                            prm_w = ccir_cfg.prm_weight
                            if ccir_contrastive_teacher_log_probs is not None and prm_w > 0:
                                ccir_S_t = (teacher_log_prob - ccir_contrastive_teacher_log_probs).detach()  # t - c
                            else:
                                ccir_S_t = torch.zeros_like(base_kl)

                            # Step 3: Full PRM signal
                            PRM_t = base_kl - prm_w * ccir_S_t  # (t - s) - ccir_weight * (t - c)

                            # Step 3.1: Solution Influence (SI) integration
                            # SI_t = t_sol - t_bare: how much knowing the solution helps predict each token.
                            # Teacher-internal (no s_t) → immune to student confidence hack.
                            SI_t = None
                            SI_centered = None
                            if si_mode != "none" and teacher_nosol_log_prob is not None:
                                SI_t = (si_sol_log_prob - teacher_nosol_log_prob).detach()

                                if si_mode == "replace":
                                    # Replace PRM entirely with centered SI (zero-sum per sequence)
                                    resp_lens_si = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                                    SI_mean = (SI_t * response_mask).sum(dim=-1, keepdim=True) / resp_lens_si
                                    SI_centered = (SI_t - SI_mean) * response_mask
                                    PRM_t = SI_centered

                                elif si_mode == "centered":
                                    # Add centered SI to existing PRM
                                    resp_lens_si = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                                    SI_mean = (SI_t * response_mask).sum(dim=-1, keepdim=True) / resp_lens_si
                                    SI_centered = (SI_t - SI_mean) * response_mask
                                    si_lam = ccir_cfg.get("si_lambda", 1.0)
                                    PRM_t = PRM_t + si_lam * SI_centered

                                elif si_mode == "raw":
                                    # Add raw SI to existing PRM
                                    si_lam = ccir_cfg.get("si_lambda", 1.0)
                                    PRM_t = PRM_t + si_lam * SI_t

                                elif si_mode == "replace_indep_norm":
                                    # Replace PRM with independently normalized SI.
                                    # PRM = norm(t_sol) - norm(t_bare) per sequence.
                                    # Removes ceiling effect: raw SI = t_sol - t_bare is bounded by -t_bare,
                                    # creating Cov(SI, t_bare) < 0. Independent standardization converts
                                    # both to relative rankings, eliminating the scale coupling.
                                    # Purely teacher-internal: no s_t involved.
                                    t_sol_d = si_sol_log_prob.detach()
                                    t_bare_d = teacher_nosol_log_prob.detach()
                                    resp_lens_si = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)

                                    # Standardize t_sol per sequence
                                    ts_mean = (t_sol_d * response_mask).sum(dim=-1, keepdim=True) / resp_lens_si
                                    ts_c = (t_sol_d - ts_mean) * response_mask
                                    ts_std = (ts_c ** 2 * response_mask).sum(dim=-1, keepdim=True).div(resp_lens_si).sqrt().clamp(min=1e-6)
                                    ts_norm = (ts_c / ts_std) * response_mask

                                    # Standardize t_bare per sequence
                                    tb_mean = (t_bare_d * response_mask).sum(dim=-1, keepdim=True) / resp_lens_si
                                    tb_c = (t_bare_d - tb_mean) * response_mask
                                    tb_std = (tb_c ** 2 * response_mask).sum(dim=-1, keepdim=True).div(resp_lens_si).sqrt().clamp(min=1e-6)
                                    tb_norm = (tb_c / tb_std) * response_mask

                                    PRM_t = (ts_norm - tb_norm) * response_mask

                            elif si_mode == "teacher_only":
                                # Teacher-only PRM: PRM = centered(t_t), zero s_t by construction.
                                # t_t = teacher_log_prob (log π_ema(y|x,sol,y<t))
                                t_det = teacher_log_prob.detach()
                                resp_lens_to = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                                t_mean = (t_det * response_mask).sum(dim=-1, keepdim=True) / resp_lens_to
                                PRM_t = (t_det - t_mean) * response_mask

                            # Step 3.15: token-level forward-style reweight on raw PRM.
                            # Let Δ_t = s_t - t_t using current student/teacher token log-probs.
                            # w_t = exp(clamp(-beta * Δ_t, -clip, clip)) = exp(clamp(beta * (t_t - s_t), ...)).
                            # When PRM_t = Δ_t (pure raw s-t), PRM_t <- w_t * PRM_t is the local
                            # negative forward-KL density under p-sampled rollouts.
                            prm_forward_mode = ccir_cfg.get("prm_forward_mode", "none")
                            if prm_forward_mode == "token_is":
                                if teacher_log_prob is None:
                                    raise ValueError("ccir.prm_forward_mode='token_is' requires teacher_log_prob")
                                valid_forward = response_mask.bool()
                                prm_forward_beta = float(ccir_cfg.get("prm_forward_beta", 1.0))
                                prm_forward_log_clip = float(ccir_cfg.get("prm_forward_log_clip", 5.0))
                                if valid_forward.any():
                                    _forward_token_input_mean = PRM_t[valid_forward].mean().item()
                                    _forward_token_input_abs_mean = PRM_t[valid_forward].abs().mean().item()
                                forward_gap = (teacher_log_prob - log_prob).detach() * response_mask  # t - s
                                _forward_token_gap = forward_gap
                                forward_log_weights_unclipped = prm_forward_beta * forward_gap
                                forward_log_weights = forward_log_weights_unclipped.clamp(
                                    min=-prm_forward_log_clip,
                                    max=prm_forward_log_clip,
                                ) * response_mask
                                _forward_token_weights = torch.exp(forward_log_weights) * response_mask
                                _forward_token_log_weights = forward_log_weights
                                if valid_forward.any():
                                    clipped = (forward_log_weights_unclipped[valid_forward] != forward_log_weights[valid_forward])
                                    _forward_token_clip_frac = clipped.float().mean().item()
                                PRM_t = PRM_t * _forward_token_weights
                                if valid_forward.any():
                                    _forward_token_output_mean = PRM_t[valid_forward].mean().item()
                                    _forward_token_output_abs_mean = PRM_t[valid_forward].abs().mean().item()

                            elif prm_forward_mode == "renyi_unbiased":
                                # Unbiased-advantage PG for Rényi-α f-divergence.
                                # u = log π_v(y|context) - log π_s(y|x),  where:
                                #   log π_v = virtual_alpha * t(x,z) + (1 - virtual_alpha) * s(x')
                                # virtual_alpha = 1.0: pure teacher-with-solution (no x' anchor)
                                # virtual_alpha < 1.0: blend in x' generic-student as anchor to prevent
                                #   max-K3 divergence into non-problem-specific modes.
                                # PRM_t = sign * (exp(clamp(alpha_r * u, -log_clip, log_clip)) - 1)
                                # sign = +1: minimize K3 / reverse KL, push student TOWARD virtual teacher
                                # sign = -1: maximize K3, push student AWAY from virtual teacher (K3-REINFORCE)
                                if teacher_log_prob is None:
                                    raise ValueError("ccir.prm_forward_mode='renyi_unbiased' requires teacher_log_prob")
                                _renyi_active = True
                                valid_forward = response_mask.bool()
                                prm_renyi_alpha = float(ccir_cfg.get("prm_renyi_alpha", 1.0))
                                prm_renyi_sign = float(ccir_cfg.get("prm_renyi_sign", 1.0))
                                prm_renyi_virtual_alpha = float(ccir_cfg.get("prm_renyi_virtual_alpha", 1.0))
                                prm_forward_log_clip = float(ccir_cfg.get("prm_forward_log_clip", 3.0))
                                # Construct u = log π_v - log π_s
                                u_t = (teacher_log_prob - log_prob).detach() * response_mask  # t - s
                                if prm_renyi_virtual_alpha < 1.0:
                                    if cross_student_log_prob is None:
                                        raise ValueError(
                                            "renyi_unbiased with prm_renyi_virtual_alpha < 1.0 requires "
                                            "cross_student_log_prob (set ccir_cross_problem=True)"
                                        )
                                    u_xp = (cross_student_log_prob - log_prob).detach() * response_mask  # s' - s
                                    forward_gap = (
                                        prm_renyi_virtual_alpha * u_t
                                        + (1.0 - prm_renyi_virtual_alpha) * u_xp
                                    ) * response_mask
                                else:
                                    forward_gap = u_t
                                # Optional: adaptive u clip (zero-out outside k·σ_ref).
                                # Reuses σ_ref from Step 1.5 warmup. Clipped tokens contribute
                                # 0 PRM (exp(0)-1 = 0) regardless of sign/alpha.
                                _renyi_u_clipped_frac = 0.0
                                if (ccir_cfg.get("prm_u_clip_mode", "none") == "adaptive"
                                        and hasattr(self, '_u_clip_sigma_ref')):
                                    _u_clip_k_r = float(ccir_cfg.get("prm_u_clip_k_sigma", 2.0))
                                    _u_clip_range_r = _u_clip_k_r * self._u_clip_sigma_ref
                                    safe_mask_r = forward_gap.abs() <= _u_clip_range_r
                                    if valid_forward.any():
                                        _renyi_u_clipped_frac = (~safe_mask_r[valid_forward]).float().mean().item()
                                    forward_gap = torch.where(safe_mask_r, forward_gap, torch.zeros_like(forward_gap))
                                _forward_token_gap = forward_gap
                                renyi_log_unclipped = prm_renyi_alpha * forward_gap
                                renyi_log_clipped = renyi_log_unclipped.clamp(
                                    min=-prm_forward_log_clip,
                                    max=prm_forward_log_clip,
                                ) * response_mask
                                PRM_t = prm_renyi_sign * (torch.exp(renyi_log_clipped) - 1.0) * response_mask
                                if valid_forward.any():
                                    clipped = (renyi_log_unclipped[valid_forward] != renyi_log_clipped[valid_forward])
                                    _renyi_clip_frac = clipped.float().mean().item()
                                    _renyi_prm_mean = PRM_t[valid_forward].mean().item()
                                    _renyi_prm_abs_mean = PRM_t[valid_forward].abs().mean().item()

                            elif prm_forward_mode == "jsd_unbiased":
                                # Unbiased PG advantage for max Jensen-Shannon divergence.
                                # Derivation: JSD(s||t) = D_f(s||t) with f(x)=0.5[x·log(2x/(x+1))+log(2/(x+1))].
                                # f'(x) = 0.5·log(2x/(x+1)); with x = exp(-u):
                                #   coef = 0.5·[log 2 - softplus(u_v)]
                                # Properties (u_v = virtual_alpha·(t-s) + (1-virtual_alpha)·(s'-s)):
                                #   u_v = 0: coef = 0 (per-token stopping when s meets virtual teacher)
                                #   u_v << 0 (s >> t, q1 region): coef → log 2 / 2 ≈ 0.347 (BOUNDED, no q1 blow-up)
                                #   u_v >> 0 (s << t, q4/EOS): coef ≈ -0.5·u_v (LINEAR, no exp blow-up)
                                # Sign convention matches renyi_unbiased: sign=-1 for anti-distill (default).
                                # We write PRM_t = 0.5·sign·(softplus(u)-log 2) so sign=-1 gives max direction.
                                if teacher_log_prob is None:
                                    raise ValueError("ccir.prm_forward_mode='jsd_unbiased' requires teacher_log_prob")
                                _jsd_active = True
                                valid_forward = response_mask.bool()
                                prm_jsd_sign = float(ccir_cfg.get("prm_renyi_sign", 1.0))
                                prm_jsd_virtual_alpha = float(ccir_cfg.get("prm_renyi_virtual_alpha", 1.0))
                                prm_forward_log_clip = float(ccir_cfg.get("prm_forward_log_clip", 5.0))
                                u_t = (teacher_log_prob - log_prob).detach() * response_mask  # t - s
                                if prm_jsd_virtual_alpha < 1.0:
                                    if cross_student_log_prob is None:
                                        raise ValueError(
                                            "jsd_unbiased with prm_renyi_virtual_alpha < 1.0 requires "
                                            "cross_student_log_prob (set ccir_cross_problem=True)"
                                        )
                                    u_xp = (cross_student_log_prob - log_prob).detach() * response_mask  # s' - s
                                    forward_gap = (
                                        prm_jsd_virtual_alpha * u_t
                                        + (1.0 - prm_jsd_virtual_alpha) * u_xp
                                    ) * response_mask
                                else:
                                    forward_gap = u_t
                                # Adaptive u clip (reuse renyi path): zero-out outliers.
                                _jsd_u_clipped_frac = 0.0
                                if (ccir_cfg.get("prm_u_clip_mode", "none") == "adaptive"
                                        and hasattr(self, '_u_clip_sigma_ref')):
                                    _u_clip_k_j = float(ccir_cfg.get("prm_u_clip_k_sigma", 2.0))
                                    _u_clip_range_j = _u_clip_k_j * self._u_clip_sigma_ref
                                    safe_mask_j = forward_gap.abs() <= _u_clip_range_j
                                    if valid_forward.any():
                                        _jsd_u_clipped_frac = (~safe_mask_j[valid_forward]).float().mean().item()
                                    forward_gap = torch.where(safe_mask_j, forward_gap, torch.zeros_like(forward_gap))
                                _forward_token_gap = forward_gap
                                # Clamp for numerical stability in softplus (matches renyi clip semantics).
                                gap_clipped = forward_gap.clamp(
                                    min=-prm_forward_log_clip,
                                    max=prm_forward_log_clip,
                                )
                                # coef (max-JSD direction) = 0.5·(log 2 - softplus(u_v))
                                # With prm_jsd_sign: PRM_t = 0.5·sign·(softplus(u_v) - log 2)
                                log2 = math.log(2.0)
                                jsd_raw = torch.nn.functional.softplus(gap_clipped) - log2  # >0 for u>0 (distill)
                                PRM_t = 0.5 * prm_jsd_sign * jsd_raw * response_mask
                                if valid_forward.any():
                                    clipped_j = (gap_clipped[valid_forward] != forward_gap[valid_forward])
                                    _jsd_clip_frac = clipped_j.float().mean().item()
                                    _jsd_prm_mean = PRM_t[valid_forward].mean().item()
                                    _jsd_prm_abs_mean = PRM_t[valid_forward].abs().mean().item()
                                    _jsd_gap_mean = forward_gap[valid_forward].mean().item()
                                    _jsd_gap_abs_mean = forward_gap[valid_forward].abs().mean().item()

                            # Step 3.2: tanh saturation — compress extreme PRM values
                            # Applied BEFORE normalization to break "rich get richer" feedback loop.
                            # PRM_t = tanh(PRM_t / tau) * tau, bounding to [-tau, tau].
                            prm_tanh_tau = ccir_cfg.get("prm_tanh_tau", None)
                            prm_pre_tanh_abs = None
                            if prm_tanh_tau is not None and prm_tanh_tau > 0:
                                prm_pre_tanh_abs = PRM_t[response_mask.bool()].abs().mean().item()
                                PRM_t = torch.tanh(PRM_t / prm_tanh_tau) * prm_tanh_tau

                            # Step 3.3: MaxEnt at Position A (before normalization)
                            # Subtract α·(s_t - s̄) from PRM_t to counteract the positive s_t feedback.
                            # Empirically Cov(PRM, s) > 0 regardless of PRM sign (t-s or s-t),
                            # so subtracting α·s removes the confidence-correlated component.
                            maxent_mode = ccir_cfg.get("maxent_coeff", "none")
                            maxent_position = ccir_cfg.get("maxent_position", "advantage")
                            maxent_correction_abs_mean = None
                            if maxent_mode != "none" and maxent_position == "prm_raw":
                                if self._maxent_alpha is None:
                                    self._maxent_alpha = ccir_cfg.get("maxent_alpha", 0.1)
                                s_t_me = old_log_prob.detach()
                                resp_lens_me = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                                s_mean_me = (s_t_me * response_mask).sum(dim=-1, keepdim=True) / resp_lens_me
                                s_centered_me = (s_t_me - s_mean_me) * response_mask
                                correction = self._maxent_alpha * s_centered_me
                                PRM_t = PRM_t - correction
                                maxent_correction_abs_mean = correction[response_mask.bool()].abs().mean().item()

                            # Step 3.5: Entropy-neutral decorrelation
                            # PRM_t is correlated with s_t = log π(y_t) (student confidence).
                            # This correlation causes "policy momentum" → entropy collapse.
                            # Decorrelation: PRM_EN = PRM - β·(s - s̄), β = Cov(PRM, s)/Var(s)
                            # ensures Cov(PRM_EN, s) = 0 → first-order entropy-neutral.
                            en_mode = ccir_cfg.get("prm_entropy_neutral", "none")
                            en_beta = None  # will be set if decorrelation is applied
                            en_cov_before = None

                            if en_mode == "decorrelate":
                                s_t = old_log_prob.detach()  # log π_old(y_t), the student confidence
                                resp_lens = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, 1]

                                # Per-sequence centered versions
                                s_mean = (s_t * response_mask).sum(dim=-1, keepdim=True) / resp_lens
                                s_centered = (s_t - s_mean) * response_mask  # [B, T]

                                prm_mean = (PRM_t * response_mask).sum(dim=-1, keepdim=True) / resp_lens
                                prm_centered = (PRM_t - prm_mean) * response_mask  # [B, T]

                                # Per-sequence regression coefficient: β = Cov(PRM, s) / Var(s)
                                cov_ps = (prm_centered * s_centered * response_mask).sum(dim=-1, keepdim=True) / resp_lens  # [B, 1]
                                var_s = (s_centered ** 2 * response_mask).sum(dim=-1, keepdim=True) / resp_lens  # [B, 1]
                                en_beta = cov_ps / var_s.clamp(min=1e-8)  # [B, 1]

                                # Store pre-decorrelation Cov for metrics
                                en_cov_before = cov_ps.mean().item()

                                # Decorrelate: remove the s-correlated component
                                PRM_t = (PRM_t - en_beta * s_centered) * response_mask

                            # Step 4: PRM normalization
                            # prm_normalize_mode supersedes legacy prm_normalize / prm_seq_demean
                            normalize_mode = ccir_cfg.get("prm_normalize_mode", "batch")

                            if normalize_mode == "sequence":
                                # Per-sequence standardization: mean=0, var=1 per sequence.
                                # Removes both first-moment (mean) and second-moment (variance)
                                # length dependence. PRM becomes a pure credit assignment weight.
                                resp_lengths = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, 1]
                                prm_seq_mean = (PRM_t * response_mask).sum(dim=-1, keepdim=True) / resp_lengths  # [B, 1]
                                prm_centered = (PRM_t - prm_seq_mean) * response_mask
                                prm_seq_var = (prm_centered ** 2 * response_mask).sum(dim=-1, keepdim=True) / resp_lengths  # [B, 1]
                                prm_seq_std = prm_seq_var.sqrt().clamp(min=1e-6)  # [B, 1]
                                PRM_processed = (prm_centered / prm_seq_std) * response_mask

                            elif normalize_mode == "batch":
                                # Legacy per-batch normalization: mean=0, var=1 across all valid tokens
                                valid_mask = response_mask.bool()
                                if self_distillation_mask is not None:
                                    valid_mask = valid_mask & self_distillation_mask.unsqueeze(-1).bool()
                                valid_prm = PRM_t[valid_mask]
                                if valid_prm.numel() > 0:
                                    prm_mean = valid_prm.mean()
                                    prm_std = valid_prm.std().clamp(min=1e-6)
                                    PRM_processed = (PRM_t - prm_mean) / prm_std
                                else:
                                    PRM_processed = torch.zeros_like(PRM_t)

                                # Legacy per-sequence de-mean (on top of batch normalization)
                                if ccir_cfg.get("prm_seq_demean", False):
                                    resp_lengths = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                                    prm_seq_mean = (PRM_processed * response_mask).sum(dim=-1, keepdim=True) / resp_lengths
                                    PRM_processed = (PRM_processed - prm_seq_mean) * response_mask

                            elif normalize_mode == "sequence_demean":
                                # Per-sequence mean removal only (no std division).
                                # Enables exact α=λ cancellation: after demean, PRM scale matches s_t scale.
                                resp_lengths = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                                prm_seq_mean = (PRM_t * response_mask).sum(dim=-1, keepdim=True) / resp_lengths
                                PRM_processed = (PRM_t - prm_seq_mean) * response_mask

                            else:
                                # No normalization: raw PRM signal
                                PRM_processed = PRM_t

                            if ccir_cfg.prm_clip is not None:
                                PRM_processed = PRM_processed.clamp(-ccir_cfg.prm_clip, ccir_cfg.prm_clip)

                            # Step 4.5: PRM-ORM correlation + alignment gate
                            # Always compute correlation when orm_weight > 0 (diagnostic metric).
                            # Gate (optional): when PRM-ORM correlation <= 0, PRM is noise → gate to 0.
                            ca_lambda_mode = ccir_cfg.get("ca_lambda_mode", "fixed")
                            if ca_lambda_mode == "adaptive":
                                bkl_abs_mean = base_kl[response_mask.bool()].abs().mean().clamp(min=1e-8)
                                ca_lambda_target = ccir_cfg.get("ca_lambda_target", 0.001)
                                ca_lambda = (ca_lambda_target / bkl_abs_mean).clamp(
                                    min=ccir_cfg.get("ca_lambda_min", 1e-4),
                                    max=ccir_cfg.get("ca_lambda_max", 1.0),
                                ).item()
                            elif ca_lambda_mode == "length_aware":
                                mean_resp_len = response_mask.sum(dim=-1).float().mean().item()
                                len_target = ccir_cfg.get("ca_lambda_length_target", 10000.0)
                                len_alpha = ccir_cfg.get("ca_lambda_length_alpha", 2.0)
                                length_ratio = mean_resp_len / max(len_target, 1.0)
                                ca_lambda = ccir_cfg.ca_lambda * (1.0 - len_alpha * max(0.0, length_ratio - 1.0))
                                ca_lambda = max(min(ca_lambda, ccir_cfg.get("ca_lambda_max", 1.0)),
                                                ccir_cfg.get("ca_lambda_min", -0.1))
                            elif ca_lambda_mode == "teacher_perp":
                                # Teacher perplexity as entropy-based negative feedback (per-sequence).
                                #   teacher_perp > target → λ>0 (s-t) → entropy↓ → perp↓ → stabilize
                                #   teacher_perp < target → λ<0 (t-s) → entropy↑ → perp↑ → stabilize
                                # Warmup: first N steps collect t_ppl baseline (λ=0, ORM only).
                                # After warmup: target = median(collected) - delta.
                                seq_t_logp = (teacher_log_prob.detach() * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)  # [B]
                                seq_teacher_perp = torch.exp(-seq_t_logp)  # [B]
                                teacher_perp = seq_teacher_perp.mean().item()
                                # Per-seq student perplexity and t/s ratio for diagnostics
                                seq_s_logp = (old_log_prob.detach() * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)  # [B]
                                seq_student_perp = torch.exp(-seq_s_logp)  # [B]
                                seq_ts_ratio = seq_teacher_perp / seq_student_perp.clamp(min=1.001)  # [B]
                                perp_target = ccir_cfg.get("ca_lambda_perp_target", 0.0)
                                perp_delta = ccir_cfg.get("ca_lambda_perp_delta", 0.10)
                                warmup_steps = ccir_cfg.get("ca_lambda_warmup_steps", 5)
                                current_step = self._global_training_steps
                                # Warmup phase: collect t_ppl, λ=0 (pure ORM)
                                if perp_target <= 0 and current_step < warmup_steps:
                                    if not hasattr(self, '_warmup_perp_values'):
                                        self._warmup_perp_values = []
                                    # Collect non-spike values
                                    batch_perp = teacher_perp
                                    if batch_perp < 3.0:
                                        self._warmup_perp_values.append(batch_perp)
                                    ca_lambda = 0.0
                                    # Reshape for broadcasting: scalar 0
                                    _ca_lambda_broadcast = None  # will use scalar ca_lambda
                                else:
                                    # Calibrate target from warmup data (once, all-reduced across ranks)
                                    if perp_target <= 0:
                                        if not hasattr(self, '_perp_target_calibrated'):
                                            if hasattr(self, '_warmup_perp_values') and self._warmup_perp_values:
                                                vals = sorted(self._warmup_perp_values)
                                                median_perp = vals[len(vals) // 2]
                                            else:
                                                median_perp = teacher_perp
                                            # All-reduce across DP ranks for consistent target
                                            median_tensor = torch.tensor([median_perp], device=old_log_prob.device)
                                            if torch.distributed.is_initialized():
                                                torch.distributed.all_reduce(median_tensor, op=torch.distributed.ReduceOp.AVG)
                                            median_perp = median_tensor.item()
                                            # Prefer ratio calibration (cross-model consistent), fallback to delta
                                            target_ratio = ccir_cfg.get("ca_lambda_perp_target_ratio", 0.0)
                                            if target_ratio > 0:
                                                self._perp_target_calibrated = max(median_perp * target_ratio, 1.01)
                                            else:
                                                self._perp_target_calibrated = max(median_perp - perp_delta, 1.01)
                                        perp_target = self._perp_target_calibrated
                                    perp_alpha = ccir_cfg.get("ca_lambda_perp_alpha", 2.0)
                                    # Per-sequence mask: spike → λ=0
                                    perp_mask_threshold = ccir_cfg.get("ca_lambda_perp_mask", 3.0)
                                    perp_masked = seq_teacher_perp > perp_mask_threshold  # [B]
                                    # Scope: per_seq (default) uses per-seq t_ppl; batch_mean uses batch-average
                                    tppl_scope = ccir_cfg.get("ca_lambda_tppl_scope", "per_seq")
                                    # Unified hysteresis + scope: always compute global batch_perp for hys gate,
                                    # then apply scope (batch_mean / per_seq) for λ magnitude.
                                    valid_seqs = ~perp_masked
                                    _device = seq_teacher_perp.device
                                    if valid_seqs.any():
                                        _local_sum = seq_teacher_perp[valid_seqs].sum().detach().clone()
                                        _local_cnt = torch.tensor(float(valid_seqs.sum().item()), device=_device)
                                    else:
                                        _local_sum = torch.tensor(0.0, device=_device)
                                        _local_cnt = torch.tensor(0.0, device=_device)
                                    _tp_sc = torch.stack([_local_sum, _local_cnt])
                                    if torch.distributed.is_initialized():
                                        torch.distributed.all_reduce(_tp_sc, op=torch.distributed.ReduceOp.SUM)
                                    if _tp_sc[1].item() > 0:
                                        batch_perp_scalar = _tp_sc[0] / _tp_sc[1]
                                    else:
                                        batch_perp_scalar = torch.tensor(perp_target, device=_device)
                                    # Hysteresis state — deterministic across ranks (same global batch_perp)
                                    _reactivate_abs = ccir_cfg.get("ca_lambda_perp_reactivate_target", 0.0)
                                    _reactivate_ratio = ccir_cfg.get("ca_lambda_perp_reactivate_ratio", 0.0)
                                    _target_ratio_cfg = ccir_cfg.get("ca_lambda_perp_target_ratio", 0.0)
                                    if _reactivate_abs > 0:
                                        _reactivate_th = float(_reactivate_abs)
                                    elif _reactivate_ratio > 0 and _target_ratio_cfg > 0:
                                        _warmup_median = perp_target / _target_ratio_cfg
                                        _reactivate_th = float(_warmup_median * _reactivate_ratio)
                                    else:
                                        _reactivate_th = 0.0  # hysteresis disabled
                                    _hys_enabled = _reactivate_th > perp_target
                                    if not hasattr(self, '_prm_active'):
                                        self._prm_active = True
                                    _bp_val = batch_perp_scalar.item()
                                    # Step-boundary hysteresis: accumulate (sum, count) across micro-batches
                                    # within a training step; decide state transition ONLY at step boundary
                                    # using the full-step average — avoids micro-batch subset noise
                                    # (e.g. 4-seq subsets crossing the threshold accidentally).
                                    if not hasattr(self, '_hys_step_sum'):
                                        self._hys_step_sum = 0.0
                                        self._hys_step_cnt = 0.0
                                        self._hys_last_step = -1
                                    _cur_step = self._global_training_steps
                                    if _hys_enabled and self._hys_last_step != _cur_step:
                                        # New step boundary: use previous step's full-batch avg to transition
                                        if self._hys_step_cnt > 0:
                                            _prev_step_avg = self._hys_step_sum / self._hys_step_cnt
                                            if self._prm_active and _prev_step_avg < perp_target:
                                                self._prm_active = False
                                            elif (not self._prm_active) and _prev_step_avg > _reactivate_th:
                                                self._prm_active = True
                                        # reset accumulators for the new step
                                        self._hys_step_sum = 0.0
                                        self._hys_step_cnt = 0.0
                                        self._hys_last_step = _cur_step
                                    # Accumulate current micro-batch's global (sum, count) into step totals
                                    self._hys_step_sum += float(_tp_sc[0].item())
                                    self._hys_step_cnt += float(_tp_sc[1].item())
                                    _lam_min = ccir_cfg.get("ca_lambda_min", -0.02)
                                    _lam_max = ccir_cfg.get("ca_lambda_max", 0.5)
                                    if _hys_enabled and not self._prm_active:
                                        # Gate closed: force all λ to λ_min regardless of scope
                                        ca_lambda_seq = torch.full_like(seq_teacher_perp, _lam_min)
                                    elif tppl_scope == "batch_mean":
                                        log_ratio_scalar = torch.log((batch_perp_scalar / perp_target).clamp(min=0.5, max=3.0))
                                        ca_lambda_scalar = (ccir_cfg.ca_lambda * perp_alpha * log_ratio_scalar).clamp(
                                            min=_lam_min, max=_lam_max)
                                        ca_lambda_seq = ca_lambda_scalar.expand_as(seq_teacher_perp).clone()
                                    else:  # per_seq: each seq computes its own λ
                                        log_ratio = torch.log((seq_teacher_perp / perp_target).clamp(min=0.5, max=3.0))
                                        ca_lambda_seq = (ccir_cfg.ca_lambda * perp_alpha * log_ratio).clamp(
                                            min=_lam_min, max=_lam_max)
                                    ca_lambda_seq[perp_masked] = 0.0
                                    # Batch-mean shift:
                                    #   mean_shift_always: force mean(λ) = 0 every step (difficulty-differential PRM)
                                    #   mean_shift (legacy): only shift when mean(λ) < 0
                                    # When meanshift is applied, cross-DP all-reduce ensures consistent batch-mean
                                    # across ranks (each rank sees only its local micro-batch).
                                    _mean_shift_amount = 0.0
                                    _mean_shift_always = ccir_cfg.get("ca_lambda_mean_shift_always", False)
                                    _mean_shift_neg = ccir_cfg.get("ca_lambda_mean_shift", False)
                                    if _mean_shift_always or _mean_shift_neg:
                                        # Global batch mean across DP ranks
                                        _local_mean = ca_lambda_seq.mean().detach().clone()
                                        _local_count = torch.tensor(
                                            float(ca_lambda_seq.numel()),
                                            device=ca_lambda_seq.device
                                        )
                                        _local_sum = _local_mean * _local_count
                                        _sum_count = torch.stack([_local_sum, _local_count])
                                        if torch.distributed.is_initialized():
                                            torch.distributed.all_reduce(_sum_count, op=torch.distributed.ReduceOp.SUM)
                                        _global_mean = (_sum_count[0] / _sum_count[1].clamp(min=1.0)).item()
                                        if _mean_shift_always or _global_mean < 0:
                                            ca_lambda_seq = ca_lambda_seq - _global_mean
                                            _mean_shift_amount = -_global_mean
                                    ca_lambda = ca_lambda_seq.mean().item()
                                    _ca_lambda_broadcast = ca_lambda_seq.unsqueeze(-1)
                            elif ca_lambda_mode == "student_perp":
                                # Student perplexity as control signal (s_ppl ≈ exp(entropy)).
                                # More robust than teacher_perp: no spike, not affected by ratio drift.
                                # s-t → entropy↓ → s_ppl↓;  t-s → entropy↑ → s_ppl↑.
                                # Symmetric negative feedback (per-sequence):
                                #   s_ppl > target → λ>0 (s-t) → entropy↓ → s_ppl↓ → stabilize
                                #   s_ppl < target → λ<0 (t-s) → entropy↑ → s_ppl↑ → stabilize
                                seq_s_logp = (old_log_prob.detach() * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)  # [B]
                                seq_student_perp = torch.exp(-seq_s_logp)  # [B]
                                perp_target = ccir_cfg.get("ca_lambda_perp_target", 0.0)
                                perp_delta = ccir_cfg.get("ca_lambda_perp_delta", 0.15)
                                # Auto-calibrate: use step@0 student_perp as baseline, offset by delta
                                if perp_target <= 0:
                                    if not hasattr(self, '_student_perp_init'):
                                        self._student_perp_init = seq_student_perp.mean().item()
                                    perp_target = max(self._student_perp_init - perp_delta, 1.01)
                                perp_alpha = ccir_cfg.get("ca_lambda_perp_alpha", 2.0)
                                log_ratio = torch.log((seq_student_perp / perp_target).clamp(min=0.5, max=3.0))
                                ca_lambda_seq = ccir_cfg.ca_lambda * perp_alpha * log_ratio  # [B]
                                ca_lambda_seq = ca_lambda_seq.clamp(
                                    min=ccir_cfg.get("ca_lambda_min", -0.01),
                                    max=ccir_cfg.get("ca_lambda_max", 0.5))
                                # For logging/conditionals: batch mean
                                ca_lambda = ca_lambda_seq.mean().item()
                                teacher_perp = seq_student_perp.mean().item()  # log as teacher_perp for compatibility
                                # Reshape for broadcasting: [B] → [B, 1]
                                _ca_lambda_broadcast = ca_lambda_seq.unsqueeze(-1)
                            elif ca_lambda_mode == "prm_strength":
                                # PRM-internal strength as control signal. Uses |base_kl| per-seq.
                                # Concept-consistent with any PRM construction: when PRM signal is strong,
                                # training has juice → λ>0; when weak → λ→0 via max(0) (natural exit).
                                # Does NOT depend on teacher_log_prob — works when teacher is absent.
                                seq_prm_strength = (base_kl.detach().abs() * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)  # [B]
                                perp_target = ccir_cfg.get("ca_lambda_perp_target", 0.0)
                                perp_delta = ccir_cfg.get("ca_lambda_perp_delta", 0.10)
                                warmup_steps = ccir_cfg.get("ca_lambda_warmup_steps", 5)
                                current_step = self._global_training_steps
                                # Warmup: collect strength baseline, λ=0
                                if perp_target <= 0 and current_step < warmup_steps:
                                    if not hasattr(self, '_prm_strength_baseline'):
                                        self._prm_strength_baseline = []
                                    batch_strength = seq_prm_strength.mean().item()
                                    if 1e-4 < batch_strength < 10.0:  # filter spikes/zeros
                                        self._prm_strength_baseline.append(batch_strength)
                                    ca_lambda = 0.0
                                    _ca_lambda_broadcast = None
                                else:
                                    # Calibrate target from warmup median (once, all-reduced across ranks)
                                    if perp_target <= 0:
                                        if not hasattr(self, '_prm_strength_target'):
                                            if hasattr(self, '_prm_strength_baseline') and self._prm_strength_baseline:
                                                vals = sorted(self._prm_strength_baseline)
                                                median_strength = vals[len(vals) // 2]
                                            else:
                                                median_strength = seq_prm_strength.mean().item()
                                            median_tensor = torch.tensor([median_strength], device=old_log_prob.device)
                                            if torch.distributed.is_initialized():
                                                torch.distributed.all_reduce(median_tensor, op=torch.distributed.ReduceOp.AVG)
                                            median_strength = median_tensor.item()
                                            # Target = baseline × (1 - delta), floor at 1e-4 for numerical safety
                                            self._prm_strength_target = max(median_strength * (1.0 - perp_delta), 1e-4)
                                        perp_target = self._prm_strength_target
                                    perp_alpha = ccir_cfg.get("ca_lambda_perp_alpha", 2.0)
                                    # Clamped log ratio prevents outlier amplification
                                    log_ratio = torch.log((seq_prm_strength / max(perp_target, 1e-6)).clamp(min=0.5, max=3.0))
                                    ca_lambda_seq = ccir_cfg.ca_lambda * perp_alpha * log_ratio  # [B]
                                    ca_lambda_seq = ca_lambda_seq.clamp(
                                        min=ccir_cfg.get("ca_lambda_min", 0.0),
                                        max=ccir_cfg.get("ca_lambda_max", 0.5))
                                    ca_lambda = ca_lambda_seq.mean().item()
                                    # Compat: log PRM strength as teacher_perp for metric-compat
                                    teacher_perp = seq_prm_strength.mean().item()
                                    _ca_lambda_broadcast = ca_lambda_seq.unsqueeze(-1)
                            elif ca_lambda_mode == "ratio_perp":
                                # Ratio (t_ppl / s_ppl) as control signal, computed in LOG SPACE.
                                # log(ratio) = log(t_ppl/s_ppl) = mean(s_logp - t_logp) per seq = base_kl per seq.
                                # Working in log space avoids exp/div numerical issues and outlier amplification.
                                # Ratio measures student-teacher distance, invariant to "model gets better".
                                seq_s_logp = (old_log_prob.detach() * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)
                                seq_t_logp = (teacher_log_prob.detach() * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)
                                # log(ratio) per sequence, clamped to avoid outlier spikes
                                seq_log_ratio = (seq_s_logp - seq_t_logp).clamp(min=-0.5, max=0.5)  # [B]
                                perp_delta = ccir_cfg.get("ca_lambda_perp_delta", 0.04)
                                # Auto-calibrate: log_ratio_target = mean(log_ratio at step 0) - delta
                                if not hasattr(self, '_log_ratio_target'):
                                    self._log_ratio_target = seq_log_ratio.mean().item() - perp_delta
                                log_ratio_target = self._log_ratio_target
                                perp_alpha = ccir_cfg.get("ca_lambda_perp_alpha", 5.0)
                                # λ = base * α * (log_ratio - log_ratio_target)
                                ca_lambda_seq = ccir_cfg.ca_lambda * perp_alpha * (seq_log_ratio - log_ratio_target)  # [B]
                                ca_lambda_seq = ca_lambda_seq.clamp(
                                    min=ccir_cfg.get("ca_lambda_min", -0.05),
                                    max=ccir_cfg.get("ca_lambda_max", 0.5))
                                # For logging: batch mean log_ratio (= approx log(batch_ratio))
                                ca_lambda = ca_lambda_seq.mean().item()
                                teacher_perp = torch.exp(seq_log_ratio).mean().item()  # log as ratio for compatibility
                                # Reshape for broadcasting: [B] → [B, 1]
                                _ca_lambda_broadcast = ca_lambda_seq.unsqueeze(-1)
                            else:
                                ca_lambda = ccir_cfg.ca_lambda

                            # Step 4.5b: Step-based λ cutoff (two-phase training)
                            step_cutoff = ccir_cfg.get("ca_lambda_step_cutoff", -1)
                            if step_cutoff > 0:
                                current_step = self._global_training_steps
                                if current_step >= step_cutoff:
                                    ca_lambda = 0.0
                                    _ca_lambda_broadcast = None  # force scalar fallback for per-seq modes

                            # Step 4.5c: s_specific-based lambda gate
                            s_specific_gate = ccir_cfg.get("ca_lambda_s_specific_gate", "none")
                            _s_specific_gate_mult = 1.0
                            if s_specific_gate != "none" and _ccir_s_specific is not None:
                                s_spec_abs_batch = _ccir_s_specific[response_mask.bool()].abs().mean().item() if response_mask.any() else 0.0

                                if s_specific_gate == "quality_ratio":
                                    # Quality ratio: gate = (quality - floor) / (1 - floor)
                                    # quality = s_spec / (s_spec + w * g_gap), self-calibrating, no warmup needed
                                    g_gap_weight = _ccir_generic_gap_weight
                                    g_gap_abs_batch = _ccir_generic_gap[response_mask.bool()].abs().mean().item() if (_ccir_generic_gap is not None and response_mask.any()) else 0.0
                                    quality = s_spec_abs_batch / max(s_spec_abs_batch + g_gap_weight * g_gap_abs_batch, 1e-8)
                                    s_spec_floor = ccir_cfg.get("ca_lambda_s_specific_floor", 0.5)
                                    _s_specific_gate_mult = max(0.0, (quality - s_spec_floor) / max(1.0 - s_spec_floor, 1e-8))
                                    _s_specific_gate_mult = min(1.0, _s_specific_gate_mult)

                                elif s_specific_gate == "proportional":
                                    # Original: gate = s_specific_abs / threshold (warmup-calibrated)
                                    s_spec_threshold = ccir_cfg.get("ca_lambda_s_specific_threshold", 0.0)
                                    s_spec_warmup = ccir_cfg.get("ca_lambda_warmup_steps", 5)
                                    current_step = self._global_training_steps
                                    if s_spec_threshold <= 0 and current_step < s_spec_warmup:
                                        if not hasattr(self, '_s_specific_warmup_values'):
                                            self._s_specific_warmup_values = []
                                        self._s_specific_warmup_values.append(s_spec_abs_batch)
                                    else:
                                        if s_spec_threshold <= 0:
                                            if not hasattr(self, '_s_specific_threshold_calibrated'):
                                                if hasattr(self, '_s_specific_warmup_values') and self._s_specific_warmup_values:
                                                    vals = sorted(self._s_specific_warmup_values)
                                                    self._s_specific_threshold_calibrated = vals[len(vals) // 2]
                                                else:
                                                    self._s_specific_threshold_calibrated = max(s_spec_abs_batch, 1e-6)
                                            s_spec_threshold = self._s_specific_threshold_calibrated
                                        s_spec_floor = ccir_cfg.get("ca_lambda_s_specific_floor", 0.0)
                                        _s_specific_gate_mult = min(1.0, s_spec_abs_batch / max(s_spec_threshold, 1e-8))
                                        _s_specific_gate_mult = max(_s_specific_gate_mult, s_spec_floor)

                                ca_lambda = ca_lambda * _s_specific_gate_mult
                                if _ca_lambda_broadcast is not None:
                                    _ca_lambda_broadcast = _ca_lambda_broadcast * _s_specific_gate_mult

                            # Step 4.5d: PRM-strength-adaptive λ decay (natural soft exit).
                            # As |base_kl| or |PRM_t| declines (training converges), λ shrinks.
                            # decay_factor = clamp(signal_ema_current / signal_baseline, floor, 1.0)
                            # Baseline frozen after warmup. Signal clamps at 1.0 (can't exceed baseline).
                            decay_mode = ccir_cfg.get("ca_lambda_decay_mode", "none")
                            _decay_factor = 1.0
                            _decay_signal_current = 0.0
                            _decay_signal_baseline = 0.0
                            if decay_mode != "none":
                                decay_warmup = int(ccir_cfg.get("ca_lambda_warmup_steps", 5))
                                decay_floor = float(ccir_cfg.get("ca_lambda_decay_floor", 0.0))
                                decay_ema = float(ccir_cfg.get("ca_lambda_decay_ema", 0.9))
                                current_step_dec = self._global_training_steps
                                # Compute signal on current batch
                                valid_dec = response_mask.bool()
                                if valid_dec.any():
                                    if decay_mode == "bkl_abs":
                                        sig_tensor = base_kl[valid_dec].detach().abs()
                                    elif decay_mode == "prm_abs":
                                        sig_tensor = PRM_t[valid_dec].detach().abs()
                                    else:
                                        sig_tensor = None
                                    if sig_tensor is not None and sig_tensor.numel() > 0:
                                        sig_local = sig_tensor.mean().unsqueeze(0).clone()
                                        if torch.distributed.is_initialized():
                                            torch.distributed.all_reduce(sig_local, op=torch.distributed.ReduceOp.AVG)
                                        sig_current = max(sig_local.item(), 1e-8)
                                        # EMA on current signal (reduces per-step noise)
                                        if not hasattr(self, '_lambda_decay_signal_ema'):
                                            self._lambda_decay_signal_ema = sig_current
                                        else:
                                            self._lambda_decay_signal_ema = (
                                                decay_ema * self._lambda_decay_signal_ema
                                                + (1.0 - decay_ema) * sig_current
                                            )
                                        _decay_signal_current = self._lambda_decay_signal_ema
                                        # Warmup: collect samples, no decay yet
                                        if current_step_dec < decay_warmup:
                                            if not hasattr(self, '_lambda_decay_baseline_samples'):
                                                self._lambda_decay_baseline_samples = []
                                            self._lambda_decay_baseline_samples.append(sig_current)
                                            _decay_factor = 1.0
                                        else:
                                            # Freeze baseline after warmup
                                            if not hasattr(self, '_lambda_decay_baseline'):
                                                if hasattr(self, '_lambda_decay_baseline_samples') and self._lambda_decay_baseline_samples:
                                                    self._lambda_decay_baseline = sum(self._lambda_decay_baseline_samples) / len(self._lambda_decay_baseline_samples)
                                                else:
                                                    self._lambda_decay_baseline = sig_current
                                                self._lambda_decay_baseline = max(self._lambda_decay_baseline, 1e-8)
                                            _decay_signal_baseline = self._lambda_decay_baseline
                                            _decay_factor = min(1.0, max(decay_floor, _decay_signal_current / _decay_signal_baseline))
                                    ca_lambda = ca_lambda * _decay_factor
                                    if _ca_lambda_broadcast is not None:
                                        _ca_lambda_broadcast = _ca_lambda_broadcast * _decay_factor

                            # Step 4.6a: Contrastive length brake
                            # PRM += beta * (t_correct - t_wrong) where contrastive has opposite length bias
                            contrastive_brake_applied = 0.0
                            brake_beta = ccir_cfg.get("contrastive_brake_beta", 0.0)
                            brake_adaptive = ccir_cfg.get("contrastive_brake_adaptive", False)
                            if (brake_beta > 0 or brake_adaptive) and teacher_nosol_log_prob is not None:
                                contrastive_signal = (teacher_log_prob - teacher_nosol_log_prob).detach()
                                if brake_adaptive:
                                    mean_resp_len_brake = response_mask.sum(dim=-1).float().mean().item()
                                    brake_target = ccir_cfg.get("contrastive_brake_length_target", 10000.0)
                                    brake_ratio = mean_resp_len_brake / max(brake_target, 1.0)
                                    effective_beta = brake_beta * max(0.0, brake_ratio - 0.8)
                                else:
                                    effective_beta = brake_beta
                                if effective_beta > 0:
                                    PRM_processed = PRM_processed + effective_beta * contrastive_signal * response_mask
                                    contrastive_brake_applied = effective_beta

                            # Step 4.6b: Position debiasing — remove position-dependent PRM bias
                            position_debias_mode = ccir_cfg.get("position_debias_mode", "none")
                            if position_debias_mode != "none":
                                seq_lens_pd = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, 1]
                                positions = torch.arange(PRM_processed.shape[1], device=PRM_processed.device).unsqueeze(0)  # [1, T]
                                position_frac = positions / seq_lens_pd  # [B, T], 0 to ~1
                                if position_debias_mode == "quartile":
                                    for q_start in [0.0, 0.25, 0.5, 0.75]:
                                        q_mask = ((position_frac >= q_start) & (position_frac < q_start + 0.25)
                                                  & response_mask.bool())
                                        if q_mask.any():
                                            PRM_processed[q_mask] -= PRM_processed[q_mask].mean()
                                elif position_debias_mode == "linear":
                                    # Subtract linear trend: PRM -= (a * pos_frac + b) fitted per batch
                                    valid_pd = response_mask.bool()
                                    if valid_pd.any():
                                        x = position_frac[valid_pd]
                                        y = PRM_processed[valid_pd]
                                        x_mean, y_mean = x.mean(), y.mean()
                                        cov_xy = ((x - x_mean) * (y - y_mean)).mean()
                                        var_x = ((x - x_mean) ** 2).mean().clamp(min=1e-8)
                                        slope = cov_xy / var_x
                                        intercept = y_mean - slope * x_mean
                                        PRM_processed = PRM_processed - (slope * position_frac + intercept) * response_mask

                            orm_weight = ccir_cfg.orm_weight
                            prm_orm_correlation = None
                            gate_value = None
                            if orm_weight > 0:
                                resp_lens_gate = response_mask.sum(dim=-1).clamp(min=1.0)
                                # Use raw base_kl (before seq-normalization) for meaningful per-seq PRM signal.
                                # PRM_processed is zero-mean per sequence after seqnorm → per-seq mean ≈ 0 → correlation trivially 0.
                                raw_PRM_per_seq = (base_kl * response_mask).sum(dim=-1) / resp_lens_gate
                                ORM_per_seq = (advantages * response_mask).sum(dim=-1) / resp_lens_gate
                                PRM_c = raw_PRM_per_seq - raw_PRM_per_seq.mean()
                                ORM_c = ORM_per_seq - ORM_per_seq.mean()
                                if PRM_c.numel() > 1:
                                    _prm_c_std = PRM_c.std().item()
                                    _orm_c_std = ORM_c.std().item()
                                if PRM_c.numel() > 1 and _prm_c_std is not None and _orm_c_std is not None and _prm_c_std > 1e-8 and _orm_c_std > 1e-8:
                                    prm_orm_correlation = torch.nn.functional.cosine_similarity(
                                        PRM_c.unsqueeze(0), ORM_c.unsqueeze(0)
                                    ).item()

                                # Only apply gating if enabled
                                if ccir_cfg.get("prm_alignment_gate", False):
                                    if prm_orm_correlation is None:
                                        # Correlation is undefined for single-sequence micro-batches.
                                        # Skip gating instead of zeroing PRM.
                                        gate_value = 1.0
                                    else:
                                        gate_value = max(prm_orm_correlation, 0.0)
                                        PRM_processed = PRM_processed * gate_value

                            # Step 4.6: Entropy gate — suppress PRM when entropy is low
                            entropy_gate_ratio = None
                            maxent_entropy_gate = ccir_cfg.get("maxent_entropy_gate", False)
                            if maxent_entropy_gate and entropy is not None:
                                H_batch = (entropy * response_mask).sum() / response_mask.sum().clamp(min=1.0)
                                H_batch_val = H_batch.item()
                                if self._maxent_h_target is None:
                                    h_target_cfg = ccir_cfg.get("maxent_h_target", None)
                                    self._maxent_h_target = h_target_cfg if h_target_cfg is not None else H_batch_val
                                entropy_gate_ratio = min(max(H_batch_val / max(self._maxent_h_target, 1e-6), 0.0), 1.0)
                                PRM_processed = PRM_processed * entropy_gate_ratio

                            # Step 4.8: FutureConf gate — weight PRM by teacher's future trajectory confidence
                            # gate_t = exp(FutureConf_t / remaining_len)
                            # FutureConf_t = Σ_{k=t}^{T} γ^{k-t} · teacher_log_prob_k
                            future_conf_gamma = ccir_cfg.get("future_conf_gamma", 0.0)
                            future_conf_gate_mean = 0.0
                            if future_conf_gamma > 0 and teacher_log_prob is not None:
                                t_masked = teacher_log_prob.detach() * response_mask  # [B, T]
                                B, T_len = t_masked.shape
                                # Reverse cumulative discounted sum: FC_T = t_T, FC_t = t_t + γ·FC_{t+1}
                                future_conf = torch.zeros_like(t_masked)
                                future_conf[:, -1] = t_masked[:, -1]
                                for k in range(T_len - 2, -1, -1):
                                    future_conf[:, k] = t_masked[:, k] + future_conf_gamma * future_conf[:, k + 1]
                                future_conf = future_conf * response_mask
                                # Normalize by remaining response length → geometric mean of future teacher probs
                                remaining = torch.cumsum(response_mask.flip(1), dim=1).flip(1).clamp(min=1.0)
                                fc_normalized = future_conf / remaining  # avg future teacher logprob
                                fc_gate = torch.exp(fc_normalized) * response_mask  # (0, 1)
                                PRM_processed = PRM_processed * fc_gate
                                future_conf_gate_mean = fc_gate[response_mask.bool()].mean().item() if response_mask.any() else 0.0

                            # Step 4.9: PRM position mask — hard cutoff beyond N tokens
                            prm_max_pos = ccir_cfg.get("prm_max_position", 0)
                            if prm_max_pos > 0:
                                response_pos = torch.cumsum(response_mask, dim=-1)  # 1-indexed position within response
                                pos_mask = (response_pos <= prm_max_pos).float() * response_mask
                                PRM_processed = PRM_processed * pos_mask

                            # Step 4.9b: Per-sequence PRM length mask.
                            # Zero PRM for sequences whose response length ≥ threshold.
                            # Long responses risk truncation → PRM signal unreliable → ORM handles.
                            # Short responses keep PRM → accelerated learning.
                            # Creates self-correcting equilibrium: PRM↑len → mask triggers → ORM↓len → PRM re-enters.
                            prm_len_mask_threshold = int(ccir_cfg.get("prm_length_mask_threshold", 0))
                            _prm_length_masked_frac = 0.0
                            if prm_len_mask_threshold > 0:
                                seq_lens_for_mask = response_mask.sum(dim=-1)  # [B]
                                long_seqs = (seq_lens_for_mask >= prm_len_mask_threshold)  # [B] bool
                                if long_seqs.any():
                                    _prm_length_masked_frac = long_seqs.float().mean().item()
                                    PRM_processed = PRM_processed * (~long_seqs).float().unsqueeze(-1)

                            # Step 4.9c: Entropy gate (batch-level, hysteresis).
                            # When batch entropy drops past ratio_low × H_warmup → PRM = 0.
                            # Reopens when H rises above ratio_high × H_warmup.
                            # Hysteresis prevents flapping near threshold.
                            # Data-calibrated: ratio_low=0.45 catches pre-crash (QN-fwdAlpha @s40=44%),
                            # ratio_high=0.60 stays above borderline (ON @s90=53%).
                            entropy_gate_mode = ccir_cfg.get("entropy_gate_mode", "none")
                            _entropy_gate_closed = 0.0
                            _entropy_gate_h_batch = 0.0
                            _entropy_gate_h_warmup = 0.0
                            if entropy_gate_mode != "none" and entropy is not None:
                                valid_eg = response_mask.bool()
                                if valid_eg.any():
                                    H_batch_tensor = (entropy * response_mask).sum() / response_mask.sum().clamp(min=1.0)
                                    H_batch_local = H_batch_tensor.detach().clone().unsqueeze(0)
                                    if torch.distributed.is_initialized():
                                        torch.distributed.all_reduce(H_batch_local, op=torch.distributed.ReduceOp.AVG)
                                    H_batch_val = H_batch_local.item()
                                    _entropy_gate_h_batch = H_batch_val
                                    eg_warmup = int(ccir_cfg.get("ca_lambda_warmup_steps", 10))
                                    current_step_eg = self._global_training_steps
                                    # Warmup: collect H baseline
                                    if current_step_eg < eg_warmup:
                                        if not hasattr(self, '_entropy_gate_warmup_values'):
                                            self._entropy_gate_warmup_values = []
                                        self._entropy_gate_warmup_values.append(H_batch_val)
                                    else:
                                        # Freeze H_warmup once (median of warmup values)
                                        if not hasattr(self, '_entropy_gate_h_warmup'):
                                            if hasattr(self, '_entropy_gate_warmup_values') and self._entropy_gate_warmup_values:
                                                vals = sorted(self._entropy_gate_warmup_values)
                                                local_median = vals[len(vals) // 2]
                                            else:
                                                local_median = H_batch_val
                                            median_tensor = torch.tensor([local_median], device=H_batch_local.device)
                                            if torch.distributed.is_initialized():
                                                torch.distributed.all_reduce(median_tensor, op=torch.distributed.ReduceOp.AVG)
                                            self._entropy_gate_h_warmup = max(median_tensor.item(), 1e-6)
                                        H_warmup = self._entropy_gate_h_warmup
                                        _entropy_gate_h_warmup = H_warmup
                                        ratio_low = float(ccir_cfg.get("entropy_gate_h_ratio_low", 0.45))
                                        ratio_high = float(ccir_cfg.get("entropy_gate_h_ratio_high", 0.60))
                                        # Initialize gate state
                                        if not hasattr(self, '_entropy_gate_is_closed'):
                                            self._entropy_gate_is_closed = False
                                        # State transition
                                        if entropy_gate_mode == "hysteresis":
                                            if self._entropy_gate_is_closed:
                                                if H_batch_val > ratio_high * H_warmup:
                                                    self._entropy_gate_is_closed = False
                                            else:
                                                if H_batch_val < ratio_low * H_warmup:
                                                    self._entropy_gate_is_closed = True
                                        elif entropy_gate_mode == "binary":
                                            self._entropy_gate_is_closed = (H_batch_val < ratio_low * H_warmup)
                                        if self._entropy_gate_is_closed:
                                            PRM_processed = PRM_processed * 0.0
                                            _entropy_gate_closed = 1.0

                            # Step 5: Advantage refinement (additive or multiplicative)
                            ca_mode = ccir_cfg.get("ca_mode", "additive")

                            if ca_mode in ("credit", "anti_credit"):
                                # Credit modes: A_t = A_seq * w_t, w_t = softmax(f_t / τ) * T
                                # Total gradient per seq = |A_seq| * T (conserved).
                                # ca_lambda acts as temperature τ (higher = more uniform).
                                tau = ca_lambda if ca_lambda > 0 else 1.0
                                seq_lens = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)

                                if ca_mode == "anti_credit":
                                    # Anti-PRM: f_t = -sign(A_ORM) * PRM / τ
                                    # Correct (A>0): f = (t_t - s_t)/τ → credit to teacher-student gap
                                    # Incorrect (A<0): f = (s_t - t_t)/τ → penalty on confident divergence
                                    # Negative feedback: s_t↑ → credit↓ (self-correcting, no hack)
                                    sign_A = torch.sign(advantages[:, :1])  # [B, 1] per-sequence sign
                                    prm_logits = -sign_A * PRM_processed / tau
                                else:
                                    # Standard credit: f_t = PRM / τ (positive feedback, hackable)
                                    prm_logits = PRM_processed / tau

                                # Mask out non-response tokens with -inf before softmax
                                prm_logits = prm_logits * response_mask + (-1e9) * (1.0 - response_mask)
                                credit_weights = torch.nn.functional.softmax(prm_logits, dim=-1) * response_mask
                                # Scale so E[w_t] = 1 (preserve gradient magnitude)
                                credit_weights = credit_weights * seq_lens

                                refined_advantages = advantages * credit_weights
                            elif ca_mode == "multiplicative":
                                # Multiplicative: A_t = A_seq * (1 + ca_lambda * PRM_norm)
                                # PRM only modulates ORM direction; when A_seq=0, A_t=0.
                                # No anchor needed — PRM_processed is already normalized.
                                _lam = _ca_lambda_broadcast if _ca_lambda_broadcast is not None else ca_lambda
                                refined_advantages = advantages * (1.0 + _lam * PRM_processed)
                            elif ca_mode == "rlsd":
                                # RLSD (Yang 2026 Self-Distilled RLVR) — magnitude-only modulation.
                                # A_t = A_seq · ((1-λ) + λ · clip(exp(sign(A_seq)·(t-s)), 1-ε, 1+ε))
                                # Direction is sign(A_seq) (env reward); evidence ratio only scales magnitude.
                                # PRM_processed is IGNORED here — RLSD doesn't use our φ-shaped PRM.
                                rlsd_eps = float(ccir_cfg.get("rlsd_eps_w", 0.2))
                                rlsd_lam = float(ccir_cfg.get("rlsd_lambda", 1.0))
                                sign_A = torch.sign(advantages).detach()
                                delta_t = (teacher_log_prob - log_prob).detach()
                                w_t = torch.exp(sign_A * delta_t)
                                w_clipped = torch.clamp(w_t, 1.0 - rlsd_eps, 1.0 + rlsd_eps)
                                modulator = (1.0 - rlsd_lam) + rlsd_lam * w_clipped if rlsd_lam < 1.0 else w_clipped
                                refined_advantages = advantages * modulator
                            else:
                                # Additive: A_t = orm_weight * A_seq + ca_lambda * PRM_processed
                                # Optional: anchor PRM magnitude to ORM magnitude
                                if ccir_cfg.get("prm_anchor_to_orm", False) and orm_weight > 0:
                                    valid_anchor = response_mask.bool()
                                    orm_scale = (orm_weight * advantages[valid_anchor]).abs().mean().clamp(min=1e-8)
                                    prm_scale = PRM_processed[valid_anchor].abs().mean().clamp(min=1e-8)
                                    PRM_processed = PRM_processed * (orm_scale / prm_scale)

                                _lam = _ca_lambda_broadcast if _ca_lambda_broadcast is not None else ca_lambda
                                refined_advantages = orm_weight * advantages + _lam * PRM_processed

                            # Step 5.5: MaxEnt entropy stabilization (Position C: on final advantage)
                            # A_maxent = A_refined - α·(s_t - mean(s_t))
                            # Empirically Cov(PRM, s) > 0 → entropy collapses.
                            # Subtracting α·(s-s̄) removes the confidence-correlated component.
                            # maxent_mode and maxent_position already set in Step 3.3
                            if maxent_mode != "none" and maxent_position == "advantage":
                                # Conditional maxent: only apply when λ > 0 (exploration mode)
                                maxent_conditional = ccir_cfg.get("maxent_conditional", False)
                                maxent_active = (not maxent_conditional) or (ca_lambda > 0)
                                maxent_correction_abs_mean = 0.0  # always define for metrics
                                if maxent_active:
                                    if self._maxent_alpha is None:
                                        self._maxent_alpha = ccir_cfg.get("maxent_alpha", 0.1)

                                    s_t = old_log_prob.detach()
                                    resp_lens_me = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                                    s_mean = (s_t * response_mask).sum(dim=-1, keepdim=True) / resp_lens_me
                                    s_centered = (s_t - s_mean) * response_mask

                                    correction = self._maxent_alpha * s_centered
                                    refined_advantages = refined_advantages - correction
                                    maxent_correction_abs_mean = correction[response_mask.bool()].abs().mean().item()

                            # Adaptive α update (runs regardless of position, needs entropy)
                            if maxent_mode == "adaptive" and entropy is not None:
                                if self._maxent_alpha is None:
                                    self._maxent_alpha = ccir_cfg.get("maxent_alpha", 0.1)
                                H_batch_me = (entropy * response_mask).sum() / response_mask.sum().clamp(min=1.0)
                                H_batch_me_val = H_batch_me.item()
                                if self._maxent_h_target is None:
                                    h_target_cfg = ccir_cfg.get("maxent_h_target", None)
                                    self._maxent_h_target = h_target_cfg if h_target_cfg is not None else H_batch_me_val
                                maxent_lr = ccir_cfg.get("maxent_lr", 0.01)
                                # α ← α - lr·(H_batch - H_target): H_batch < H_target → α increases
                                # (more s_t subtraction to counteract entropy collapse from Cov(PRM, s) > 0)
                                self._maxent_alpha = max(0.0, self._maxent_alpha - maxent_lr * (H_batch_me_val - self._maxent_h_target))

                            # Step 5.6: EN-decorrelation at Position C (on final advantage)
                            # Removes s_t-correlated component from refined_advantages (not PRM_t).
                            # Catches all sources of s_t correlation (PRM + seqnorm artifacts).
                            en_adv_beta = None
                            if en_mode == "decorrelate_advantage":
                                s_t_ea = old_log_prob.detach()
                                resp_lens_ea = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                                s_mean_ea = (s_t_ea * response_mask).sum(dim=-1, keepdim=True) / resp_lens_ea
                                s_centered_ea = (s_t_ea - s_mean_ea) * response_mask

                                adv_mean_ea = (refined_advantages * response_mask).sum(dim=-1, keepdim=True) / resp_lens_ea
                                adv_centered_ea = (refined_advantages - adv_mean_ea) * response_mask

                                cov_as = (adv_centered_ea * s_centered_ea * response_mask).sum(dim=-1, keepdim=True) / resp_lens_ea
                                var_s_ea = (s_centered_ea ** 2 * response_mask).sum(dim=-1, keepdim=True) / resp_lens_ea
                                en_adv_beta = cov_as / var_s_ea.clamp(min=1e-8)

                                refined_advantages = (refined_advantages - en_adv_beta * s_centered_ea) * response_mask

                            # Step 5.8: KL constraint against reference policy (π_ref = frozen initial)
                            # Subtracts β_kl * Δlog_p = β_kl * (s - t_nosol) from advantage.
                            # Directly damps self-reinforcement component of s-t PRM.
                            kl_ref_beta = ccir_cfg.get("kl_ref_beta", 0.0)
                            kl_ref_penalty_mean = 0.0
                            if kl_ref_beta > 0 and teacher_nosol_log_prob is not None:
                                delta_log_p = (log_prob - teacher_nosol_log_prob).detach()
                                kl_ref_penalty = kl_ref_beta * delta_log_p * response_mask
                                refined_advantages = refined_advantages - kl_ref_penalty
                                kl_ref_penalty_mean = kl_ref_penalty[response_mask.bool()].mean().item() if response_mask.any() else 0.0

                            # Step 5.9: Length penalty on advantage
                            lp_alpha = ccir_cfg.get("length_penalty_alpha", 0.0)
                            if lp_alpha > 0:
                                lp_target = ccir_cfg.get("length_penalty_target", 10000.0)
                                per_seq_len = response_mask.sum(dim=-1, keepdim=True).float()  # [B, 1]
                                len_overshoot = ((per_seq_len - lp_target) / max(lp_target, 1.0)).clamp(min=0.0)  # [B, 1]
                                refined_advantages = refined_advantages - lp_alpha * len_overshoot * response_mask

                            # Step 6: Vanilla PPO loss with refined advantages
                            policy_loss_fn = get_policy_loss_fn("vanilla")
                            pg_loss, pg_metrics = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=refined_advantages,
                                response_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                            )

                            # Step 7: Metrics
                            resp_mask_bool = response_mask.bool()
                            if response_mask.any():
                                valid = resp_mask_bool
                                # --- Raw components ---
                                pg_metrics["grpo_ca/base_kl_mean"] = base_kl[valid].mean().item()
                                pg_metrics["grpo_ca/ca_lambda"] = ca_lambda
                                if _ca_lambda_broadcast is not None and ca_lambda_seq is not None:
                                    pg_metrics["grpo_ca/ca_lambda_std"] = ca_lambda_seq.std().item()
                                    pg_metrics["grpo_ca/ca_lambda_min_seq"] = ca_lambda_seq.min().item()
                                    pg_metrics["grpo_ca/ca_lambda_max_seq"] = ca_lambda_seq.max().item()
                                    pg_metrics["grpo_ca/ca_lambda_pos_frac"] = (ca_lambda_seq > 0).float().mean().item()
                                else:
                                    pg_metrics["grpo_ca/ca_lambda_std"] = 0.0
                                    pg_metrics["grpo_ca/ca_lambda_min_seq"] = ca_lambda
                                    pg_metrics["grpo_ca/ca_lambda_max_seq"] = ca_lambda
                                    pg_metrics["grpo_ca/ca_lambda_pos_frac"] = 1.0 if ca_lambda > 0 else 0.0
                                # Teacher perp control diagnostics
                                pg_metrics["grpo_ca/perp_target"] = getattr(self, '_perp_target_calibrated', 0.0)
                                pg_metrics["grpo_ca/prm_strength_target"] = getattr(self, '_prm_strength_target', 0.0)
                                _tp_warmup_active = (ca_lambda_mode == "teacher_perp" and hasattr(self, '_warmup_perp_values') and not hasattr(self, '_perp_target_calibrated'))
                                _ps_warmup_active = (ca_lambda_mode == "prm_strength" and hasattr(self, '_prm_strength_baseline') and not hasattr(self, '_prm_strength_target'))
                                pg_metrics["grpo_ca/warmup_active"] = 1.0 if (_tp_warmup_active or _ps_warmup_active) else 0.0
                                pg_metrics["grpo_ca/perp_masked_frac"] = (seq_teacher_perp > ccir_cfg.get("ca_lambda_perp_mask", 3.0)).float().mean().item() if seq_teacher_perp is not None else 0.0
                                # Per-seq teacher_perp distribution — early warning for mask threshold approach
                                if seq_teacher_perp is not None:
                                    pg_metrics["grpo_ca/teacher_perp_seq_max"] = seq_teacher_perp.max().item()
                                    pg_metrics["grpo_ca/teacher_perp_seq_std"] = seq_teacher_perp.std().item() if seq_teacher_perp.numel() > 1 else 0.0
                                    # p90 via quantile (approximate, avoids sort on large batch)
                                    if seq_teacher_perp.numel() >= 10:
                                        pg_metrics["grpo_ca/teacher_perp_seq_p90"] = seq_teacher_perp.quantile(0.9).item()
                                    else:
                                        pg_metrics["grpo_ca/teacher_perp_seq_p90"] = seq_teacher_perp.max().item()
                                else:
                                    pg_metrics["grpo_ca/teacher_perp_seq_max"] = 0.0
                                    pg_metrics["grpo_ca/teacher_perp_seq_std"] = 0.0
                                    pg_metrics["grpo_ca/teacher_perp_seq_p90"] = 0.0
                                pg_metrics["grpo_ca/mean_shift_amount"] = _mean_shift_amount
                                pg_metrics["grpo_ca/prm_active"] = float(getattr(self, '_prm_active', True))
                                pg_metrics["grpo_ca/s_specific_gate_mult"] = _s_specific_gate_mult
                                # PRM-strength adaptive λ decay diagnostics
                                pg_metrics["grpo_ca/lambda_decay_factor"] = _decay_factor
                                pg_metrics["grpo_ca/lambda_decay_signal_current"] = _decay_signal_current
                                pg_metrics["grpo_ca/lambda_decay_signal_baseline"] = _decay_signal_baseline
                                pg_metrics["grpo_ca/prm_length_masked_frac"] = _prm_length_masked_frac
                                pg_metrics["grpo_ca/entropy_gate_closed"] = _entropy_gate_closed
                                pg_metrics["grpo_ca/entropy_gate_h_batch"] = _entropy_gate_h_batch
                                pg_metrics["grpo_ca/entropy_gate_h_warmup"] = _entropy_gate_h_warmup
                                # Position profile diagnostic: PRM mean by quartile (always emit all keys)
                                seq_lens_diag = response_mask.sum(dim=-1, keepdim=True).float().clamp(min=1.0)
                                pos_diag = torch.arange(PRM_processed.shape[1], device=PRM_processed.device).float().unsqueeze(0)
                                pos_frac_diag = pos_diag / seq_lens_diag
                                for lo, hi, label in [
                                    (0.0, 0.25, "q1"), (0.25, 0.5, "q2"),
                                    (0.5, 0.75, "q3"), (0.75, 1.01, "q4"),
                                ]:
                                    qm = ((pos_frac_diag >= lo) & (pos_frac_diag < hi) & valid)
                                    pg_metrics[f"grpo_ca/PRM_pos_{label}"] = PRM_processed[qm].mean().item() if qm.any() else 0.0
                                pg_metrics["grpo_ca/contrastive_brake_beta"] = contrastive_brake_applied
                                pg_metrics["grpo_ca/kl_ref_penalty_mean"] = kl_ref_penalty_mean
                                pg_metrics["grpo_ca/future_conf_gate_mean"] = future_conf_gate_mean
                                # Teacher perplexity (always log for diagnostics)
                                if teacher_log_prob is not None and response_mask.any():
                                    _mean_t_lp = (teacher_log_prob.detach() * response_mask).sum() / response_mask.sum().clamp(min=1)
                                    pg_metrics["grpo_ca/teacher_perplexity"] = torch.exp(-_mean_t_lp).item()
                                else:
                                    pg_metrics["grpo_ca/teacher_perplexity"] = 0.0
                                if teacher_log_prob is not None:
                                    s_minus_t = (log_prob.detach() - teacher_log_prob.detach()) * response_mask
                                    t_minus_s = -s_minus_t
                                    s_minus_t_valid = s_minus_t[valid]
                                    t_minus_s_valid = t_minus_s[valid]
                                    pg_metrics["grpo_ca/s_minus_t_mean"] = s_minus_t_valid.mean().item()
                                    pg_metrics["grpo_ca/s_minus_t_std"] = s_minus_t_valid.std().item() if s_minus_t_valid.numel() > 1 else 0.0
                                    pg_metrics["grpo_ca/t_minus_s_mean"] = t_minus_s_valid.mean().item()
                                    pg_metrics["grpo_ca/t_minus_s_std"] = t_minus_s_valid.std().item() if t_minus_s_valid.numel() > 1 else 0.0
                                    pg_metrics["grpo_ca/reverse_kl_density_mean"] = s_minus_t_valid.mean().item()
                                    pg_metrics["grpo_ca/reverse_kl_density_abs_mean"] = s_minus_t_valid.abs().mean().item()
                                else:
                                    pg_metrics["grpo_ca/s_minus_t_mean"] = 0.0
                                    pg_metrics["grpo_ca/s_minus_t_std"] = 0.0
                                    pg_metrics["grpo_ca/t_minus_s_mean"] = 0.0
                                    pg_metrics["grpo_ca/t_minus_s_std"] = 0.0
                                    pg_metrics["grpo_ca/reverse_kl_density_mean"] = 0.0
                                    pg_metrics["grpo_ca/reverse_kl_density_abs_mean"] = 0.0
                                # Per-seq student perplexity and t/s ratio diagnostics
                                if seq_student_perp is not None and seq_ts_ratio is not None:
                                    pg_metrics["grpo_ca/student_perplexity"] = seq_student_perp.mean().item()
                                    pg_metrics["grpo_ca/ts_ratio_mean"] = seq_ts_ratio.mean().item()
                                    pg_metrics["grpo_ca/ts_ratio_std"] = seq_ts_ratio.std().item()
                                    pg_metrics["grpo_ca/ts_ratio_min"] = seq_ts_ratio.min().item()
                                    pg_metrics["grpo_ca/ts_ratio_max"] = seq_ts_ratio.max().item()
                                pg_metrics["grpo_ca/length_penalty_mean"] = (
                                    (lp_alpha * len_overshoot).mean().item() if lp_alpha > 0 else 0.0
                                )
                                pg_metrics["grpo_ca/forward_token_active"] = 1.0 if _forward_token_weights is not None else 0.0
                                pg_metrics["grpo_ca/forward_token_beta"] = (
                                    float(ccir_cfg.get("prm_forward_beta", 1.0)) if _forward_token_weights is not None else 0.0
                                )
                                pg_metrics["grpo_ca/forward_token_log_clip"] = (
                                    float(ccir_cfg.get("prm_forward_log_clip", 5.0)) if _forward_token_weights is not None else 0.0
                                )
                                pg_metrics["grpo_ca/forward_token_clip_frac"] = _forward_token_clip_frac
                                pg_metrics["grpo_ca/forward_token_prm_input_mean"] = _forward_token_input_mean
                                pg_metrics["grpo_ca/forward_token_prm_input_abs_mean"] = _forward_token_input_abs_mean
                                pg_metrics["grpo_ca/forward_token_prm_output_mean"] = _forward_token_output_mean
                                pg_metrics["grpo_ca/forward_token_prm_output_abs_mean"] = _forward_token_output_abs_mean
                                if _forward_token_gap is not None:
                                    forward_gap_valid = _forward_token_gap[valid]
                                    pg_metrics["grpo_ca/forward_token_gap_mean"] = forward_gap_valid.mean().item()
                                    pg_metrics["grpo_ca/forward_token_gap_std"] = (
                                        forward_gap_valid.std().item() if forward_gap_valid.numel() > 1 else 0.0
                                    )
                                else:
                                    pg_metrics["grpo_ca/forward_token_gap_mean"] = 0.0
                                    pg_metrics["grpo_ca/forward_token_gap_std"] = 0.0
                                if _forward_token_weights is not None and _forward_token_log_weights is not None:
                                    forward_weight_valid = _forward_token_weights[valid]
                                    forward_logw_valid = _forward_token_log_weights[valid]
                                    pg_metrics["grpo_ca/forward_token_weight_mean"] = forward_weight_valid.mean().item()
                                    pg_metrics["grpo_ca/forward_token_weight_std"] = (
                                        forward_weight_valid.std().item() if forward_weight_valid.numel() > 1 else 0.0
                                    )
                                    pg_metrics["grpo_ca/forward_token_weight_min"] = forward_weight_valid.min().item()
                                    pg_metrics["grpo_ca/forward_token_weight_max"] = forward_weight_valid.max().item()
                                    pg_metrics["grpo_ca/forward_token_logw_mean"] = forward_logw_valid.mean().item()
                                    pg_metrics["grpo_ca/forward_token_logw_std"] = (
                                        forward_logw_valid.std().item() if forward_logw_valid.numel() > 1 else 0.0
                                    )
                                else:
                                    pg_metrics["grpo_ca/forward_token_weight_mean"] = 1.0
                                    pg_metrics["grpo_ca/forward_token_weight_std"] = 0.0
                                    pg_metrics["grpo_ca/forward_token_weight_min"] = 1.0
                                    pg_metrics["grpo_ca/forward_token_weight_max"] = 1.0
                                    pg_metrics["grpo_ca/forward_token_logw_mean"] = 0.0
                                    pg_metrics["grpo_ca/forward_token_logw_std"] = 0.0
                                # Rényi-α unbiased PRM diagnostics
                                pg_metrics["grpo_ca/renyi_active"] = 1.0 if _renyi_active else 0.0
                                pg_metrics["grpo_ca/renyi_alpha"] = (
                                    float(ccir_cfg.get("prm_renyi_alpha", 1.0)) if _renyi_active else 0.0
                                )
                                pg_metrics["grpo_ca/renyi_log_clip"] = (
                                    float(ccir_cfg.get("prm_forward_log_clip", 3.0)) if _renyi_active else 0.0
                                )
                                pg_metrics["grpo_ca/renyi_sign"] = (
                                    float(ccir_cfg.get("prm_renyi_sign", 1.0)) if _renyi_active else 0.0
                                )
                                pg_metrics["grpo_ca/renyi_virtual_alpha"] = (
                                    float(ccir_cfg.get("prm_renyi_virtual_alpha", 1.0)) if _renyi_active else 0.0
                                )
                                pg_metrics["grpo_ca/renyi_clip_frac"] = _renyi_clip_frac
                                # Adaptive u clip diagnostics
                                pg_metrics["grpo_ca/u_clip_active"] = 1.0 if _u_clip_active else 0.0
                                pg_metrics["grpo_ca/u_clip_sigma_ref"] = _u_clip_sigma_ref
                                pg_metrics["grpo_ca/u_clip_range"] = _u_clip_range
                                pg_metrics["grpo_ca/u_clip_frac"] = _u_clip_frac
                                pg_metrics["grpo_ca/u_std_ratio"] = _u_std_ratio
                                pg_metrics["grpo_ca/renyi_prm_mean"] = _renyi_prm_mean
                                pg_metrics["grpo_ca/renyi_prm_abs_mean"] = _renyi_prm_abs_mean
                                pg_metrics["grpo_ca/renyi_u_clipped_frac"] = _renyi_u_clipped_frac
                                # JSD unbiased PRM diagnostics
                                pg_metrics["grpo_ca/jsd_active"] = 1.0 if _jsd_active else 0.0
                                pg_metrics["grpo_ca/jsd_sign"] = (
                                    float(ccir_cfg.get("prm_renyi_sign", 1.0)) if _jsd_active else 0.0
                                )
                                pg_metrics["grpo_ca/jsd_virtual_alpha"] = (
                                    float(ccir_cfg.get("prm_renyi_virtual_alpha", 1.0)) if _jsd_active else 0.0
                                )
                                pg_metrics["grpo_ca/jsd_clip_frac"] = _jsd_clip_frac
                                pg_metrics["grpo_ca/jsd_prm_mean"] = _jsd_prm_mean
                                pg_metrics["grpo_ca/jsd_prm_abs_mean"] = _jsd_prm_abs_mean
                                pg_metrics["grpo_ca/jsd_u_clipped_frac"] = _jsd_u_clipped_frac
                                pg_metrics["grpo_ca/jsd_gap_mean"] = _jsd_gap_mean
                                pg_metrics["grpo_ca/jsd_gap_abs_mean"] = _jsd_gap_abs_mean
                                pg_metrics["grpo_ca/ccir_S_t_mean"] = ccir_S_t[valid].mean().item()
                                pg_metrics["grpo_ca/ccir_S_t_std"] = ccir_S_t[valid].std().item()
                                # CCIR cross-problem decomposition: s_specific vs generic_gap
                                if _ccir_s_specific is not None:
                                    pg_metrics["grpo_ca/ccir_s_specific_mean"] = _ccir_s_specific[valid].mean().item()
                                    pg_metrics["grpo_ca/ccir_s_specific_abs"] = _ccir_s_specific[valid].abs().mean().item()
                                if _ccir_generic_gap is not None:
                                    pg_metrics["grpo_ca/ccir_generic_gap_mean"] = _ccir_generic_gap[valid].mean().item()
                                    pg_metrics["grpo_ca/ccir_generic_gap_abs"] = _ccir_generic_gap[valid].abs().mean().item()
                                pg_metrics["grpo_ca/ccir_generic_gap_weight"] = _ccir_generic_gap_weight
                                pg_metrics["grpo_ca/PRM_raw_mean"] = PRM_t[valid].mean().item()
                                pg_metrics["grpo_ca/PRM_raw_std"] = PRM_t[valid].std().item()
                                pg_metrics["grpo_ca/PRM_processed_mean"] = PRM_processed[valid].mean().item()
                                pg_metrics["grpo_ca/PRM_processed_abs_mean"] = PRM_processed[valid].abs().mean().item()
                                pg_metrics["grpo_ca/A_seq_mean"] = advantages[valid].mean().item()
                                pg_metrics["grpo_ca/A_seq_abs_mean"] = advantages[valid].abs().mean().item()
                                # --- Final refined advantages ---
                                pg_metrics["grpo_ca/refined_adv_mean"] = refined_advantages[valid].mean().item()
                                pg_metrics["grpo_ca/refined_adv_abs_mean"] = refined_advantages[valid].abs().mean().item()
                                if ca_mode in ("credit", "anti_credit"):
                                    # --- Credit weight stats ---
                                    # credit_weights: per-token weight (E[w]=1, sum=seq_len)
                                    cw = credit_weights[valid]
                                    pg_metrics["grpo_ca/credit_weight_mean"] = cw.mean().item()
                                    pg_metrics["grpo_ca/credit_weight_std"] = cw.std().item()
                                    pg_metrics["grpo_ca/credit_weight_max"] = cw.max().item()
                                    pg_metrics["grpo_ca/credit_weight_min"] = cw[cw > 0].min().item() if (cw > 0).any() else 0.0
                                    # Effective number of tokens: exp(entropy of weights) / seq_len
                                    # 1.0 = perfectly uniform, 0.0 = all mass on one token
                                    w_norm = credit_weights / credit_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                                    w_ent = -(w_norm * (w_norm + 1e-10).log() * response_mask).sum(dim=-1)
                                    max_ent = seq_lens.squeeze(-1).clamp(min=1.0).log()
                                    credit_uniformity = (w_ent / max_ent.clamp(min=1e-8)).mean().item()
                                    pg_metrics["grpo_ca/credit_uniformity"] = credit_uniformity
                                    pg_metrics["grpo_ca/credit_temperature"] = tau
                                    if ca_mode == "anti_credit":
                                        # Anti-credit specific: sign distribution and gap stats
                                        adv_first = advantages[:, 0]  # per-sequence advantage
                                        n_pos = (adv_first > 0).sum().item()
                                        n_neg = (adv_first < 0).sum().item()
                                        n_zero = (adv_first == 0).sum().item()
                                        pg_metrics["grpo_ca/anti_credit_n_pos"] = n_pos
                                        pg_metrics["grpo_ca/anti_credit_n_neg"] = n_neg
                                        pg_metrics["grpo_ca/anti_credit_n_zero"] = n_zero
                                        # Teacher-student gap on correct/incorrect seqs
                                        # Always emit to avoid inhomogeneous list lengths across shards
                                        correct_mask = (advantages > 0) & response_mask.bool()
                                        pg_metrics["grpo_ca/anti_credit_gap_correct"] = (
                                            PRM_processed[correct_mask].mean().item() if correct_mask.any() else 0.0
                                        )
                                        incorrect_mask = (advantages < 0) & response_mask.bool()
                                        pg_metrics["grpo_ca/anti_credit_gap_incorrect"] = (
                                            PRM_processed[incorrect_mask].mean().item() if incorrect_mask.any() else 0.0
                                        )
                                elif ca_mode == "multiplicative":
                                    # --- Multiplicative modulation stats ---
                                    modulation = ca_lambda * PRM_processed[valid]
                                    pg_metrics["grpo_ca/modulation_mean"] = modulation.mean().item()
                                    pg_metrics["grpo_ca/modulation_abs_mean"] = modulation.abs().mean().item()
                                    pg_metrics["grpo_ca/modulation_std"] = modulation.std().item()
                                elif ca_mode == "rlsd":
                                    # --- RLSD evidence-ratio modulation stats ---
                                    sign_A = torch.sign(advantages[valid]).detach()
                                    delta_t = (teacher_log_prob[valid] - log_prob[valid]).detach()
                                    w_t = torch.exp(sign_A * delta_t)
                                    rlsd_eps = float(ccir_cfg.get("rlsd_eps_w", 0.2))
                                    w_clip_frac = ((w_t > 1.0 + rlsd_eps) | (w_t < 1.0 - rlsd_eps)).float().mean().item()
                                    pg_metrics["grpo_ca/rlsd_w_mean"] = w_t.mean().item()
                                    pg_metrics["grpo_ca/rlsd_w_std"] = w_t.std().item()
                                    pg_metrics["grpo_ca/rlsd_w_clip_frac"] = w_clip_frac
                                else:
                                    # --- Additive terms: A_t = orm_w * A_seq + ca_λ * PRM ---
                                    orm_term = orm_weight * advantages[valid]
                                    prm_term = ca_lambda * PRM_processed[valid]
                                    pg_metrics["grpo_ca/orm_term_mean"] = orm_term.mean().item()
                                    pg_metrics["grpo_ca/orm_term_abs_mean"] = orm_term.abs().mean().item()
                                    pg_metrics["grpo_ca/prm_term_mean"] = prm_term.mean().item()
                                    pg_metrics["grpo_ca/prm_term_abs_mean"] = prm_term.abs().mean().item()
                                    # --- Dominance ratio: |prm_term| / (|orm_term| + |prm_term|) ---
                                    orm_abs = orm_term.abs().mean()
                                    prm_abs = prm_term.abs().mean()
                                    pg_metrics["grpo_ca/prm_dominance"] = (prm_abs / (orm_abs + prm_abs + 1e-8)).item()
                            # --- Entropy-neutral decorrelation metrics ---
                            if en_mode == "decorrelate" and en_beta is not None:
                                pg_metrics["grpo_ca/en_beta_mean"] = en_beta.mean().item()
                                pg_metrics["grpo_ca/en_beta_std"] = en_beta.std().item()
                                pg_metrics["grpo_ca/en_cov_prm_s_before"] = en_cov_before if en_cov_before is not None else 0.0
                                # Post-decorrelation Cov(PRM_decorr, s) — should be ≈ 0
                                if response_mask.any():
                                    s_t_check = old_log_prob.detach()
                                    resp_lens_check = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                                    s_mean_check = (s_t_check * response_mask).sum(dim=-1, keepdim=True) / resp_lens_check
                                    s_c = (s_t_check - s_mean_check) * response_mask
                                    # PRM_t is already decorrelated at this point (overwritten in step 3.5)
                                    prm_m = (PRM_t * response_mask).sum(dim=-1, keepdim=True) / resp_lens_check
                                    prm_c = (PRM_t - prm_m) * response_mask
                                    cov_after = (prm_c * s_c * response_mask).sum(dim=-1, keepdim=True) / resp_lens_check
                                    pg_metrics["grpo_ca/en_cov_prm_s_after"] = cov_after.mean().item()
                            # --- EN at Position C metrics ---
                            if en_mode == "decorrelate_advantage" and en_adv_beta is not None:
                                pg_metrics["grpo_ca/en_adv_beta_mean"] = en_adv_beta.mean().item()
                                pg_metrics["grpo_ca/en_adv_beta_std"] = en_adv_beta.std().item()
                            pg_metrics["grpo_ca/en_mode"] = {"none": 0.0, "decorrelate": 1.0, "decorrelate_advantage": 2.0}.get(en_mode, -1.0)
                            pg_metrics["grpo_ca/ca_mode"] = {"additive": 0.0, "multiplicative": 1.0, "credit": 2.0, "anti_credit": 3.0, "rlsd": 4.0}.get(ca_mode, -1.0)
                            pg_metrics["grpo_ca/prm_normalize_mode"] = {"none": 0.0, "batch": 1.0, "sequence": 2.0, "sequence_demean": 3.0}.get(normalize_mode, -1.0)
                            ccir_cp_mode_metric = ccir_cfg.get("ccir_cross_problem_mode", "full")
                            pg_metrics["grpo_ca/ccir_cross_problem_mode"] = {"full": 0.0, "blend": 1.0, "blend_current_teacher": 2.0}.get(ccir_cp_mode_metric, -1.0)
                            pg_metrics["grpo_ca/ccir_cross_problem_alpha"] = float(ccir_cfg.get("ccir_cross_problem_alpha", 0.5)) if ccir_cp_mode_metric == "blend_current_teacher" else 0.0
                            pg_metrics["grpo_ca/prm_anchor_to_orm"] = float(ccir_cfg.get("prm_anchor_to_orm", False))
                            pg_metrics["grpo_ca/prm_seq_demean"] = float(ccir_cfg.get("prm_seq_demean", False))
                            pg_metrics["grpo_ca/prm_construction"] = {"raw": 0.0, "indep_norm": 1.0}.get(prm_construction, -1.0)
                            pg_metrics["grpo_ca/prm_weight"] = prm_w
                            pg_metrics["grpo_ca/orm_weight"] = orm_weight
                            # --- Raw PRM per-seq bias diagnostics (critical for prm_normalize_mode=none) ---
                            # Without seq-norm, each seq's mean(PRM_raw) ≈ log(t_ppl_i/s_ppl_i) enters
                            # advantage unchanged → systematic pg_loss bias. Emit regardless of norm mode
                            # to enable direct A/B comparison between raw and seq-normed runs.
                            if base_kl is not None and response_mask.any():
                                _lens_diag = response_mask.sum(dim=-1).clamp(min=1.0)
                                _bkl_seq_mean = (base_kl * response_mask).sum(dim=-1) / _lens_diag  # [B]
                                mini_batch_prm_seq_means.append(_bkl_seq_mean.detach().float().cpu())
                                if orm_weight > 0:
                                    _orm_seq_mean = (advantages * response_mask).sum(dim=-1) / _lens_diag  # [B]
                                    mini_batch_orm_seq_means.append(_orm_seq_mean.detach().float().cpu())
                                # Token-level (base_kl, advantages) sums for global PRM-ORM correlation.
                                # Stay on GPU; all-reduce + reduce happens once per mini-batch after micro-batch loop.
                                if orm_weight > 0:
                                    _valid_tok = response_mask.bool()
                                    if _valid_tok.any():
                                        _pk_tok = base_kl[_valid_tok].detach().float()
                                        _ov_tok = advantages[_valid_tok].detach().float()
                                        _stats = torch.stack([
                                            torch.tensor(float(_pk_tok.numel()), device=_pk_tok.device),
                                            _pk_tok.sum(),
                                            _ov_tok.sum(),
                                            (_pk_tok * _ov_tok).sum(),
                                            (_pk_tok * _pk_tok).sum(),
                                            (_ov_tok * _ov_tok).sum(),
                                        ])
                                        if mini_batch_corr_stats is None:
                                            mini_batch_corr_stats = _stats.detach().clone()
                                        else:
                                            mini_batch_corr_stats = mini_batch_corr_stats + _stats.detach()
                                pg_metrics["grpo_ca/base_kl_seq_mean_abs_mean"] = _bkl_seq_mean.abs().mean().item()
                                pg_metrics["grpo_ca/base_kl_seq_mean_pos_frac"] = (_bkl_seq_mean > 0).float().mean().item()
                                pg_metrics["grpo_ca/base_kl_seq_mean_max"] = _bkl_seq_mean.max().item()
                                pg_metrics["grpo_ca/base_kl_seq_mean_min"] = _bkl_seq_mean.min().item()
                                # Estimated pg_loss contribution from PRM bias: mean(λ_i × seq_mean_i)
                                # Handles ca_lambda as both scalar (float) and per-seq tensor (_ca_lambda_broadcast)
                                _lam_per_seq = None
                                if _ca_lambda_broadcast is not None:
                                    _lam_per_seq = _ca_lambda_broadcast.squeeze(-1) if _ca_lambda_broadcast.dim() > 1 else _ca_lambda_broadcast
                                elif isinstance(ca_lambda, (int, float)):
                                    _lam_per_seq = torch.full_like(_bkl_seq_mean, float(ca_lambda))
                                if _lam_per_seq is not None and _lam_per_seq.shape == _bkl_seq_mean.shape:
                                    pg_metrics["grpo_ca/prm_bias_estimate"] = (_lam_per_seq * _bkl_seq_mean).mean().abs().item()
                                else:
                                    pg_metrics["grpo_ca/prm_bias_estimate"] = 0.0
                            else:
                                pg_metrics["grpo_ca/base_kl_seq_mean_abs_mean"] = 0.0
                                pg_metrics["grpo_ca/base_kl_seq_mean_pos_frac"] = 0.0
                                pg_metrics["grpo_ca/base_kl_seq_mean_max"] = 0.0
                                pg_metrics["grpo_ca/base_kl_seq_mean_min"] = 0.0
                                pg_metrics["grpo_ca/prm_bias_estimate"] = 0.0
                            # --- Solution Influence (SI) / teacher_only metrics ---
                            if si_mode != "none":
                                pg_metrics["grpo_ca/si_mode"] = {"none": 0, "centered": 1, "raw": 2, "replace": 3, "teacher_only": 4, "replace_indep_norm": 5}.get(si_mode, -1)
                            if si_mode != "none" and SI_t is not None:
                                si_valid = SI_t[valid]
                                pg_metrics["grpo_ca/SI_raw_mean"] = si_valid.mean().item()
                                pg_metrics["grpo_ca/SI_raw_std"] = si_valid.std().item()
                                pg_metrics["grpo_ca/SI_raw_abs_mean"] = si_valid.abs().mean().item()
                                # Per-sequence average SI: positive = solution helps predict response
                                SI_per_seq = (SI_t * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)
                                pg_metrics["grpo_ca/SI_seq_pos_frac"] = (SI_per_seq > 0).float().mean().item()
                                pg_metrics["grpo_ca/SI_seq_mean"] = SI_per_seq.mean().item()
                                if SI_centered is not None:
                                    pg_metrics["grpo_ca/SI_centered_abs_mean"] = SI_centered[valid].abs().mean().item()
                                pg_metrics["grpo_ca/si_mode"] = {"none": 0, "centered": 1, "raw": 2, "replace": 3, "teacher_only": 4}.get(si_mode, -1)
                                # Sol/nosol individual log-prob means (verify both forward passes ran)
                                pg_metrics["grpo_ca/SI_sol_logprob_mean"] = si_sol_log_prob[valid].mean().item()
                                pg_metrics["grpo_ca/SI_nosol_logprob_mean"] = teacher_nosol_log_prob[valid].mean().item()
                                # SI split by GRPO advantage sign (correct vs incorrect direction)
                                # For feedback_contrastive: correct → SI>0, incorrect → SI<0
                                A_per_seq = (advantages * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)
                                correct_seqs = A_per_seq > 0
                                incorrect_seqs = A_per_seq < 0
                                pg_metrics["grpo_ca/SI_correct_seq_mean"] = (
                                    SI_per_seq[correct_seqs].mean().item() if correct_seqs.any() else 0.0)
                                pg_metrics["grpo_ca/SI_incorrect_seq_mean"] = (
                                    SI_per_seq[incorrect_seqs].mean().item() if incorrect_seqs.any() else 0.0)
                                # Correlation between SI direction and GRPO advantage
                                if SI_per_seq.numel() > 1 and A_per_seq.std() > 1e-8 and SI_per_seq.std() > 1e-8:
                                    pg_metrics["grpo_ca/SI_advantage_corr"] = torch.corrcoef(
                                        torch.stack([SI_per_seq, A_per_seq])
                                    )[0, 1].item()
                                else:
                                    pg_metrics["grpo_ca/SI_advantage_corr"] = 0.0
                            # --- tanh saturation metrics ---
                            if prm_pre_tanh_abs is not None:
                                pg_metrics["grpo_ca/prm_pre_tanh_abs"] = prm_pre_tanh_abs
                                sat_frac = ((PRM_t[resp_mask_bool].abs() / prm_tanh_tau) > 0.9).float().mean().item()
                                pg_metrics["grpo_ca/prm_tanh_saturation_frac"] = sat_frac
                            # --- PRM-ORM correlation and alignment gate metrics ---
                            if gate_value is not None:
                                pg_metrics["grpo_ca/prm_gate_value"] = gate_value
                            # --- MaxEnt metrics ---
                            if maxent_mode != "none":
                                pg_metrics["grpo_ca/maxent_alpha"] = self._maxent_alpha
                                if maxent_correction_abs_mean is not None:
                                    pg_metrics["grpo_ca/maxent_correction_abs_mean"] = maxent_correction_abs_mean
                            if self._maxent_h_target is not None:
                                pg_metrics["grpo_ca/maxent_h_target"] = self._maxent_h_target
                            if maxent_mode == "adaptive" and entropy is not None:
                                pg_metrics["grpo_ca/maxent_h_batch"] = H_batch_me_val
                                pg_metrics["grpo_ca/maxent_entropy_gap"] = H_batch_me_val - self._maxent_h_target
                            if entropy_gate_ratio is not None:
                                pg_metrics["grpo_ca/maxent_entropy_gate_ratio"] = entropy_gate_ratio
                            pg_metrics["grpo_ca/pg_loss"] = pg_loss.detach().item()
                            pg_metrics["self_distillation/empty_target_batch"] = self_distillation_mask.sum().item() == 0
                            micro_batch_metrics.update(pg_metrics)
                        else:
                            pg_loss, pg_metrics = compute_self_distillation_loss(
                                student_log_probs=log_prob,
                                teacher_log_probs=teacher_log_prob,
                                response_mask=response_mask,
                                self_distillation_config=self_distillation_cfg,
                                old_log_probs=old_log_prob,
                                student_all_log_probs=student_all_logps,
                                teacher_all_log_probs=teacher_all_logps,
                                student_topk_log_probs=student_topk_logps,
                                teacher_topk_log_probs=teacher_topk_logps,
                                self_distillation_mask=self_distillation_mask,
                                loss_agg_mode=loss_agg_mode,
                                rollout_is_weights=rollout_is_weights,
                                contrastive_teacher_log_probs=ccir_contrastive_teacher_log_probs,
                                contrastive_teacher_topk_log_probs=ccir_contrastive_teacher_topk_log_probs,
                                ccir_config=ccir_cfg,
                            )

                            pg_metrics["self_distillation/empty_target_batch"] = self_distillation_mask.sum().item() == 0
                            micro_batch_metrics.update(pg_metrics)
                    else:
                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        policy_loss_fn = get_policy_loss_fn(loss_mode)

                        # Compute policy loss (any function is expected to return 2 values)
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                if torch.isfinite(grad_norm).item():
                    did_update = True
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                if mini_batch_prm_seq_means:
                    mb_prm_seq_mean = torch.cat(mini_batch_prm_seq_means, dim=0)
                    mini_batch_metrics["grpo_ca/base_kl_seq_mean_std"] = (
                        mb_prm_seq_mean.std().item() if mb_prm_seq_mean.numel() > 1 else 0.0
                    )
                    prm_c = mb_prm_seq_mean - mb_prm_seq_mean.mean()
                    mini_batch_metrics["grpo_ca/prm_c_std"] = prm_c.std().item() if prm_c.numel() > 1 else 0.0
                    if mini_batch_orm_seq_means:
                        mb_orm_seq_mean = torch.cat(mini_batch_orm_seq_means, dim=0)
                        orm_c = mb_orm_seq_mean - mb_orm_seq_mean.mean()
                        mini_batch_metrics["grpo_ca/orm_c_std"] = orm_c.std().item() if orm_c.numel() > 1 else 0.0
                # Token-level global PRM-ORM correlation (all-reduce across DP ranks).
                # Uses all valid tokens from all micro-batches × all DP ranks = full 256-seq × ~10K tokens.
                # This n >> 16-seq × 16-rank mini-batch correlation → σ drops from ~0.25 to ~0.005.
                if mini_batch_corr_stats is not None:   # grpo_ca only (was emitted as 0.0 for every loss mode)
                    mini_batch_metrics["grpo_ca/prm_orm_correlation"] = 0.0
                    _corr_stats = mini_batch_corr_stats.detach().clone()
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(_corr_stats, op=torch.distributed.ReduceOp.SUM)
                    _n_g, _sx_g, _sy_g, _sxy_g, _sx2_g, _sy2_g = _corr_stats.tolist()
                    if _n_g > 1:
                        _mx = _sx_g / _n_g
                        _my = _sy_g / _n_g
                        _var_x = max((_sx2_g / _n_g) - _mx * _mx, 0.0)
                        _var_y = max((_sy2_g / _n_g) - _my * _my, 0.0)
                        _cov_xy = (_sxy_g / _n_g) - _mx * _my
                        if _var_x > 1e-12 and _var_y > 1e-12:
                            mini_batch_metrics["grpo_ca/prm_orm_correlation"] = (
                                _cov_xy / (_var_x ** 0.5 * _var_y ** 0.5)
                            )
                    mini_batch_metrics["grpo_ca/prm_orm_corr_n_tokens"] = float(_n_g)
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        if did_update:
            self._update_teacher()
            self._global_training_steps += 1
        return metrics
