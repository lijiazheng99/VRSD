# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig

from .engine import FSDPEngineConfig, McoreEngineConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "SelfDistillationConfig",
    "CCIRConfig",
    "PolicyLossConfig",
    "RouterReplayConfig",
    "ActorConfig",
    "FSDPActorConfig",
    "McoreActorConfig",
]


@dataclass
class SelfDistillationConfig(BaseConfig):
    """Configuration for self-distillation loss.

    Args:
        Distillation is enabled when policy_loss.loss_mode == "sdpo" or "grpo_ccir".
        full_logit_distillation (bool): Whether to use full-logit KL distillation.
        alpha (float): KL interpolation coefficient. 0.0=forward KL, 1.0=reverse KL, in-between=JSD.
        success_reward_threshold (float): Minimum sequence reward to be considered successful.
        teacher_regularization (str): Teacher regularization mode. Options: "ema", "trust-region".
        teacher_update_rate (float): EMA update rate for teacher weights, or trust-region mixing coefficient.
        distillation_topk (Optional[int]): If set, use top-k logits for distillation.
        distillation_add_tail (bool): Whether to add a tail bucket for top-k distillation.
        max_reprompt_len (int): Maximum length of the reprompted prompt.
        reprompt_truncation (str): Truncation method for the reprompted prompt (recommended to use "right" or "error").
        dont_reprompt_on_self_success (bool): Whether to not reprompt on self-success.
        remove_thinking_from_demonstration (bool): Whether to remove <think>...</think> tags from successful demonstrations before reprompting.
        max_solution_tokens (Optional[int]): If set, truncate solution text to at most this many tokens, keeping the tail (to preserve final answer). None disables truncation.
        is_clip (Optional[float]): Clip value for distillation IS ratio; None disables IS weighting.
        reprompt_template (str): Template for reprompting. Uses {prompt}, {solution}, {feedback} placeholders.
        solution_template (str): Template for formatting solution section. Uses {successful_previous_attempt} placeholder.
        feedback_template (str): Template for formatting feedback section. Uses {feedback_raw} placeholder.
        include_environment_feedback (bool): Whether to include environment feedback in reprompting for wrong attempts.
        environment_feedback_only_without_solution (bool): If True, only use feedback when no solution is available (ignore feedback when solution exists).
        reprompt_template_feedback (str): Template for reprompting with feedback but no solution.
        reprompt_template_feedback_solution (str): Template for reprompting with both feedback and solution.
        teacher_system_prompt (Optional[str]): If set, inject this text as a system prompt for the
            TEACHER only (the student still sees the bare prompt). Used to study whether a
            reflection / workflow disposition placed in the teacher's system position transfers via
            self-distillation. Ignored if teacher_system_prompt_file is set.
        teacher_system_prompt_file (Optional[str]): Path to a text file whose contents are used as the
            teacher system prompt. Takes precedence over teacher_system_prompt. Keeps long/multiline
            prompts out of the CLI and version-controlled.
        privileged_source (str): Which privileged block the teacher conditions on. "sibling"
            (default) = the self-generated correct rollout (current SDPO). "cot" / "summary" /
            "rubric" = substitute an offline ground-truth-derived abstraction (from
            an offline preprocessing script (not included)) for the sibling, looked up by question, only on
            samples that already had a successful sibling (the self_distillation_mask is unchanged).
            For the abstraction A/B; env PRIVILEGED_SOURCE overrides this.
        privileged_context_dir (Optional[str]): Directory of <dataset>.json privileged-context files.
            Defaults to $SDPO_ROOT/datasets/privileged_context. Only used when privileged_source != "sibling".
    """

    full_logit_distillation: bool = True
    alpha: float = 0.0
    success_reward_threshold: float = 1.0
    teacher_regularization: str = "ema"
    teacher_update_rate: float = 0.05
    distillation_topk: Optional[int] = None
    distillation_add_tail: bool = True
    max_reprompt_len: int = 10240
    reprompt_truncation: str = "right"
    dont_reprompt_on_self_success: bool = False
    remove_thinking_from_demonstration: bool = False
    max_solution_tokens: Optional[int] = None
    is_clip: Optional[float] = None
    reprompt_template: str = (
        "{prompt}{solution}{feedback}\n\n"
        "Correctly solve the original question.\n"
    )
    solution_template: str = (
        "\n"
        "Correct solution:\n\n"
        "{successful_previous_attempt}\n\n"
    )
    feedback_template: str = (
        "\n"
        "The following is feedback from your unsuccessful earlier attempt:\n\n"
        "{feedback_raw}\n\n"
    )
    wrong_solution_template: str = (
        "\n"
        "A previous incorrect attempt:\n\n"
        "{successful_previous_attempt}\n\n"
    )
    # None = SDPO's base behaviour (reprompt_template with an empty solution section); the AntiSD arm sets
    # the authors' dedicated template via config/sd_templates/antisd.yaml.
    reprompt_template_feedback_only: Optional[str] = None
    include_environment_feedback: bool = False
    environment_feedback_only_without_solution: bool = False
    teacher_system_prompt: Optional[str] = None
    teacher_system_prompt_file: Optional[str] = None
    privileged_source: str = "sibling"
    privileged_context_dir: Optional[str] = None
    # --- AntiSD-port switches (default = SDPO's base behaviour; antisd_recipe.sh turns them on) ---
    truncate_solution_to_budget: bool = False      # AntiSD: head/tail-compress over-budget solutions before templating
    reprompt_head_tail_truncation: bool = False    # AntiSD: no tokenizer truncation, head+tail to max_reprompt_len after
    require_solution_for_distillation: bool = False
    remove_answer_from_solution: bool = False
    solution_selection: str = "random"
    truncate_solution_at_correct_answer: bool = False
    provide_ground_truth_in_feedback: bool = False
    solution_source: str = "group_only"   # ours = SDPO base (sibling only); the AntiSD recipe passes group_only explicitly too
    solution_content: str = "full"  # "full" | "feedback_only"
    # Solution mode: transforms applied to solution text before teacher prompt building.
    # "normal": use actual solutions (default)
    # "cross_problem": shuffle solutions across batch (different problem's solution)
    # "none": no solution, teacher sees prompt + suffix only
    # "shuffle_sentences": shuffle sentences within solution (break structural coherence)
    # "answer_only": keep only \boxed{answer}, strip all reasoning
    # "fixed_detailed": hardcoded detailed multi-step reasoning for ALL samples
    # "fixed_generic": hardcoded brief reasoning scaffold for ALL samples
    # "fixed_unrelated": hardcoded non-reasoning text for ALL samples
    solution_mode: str = "normal"
    teacher_prompt_suffix: str = ""  # appended to bare teacher prompt when no solution/feedback
    # Reprompt style: controls how solution context is injected into the teacher prompt.
    # "suffix" (default): solution appended after prompt in the same user message.
    # "system_prefix": solution injected into system message, user message = bare prompt.
    # "multi_turn": solution as prior assistant turn, feedback as follow-up user turn.
    #   [user: problem] [assistant: sibling_solution] [user: feedback. Try again.]
    reprompt_style: str = "suffix"
    # Template for system_prefix mode. Uses {solution} and {feedback} placeholders.
    reprompt_system_prefix_template: str = (
        "Here is a previous attempt at the problem that follows:"
        "{solution}{feedback}"
    )

    def __post_init__(self):
        valid_privileged_source = ["sibling", "cot", "summary", "rubric"]
        if self.privileged_source not in valid_privileged_source:
            raise ValueError(
                "self_distillation.privileged_source must be one of "
                f"{valid_privileged_source}, got {self.privileged_source}"
            )
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"self_distillation.alpha must be in [0,1], got {self.alpha}")
        valid_teacher_regularization = ["ema", "trust-region"]
        if self.teacher_regularization not in valid_teacher_regularization:
            raise ValueError(
                "self_distillation.teacher_regularization must be one of "
                f"{valid_teacher_regularization}, got {self.teacher_regularization}"
            )
        if not 0.0 <= self.teacher_update_rate <= 1.0:
            raise ValueError(
                f"self_distillation.teacher_update_rate must be in [0,1], got {self.teacher_update_rate}"
            )
        if self.distillation_topk is not None and self.distillation_topk <= 0:
            raise ValueError(
                f"self_distillation.distillation_topk must be a positive integer, got {self.distillation_topk}"
            )
        if self.is_clip is not None and self.is_clip <= 0:
            raise ValueError(f"self_distillation.is_clip must be positive, got {self.is_clip}")
        valid_solution_selection = ["random", "prefer_short"]
        if self.solution_selection not in valid_solution_selection:
            raise ValueError(
                f"self_distillation.solution_selection must be one of "
                f"{valid_solution_selection}, got {self.solution_selection}"
            )
        valid_solution_sources = ["group_first", "external_first", "external_only", "group_only"]
        if self.solution_source not in valid_solution_sources:
            raise ValueError(
                f"self_distillation.solution_source must be one of "
                f"{valid_solution_sources}, got {self.solution_source}"
            )
        valid_solution_content = ["full", "feedback_only"]
        if self.solution_content not in valid_solution_content:
            raise ValueError(
                f"self_distillation.solution_content must be one of "
                f"{valid_solution_content}, got {self.solution_content}"
            )


@dataclass
class CCIRConfig(BaseConfig):
    """Configuration for CCIR (Contrastive Causal Information Reward).

    When enabled, provides contrastive causal information reward (CCIR).
    In "sdpo" mode with full_logit_distillation + distillation_topk, computes distributional
    S_t(v) = log π_t(v|x,z) - log π_t(v|x',z) over topk vocab items and weights the
    per-vocab KL loss by (1 + prm_weight * S_t(v)).
    In "sdpo" mode without topk, falls back to per-token scalar S_t additive PG term.
    In "grpo_ccir" mode, combines GRPO policy gradient with CCIR-weighted topk KL distillation:
        L = L_GRPO(A_seq) + kl_coeff · L_CCIR_KL(S_t_topk)
    where L_CCIR_KL is computed by compute_self_distillation_loss with distributional CCIR weighting.
    In "grpo_st" mode, uses S_t to refine GRPO per-token advantages (no vocab-level KL):
        A_refined(t) = A_seq * (1 + st_advantage_alpha * S_t_norm(t))   when A_seq != 0
        A_refined(t) = st_fallback_beta * S_t_norm(t)                    when A_seq == 0
    In "grpo_ca" mode (credit assignment), fuses SDPO's full PRM signal with GRPO advantage:
        PRM_t = (s - t) - prm_weight * (t - c)    [full SDPO advantage]
        ca_mode="additive":       A_refined(t) = orm_weight * A_seq + ca_lambda * normalize(PRM_t)
        ca_mode="multiplicative": A_refined(t) = A_seq * (1 + ca_lambda * normalize(PRM_t))
        ca_mode="rlsd":           A_refined(t) = A_seq * ((1-λ) + λ * clip(exp(sign(A)*(t-s)), 1-ε, 1+ε))
    Multiplicative mode constrains PRM to only modulate ORM's direction (prevents length explosion).
    Additive mode allows PRM to create independent optimization direction.
    RLSD mode (Yang 2026) is magnitude-only: direction is always sign(A_seq) from environment reward;
    the evidence ratio (P_T/P_S)^{sign(A)} only modulates magnitude. Used as Prop 3 ablation.

    Args:
        enabled: Whether to enable CCIR contrastive weighting.
        num_contrastive: Number of prompt shuffles (K). K=1 is most efficient.
        temperature: DEPRECATED. Was used for sigmoid weighting in earlier CCIR versions. Kept for config compatibility.
        min_weight: DEPRECATED. Was used as minimum CCIR weight floor. Kept for config compatibility.
        prm_weight: Weight λ for CCIR S_t within self-distillation.
            In sdpo distributional mode: KL weight = 1 + λ·S_t(v). λ=0 gives uniform (unweighted) KL.
            In grpo_ca mode: PRM_t = (s - t) - λ·(t - c). λ=0 gives pure KL gap (no CCIR).
        kl_coeff: Coefficient γ controlling KL regularization strength relative to GRPO.
            L = L_GRPO + γ·L_CCIR_KL. γ=0 recovers pure GRPO.
        st_advantage_alpha: Multiplicative S_t modulation coefficient for grpo_st mode.
            Controls how much S_t redistributes credit within a sequence when outcome signal exists.
        st_fallback_beta: Additive S_t coefficient for grpo_st mode when all rollouts fail (A_seq=0).
            Provides dense gradient signal from S_t when GRPO advantage is zero.
        ca_lambda: PRM → ORM injection strength for grpo_ca mode.
            A_t = orm_weight * A_seq + ca_lambda * normalize(PRM_t). Controls how much the full
            token-level PRM signal modulates the sequence-level GRPO advantage.
        orm_weight: Weight on GRPO's sequence-level advantage in grpo_ca mode.
            A_t = orm_weight * A_seq + ca_lambda * normalize(PRM_t).
            orm_weight=1 (default): standard GRPO + PRM.
            orm_weight=0: pure PRM signal, no outcome reward.
        prm_normalize: Whether to normalize PRM_t to zero-mean unit-variance (grpo_ca mode).
            True: PRM_norm = (PRM_t - mean) / std. Signal magnitude is constant regardless of
            student-teacher gap. Risks amplifying noise as model converges.
            False: use raw PRM_t. Signal naturally decays as student approaches teacher.
            NOTE: This is the legacy per-batch normalization. For per-sequence normalization,
            use prm_normalize_mode="sequence" instead (which supersedes this flag).
        prm_normalize_mode: PRM normalization granularity (grpo_ca mode).
            "batch": Per-batch zero-mean unit-variance (same as prm_normalize=True). All sequences
                share the same mean/std. Length-dependent PRM bias is NOT removed.
            "sequence": Per-sequence zero-mean unit-variance. Each sequence's PRM is independently
                standardized. Removes both first-moment (mean) and second-moment (variance)
                length dependence. PRM becomes a pure credit assignment weight: mean=0, var=1
                per sequence regardless of response length or hint quality.
                Supersedes prm_normalize and prm_seq_demean when set.
            "none": No normalization. Raw PRM_t = (s-t) - w*(t-c).
        prm_clip: Truncation magnitude for PRM signal (grpo_ca mode).
            When set, clamps PRM (after optional normalization) to [-prm_clip, prm_clip].
            Prevents extreme PRM values from dominating. null disables clipping.
        prm_anchor_to_orm: Whether to rescale PRM signal to match ORM signal magnitude (grpo_ca mode).
            When True, PRM is multiplied by |orm_term|_mean / |prm_term|_mean so that
            ca_lambda controls the true ORM:PRM ratio regardless of advantage collapse.
            Prevents PRM from dominating when A_seq shrinks during training.
        prm_seq_demean: Whether to remove per-sequence mean from PRM signal (grpo_ca mode).
            When True, PRM_t is replaced by PRM_t - mean_t(PRM_t) per sequence, so PRM only
            does within-sequence credit redistribution (which tokens matter more/less) without
            adding a net positive/negative bias per sequence. ORM controls the overall direction
            (reinforce/suppress), PRM controls intra-sequence token credit assignment.
            NOTE: prm_normalize_mode="sequence" supersedes this (it does demean + variance norm).
        ca_mode: Advantage composition mode for grpo_ca.
            "additive": A_t = orm_weight * A_seq + ca_lambda * PRM_processed.
                PRM creates an independent optimization direction. Prone to length explosion.
            "multiplicative": A_t = A_seq * (1 + ca_lambda * PRM_processed).
                PRM only modulates ORM's direction; cannot create independent signal.
                When A_seq=0 (uniform group), A_t=0 — PRM cannot override ORM.
                Prevents length explosion by construction.
        prm_entropy_neutral: Entropy-neutral PRM decorrelation mode (grpo_ca mode).
            PRM_t = s_t - t_t is correlated with student confidence s_t = log π(y_t).
            This correlation causes PRM to act as "policy momentum" — systematically
            reinforcing confident tokens — leading to entropy collapse.

            Decorrelation removes the component of PRM linearly correlated with s_t:
                PRM_EN = PRM - β·(s - mean(s)),  β = Cov(PRM, s) / Var(s)
            By construction Cov(PRM_EN, s) = 0, so the first-order entropy effect vanishes.

            Options:
            "none": No decorrelation. Original PRM_t = s_t - t_t.
            "decorrelate": Full decorrelation. Remove the s_t-correlated component from PRM.
                β = 1 - Cov(s,t)/Var(s) is computed per sequence and adapts automatically:
                β ≈ 0 early (teacher ≈ student) → full PRM speed;
                β grows as teacher lags → momentum suppressed.
        ca_beta: DEPRECATED. CCIR debiasing is now hardcoded as β=1 in grpo_ca.
            The inner CCIR weight is controlled by prm_weight instead.
    """
    enabled: bool = False
    num_contrastive: int = 1
    temperature: float = 1.0
    min_weight: float = 0.1
    prm_weight: float = 0.1
    kl_coeff: float = 0.1
    st_advantage_alpha: float = 0.3
    st_fallback_beta: float = 0.05
    ca_lambda: float = 0.1
    ca_lambda_mode: str = "fixed"        # "fixed", "adaptive", "length_aware", "teacher_perp", "student_perp", "ratio_perp"
    # Scope for teacher_perp λ controller:
    #   "per_seq": each sequence gets its own λ from its own t_ppl (default, adaptive)
    #   "batch_mean": all sequences share λ computed from batch-mean t_ppl (uniform, simpler)
    ca_lambda_tppl_scope: str = "per_seq"
    ca_lambda_target: float = 0.001      # target for |base_kl|_mean * lambda (adaptive mode)
    ca_lambda_min: float = -0.01          # lower clamp for lambda (limits t-s strength)
    ca_lambda_max: float = 0.5           # upper clamp for lambda
    # Length-aware λ: λ = base * (1 - length_alpha * max(0, mean_len/length_target - 1))
    ca_lambda_length_target: float = 10000.0  # target response length
    ca_lambda_length_alpha: float = 2.0       # how fast λ decreases with length overshoot
    # Teacher-perplexity-aware λ: λ = base * (1 - perp_alpha * max(0, teacher_perp/perp_target - 1))
    # Directly measures whether teacher can still understand the responses.
    # Unifies LenFlip (length proxy) and adaptive (bkl proxy) into a teacher-centric framework.
    ca_lambda_perp_target: float = 0.0          # zero-crossing perp (0=auto: calibrate from warmup)
    # Auto-calibration of target (when ca_lambda_perp_target <= 0):
    #   If ca_lambda_perp_target_ratio > 0: target = warmup_median × ratio (cross-model consistent)
    #   Else: target = warmup_median − ca_lambda_perp_delta (absolute offset)
    ca_lambda_perp_target_ratio: float = 0.0   # relative target (0=disabled, else ratio × warmup_median)
    ca_lambda_perp_delta: float = 0.10         # absolute offset below warmup median (fallback auto mode)
    # Hysteresis (Schmitt) reactivation threshold for batch_mean tppl_scope:
    #   deactivate (λ=λ_min) when batch_perp < ca_lambda_perp_target
    #   reactivate (normal λ formula) only when batch_perp > reactivate_threshold
    # Semantics (priority): explicit reactivate_target > reactivate_ratio × warmup_median > disabled
    # warmup_median is recovered internally as perp_target / perp_target_ratio when ratio-mode.
    ca_lambda_perp_reactivate_target: float = 0.0  # absolute reactivate perp (0=disabled / ratio-fallback)
    ca_lambda_perp_reactivate_ratio: float = 0.0   # relative: reactivate at ratio × warmup_median (0=disabled)
    ca_lambda_perp_alpha: float = 2.0         # sensitivity: λ = base * α * log(perp/target)
    ca_lambda_perp_mask: float = 3.0          # per-seq mask: PRM=0 when teacher_perp > this threshold
    ca_lambda_warmup_steps: int = 5           # warmup steps: collect t_ppl, λ=0 (ORM only)
    ca_lambda_mean_shift: bool = False        # batch-mean shift: if mean(λ)<0, shift all λ_i up so mean=0
    # Always meanshift (stronger than ca_lambda_mean_shift): force mean(λ)=0 every step, regardless of sign.
    # Preserves per-seq ranking; net batch-level push = 0. Purely difficulty-differential PRM.
    ca_lambda_mean_shift_always: bool = False
    # Step-based λ cutoff: set λ=0 after this many training steps (-1 = disabled).
    # Enables two-phase training: s-t PRM acceleration first, then pure GRPO.
    ca_lambda_step_cutoff: int = -1
    # s_specific gate: modulate λ by problem-specific signal strength.
    # When s_specific declines, PRM is becoming generic → reduce λ.
    ca_lambda_s_specific_gate: str = "none"  # "none" or "proportional"
    ca_lambda_s_specific_threshold: float = 0.0  # 0 = auto-calibrate from warmup
    ca_lambda_s_specific_floor: float = 0.0  # minimum gate multiplier
    # PRM-strength-adaptive λ decay. Multiplies final λ by clamp(signal/baseline, 0, 1).
    # As PRM magnitude declines (training converges, u → 0), λ shrinks proportionally.
    # Natural "soft exit": λ → 0 as signal vanishes, without hard step cutoff.
    # Combined with Lagrangian controllers (t_ppl etc.), this modulates magnitude only.
    #   "none": disabled (no decay).
    #   "bkl_abs": signal = |base_kl|.mean() = mean |u|. Recommended. Works for any PRM shape.
    #   "prm_abs": signal = |PRM_t|.mean() post-construction. Shape-specific (PRM's actual scale).
    ca_lambda_decay_mode: str = "none"
    # Baseline calibration: average signal over first N steps. Reuses ca_lambda_warmup_steps
    # when > 0, else falls back to this value. After warmup, baseline is frozen and
    # decay_factor = clamp(signal_current / signal_baseline, decay_floor, 1.0).
    ca_lambda_decay_floor: float = 0.0       # minimum decay factor (0 = full exit allowed)
    ca_lambda_decay_ema: float = 0.9         # EMA smoothing on signal (0 = no smoothing, 1 = frozen)
    # CCIR cross-problem: evaluate responses under different problem contexts
    # to separate x-specific vs x-generic PRM signal.
    ccir_cross_problem: bool = False          # enable cross-problem forward passes
    ccir_cross_problem_mode: str = "full"     # "full": [s(x)-s(x')]-β[t(x,y')-t(x',y')]; "blend": s(x)-0.5[s(x')+t(x',y')]; "blend_current_teacher": s(x)-[(1-a)s(x')+a t(x)]
    ccir_cross_problem_beta: float = 1.0      # weight on teacher x-specific component (full mode)
    ccir_cross_problem_alpha: float = 0.5     # blend weight on current-problem teacher t(x) for blend_current_teacher
    # KL constraint against reference policy (π_ref = frozen initial model via t_nosol).
    # Subtracts β_kl * Δlog_p = β_kl * (s - t_nosol) from advantage.
    # When β_kl = ca_lambda: Δlog_p cancels, leaving pure content credit (ORM - λ·PMI).
    kl_ref_beta: float = 0.0               # 0 = disabled. >0 = KL penalty strength
    # FutureConf gate: weight PRM by teacher's confidence in future trajectory.
    # gate_t = exp(FutureConf_t / remaining_len), where
    # FutureConf_t = Σ_{k=t}^{T} γ^{k-t} · log π_teacher(y_k | x, z, y<k)
    # High gate → teacher endorses future trajectory → PRM trustworthy.
    # Low gate → teacher lost track → PRM is noise → fall back to ORM.
    future_conf_gamma: float = 0.0          # 0 = disabled. >0 = discount factor (e.g. 0.99)
    # PRM position mask: only apply PRM to the first N tokens of response, mask the rest to 0.
    # Teacher loses informativeness on long responses; this directly cuts PRM where it becomes noise.
    prm_max_position: int = 0               # 0 = disabled. >0 = max tokens (e.g. 4096, 8192)
    # PRM per-sequence length mask: zero PRM for any sequence whose total length ≥ threshold.
    # When responses are long enough to risk truncation, PRM signal is unreliable — let ORM handle.
    # Short sequences keep PRM (accelerates learning). Creates self-correcting length equilibrium.
    # 0 = disabled. Recommended: 0.75 * max_response_length (e.g. 12000 for 16K rollout).
    prm_length_mask_threshold: int = 0
    # Entropy gate: turn off PRM when batch-mean entropy drops relative to warmup baseline.
    # Hysteresis prevents flapping. Based on data: crashes happen at H/H_init < 45%,
    # stable runs stay > 60%. Thresholds chosen to catch QN-fwdAlpha-style crashes
    # (H@s40=44% → crash@s50) while not misfiring on healthy declines (ON @ 53%).
    # "none": disabled. "binary": H < low_ratio → PRM=0, H >= low_ratio → PRM on.
    # "hysteresis": close at low_ratio, reopen at high_ratio (recommended).
    entropy_gate_mode: str = "none"
    entropy_gate_h_ratio_low: float = 0.45     # close threshold (H/H_warmup < this → PRM=0)
    entropy_gate_h_ratio_high: float = 0.60    # reopen threshold (H/H_warmup > this → PRM on)
    # Contrastive length brake: PRM += brake_beta * (t_correct - t_wrong)
    contrastive_brake_beta: float = 0.0       # 0 = disabled. >0 = fixed brake strength
    contrastive_brake_adaptive: bool = False   # if True, beta scales with length overshoot
    contrastive_brake_length_target: float = 10000.0
    # Position debiasing: remove position-dependent PRM bias
    position_debias_mode: str = "none"         # "none", "quartile", "linear"
    # Length penalty on advantage
    length_penalty_alpha: float = 0.0          # 0 = disabled. >0 = penalty strength
    length_penalty_target: float = 10000.0     # penalty kicks in above this
    orm_weight: float = 1.0
    prm_normalize: bool = True
    prm_normalize_mode: str = "batch"
    prm_clip: Optional[float] = None
    prm_anchor_to_orm: bool = False
    prm_seq_demean: bool = False
    ca_mode: str = "additive"
    # RLSD (Yang 2026 Self-Distilled RLVR) ablation params (only active when ca_mode="rlsd")
    # A_t = A_seq · ((1 − rlsd_lambda) + rlsd_lambda · clip(exp(sign(A_seq) · (t_t − s_t)), 1−ε, 1+ε))
    rlsd_eps_w: float = 0.2
    rlsd_lambda: float = 1.0
    prm_entropy_neutral: str = "none"
    ca_beta: float = 1.0
    prm_tanh_tau: Optional[float] = None
    # Token-level forward-style raw PRM reweighting.
    # Let Δ_t = s_t - t_t from the current student/teacher token log-probs.
    # "token_is": w_t = exp(clamp(-beta * Δ_t, -log_clip, log_clip)) = exp(clamp(beta * (t_t - s_t), ...))
    # Then PRM_t <- w_t * PRM_t before tanh / maxent / normalization.
    # Exact local forward-KL density when PRM_t = Δ_t (pure raw s-t) and no extra SI/CCIR terms are added.
    # "token_is": PRM_t <- w_t * PRM_t where w_t = exp(clamp(beta*(t-s), ...)). Biased IS reweight.
    # "renyi_unbiased": PRM_t <- exp(clamp(alpha_r*(t-s), ...)) - 1. Unbiased PG advantage for
    #   Rényi-α f-divergence minimization. alpha_r=1.0 is forward KL (A=exp(u)-1 via centering),
    #   alpha_r=0.5 is Hellinger (A=sqrt(pi_t/pi_s)-1). Signed via -1 subtraction, E[PRM]=0 for α=1.
    # "jsd_unbiased": PRM_t <- 0.5·sign·(softplus(clamp(u_v)) - log 2). Unbiased PG advantage for
    #   Jensen-Shannon divergence maximization. Principled coef = 0.5·(log 2 - softplus(u)):
    #   bounded by log 2 / 2 ≈ 0.347 on u<<0 (q1) side (no exp blow-up); linear on u>>0 (q4) side.
    #   Natural stop at u_v = 0. sign=-1 anti-distill (default). u_v respects virtual_alpha.
    prm_forward_mode: str = "none"
    prm_forward_beta: float = 1.0
    prm_forward_log_clip: float = 5.0
    prm_renyi_alpha: float = 1.0
    # Sign applied to renyi_unbiased PRM = prm_renyi_sign * (exp(alpha_r * u) - 1).
    # +1.0: A = exp(u)-1, minimizes reverse KL (K3), pushes student toward teacher (distill).
    # -1.0: A = 1-exp(u), maximizes K3 (K3-REINFORCE direction), preserves student identity
    #       (same direction as fwdRawTok / plain s-t PG advantage).
    prm_renyi_sign: float = 1.0
    # Adaptive u clip based on warmup-calibrated σ_ref.
    # "none": disabled. "adaptive": after warmup, clamp base_kl (= s-t = -u) to ±k·σ_ref.
    # σ_ref = mean(std(u) over warmup batches), averaged across DP ranks.
    prm_u_clip_mode: str = "none"
    prm_u_clip_k_sigma: float = 2.0
    # If > 0, skip warmup and use this fixed σ_ref directly (pre-computed from historical run).
    # Useful for fast restart with known calibration values.
    prm_u_clip_sigma_ref_fixed: float = 0.0
    # Virtual teacher weight in renyi_unbiased mode:
    #   log π_v = prm_renyi_virtual_alpha * t(x,z) + (1 - prm_renyi_virtual_alpha) * s(x')
    # Used u = log π_v - log π_s instead of u = t - s.
    # 1.0 (default): pure teacher (no virtual, original renyi_unbiased).
    # 0.25: 75% x'-anchor + 25% teacher-away (matches v27-blendT-a025 config).
    # Requires ccir_cross_problem=True to provide s(x') = cross_student_log_prob.
    prm_renyi_virtual_alpha: float = 1.0
    prm_alignment_gate: bool = False
    si_mode: str = "none"
    si_lambda: float = 1.0
    # PRM construction method (how t and s are compared in Step 1).
    # "raw": PRM = t(sol) - s (direct subtraction in log space). Default.
    # "indep_norm": PRM = standardize(t) - standardize(s) per sequence.
    #   Removes the log-prob ceiling effect (Cov(Δ, s) < 0 from saturation).
    # "reverse": PRM = s - t(sol). Old sign convention (v3-v10).
    # "self_reward": PRM = s. Pure student confidence, no teacher. Tests self-reward hypothesis.
    # "teacher_contrastive": PRM = t(sol) - t(nosol). No student involvement.
    #   Requires si_reference to control nosol context (e.g. "solution_contrastive" for wrong sol).
    # "s_minus_t_wrong": PRM = s - t(nosol). Student minus teacher-with-wrong-solution.
    #   Requires nosol forward (si_reference="solution_contrastive").
    # "t_wrong_minus_s": PRM = t(nosol) - s. Teacher-with-wrong-solution minus student.
    #   Requires nosol forward (si_reference="solution_contrastive").
    prm_construction: str = "raw"
    si_reference: str = "bare"
    si_wrong_label: str = "same"
    si_model: str = "ema"
    # Amplification factor for contrastive component in "reverse_combined" prm_construction.
    # PRM = (s - t) + contrastive_lambda * (t_correct - t_wrong)
    contrastive_lambda: float = 5.0
    # Gamma blending coefficient for "reverse" prm_construction.
    # PRM = s - gamma * t. Interpolates between pure self-reward (gamma=0) and full s-t (gamma=1).
    # gamma=0: PRM=s (self-reward, delayed length explosion)
    # gamma=1: PRM=s-t (fast acceleration, fast length explosion)
    # Optimal gamma balances task structure (from s) vs template bias (from t).
    prm_gamma: float = 1.0
    # MaxEnt RL: entropy-aware advantage correction.
    # "none": disabled. "fixed": constant α. "adaptive": SAC dual gradient on α.
    maxent_coeff: str = "none"
    # Only apply maxent when ca_lambda > 0 (exploration mode). Prevents entropy ratcheting
    # during LenFlip oscillation or after step cutoff.
    maxent_conditional: bool = False
    # Initial α value. For "fixed" mode, this IS α. For "adaptive", this is the starting point.
    maxent_alpha: float = 0.1
    # Target entropy H*. None = auto-initialize from first batch's mean entropy.
    maxent_h_target: Optional[float] = None
    # Learning rate for adaptive α update: α ← α - maxent_lr·(H_batch - H_target).
    maxent_lr: float = 0.01
    # Entropy gate: scale PRM by clamp(H_batch / H_target, 0, 1). Composable with A/C.
    maxent_entropy_gate: bool = False
    # Position of MaxEnt correction in the pipeline.
    # "advantage": subtract α·(s_t - s̄) from refined_advantages (after Step 5). DEFAULT.
    # "prm_raw": subtract α·(s_t - s̄) from PRM_t (before Step 4 normalization).
    maxent_position: str = "advantage"

    def __post_init__(self):
        if self.num_contrastive < 1:
            raise ValueError(f"ccir.num_contrastive must be >= 1, got {self.num_contrastive}")
        if self.temperature <= 0:
            raise ValueError(f"ccir.temperature must be > 0, got {self.temperature}")
        if not 0.0 <= self.min_weight <= 1.0:
            raise ValueError(f"ccir.min_weight must be in [0,1], got {self.min_weight}")
        valid_ca_modes = ["additive", "multiplicative", "credit", "anti_credit", "rlsd"]
        if self.ca_mode not in valid_ca_modes:
            raise ValueError(f"ccir.ca_mode must be one of {valid_ca_modes}, got {self.ca_mode}")
        valid_normalize_modes = ["batch", "sequence", "sequence_demean", "none"]
        if self.prm_normalize_mode not in valid_normalize_modes:
            raise ValueError(f"ccir.prm_normalize_mode must be one of {valid_normalize_modes}, got {self.prm_normalize_mode}")
        valid_en_modes = ["none", "decorrelate", "decorrelate_advantage"]
        if self.prm_entropy_neutral not in valid_en_modes:
            raise ValueError(f"ccir.prm_entropy_neutral must be one of {valid_en_modes}, got {self.prm_entropy_neutral}")
        valid_forward_modes = ["none", "token_is", "renyi_unbiased", "jsd_unbiased"]
        if self.prm_forward_mode not in valid_forward_modes:
            raise ValueError(f"ccir.prm_forward_mode must be one of {valid_forward_modes}, got {self.prm_forward_mode}")
        if self.prm_forward_beta < 0:
            raise ValueError(f"ccir.prm_forward_beta must be >= 0, got {self.prm_forward_beta}")
        if self.prm_forward_log_clip <= 0:
            raise ValueError(f"ccir.prm_forward_log_clip must be > 0, got {self.prm_forward_log_clip}")
        if self.prm_renyi_alpha <= 0:
            raise ValueError(f"ccir.prm_renyi_alpha must be > 0, got {self.prm_renyi_alpha}")
        if self.prm_renyi_sign not in (1.0, -1.0):
            raise ValueError(f"ccir.prm_renyi_sign must be +1.0 or -1.0, got {self.prm_renyi_sign}")
        if not 0.0 <= self.prm_renyi_virtual_alpha <= 1.0:
            raise ValueError(f"ccir.prm_renyi_virtual_alpha must be in [0, 1], got {self.prm_renyi_virtual_alpha}")
        valid_u_clip_modes = ["none", "adaptive"]
        if self.prm_u_clip_mode not in valid_u_clip_modes:
            raise ValueError(f"ccir.prm_u_clip_mode must be one of {valid_u_clip_modes}, got {self.prm_u_clip_mode}")
        if self.prm_u_clip_k_sigma <= 0:
            raise ValueError(f"ccir.prm_u_clip_k_sigma must be > 0, got {self.prm_u_clip_k_sigma}")
        valid_decay_modes = ["none", "bkl_abs", "prm_abs"]
        if self.ca_lambda_decay_mode not in valid_decay_modes:
            raise ValueError(f"ccir.ca_lambda_decay_mode must be one of {valid_decay_modes}, got {self.ca_lambda_decay_mode}")
        valid_tppl_scopes = ["per_seq", "batch_mean"]
        if self.ca_lambda_tppl_scope not in valid_tppl_scopes:
            raise ValueError(f"ccir.ca_lambda_tppl_scope must be one of {valid_tppl_scopes}, got {self.ca_lambda_tppl_scope}")
        if not 0.0 <= self.ca_lambda_decay_floor <= 1.0:
            raise ValueError(f"ccir.ca_lambda_decay_floor must be in [0,1], got {self.ca_lambda_decay_floor}")
        if not 0.0 <= self.ca_lambda_decay_ema <= 1.0:
            raise ValueError(f"ccir.ca_lambda_decay_ema must be in [0,1], got {self.ca_lambda_decay_ema}")
        valid_entropy_gate_modes = ["none", "binary", "hysteresis"]
        if self.entropy_gate_mode not in valid_entropy_gate_modes:
            raise ValueError(f"ccir.entropy_gate_mode must be one of {valid_entropy_gate_modes}, got {self.entropy_gate_mode}")
        if not 0.0 <= self.entropy_gate_h_ratio_low <= 1.0:
            raise ValueError(f"ccir.entropy_gate_h_ratio_low must be in [0,1], got {self.entropy_gate_h_ratio_low}")
        if not 0.0 <= self.entropy_gate_h_ratio_high <= 1.0:
            raise ValueError(f"ccir.entropy_gate_h_ratio_high must be in [0,1], got {self.entropy_gate_h_ratio_high}")
        if self.entropy_gate_h_ratio_low > self.entropy_gate_h_ratio_high:
            raise ValueError(f"entropy_gate_h_ratio_low ({self.entropy_gate_h_ratio_low}) must be <= high ({self.entropy_gate_h_ratio_high})")
        valid_si_modes = ["none", "centered", "raw", "replace", "replace_indep_norm", "teacher_only"]
        if self.si_mode not in valid_si_modes:
            raise ValueError(f"ccir.si_mode must be one of {valid_si_modes}, got {self.si_mode}")
        valid_si_references = ["bare", "wrong_sibling", "feedback_contrastive", "solution_contrastive"]
        if self.si_reference not in valid_si_references:
            raise ValueError(f"ccir.si_reference must be one of {valid_si_references}, got {self.si_reference}")
        valid_si_wrong_labels = ["same", "honest"]
        if self.si_wrong_label not in valid_si_wrong_labels:
            raise ValueError(f"ccir.si_wrong_label must be one of {valid_si_wrong_labels}, got {self.si_wrong_label}")
        valid_si_models = ["ema", "student"]
        if self.si_model not in valid_si_models:
            raise ValueError(f"ccir.si_model must be one of {valid_si_models}, got {self.si_model}")
        valid_prm_constructions = [
            "raw", "indep_norm", "reverse", "self_reward",
            "teacher_contrastive", "teacher_contrastive_reversed",
            "s_minus_t_wrong", "t_wrong_minus_s",
            "s_minus_s_shuffled", "s_minus_s_other", "random_credit",
            "reverse_combined", "ccir_cross_problem",
        ]
        if self.prm_construction not in valid_prm_constructions:
            raise ValueError(f"ccir.prm_construction must be one of {valid_prm_constructions}, got {self.prm_construction}")
        valid_maxent_coeffs = ["none", "fixed", "adaptive"]
        if self.maxent_coeff not in valid_maxent_coeffs:
            raise ValueError(f"ccir.maxent_coeff must be one of {valid_maxent_coeffs}, got {self.maxent_coeff}")
        if self.maxent_alpha < 0:
            raise ValueError(f"ccir.maxent_alpha must be >= 0, got {self.maxent_alpha}")
        if self.maxent_lr <= 0:
            raise ValueError(f"ccir.maxent_lr must be > 0, got {self.maxent_lr}")
        valid_maxent_positions = ["advantage", "prm_raw"]
        if self.maxent_position not in valid_maxent_positions:
            raise ValueError(f"ccir.maxent_position must be one of {valid_maxent_positions}, got {self.maxent_position}")
        valid_ca_lambda_modes = ["fixed", "adaptive", "length_aware", "teacher_perp", "student_perp", "ratio_perp", "prm_strength"]
        if self.ca_lambda_mode not in valid_ca_lambda_modes:
            raise ValueError(f"ccir.ca_lambda_mode must be one of {valid_ca_lambda_modes}, got {self.ca_lambda_mode}")
        if self.ca_lambda_step_cutoff != -1 and self.ca_lambda_step_cutoff <= 0:
            raise ValueError(f"ccir.ca_lambda_step_cutoff must be -1 (disabled) or positive, got {self.ca_lambda_step_cutoff}")
        valid_ccir_cp_modes = ["full", "blend", "blend_current_teacher"]
        if self.ccir_cross_problem_mode not in valid_ccir_cp_modes:
            raise ValueError(f"ccir.ccir_cross_problem_mode must be one of {valid_ccir_cp_modes}, got {self.ccir_cross_problem_mode}")
        if not 0.0 <= self.ccir_cross_problem_alpha <= 1.0:
            raise ValueError(f"ccir.ccir_cross_problem_alpha must be in [0,1], got {self.ccir_cross_problem_alpha}")


@dataclass
class RouterReplayConfig(BaseConfig):
    """Configuration for router replay in MoE models.

    This configuration controls the routing behavior for Mixture of Experts (MoE) models,
    allowing for deterministic training through route recording and replay.

    Args:
        mode (str): Router replay mode. Options: 'disabled', 'R2', 'R3'.
            - 'disabled': No router replay functionality
            - 'R2': Use Router Replay routing strategy
            - 'R3': Use Rollout Router Replay routing strategy
        record_file (Optional[str]): File path to save recorded routing decisions.
            Required when mode is 'record', 'R2', or 'R3'.
        replay_file (Optional[str]): File path to load recorded routing decisions for replay.
            Required when mode is 'replay'.
    """

    mode: str = "disabled"
    record_file: Optional[str] = None
    replay_file: Optional[str] = None

    def __post_init__(self):
        """Validate router replay configuration."""
        valid_modes = ["disabled", "R2", "R3"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg', 'sdpo'.
        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.
        clip_cov_lb (float): Lower bound for clip-cov loss.
        clip_cov_ub (float): Upper bound for clip-cov loss.
        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.
        ppo_kl_coef (float): KL divergence penalty coefficient.
    """

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1


@dataclass
class ActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size (Optional[int]): Micro-batch size for PPO training.
            If None, uses ppo_micro_batch_size_per_gpu.
        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for PPO training.
        use_dynamic_bsz (bool): Whether to use dynamic batch sizing.
        ppo_max_token_len_per_gpu (int): Maximum token length per GPU for PPO training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        clip_ratio_low (float): Lower bound for PPO clipping ratio.
        clip_ratio_high (float): Upper bound for PPO clipping ratio.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        clip_ratio_c (float): Clipping ratio for critic loss.
        loss_agg_mode (str): Loss aggregation mode. Options: 'token-mean', 'sample-mean'.
        loss_scale_factor (Optional[int]): Scale factor for 'seq-mean-token-sum-norm' loss aggregation mode.
            If None, uses response_length. Set to a constant to ensure consistent normalization.
        entropy_coeff (float): Entropy coefficient for regularization.
        tau_pos (float): Positive tau for SAPO smoothing (>= 1.0 keeps rewards stable).
        tau_neg (float): Negative tau for SAPO smoothing (> tau_pos for asymmetry).
        use_kl_loss (bool): Whether to use KL divergence loss.
        use_torch_compile (bool): Whether to use torch.compile for optimization.
        kl_loss_coef (float): KL divergence loss coefficient.
        kl_loss_type (str): Type of KL loss to use.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (OptimizerConfig): Configuration for optimizer.
        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
        data_loader_seed (int): Seed for data loader. If None, uses global seed.
        router_replay (RouterReplayConfig): Configuration for router replay in MoE models.
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None  # deprecate
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: Optional[int] = None
    entropy_coeff: float = 0
    tau_pos: float = 1.0
    tau_neg: float = 1.05
    calculate_entropy: bool = False
    use_kl_loss: bool = False
    # Whether to enable PrefixGrouper-based shared-prefix forward
    use_prefix_grouper: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 1
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING  # must be override by sampling config
    model_config: HFModelConfig = field(default_factory=BaseConfig)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)
    self_distillation: SelfDistillationConfig = field(default_factory=SelfDistillationConfig)
    ccir: CCIRConfig = field(default_factory=CCIRConfig)

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate actor configuration with runtime parameters."""
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.
        load_weight (bool): Whether to load model weights from checkpoint.
        megatron (dict[str, Any]): Configuration for Megatron parallelism settings.
        profile (dict[str, Any]): Configuration for profiling settings.
    """

    strategy: str = "megatron"
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.megatron


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): [DEPRECATED] Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False
    calculate_sum_pi_squared: bool = False
    sum_pi_squared_checkpointing: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.fsdp_config

        # backward compatibility
        if self.ulysses_sequence_parallel_size > 1:
            self.fsdp_config.ulysses_sequence_parallel_size = self.ulysses_sequence_parallel_size

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)

        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and not model_config.get("use_remove_padding", False):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )
