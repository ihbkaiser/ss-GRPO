from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass

import torch

from flashgrpo_b200.decoding.acceptance import exact_accept_paths_batch, sample_from_logits
from flashgrpo_b200.decoding.hrdcr import (
    HRDCRFeedback,
    HRDCRPredictionBuffer,
    HRDCRStateManager,
    merge_auxiliary_records,
)
from flashgrpo_b200.decoding.kv_extraction import extract_accepted_path_kv
from flashgrpo_b200.decoding.medusa_tree import (
    CandidateTree,
    Head3QualityCalibrator,
    TreePlan,
    build_batch_trees,
    candidate_sets_by_head,
    dense_node_count,
    fit_topk_to_budget,
    plan_tree,
)
from flashgrpo_b200.decoding.reflex import (
    CoveragePredictionBuffer,
    LMHeadFeedback,
    PredictionBuffer,
    PromptDepthFeedbackAccumulator,
    ReflexAuxiliaryRecordBuffer,
    ReflexBatchStats,
    ReflexStateManager,
    VerificationUtilityScheduler,
)
from flashgrpo_b200.decoding.tree_attention import build_tree_attention_inputs
from flashgrpo_b200.models.qwen_flashgrpo_wrapper import (
    autocast_dtype,
    forward_tokens,
    forward_tree,
    logical_lengths as mask_logical_lengths,
    model_device,
    prefill,
    repeat_interleave_cache,
    select_cache_batch,
    unwrap_causal_lm,
)


@dataclass
class FlashMedusaConfig:
    num_medusa_heads: int = 3
    tree_mode: str = "concurrency_aware"
    tree_layout: str = "dense"
    acceptance: str = "exact_target"
    cache_update_mode: str = "extract_path"
    allow_recompute_fallback: bool = True
    cpeak_nodes: int = 512
    min_tree_nodes_per_seq: int = 1
    max_tree_nodes_per_seq: int = 10
    max_tree_depth: int = 4
    fixed_tree_topk_by_depth: tuple[int, ...] = (4, 3, 2)
    sparse_nodes_by_head: tuple[int, ...] = (5, 4, 0)
    sparse_min_head3_nodes: int = 0
    sparse_head3_min_budget: int = 8
    sparse_head3_exploration_fraction: float = 0.03
    sparse_head3_warmup_exploration_fraction: float = 0.10
    sparse_head3_warmup_records: int = 1024
    sparse_head3_min_calibration_records: int = 1024
    sparse_branch_score_temperature: float = 1.0
    sparse_diversity_penalty: float = 0.05
    sparse_head3_quality_bins: int = 10
    sparse_head3_node_cost: float = 0.35
    sparse_head3_quality_ema_beta: float = 0.95
    sparse_head3_top1_weight: float = 0.30
    sparse_head3_margin_weight: float = 0.20
    sparse_head3_entropy_weight: float = 0.15
    sparse_head3_path_weight: float = 0.10
    sparse_head3_acceptance_weight: float = 0.20
    sparse_head3_regret_weight: float = 0.15
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int | None = None
    clone_tree_cache: bool = True
    oom_shrink_factor: float = 0.5
    enable_medusa_spec_after: int = 0
    proposal_mode: str = "medusa"
    chain_enable_after: int = 0
    chain_bootstrap_from_medusa: bool = True
    adaptive_tree_enabled: bool = False
    adaptive_confidence_metric: str = "top1_prob"
    adaptive_confidence_quantile: float = 0.25
    adaptive_confidence_low: float = 0.15
    adaptive_confidence_high: float = 0.45
    adaptive_min_topk_by_depth: tuple[int, ...] = (1, 1, 1)
    adaptive_depth_weight_decay: float = 0.5
    head_inference_autocast: bool = True
    project_internal_tree_logits_only: bool = True
    inplace_kv_compaction: bool = True
    reflex_enabled: bool = False
    reflex_state_space: str = "projected"
    reflex_fast_state_dim: int = 128
    reflex_beta: float = 0.95
    reflex_eta: float = 0.1
    reflex_top_m_feedback: int = 64
    reflex_feedback_stride: int = 1
    reflex_feedback_stride_min: int = 1
    reflex_target_topk: int = 32
    reflex_feedback_union_cap: int = 128
    reflex_tv_gate_low: float = 0.05
    reflex_tv_gate_high: float = 0.20
    reflex_horizon_weight_decay: float = 0.85
    reflex_half_life_tokens: float = 48.0
    reflex_feedback_variance_beta: float = 0.99
    reflex_feedback_rms_clip: float = 3.0
    reflex_feature_feedback_weight: float = 0.0
    reflex_feature_agreement_floor: float = 0.0
    reflex_coverage_feedback_weight: float = 0.0
    reflex_feedback_objective: str = "distribution"
    reflex_feedback_temperature: float = 1.0
    reflex_distribution_weight: float = 0.25
    reflex_boundary_width: int = 2
    reflex_severity_tv_weight: float = 0.5
    reflex_severity_out_weight: float = 0.5
    reflex_severity_min: float = 0.02
    reflex_horizon_resolved: bool = False
    reflex_strict_horizon_pipeline: bool = False
    reflex_consensus_strength: float = 0.25
    reflex_consensus_floor: float = 0.0
    reflex_head_shrinkage_updates: float = 8.0
    reflex_preconditioner_mix: float = 0.25
    reflex_hint_quality_beta: float = 0.90
    reflex_hint_quality_floor: float = 0.0
    reflex_hint_quality_temperature: float = 0.10
    reflex_hint_cold_start: float = 0.25
    reflex_trust_n0: float = 4.0
    reflex_sketch_rank: int = 24
    reflex_sketch_seed: int = 29
    reflex_context_rank: int = 0
    reflex_context_mix: float = 0.5
    reflex_context_min_mass: float = 1e-3
    reflex_context_learning_rate: float = 0.5
    reflex_context_seed: int = 17
    reflex_state_rms_clip: float = 2.0
    reflex_numerical_reset_rms: float = 2.5
    reflex_relative_rms_delta_base: float = 0.01
    reflex_correction_ratio_min: float = 0.005
    reflex_correction_ratio_max: float = 0.020
    reflex_min_effective_updates: float = 4.0
    reflex_min_alignment_count: float = 4.0
    reflex_min_state_rms: float = 0.005
    reflex_state_reference_rms: float = 0.03
    reflex_alignment_floor: float = 0.0
    reflex_alignment_full: float = 0.10
    reflex_alignment_lcb_z: float = 1.0
    reflex_safety_min_probe_count: int = 512
    reflex_safety_bad_probe_patience: int = 2
    reflex_safety_ratio_decay: float = 0.5
    reflex_safety_reenable_probe_interval: int = 512
    reflex_state_ema_decay_by_head: tuple[float, ...] = (0.85, 0.90, 0.95)
    reflex_enabled_at_start_by_head: tuple[bool, ...] = (True, False, False)
    reflex_min_effective_updates_by_head: tuple[float, ...] = (2.0, 4.0, 4.0)
    reflex_min_alignment_count_by_head: tuple[float, ...] = (8.0, 32.0, 64.0)
    reflex_correction_ratio_min_by_head: tuple[float, ...] = (0.010, 0.005, 0.005)
    reflex_correction_ratio_max_by_head: tuple[float, ...] = (0.030, 0.015, 0.010)
    reflex_safety_candidate_mass_deadband: float = 0.0001
    reflex_safety_net_win_rate_deadband: float = 0.0005
    reflex_safety_good_probe_patience: int = 3
    reflex_safety_recovery_factor: float = 1.25
    reflex_safety_minimum_active_ratio: float = 0.125
    reflex_dynamic_tree_enabled: bool = True
    reflex_sparse_boundary_check_enabled: bool = True
    reflex_sparse_boundary_margin: float = 0.20
    reflex_injection_gate_mode: str = "legacy"
    reflex_horizon_delta_rule: str = "inverse_sqrt"
    reflex_warmup_effective_updates: float = 16.0
    reflex_magnitude_gate_floor: float = 0.25
    reflex_guard_enabled: bool = True
    reflex_guard_calibration_rollouts: int = 20
    reflex_guard_aal_drop_fraction: float = 0.05
    reflex_guard_patience: int = 2
    reflex_guard_disable_rollouts: int = 50
    reflex_candidate_probe_interval: int = 256
    reflex_counterfactual_max_sequences: int = 8
    metrics_timing_sample_interval: int = 32
    reflex_feedback_clip_norm: float = 2.0
    reflex_hidden_feedback_clip_norm: float = 0.0
    reflex_fast_state_clip_norm: float = 8.0
    reflex_correction_clip_norm: float = 1.0
    reflex_normalize_correction: bool = True
    reflex_feedback_ce_gate: bool = True
    reflex_feedback_ce_tau: float = 4.0
    reflex_feedback_ce_threshold: float = 0.4
    reflex_normalize_feedback: bool = True
    reflex_feedback_enabled: bool = False
    reflex_proposal_injection_enabled: bool = False
    reflex_proposal_injection_scale: float = 0.0
    reflex_proposal_injection_after: int = 0
    reflex_proposal_injection_warmup: int = 0
    reflex_anchor_conditioning_enabled: bool = False
    reflex_aux_cache_enabled: bool = False
    reflex_aux_cache_max_records: int = 8192
    reflex_aux_cache_stride: int = 1
    reflex_aux_store_fast_state: bool = False
    reflex_sparse_teacher_enabled: bool = False
    reflex_utility_scheduler_enabled: bool = False
    reflex_utility_ema_beta: float = 0.90
    reflex_utility_warmup_rounds: int = 8
    reflex_utility_min_active_heads: int = 2
    reflex_utility_min_depth_acceptance: float = 0.06
    reflex_utility_min_node_utility: float = 0.015
    reflex_utility_exploration_interval: int = 64
    persistent_memory_enabled: bool = False
    persistent_memory_bucket_boundaries: tuple[int, ...] = (128, 256, 512, 1024)
    persistent_memory_local_agreement_after: float = 4.0
    persistent_memory_local_disagreement_floor: float = -0.20
    persistent_memory_min_weight: float = 0.0
    # Retained for B200 motivation/causal ablations. HRDCR uses immediate.
    reflex_adaptation_mode: str = "immediate"
    motivation_trace_enabled: bool = False
    motivation_trace_window_tokens: int = 32


class FlashMedusaDecoder:
    def __init__(self, target_model, medusa_heads, tokenizer, config: FlashMedusaConfig):
        self.target_model = target_model
        self.medusa_heads = medusa_heads
        self.tokenizer = tokenizer
        self.config = config
        if config.acceptance != "exact_target":
            raise NotImplementedError("Only exact_target acceptance is implemented in the main FlashGRPO path")
        if config.cache_update_mode not in {"extract_path", "recompute_accepted"}:
            raise ValueError(f"Unsupported cache_update_mode={config.cache_update_mode}")
        if config.proposal_mode not in {"medusa", "chain"}:
            raise ValueError(f"Unsupported proposal_mode={config.proposal_mode}")
        if config.reflex_adaptation_mode not in {"disabled", "delayed", "immediate"}:
            raise ValueError("reflex_adaptation_mode must be disabled, delayed, or immediate")
        if str(config.reflex_injection_gate_mode).lower() not in {"legacy", "normalized"}:
            raise ValueError("reflex_injection_gate_mode must be legacy or normalized")
        self._reflex_guard_baseline = 0.0
        self._reflex_guard_samples = 0
        self._reflex_guard_bad_windows = 0
        self._reflex_guard_disabled_until = -1
        self._verification_utility_scheduler: VerificationUtilityScheduler | None = None
        self._head3_quality_calibrator: Head3QualityCalibrator | None = None
        self._hrdcr_safety_state: dict[str, torch.Tensor | int] | None = None
        self._reflex_context_projection: torch.Tensor | None = None
        self._reflex_runtime_stats: dict[str, torch.Tensor | float] | None = None

    def reset_reflex_degradation_guard(self) -> None:
        self._reflex_guard_baseline = 0.0
        self._reflex_guard_samples = 0
        self._reflex_guard_bad_windows = 0
        self._reflex_guard_disabled_until = -1

    def reset_verification_utility_scheduler(
        self,
        selected_heads: list[int] | None = None,
    ) -> None:
        """Reset changed proposal utility while preserving unrelated Head-3 evidence.

        ``selected_heads`` uses zero-based MEDUSA head indices.
        """

        self._verification_utility_scheduler = None
        if (
            self._head3_quality_calibrator is not None
            and (selected_heads is None or 2 in selected_heads)
        ):
            self._head3_quality_calibrator.decay_evidence(0.5)

    @torch.no_grad()
    def notify_hrdcr_auxiliary_update(
        self,
        selected_heads: list[int],
        *,
        evidence_decay: float = 0.5,
    ) -> None:
        """Down-weight, but never erase, safety evidence for updated heads."""

        state = self._hrdcr_safety_state
        if not state or not selected_heads:
            return
        head_indices = sorted(
            {
                int(head) - 1
                for head in selected_heads
                if 1 <= int(head) <= int(self.config.num_medusa_heads)
            }
        )
        if not head_indices:
            return
        sample = next(
            (value for value in state.values() if torch.is_tensor(value)),
            None,
        )
        if sample is None:
            return
        index = torch.as_tensor(
            head_indices, device=sample.device, dtype=torch.long
        )
        decay = min(1.0, max(0.0, float(evidence_decay)))
        for name in (
            "safety_mass_gain_sum",
            "safety_mass_gain_sq_sum",
            "safety_net_win_sum",
            "safety_net_win_sq_sum",
            "safety_alignment_sum",
            "safety_alignment_sq_sum",
        ):
            value = state.get(name)
            if torch.is_tensor(value):
                value.index_copy_(
                    0,
                    index.to(value.device),
                    value.index_select(0, index.to(value.device)) * decay,
                )
        for name in (
            "safety_probe_count",
            "safety_alignment_count",
        ):
            value = state.get(name)
            if torch.is_tensor(value):
                scaled = torch.floor(
                    value.index_select(0, index.to(value.device)).float()
                    * decay
                ).to(dtype=value.dtype)
                value.index_copy_(0, index.to(value.device), scaled)
        for name in (
            "safety_fresh_probe_count",
            "safety_bad_windows",
            "safety_good_windows",
        ):
            value = state.get(name)
            if torch.is_tensor(value):
                value.index_fill_(0, index.to(value.device), 0)

    def _get_head3_quality_calibrator(self) -> Head3QualityCalibrator | None:
        cfg = self.config
        if cfg.tree_layout != "sparse_asymmetric" or min(cfg.num_medusa_heads, self.medusa_heads.num_heads) < 3:
            return None
        if self._head3_quality_calibrator is None:
            self._head3_quality_calibrator = Head3QualityCalibrator(
                num_bins=int(cfg.sparse_head3_quality_bins),
                min_calibration_records=int(cfg.sparse_head3_min_calibration_records),
                exploration_fraction=float(cfg.sparse_head3_exploration_fraction),
                node_cost=float(cfg.sparse_head3_node_cost),
                ema_beta=float(cfg.sparse_head3_quality_ema_beta),
                top1_weight=float(cfg.sparse_head3_top1_weight),
                margin_weight=float(cfg.sparse_head3_margin_weight),
                entropy_weight=float(cfg.sparse_head3_entropy_weight),
                path_weight=float(cfg.sparse_head3_path_weight),
                acceptance_weight=float(cfg.sparse_head3_acceptance_weight),
                regret_weight=float(cfg.sparse_head3_regret_weight),
            )
        return self._head3_quality_calibrator

    def _get_verification_utility_scheduler(self) -> VerificationUtilityScheduler | None:
        cfg = self.config
        if not (bool(cfg.reflex_enabled) and bool(cfg.reflex_utility_scheduler_enabled)):
            return None
        if self._verification_utility_scheduler is None:
            self._verification_utility_scheduler = VerificationUtilityScheduler(
                min(cfg.num_medusa_heads, self.medusa_heads.num_heads),
                ema_beta=float(cfg.reflex_utility_ema_beta),
                warmup_rounds=int(cfg.reflex_utility_warmup_rounds),
                min_active_heads=int(cfg.reflex_utility_min_active_heads),
                min_depth_acceptance=float(cfg.reflex_utility_min_depth_acceptance),
                min_node_utility=float(cfg.reflex_utility_min_node_utility),
                exploration_interval=int(cfg.reflex_utility_exploration_interval),
            )
        return self._verification_utility_scheduler

    def _reflex_enabled(self) -> bool:
        if not bool(self.config.reflex_enabled) or self.config.reflex_adaptation_mode == "disabled":
            return False
        if self.config.reflex_state_space == "hidden":
            return True
        return (
            int(self.config.reflex_fast_state_dim) > 0
            and getattr(self.medusa_heads, "reflex_fast_state_dim", 0) > 0
        )

    def _reflex_effective_injection_scale(self, generation_step: int) -> float:
        if not (
            self._reflex_enabled()
            and self.config.reflex_adaptation_mode == "immediate"
            and bool(self.config.reflex_proposal_injection_enabled)
            and float(self.config.reflex_proposal_injection_scale) != 0.0
        ):
            return 0.0
        after = int(self.config.reflex_proposal_injection_after)
        step = int(generation_step)
        if step < int(self._reflex_guard_disabled_until):
            return 0.0
        if step < after:
            return 0.0
        scale = float(self.config.reflex_proposal_injection_scale)
        warmup = max(0, int(self.config.reflex_proposal_injection_warmup))
        if warmup > 0:
            scale *= min(1.0, max(0.0, float(step - after) / float(warmup)))
        return scale

    def _update_reflex_degradation_guard(self, average_accept_length: float, generation_step: int) -> None:
        if (
            not bool(self.config.reflex_guard_enabled)
            or not self._reflex_enabled()
            or not math.isfinite(float(average_accept_length))
        ):
            return
        calibration = max(1, int(self.config.reflex_guard_calibration_rollouts))
        value = float(average_accept_length)
        if self._reflex_guard_samples < calibration:
            self._reflex_guard_samples += 1
            self._reflex_guard_baseline += (value - self._reflex_guard_baseline) / self._reflex_guard_samples
            return
        if int(generation_step) < int(self._reflex_guard_disabled_until):
            return
        threshold = self._reflex_guard_baseline * (1.0 - float(self.config.reflex_guard_aal_drop_fraction))
        if value < threshold:
            self._reflex_guard_bad_windows += 1
        else:
            self._reflex_guard_bad_windows = 0
            self._reflex_guard_baseline = 0.98 * self._reflex_guard_baseline + 0.02 * value
        if self._reflex_guard_bad_windows >= max(1, int(self.config.reflex_guard_patience)):
            self._reflex_guard_disabled_until = int(generation_step) + max(
                1,
                int(self.config.reflex_guard_disable_rollouts),
            )
            self._reflex_guard_bad_windows = 0

    def _reflex_injection_enabled(self, generation_step: int) -> bool:
        return self._reflex_effective_injection_scale(generation_step) != 0.0

    def _reflex_guard_disabled(self, generation_step: int) -> bool:
        return int(generation_step) < int(self._reflex_guard_disabled_until)

    def _reflex_feedback_path_enabled(self, generation_step: int) -> bool:
        return bool(
            self._reflex_enabled()
            and self.config.reflex_feedback_enabled
            and (
                bool(self.config.reflex_strict_horizon_pipeline)
                or not self._reflex_guard_disabled(generation_step)
            )
        )

    @staticmethod
    def _dynamic_feedback_stride(
        *,
        feedback_stride: int,
        feedback_stride_min: int,
        active_batch_size: int,
        initial_batch_size: int,
    ) -> int:
        return max(
            max(1, int(feedback_stride_min)),
            int(
                math.ceil(
                    max(1, int(feedback_stride))
                    * max(1, int(active_batch_size))
                    / max(1, int(initial_batch_size))
                )
            ),
        )

    def _reset_reflex_runtime_stats(self, device: torch.device) -> None:
        self._reflex_runtime_stats = {
            "raw_fast_state_rms_sum": torch.zeros((), device=device, dtype=torch.float32),
            "hint_trust_sum": torch.zeros((), device=device, dtype=torch.float32),
            "warm_gate_sum": torch.zeros((), device=device, dtype=torch.float32),
            "correction_rms_sum": torch.zeros((), device=device, dtype=torch.float32),
            "correction_ratio_sum": torch.zeros((), device=device, dtype=torch.float32),
            "active_correction_ratio_sum": torch.zeros(
                (), device=device, dtype=torch.float32
            ),
            "active_injections": torch.zeros((), device=device, dtype=torch.long),
            "correction_observations": torch.zeros((), device=device, dtype=torch.long),
            "reflex_total_time": 0.0,
        }

    def _add_reflex_time(self, elapsed: float) -> None:
        if self._reflex_runtime_stats is not None:
            self._reflex_runtime_stats["reflex_total_time"] = float(
                self._reflex_runtime_stats["reflex_total_time"]
            ) + float(elapsed)

    def _finalize_reflex_runtime_stats(self) -> dict[str, float]:
        stats = self._reflex_runtime_stats
        if stats is None:
            return {
                "raw_fast_state_rms": 0.0,
                "hint_trust": 0.0,
                "warm_gate": 0.0,
                "correction_rms": 0.0,
                "correction_to_hidden_rms_ratio": 0.0,
                "active_injection_fraction": 0.0,
                "conditional_correction_to_hidden_rms_ratio": 0.0,
                "unconditional_correction_to_hidden_rms_ratio": 0.0,
                "correction_observations": 0.0,
                "reflex_total_time": 0.0,
            }
        observations = max(int(stats["correction_observations"].detach().cpu()), 1)
        active = max(int(stats["active_injections"].detach().cpu()), 1)
        unconditional_ratio = float(stats["correction_ratio_sum"].detach().cpu()) / observations
        conditional_ratio = (
            float(stats["active_correction_ratio_sum"].detach().cpu()) / active
        )
        return {
            "raw_fast_state_rms": float(stats["raw_fast_state_rms_sum"].detach().cpu()) / observations,
            "hint_trust": float(stats["hint_trust_sum"].detach().cpu()) / observations,
            "warm_gate": float(stats["warm_gate_sum"].detach().cpu()) / observations,
            "correction_rms": float(stats["correction_rms_sum"].detach().cpu()) / observations,
            "correction_to_hidden_rms_ratio": unconditional_ratio,
            "active_injection_fraction": float(
                stats["active_injections"].detach().cpu()
            )
            / observations,
            "conditional_correction_to_hidden_rms_ratio": conditional_ratio,
            "unconditional_correction_to_hidden_rms_ratio": unconditional_ratio,
            "correction_observations": float(
                stats["correction_observations"].detach().cpu()
            ),
            "reflex_total_time": float(stats["reflex_total_time"]),
        }

    def _scaled_fast_state(self, fast_state: torch.Tensor | None, generation_step: int) -> torch.Tensor | None:
        scale = self._reflex_effective_injection_scale(generation_step)
        if fast_state is None or scale == 0.0:
            return None
        return fast_state

    def _reflex_context_keys(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        rank = max(0, int(self.config.reflex_context_rank))
        if rank == 0 or hidden_states.numel() == 0:
            return None
        hidden = hidden_states.detach().float()
        if hidden.dim() == 3 and int(hidden.shape[1]) == 1:
            hidden = hidden[:, 0]
        if hidden.dim() != 2:
            raise ValueError("Reflex context keys require [batch, hidden] states")
        projection_rank = (rank + 1) // 2
        projection = self._reflex_context_projection
        if (
            projection is None
            or int(projection.shape[0]) != int(hidden.shape[-1])
            or int(projection.shape[1]) != projection_rank
            or projection.device != hidden.device
        ):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(self.config.reflex_context_seed))
            projection = torch.randn(
                (int(hidden.shape[-1]), projection_rank),
                generator=generator,
                dtype=torch.float32,
            )
            projection = projection / projection.norm(dim=0, keepdim=True).clamp_min(1e-6)
            self._reflex_context_projection = projection.to(device=hidden.device)
            projection = self._reflex_context_projection
        projected = hidden / hidden.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        projected = projected @ projection
        # Paired signed ReLU features implement a sparse angular kernel. The
        # resulting similarities remain non-negative, while opposite or
        # unrelated contexts no longer collapse into the same dense key.
        keys = torch.cat((torch.relu(projected), torch.relu(-projected)), dim=-1)[..., :rank]
        return keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _apply_persistent_memory_prior(
        self,
        local_state: torch.Tensor | None,
        effective_updates: torch.Tensor | None,
        *,
        active_original_indices: list[int],
        logical_lens: torch.Tensor,
        initial_logical_lens: torch.Tensor,
        persistent_memory: torch.Tensor | None,
        persistent_strength: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, float | int]]:
        if (
            local_state is None
            or persistent_memory is None
            or persistent_strength is None
            or not bool(self.config.persistent_memory_enabled)
        ):
            return local_state, {}
        if not active_original_indices or persistent_memory.numel() == 0:
            return local_state, {}
        device = local_state.device
        ids = torch.as_tensor(active_original_indices, dtype=torch.long, device=device)
        if ids.numel() == 0:
            return local_state, {}
        boundaries = torch.as_tensor(
            list(self.config.persistent_memory_bucket_boundaries),
            dtype=torch.long,
            device=device,
        )
        initial = initial_logical_lens.index_select(0, ids.to(initial_logical_lens.device)).to(device=device)
        depths = (logical_lens.to(device=device) - initial).clamp_min(0).long()
        buckets = torch.bucketize(depths, boundaries).clamp(0, int(persistent_memory.shape[1]) - 1)
        memory_rows = persistent_memory.index_select(0, ids)
        strength_rows = persistent_strength.index_select(0, ids)
        row_idx = torch.arange(ids.numel(), dtype=torch.long, device=device)
        memory = memory_rows[row_idx, buckets].to(device=device, dtype=torch.float32)
        strength = strength_rows[row_idx, buckets].to(device=device, dtype=torch.float32).view(-1, 1)
        active = strength.squeeze(-1).gt(0)
        if not bool(active.any().item()):
            return local_state, {"persistent_memory_active_sequences": 0}

        gate = torch.ones_like(strength)
        after = float(self.config.persistent_memory_local_agreement_after)
        if effective_updates is not None and after > 0:
            mature = effective_updates.to(device=device, dtype=torch.float32).view(-1, 1).ge(after)
            if bool(mature.any().item()):
                local = torch.nan_to_num(local_state.float(), nan=0.0, posinf=0.0, neginf=0.0)
                mem = torch.nan_to_num(memory.float(), nan=0.0, posinf=0.0, neginf=0.0)
                denom = local.norm(dim=-1, keepdim=True).clamp_min(1e-6) * mem.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                cosine = (local * mem).sum(dim=-1, keepdim=True) / denom
                floor = float(self.config.persistent_memory_local_disagreement_floor)
                agreement = torch.clamp((cosine - floor) / max(1.0 - floor, 1e-6), min=0.0, max=1.0)
                gate = torch.where(mature, agreement, gate)
        prior = strength * gate * memory
        effective = local_state.float() + prior
        nonzero = active & gate.squeeze(-1).gt(0)
        return effective.to(dtype=local_state.dtype), {
            "persistent_memory_active_sequences": int(nonzero.sum().detach().cpu()),
            "persistent_memory_strength_mean": float(strength[active.view(-1)].mean().detach().cpu()),
            "persistent_memory_strength_max": float(strength.max().detach().cpu()),
            "persistent_memory_gate_mean": float(gate[active.view(-1)].mean().detach().cpu()),
        }

    @torch.no_grad()
    def _apply_reflex_correction(
        self,
        base_hidden: torch.Tensor,
        fast_state: torch.Tensor | None,
        effective_updates: torch.Tensor | None,
        head_idx: int,
        generation_step: int,
        hint_trust: torch.Tensor | None = None,
        ratio_scale: torch.Tensor | None = None,
        important_ids: torch.Tensor | None = None,
        boundary_ids: torch.Tensor | None = None,
        lm_head=None,
    ) -> torch.Tensor:
        reflex_start = time.perf_counter()
        scale = self._reflex_effective_injection_scale(generation_step)
        if fast_state is None or scale == 0.0:
            return base_hidden
        cfg = self.config
        if fast_state.dim() == 3:
            if head_idx >= int(fast_state.shape[1]):
                return base_hidden
            fast_state = fast_state[:, int(head_idx), :]
        if effective_updates is not None and effective_updates.dim() == 2:
            if head_idx >= int(effective_updates.shape[1]):
                effective_updates = None
            else:
                effective_updates = effective_updates[:, int(head_idx)]
        if hint_trust is not None and hint_trust.dim() == 2:
            if head_idx >= int(hint_trust.shape[1]):
                hint_trust = None
            else:
                hint_trust = hint_trust[:, int(head_idx)]
        if ratio_scale is not None and ratio_scale.dim() == 2:
            if head_idx >= int(ratio_scale.shape[1]):
                ratio_scale = None
            else:
                ratio_scale = ratio_scale[:, int(head_idx)]
        if cfg.reflex_state_space != "hidden":
            corrected = self.medusa_heads.add_reflex_delta(
                base_hidden,
                fast_state,
                head_idx,
                max_norm=float(cfg.reflex_correction_clip_norm),
                scale=float(scale),
                normalize=bool(cfg.reflex_normalize_correction),
            )
            self._add_reflex_time(time.perf_counter() - reflex_start)
            return corrected

        state = torch.nan_to_num(fast_state.float(), nan=0.0, posinf=0.0, neginf=0.0)
        state_rms = state.square().mean(dim=-1, keepdim=True).sqrt()
        base_rms = torch.nan_to_num(base_hidden.float(), nan=0.0, posinf=0.0, neginf=0.0).square().mean(
            dim=-1, keepdim=True
        ).sqrt()
        if bool(cfg.reflex_strict_horizon_pipeline):
            confidence = (
                torch.zeros_like(state_rms)
                if hint_trust is None
                else torch.nan_to_num(
                    hint_trust.to(device=state.device, dtype=torch.float32).view(-1, 1),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0)
            )
            safety = (
                torch.ones_like(confidence)
                if ratio_scale is None
                else torch.nan_to_num(
                    ratio_scale.to(device=state.device, dtype=torch.float32).view(-1, 1),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0)
            )
            eligible = confidence.gt(0.0) & state_rms.ge(float(cfg.reflex_min_state_rms))
            hidden_finite = torch.isfinite(base_hidden).all(dim=-1)
            if hidden_finite.dim() > 1:
                hidden_finite = hidden_finite.all(dim=-1)
            eligible &= hidden_finite.view(-1, 1)
            if effective_updates is not None:
                min_updates = float(
                    cfg.reflex_min_effective_updates_by_head[head_idx]
                    if head_idx < len(cfg.reflex_min_effective_updates_by_head)
                    else cfg.reflex_min_effective_updates
                )
                eligible &= effective_updates.float().view(-1, 1).ge(
                    min_updates
                )
            ratio_min = max(
                0.0,
                float(
                    cfg.reflex_correction_ratio_min_by_head[head_idx]
                    if head_idx < len(cfg.reflex_correction_ratio_min_by_head)
                    else cfg.reflex_correction_ratio_min
                ),
            )
            ratio_max = max(
                ratio_min,
                float(
                    cfg.reflex_correction_ratio_max_by_head[head_idx]
                    if head_idx < len(cfg.reflex_correction_ratio_max_by_head)
                    else cfg.reflex_correction_ratio_max
                ),
            )
            ratio_cap = abs(float(scale)) * safety * ratio_max
            target_ratio = (
                float(scale)
                * safety
                * (ratio_min + (ratio_max - ratio_min) * confidence)
            ).clamp(min=0.0)
            target_ratio = torch.minimum(target_ratio, ratio_cap)
            target_ratio = target_ratio.masked_fill(safety.le(0.0), 0.0)
            target_ratio = target_ratio.masked_fill(~eligible, 0.0)
            state_unit = state / state_rms.clamp_min(1e-6)
            if base_hidden.dim() == 3:
                state_unit = state_unit.unsqueeze(1)
                target_ratio = target_ratio.unsqueeze(1)
                confidence_for_stats = confidence.unsqueeze(1)
            else:
                confidence_for_stats = confidence
            correction = target_ratio * base_rms * state_unit
            correction_rms = correction.square().mean(dim=-1, keepdim=True).sqrt()
            actual_ratio = correction_rms / base_rms.clamp_min(1e-6)
            correction = correction * (
                target_ratio / actual_ratio.clamp_min(1e-6)
            )
            correction_rms = correction.square().mean(dim=-1, keepdim=True).sqrt()
            hard_cap = ratio_cap * base_rms
            if base_hidden.dim() == 3 and hard_cap.dim() == 2:
                hard_cap = hard_cap.unsqueeze(1)
            correction = correction * torch.clamp(hard_cap / correction_rms.clamp_min(1e-6), max=1.0)
            if (
                bool(cfg.reflex_sparse_boundary_check_enabled)
                and lm_head is not None
                and important_ids is not None
                and boundary_ids is not None
                and base_hidden.dim() == 2
            ):
                important = important_ids.to(device=state.device, dtype=torch.long).view(-1)
                boundary = boundary_ids.to(device=state.device, dtype=torch.long).view(-1)
                vocab = int(lm_head.weight.shape[0])
                valid_pair = (
                    important.ge(0)
                    & important.lt(vocab)
                    & boundary.ge(0)
                    & boundary.lt(vocab)
                    & eligible.view(-1)
                )
                safe_important = important.masked_fill(~valid_pair, 0)
                safe_boundary = boundary.masked_fill(~valid_pair, 0)
                pair_direction = (
                    lm_head.weight.index_select(0, safe_important)
                    - lm_head.weight.index_select(0, safe_boundary)
                ).float()
                base_gap = torch.einsum("rh,rh->r", pair_direction, base_hidden.float())
                delta_gap = torch.einsum("rh,rh->r", pair_direction, correction.float())
                improves = valid_pair & delta_gap.gt(0.0)
                degrades = valid_pair & ~delta_gap.gt(0.0)
                needed = (
                    (float(cfg.reflex_sparse_boundary_margin) - base_gap)
                    / delta_gap.clamp_min(1e-8)
                ).clamp_min(1.0)
                current_ratio = target_ratio.view(-1).clamp_min(1e-8)
                suggested = torch.minimum(
                    needed, ratio_cap.view(-1) / current_ratio
                ).clamp_min(1.0)
                # Missing/invalid hints are not negative evidence. Preserve the
                # trust-region correction unless a valid boundary pair shows
                # that it moves logits in the wrong direction.
                rescale = torch.ones_like(current_ratio)
                rescale = torch.where(improves, suggested, rescale)
                rescale = rescale.masked_fill(degrades, 0.0)
                correction = correction * rescale.unsqueeze(-1)
                correction_rms = correction.float().square().mean(
                    dim=-1, keepdim=True
                ).sqrt()
                correction = correction * torch.clamp(
                    hard_cap / correction_rms.clamp_min(1e-6), max=1.0
                )
            correction = torch.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
            corrected = base_hidden + correction.to(device=base_hidden.device, dtype=base_hidden.dtype)
            if self._reflex_runtime_stats is not None:
                final_rms = correction.float().square().mean(dim=-1).sqrt().reshape(-1)
                base_flat = base_rms.reshape(-1).clamp_min(1e-6)
                runtime = self._reflex_runtime_stats
                runtime["raw_fast_state_rms_sum"].add_(state_rms.sum())
                runtime["hint_trust_sum"].add_(confidence_for_stats.sum())
                runtime["warm_gate_sum"].add_(eligible.sum())
                runtime["correction_rms_sum"].add_(final_rms.sum())
                ratios = final_rms / base_flat
                active_mask = ratios.gt(0.0)
                runtime["correction_ratio_sum"].add_(ratios.sum())
                runtime["active_correction_ratio_sum"].add_(
                    ratios.masked_fill(~active_mask, 0.0).sum()
                )
                runtime["active_injections"].add_(active_mask.sum())
                runtime["correction_observations"].add_(state_rms.numel())
            self._add_reflex_time(time.perf_counter() - reflex_start)
            return corrected
        if effective_updates is None:
            effective_updates = state_rms.new_zeros((state_rms.shape[0],))
        warmup = max(float(cfg.reflex_warmup_effective_updates), 1e-6)
        warm_gate = 1.0 - torch.exp(-effective_updates.float().view(-1, 1) / warmup)
        if str(cfg.reflex_horizon_delta_rule) == "trust_calibrated":
            # Horizon-resolved hint quality already estimates whether historical
            # verifier gradients transfer to this head. Keep one common trust
            # region instead of penalizing deeper heads twice.
            delta_k = float(cfg.reflex_relative_rms_delta_base)
        else:
            delta_k = float(cfg.reflex_relative_rms_delta_base) / math.sqrt(max(1, int(head_idx) + 1))
        mode = str(cfg.reflex_injection_gate_mode).lower()
        if mode == "normalized":
            trust = (
                torch.ones_like(warm_gate)
                if hint_trust is None
                else torch.nan_to_num(
                    hint_trust.to(device=state.device, dtype=torch.float32).view(-1, 1),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp(min=0.0, max=1.0)
            )
            state_unit = state / state_rms.clamp_min(1e-6)
            if base_hidden.dim() == 3:
                state_unit = state_unit.unsqueeze(1)
                warm_for_correction = warm_gate.unsqueeze(1)
                trust_for_correction = trust.unsqueeze(1)
            else:
                warm_for_correction = warm_gate
                trust_for_correction = trust
            correction = (
                float(scale)
                * delta_k
                * warm_for_correction
                * trust_for_correction
                * base_rms
                * state_unit
            )
        else:
            trust = torch.ones_like(warm_gate)
            magnitude_gate = state_rms / (state_rms + float(cfg.reflex_magnitude_gate_floor))
            alpha = (
                float(scale)
                * warm_gate
                * magnitude_gate
                * delta_k
                * base_rms
                / state_rms.clamp_min(1e-6)
            )
            correction = alpha * state
            if base_hidden.dim() == 3 and correction.dim() == 2:
                correction = correction.unsqueeze(1)
        # This clamp is tensor-only and enforces the relative RMS safety cap.
        correction_rms = correction.square().mean(dim=-1, keepdim=True).sqrt()
        cap = 1.01 * abs(float(scale) * delta_k) * base_rms
        correction = correction * torch.clamp(cap / correction_rms.clamp_min(1e-6), max=1.0)
        corrected = base_hidden + correction.to(device=base_hidden.device, dtype=base_hidden.dtype)
        if self._reflex_runtime_stats is not None:
            final_correction_rms = correction.float().square().mean(dim=-1).sqrt().reshape(-1)
            base_ratio_rms = base_rms.reshape(-1).clamp_min(1e-6)
            runtime = self._reflex_runtime_stats
            runtime["raw_fast_state_rms_sum"].add_(state_rms.sum())
            runtime["hint_trust_sum"].add_(trust.sum())
            runtime["warm_gate_sum"].add_(warm_gate.sum())
            runtime["correction_rms_sum"].add_(final_correction_rms.sum())
            runtime["correction_ratio_sum"].add_((final_correction_rms / base_ratio_rms).sum())
            runtime["correction_observations"].add_(state_rms.numel())
        self._add_reflex_time(time.perf_counter() - reflex_start)
        return corrected

    @torch.no_grad()
    def _medusa_logits_for_last_hidden(
        self,
        last_hidden: torch.Tensor,
        *,
        lm_head,
        max_heads: int,
        fast_state: torch.Tensor | None,
        effective_updates: torch.Tensor | None,
        generation_step: int,
        hint_trust: torch.Tensor | None = None,
        ratio_scale: torch.Tensor | None = None,
        important_ids: torch.Tensor | None = None,
        boundary_ids: torch.Tensor | None = None,
        root_tokens: torch.Tensor | None = None,
        embedding_layer=None,
        return_projected: bool = False,
        return_raw_projected: bool = False,
    ):
        max_heads = max(0, min(int(max_heads), self.medusa_heads.num_heads))
        if max_heads == 0:
            if return_raw_projected:
                return [], [], []
            return ([], []) if return_projected else []
        device_type = "cuda" if last_hidden.device.type == "cuda" else last_hidden.device.type
        with torch.amp.autocast(
            device_type,
            dtype=autocast_dtype(unwrap_causal_lm(self.target_model)),
            enabled=bool(self.config.head_inference_autocast and last_hidden.device.type == "cuda"),
        ):
            anchor_embeddings = None
            if (
                bool(self.config.reflex_anchor_conditioning_enabled)
                and getattr(self.medusa_heads, "anchor_conditioner", None) is not None
                and root_tokens is not None
            ):
                if embedding_layer is None:
                    raise ValueError("embedding_layer is required for anchor-conditioned Reflex")
                anchor_embeddings = embedding_layer(root_tokens.to(device=last_hidden.device)).detach()
            projected: list[torch.Tensor] = []
            raw_projected: list[torch.Tensor] = []
            for head_idx in range(max_heads):
                medusa_hidden = self.medusa_heads.heads[head_idx].project_hidden(last_hidden)
                if anchor_embeddings is not None:
                    medusa_hidden = self.medusa_heads.anchor_conditioner(
                        medusa_hidden,
                        anchor_embeddings,
                        head_idx,
                    )
                raw_projected.append(medusa_hidden)
                medusa_hidden = self._apply_reflex_correction(
                    medusa_hidden,
                    fast_state,
                    effective_updates,
                    head_idx,
                    generation_step,
                    hint_trust,
                    ratio_scale,
                    (
                        important_ids[:, head_idx]
                        if important_ids is not None
                        and important_ids.dim() == 2
                        and head_idx < int(important_ids.shape[1])
                        else important_ids
                    ),
                    (
                        boundary_ids[:, head_idx]
                        if boundary_ids is not None
                        and boundary_ids.dim() == 2
                        and head_idx < int(boundary_ids.shape[1])
                        else boundary_ids
                    ),
                    lm_head,
                )
                projected.append(medusa_hidden)

            # Tied heads all share the 152k-vocabulary projection. One larger
            # GEMM has lower launch overhead than one projection per head while
            # producing identical logits up to the chosen inference dtype.
            if all(self.medusa_heads.heads[idx].output is None for idx in range(max_heads)):
                batch = int(last_hidden.shape[0])
                flat_hidden = torch.cat(projected, dim=0)
                lm_dtype = getattr(lm_head.weight, "dtype", flat_hidden.dtype)
                flat_logits = lm_head(flat_hidden.to(dtype=lm_dtype))
                logits = list(flat_logits.split(batch, dim=0))
                if return_raw_projected:
                    return logits, projected, raw_projected
                return (logits, projected) if return_projected else logits

            logits_by_head: list[torch.Tensor] = []
            for head_idx, medusa_hidden in enumerate(projected):
                output = self.medusa_heads.heads[head_idx].output
                if output is not None:
                    logits_by_head.append(output(medusa_hidden))
                else:
                    lm_dtype = getattr(lm_head.weight, "dtype", medusa_hidden.dtype)
                    logits_by_head.append(lm_head(medusa_hidden.to(dtype=lm_dtype)))
            if return_raw_projected:
                return logits_by_head, projected, raw_projected
            return (logits_by_head, projected) if return_projected else logits_by_head

    def _normalize_reflex_feedback(self, feedback: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        feedback = torch.nan_to_num(feedback.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if bool(cfg.reflex_normalize_feedback):
            rms = feedback.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            feedback = feedback / rms
        if float(cfg.reflex_feedback_clip_norm) > 0:
            norm = feedback.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            feedback = feedback * torch.clamp(float(cfg.reflex_feedback_clip_norm) / norm, max=1.0)
        return feedback

    def _confidence_from_logits(self, logits: torch.Tensor) -> float:
        return self._confidences_from_logits([logits])[0]

    def _confidences_from_logits(self, logits_by_head: list[torch.Tensor]) -> list[float]:
        if not logits_by_head:
            return []
        cfg = self.config
        per_head_scores: list[torch.Tensor] = []
        for logits in logits_by_head:
            if cfg.adaptive_confidence_metric == "margin":
                values = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1).values
                if values.shape[-1] == 1:
                    score = values[..., 0]
                else:
                    score = values[..., 0] - values[..., 1]
            else:
                top1 = torch.max(logits, dim=-1).values.float()
                score = torch.exp(top1 - torch.logsumexp(logits, dim=-1).float())
            per_head_scores.append(torch.nan_to_num(score.float(), nan=0.0, posinf=0.0, neginf=0.0))
        scores = torch.stack(per_head_scores, dim=0)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        q = min(1.0, max(0.0, float(cfg.adaptive_confidence_quantile)))
        return torch.quantile(scores.detach(), q, dim=-1).cpu().tolist()

    def _topk_from_confidence(self, confidence: float, base_k: int, depth_idx: int) -> int:
        cfg = self.config
        base_k = max(1, int(base_k))
        min_defaults = list(cfg.adaptive_min_topk_by_depth) or [1]
        min_k = max(1, min(base_k, int(min_defaults[min(depth_idx, len(min_defaults) - 1)])))
        if (not cfg.adaptive_tree_enabled) or base_k <= min_k:
            return base_k
        if not math.isfinite(float(confidence)):
            return min_k
        low = float(cfg.adaptive_confidence_low)
        high = float(cfg.adaptive_confidence_high)
        if high <= low:
            return min_k if confidence >= high else base_k
        if confidence <= low:
            return base_k
        if confidence >= high:
            return min_k
        ratio = (confidence - low) / (high - low)
        k_float = base_k - ratio * (base_k - min_k)
        return max(min_k, min(base_k, int(round(k_float))))

    def _plan_with_topk(self, plan: TreePlan, topk_by_depth: list[int], *, actual_nodes: int | None = None) -> TreePlan:
        topk_by_depth = [max(1, int(k)) for k in topk_by_depth]
        return TreePlan(
            node_budget_per_seq=plan.node_budget_per_seq,
            active_heads=len(topk_by_depth),
            topk_by_depth=topk_by_depth,
            actual_nodes=int(actual_nodes if actual_nodes is not None else dense_node_count(topk_by_depth)),
            mode=plan.mode,
            layout=plan.layout,
            nodes_by_head=(list(topk_by_depth) if plan.layout == "sparse_asymmetric" else []),
            min_head3_nodes=plan.min_head3_nodes,
            head3_min_budget=plan.head3_min_budget,
            branch_score_temperature=plan.branch_score_temperature,
            diversity_penalty=plan.diversity_penalty,
        )

    def _adapt_plan_from_logits(self, medusa_logits: list[torch.Tensor], plan: TreePlan) -> tuple[TreePlan, dict]:
        if not self.config.adaptive_tree_enabled or not medusa_logits or not plan.topk_by_depth:
            return plan, {}
        head_count = min(len(plan.topk_by_depth), len(medusa_logits))
        confidences = self._confidences_from_logits(medusa_logits[:head_count])
        adapted = []
        for depth_idx, (base_k, confidence) in enumerate(zip(plan.topk_by_depth, confidences)):
            adapted.append(self._topk_from_confidence(confidence, int(base_k), depth_idx))
        if not adapted:
            return self._plan_with_topk(plan, []), {"confidence": confidences, "topk": []}
        adapted = fit_topk_to_budget(
            adapted,
            plan.node_budget_per_seq,
            min_topk_by_depth=list(self.config.adaptive_min_topk_by_depth)[: len(adapted)],
            depth_weight_decay=float(self.config.adaptive_depth_weight_decay),
        )
        return self._plan_with_topk(plan, adapted), {"confidence": confidences, "topk": adapted}

    @staticmethod
    def _internal_tree_logit_layout(
        trees: list[CandidateTree],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
        """Map only nodes with children to packed LM-head rows."""

        max_nodes = max((tree.node_count for tree in trees), default=1)
        rows: list[int] = []
        nodes: list[int] = []
        slots_cpu = [[-1] * max_nodes for _ in trees]
        for row, tree in enumerate(trees):
            for node_idx in range(tree.node_count):
                if not tree.children.get(node_idx):
                    continue
                slots_cpu[row][node_idx] = len(rows)
                rows.append(row)
                nodes.append(node_idx)
        row_index = torch.tensor(rows, dtype=torch.long, device=device)
        node_index = torch.tensor(nodes, dtype=torch.long, device=device)
        return row_index, node_index, slots_cpu

    def _project_packed_tree_logits(
        self,
        tree_hidden: torch.Tensor,
        trees: list[CandidateTree],
        lm_head,
    ) -> tuple[torch.Tensor, list[list[int]]]:
        row_index, node_index, slots_cpu = self._internal_tree_logit_layout(
            trees,
            device=tree_hidden.device,
        )
        hidden = tree_hidden[row_index, node_index]
        base = unwrap_causal_lm(self.target_model)
        device_type = "cuda" if hidden.device.type == "cuda" else hidden.device.type
        with torch.amp.autocast(
            device_type,
            dtype=autocast_dtype(base),
            enabled=(hidden.device.type == "cuda"),
        ):
            logits = lm_head(hidden.to(dtype=getattr(lm_head.weight, "dtype", hidden.dtype)))
        return logits, slots_cpu

    def _build_chain_batch_trees(
        self,
        root_tokens: torch.Tensor,
        current_hidden: torch.Tensor,
        plan: TreePlan,
        *,
        lm_head,
        embedding_layer,
        fast_state: torch.Tensor | None = None,
        effective_updates: torch.Tensor | None = None,
        hint_trust: torch.Tensor | None = None,
        ratio_scale: torch.Tensor | None = None,
        generation_step: int = 0,
    ) -> tuple[list[CandidateTree], TreePlan, dict]:
        cfg = self.config
        device = current_hidden.device
        batch = int(root_tokens.shape[0])
        row_tokens = [[int(token)] for token in root_tokens.detach().cpu().tolist()]
        row_parents = [[-1] for _ in range(batch)]
        row_depths = [[1] for _ in range(batch)]
        row_scores = [[0.0] for _ in range(batch)]

        if cfg.chain_bootstrap_from_medusa and self.medusa_heads.num_heads > 0:
            parent_states = self.medusa_heads.heads[0].project_hidden(current_hidden.detach())
        else:
            parent_states = self.medusa_heads.chain_next_state(current_hidden.detach(), root_tokens, embedding_layer)
        parent_rows = torch.arange(batch, dtype=torch.long, device=device)
        parent_node_ids = [0 for _ in range(batch)]
        parent_fast_state = self._scaled_fast_state(fast_state, generation_step)
        parent_effective_updates = effective_updates
        parent_hint_trust = hint_trust
        parent_ratio_scale = ratio_scale
        adapted_topk: list[int] = []
        confidences: list[float] = []
        record_logits: list[torch.Tensor] = []

        for depth_idx, base_k in enumerate(plan.topk_by_depth):
            if parent_states.numel() == 0:
                break
            logit_states = self._apply_reflex_correction(
                parent_states,
                parent_fast_state,
                parent_effective_updates,
                depth_idx,
                generation_step,
                parent_hint_trust,
                parent_ratio_scale,
            )
            logits = torch.nan_to_num(
                self.medusa_heads.chain_logits_from_state(logit_states, lm_head).float(),
                nan=-1.0e9,
                posinf=1.0e9,
                neginf=-1.0e9,
            )
            if depth_idx == 0 and parent_rows.numel() == batch and bool((parent_rows == torch.arange(batch, device=device)).all().item()):
                record_logits.append(logits.detach())
            confidence = self._confidence_from_logits(logits)
            confidences.append(confidence)
            k = self._topk_from_confidence(confidence, int(base_k), depth_idx)
            if k <= 0:
                break
            adapted_topk.append(k)
            values, indices = torch.topk(logits, k=min(k, logits.shape[-1]), dim=-1)
            parent_rows_cpu = parent_rows.detach().cpu().tolist()
            token_rows = indices.detach().cpu().tolist()
            score_rows = values.detach().cpu().tolist()
            next_parent_state_indices: list[int] = []
            next_token_ids: list[int] = []
            next_rows: list[int] = []
            next_node_ids: list[int] = []
            for parent_state_idx, row in enumerate(parent_rows_cpu):
                if len(row_tokens[row]) >= plan.node_budget_per_seq:
                    continue
                parent_node = int(parent_node_ids[parent_state_idx])
                for token, score in zip(token_rows[parent_state_idx], score_rows[parent_state_idx]):
                    if len(row_tokens[row]) >= plan.node_budget_per_seq:
                        break
                    row_tokens[row].append(int(token))
                    row_parents[row].append(parent_node)
                    row_depths[row].append(depth_idx + 2)
                    row_scores[row].append(float(score))
                    next_parent_state_indices.append(parent_state_idx)
                    next_token_ids.append(int(token))
                    next_rows.append(int(row))
                    next_node_ids.append(len(row_tokens[row]) - 1)
            if not next_token_ids:
                break
            parent_index = torch.tensor(next_parent_state_indices, dtype=torch.long, device=device)
            token_tensor = torch.tensor(next_token_ids, dtype=torch.long, device=device)
            parent_states = self.medusa_heads.chain_next_state(
                parent_states.index_select(0, parent_index),
                token_tensor,
                embedding_layer,
            )
            parent_fast_state = parent_fast_state.index_select(0, parent_index) if parent_fast_state is not None else None
            parent_effective_updates = (
                parent_effective_updates.index_select(0, parent_index)
                if parent_effective_updates is not None
                else None
            )
            parent_hint_trust = (
                parent_hint_trust.index_select(0, parent_index)
                if parent_hint_trust is not None
                else None
            )
            parent_ratio_scale = (
                parent_ratio_scale.index_select(0, parent_index)
                if parent_ratio_scale is not None
                else None
            )
            parent_rows = torch.tensor(next_rows, dtype=torch.long, device=device)
            parent_node_ids = next_node_ids
            del logits, values, indices, parent_index, token_tensor

        trees = [
            CandidateTree(tokens=row_tokens[row], parents=row_parents[row], depths=row_depths[row], scores=row_scores[row])
            for row in range(batch)
        ]
        actual_nodes = max((tree.node_count for tree in trees), default=1)
        chain_plan = self._plan_with_topk(plan, adapted_topk, actual_nodes=actual_nodes)
        return trees, chain_plan, {"confidence": confidences, "topk": adapted_topk, "record_logits": record_logits}

    def _last_valid_hidden(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_idx = attention_mask.long().sum(dim=-1).clamp_min(1) - 1
        # The training collator left-pads prompts, so the last non-padding token
        # is at the right edge. This fallback also handles non-left-padded smoke tests.
        if bool((attention_mask[:, -1] == 1).all().item()):
            return hidden_states[:, -1, :]
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, hidden_states.shape[-1])
        return hidden_states.gather(1, gather_idx).squeeze(1)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        repeated_generate_nums: int | None = None,
        max_length: int = 2048,
        statistical_time: bool = False,
        generation_step: int = 0,
        collect_reflex_aux_cache: bool | None = None,
        persistent_reflex_memory: dict | None = None,
    ) -> dict:
        cfg = self.config
        device = model_device(self.target_model)
        repeats = max(1, int(repeated_generate_nums or 1))
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        base = unwrap_causal_lm(self.target_model)
        lm_head = base.lm_head
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = pad_token_id
        prompt_memory_tensor = None
        prompt_memory_strength = None
        prompt_memory_bounds = tuple(int(x) for x in cfg.persistent_memory_bucket_boundaries)
        if (
            bool(cfg.persistent_memory_enabled)
            and not self._reflex_guard_disabled(generation_step)
            and persistent_reflex_memory
            and torch.is_tensor(persistent_reflex_memory.get("memory"))
            and torch.is_tensor(persistent_reflex_memory.get("strength"))
        ):
            raw_memory = persistent_reflex_memory["memory"]
            raw_strength = persistent_reflex_memory["strength"]
            if int(raw_memory.shape[0]) == int(input_ids.shape[0]) and int(raw_strength.shape[0]) == int(input_ids.shape[0]):
                prompt_memory_tensor = raw_memory.to(device=device, dtype=torch.float32, non_blocking=True)
                prompt_memory_strength = raw_strength.to(device=device, dtype=torch.float32, non_blocking=True)
                prompt_memory_bounds = tuple(int(x) for x in persistent_reflex_memory.get("bucket_bounds", prompt_memory_bounds))

        total_start = time.time()
        if self._reflex_enabled() and not self._reflex_guard_disabled(generation_step):
            self._reset_reflex_runtime_stats(device)
        else:
            self._reflex_runtime_stats = None
        if statistical_time and torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_start = time.time()
        prefill_out = prefill(self.target_model, input_ids, attention_mask)
        if statistical_time and torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_time = time.time() - prefill_start
        past_key_values = prefill_out["past_key_values"]
        prefill_hidden = prefill_out["hidden_states"]
        current_hidden = self._last_valid_hidden(prefill_hidden, attention_mask)
        del prefill_hidden, prefill_out
        with torch.amp.autocast(
            "cuda" if device.type == "cuda" else device.type,
            dtype=autocast_dtype(base),
            enabled=(device.type == "cuda"),
        ):
            current_logits = lm_head(current_hidden.to(dtype=getattr(lm_head.weight, "dtype", current_hidden.dtype)))
        if repeats > 1:
            past_key_values = repeat_interleave_cache(past_key_values, repeats, causal_lm=self.target_model)
            current_hidden = current_hidden.repeat_interleave(repeats, dim=0).contiguous()
            current_logits = current_logits.repeat_interleave(repeats, dim=0).contiguous()
            attention_mask = attention_mask.repeat_interleave(repeats, dim=0).contiguous()
            if prompt_memory_tensor is not None:
                prompt_memory_tensor = prompt_memory_tensor.repeat_interleave(repeats, dim=0).contiguous()
                prompt_memory_strength = prompt_memory_strength.repeat_interleave(repeats, dim=0).contiguous()

        total_sequences = attention_mask.shape[0]
        generated: list[list[int]] = [[] for _ in range(total_sequences)]
        trace_enabled = bool(cfg.motivation_trace_enabled)
        trace_window = max(1, int(cfg.motivation_trace_window_tokens))
        trace_buckets: dict[tuple[int, int], dict] = {}

        def trace_bucket(sequence_id: int, generated_position: int) -> dict:
            window_index = max(0, int(generated_position)) // trace_window
            key = (int(sequence_id), window_index)
            if key not in trace_buckets:
                trace_buckets[key] = {
                    "sequence_id": int(sequence_id),
                    "window_index": int(window_index),
                    "token_start": int(window_index * trace_window),
                    "token_end": int((window_index + 1) * trace_window - 1),
                    "verify_rounds": 0,
                    "accepted_length_sum": 0,
                    "accepted_medusa_tokens": 0,
                    "proposed_medusa_tokens": 0,
                    "feedback_events": 0,
                    "fast_state_rms_sum": 0.0,
                    "fast_state_rms_observations": 0,
                }
            return trace_buckets[key]

        active_original_indices = list(range(total_sequences))
        # A target sample that misses the proposal tree is already the exact
        # next token. Reuse it as the next root instead of drawing again from
        # the same distribution, which would favor tokens inside the tree.
        pending_root_tokens: dict[int, int] = {}
        full_attention_mask = attention_mask.long()
        logical_lens = mask_logical_lengths(full_attention_mask)
        initial_logical_lens = logical_lens.clone()
        initial_logical_lens_cpu = initial_logical_lens.detach().cpu().tolist()
        reflex_enabled = self._reflex_enabled()
        reflex_state_enabled = self._reflex_enabled()
        strict_hrdcr = bool(cfg.reflex_strict_horizon_pipeline)
        reflex_injection_enabled = self._reflex_injection_enabled(generation_step)
        reflex_guard_disabled = self._reflex_guard_disabled(generation_step)
        reflex_feedback_enabled = self._reflex_feedback_path_enabled(generation_step)
        coverage_feedback_enabled = bool(
            not strict_hrdcr
            and reflex_feedback_enabled
            and str(cfg.reflex_feedback_objective).lower() == "coverage"
        )
        distribution_feedback_enabled = bool(
            reflex_feedback_enabled and not coverage_feedback_enabled
        )
        normalized_injection = str(cfg.reflex_injection_gate_mode).lower() == "normalized"
        collect_reflex_aux_cache = (
            bool(cfg.reflex_aux_cache_enabled)
            if collect_reflex_aux_cache is None
            else bool(collect_reflex_aux_cache)
        )
        # Auxiliary head refresh is an independent ablation axis. It may cache
        # verified hidden/teacher pairs even when m_t injection is disabled.
        collect_reflex_aux_cache = bool(
            collect_reflex_aux_cache and (strict_hrdcr or not reflex_guard_disabled)
        )
        state_dim = int(current_hidden.shape[-1]) if cfg.reflex_state_space == "hidden" else int(cfg.reflex_fast_state_dim)
        reflex_state_active = bool(
            reflex_state_enabled
            and (strict_hrdcr or not reflex_guard_disabled)
            and (
                reflex_feedback_enabled
                or reflex_injection_enabled
                or bool(cfg.persistent_memory_enabled)
                or (collect_reflex_aux_cache and bool(cfg.reflex_aux_store_fast_state))
            )
        )
        reflex_manager = (
            HRDCRStateManager(
                total_sequences,
                min(int(cfg.num_medusa_heads), int(self.medusa_heads.num_heads)),
                state_dim,
                device=device,
                half_life_tokens=float(cfg.reflex_half_life_tokens),
                alignment_beta=float(cfg.reflex_hint_quality_beta),
                trust_n0=float(cfg.reflex_trust_n0),
                sketch_rank=int(cfg.reflex_sketch_rank),
                sketch_seed=int(cfg.reflex_sketch_seed),
                min_effective_updates=float(cfg.reflex_min_effective_updates),
                min_alignment_count=float(cfg.reflex_min_alignment_count),
                min_state_rms=float(cfg.reflex_min_state_rms),
                state_reference_rms=float(cfg.reflex_state_reference_rms),
                alignment_floor=float(cfg.reflex_alignment_floor),
                alignment_full=float(cfg.reflex_alignment_full),
                alignment_lcb_z=float(cfg.reflex_alignment_lcb_z),
                safety_min_probe_count=int(cfg.reflex_safety_min_probe_count),
                safety_bad_probe_patience=int(cfg.reflex_safety_bad_probe_patience),
                safety_ratio_decay=float(cfg.reflex_safety_ratio_decay),
                safety_reenable_probe_interval=int(cfg.reflex_safety_reenable_probe_interval),
                state_ema_decay_by_head=cfg.reflex_state_ema_decay_by_head,
                enabled_at_start_by_head=cfg.reflex_enabled_at_start_by_head,
                min_effective_updates_by_head=cfg.reflex_min_effective_updates_by_head,
                min_alignment_count_by_head=cfg.reflex_min_alignment_count_by_head,
                safety_candidate_mass_deadband=float(
                    cfg.reflex_safety_candidate_mass_deadband
                ),
                safety_net_win_rate_deadband=float(
                    cfg.reflex_safety_net_win_rate_deadband
                ),
                safety_good_probe_patience=int(
                    cfg.reflex_safety_good_probe_patience
                ),
                safety_recovery_factor=float(cfg.reflex_safety_recovery_factor),
                safety_minimum_active_ratio=float(
                    cfg.reflex_safety_minimum_active_ratio
                ),
                head3_exploration_fraction=float(
                    cfg.sparse_head3_exploration_fraction
                ),
                head3_warmup_exploration_fraction=float(
                    cfg.sparse_head3_warmup_exploration_fraction
                ),
                head3_warmup_records=int(cfg.sparse_head3_warmup_records),
            )
            if (reflex_state_active and strict_hrdcr)
            else ReflexStateManager(
                total_sequences,
                state_dim,
                device=device,
                half_life_tokens=float(cfg.reflex_half_life_tokens),
                eta=float(cfg.reflex_eta),
                feedback_variance_beta=float(cfg.reflex_feedback_variance_beta),
                feedback_rms_clip=float(cfg.reflex_feedback_rms_clip),
                state_rms_clip=float(cfg.reflex_state_rms_clip),
                numerical_reset_rms=float(cfg.reflex_numerical_reset_rms),
                num_heads=min(int(cfg.num_medusa_heads), int(self.medusa_heads.num_heads)),
                horizon_resolved=bool(cfg.reflex_horizon_resolved),
                consensus_strength=float(cfg.reflex_consensus_strength),
                consensus_floor=float(cfg.reflex_consensus_floor),
                head_shrinkage_updates=float(cfg.reflex_head_shrinkage_updates),
                preconditioner_mix=float(cfg.reflex_preconditioner_mix),
                hint_quality_beta=float(cfg.reflex_hint_quality_beta),
                hint_quality_floor=float(cfg.reflex_hint_quality_floor),
                hint_quality_temperature=float(cfg.reflex_hint_quality_temperature),
                hint_cold_start=float(cfg.reflex_hint_cold_start),
                context_rank=int(cfg.reflex_context_rank),
                context_mix=float(cfg.reflex_context_mix),
                context_min_mass=float(cfg.reflex_context_min_mass),
                context_learning_rate=float(cfg.reflex_context_learning_rate),
            )
            if reflex_state_active
            else None
        )
        reflex_aux_buffer = (
            ReflexAuxiliaryRecordBuffer(max_records=int(cfg.reflex_aux_cache_max_records))
            if (collect_reflex_aux_cache and not strict_hrdcr)
            else None
        )
        if isinstance(reflex_manager, HRDCRStateManager):
            reflex_manager.load_safety_state(self._hrdcr_safety_state)
            if bool(cfg.reflex_dynamic_tree_enabled):
                hrdcr_base_nodes, hrdcr_base_layout_name, _ = (
                    reflex_manager.dynamic_tree_layout()
                )
            else:
                hrdcr_base_nodes = list(cfg.sparse_nodes_by_head)
                hrdcr_base_layout_name = "configured"
            head3_warm = (
                reflex_manager.num_heads >= 3
                and int(reflex_manager.mature_feedback_count[2].detach().cpu())
                < int(cfg.sparse_head3_warmup_records)
            )
            head3_exploration_fraction = (
                float(cfg.sparse_head3_warmup_exploration_fraction)
                if head3_warm
                else float(cfg.sparse_head3_exploration_fraction)
            )
            head3_exploration_period = max(
                1,
                int(round(1.0 / max(head3_exploration_fraction, 1e-9))),
            )
            hrdcr_layout_refresh_interval = 256
            hrdcr_last_layout_refresh_round = 0
        else:
            hrdcr_base_nodes = list(cfg.sparse_nodes_by_head)
            hrdcr_base_layout_name = "configured"
            head3_exploration_fraction = 0.0
            head3_exploration_period = 1
            hrdcr_layout_refresh_interval = 256
            hrdcr_last_layout_refresh_round = 0
        collect_sparse_teacher = bool(
            reflex_aux_buffer is not None and cfg.reflex_sparse_teacher_enabled
        )
        hrdcr_prediction_buffer = (
            HRDCRPredictionBuffer(
                proposal_topk=int(cfg.reflex_top_m_feedback),
                max_records=int(cfg.reflex_aux_cache_max_records),
            )
            if (strict_hrdcr and reflex_feedback_enabled)
            else None
        )
        hrdcr_aux_batches: list[dict[str, torch.Tensor]] = []
        coverage_prediction_buffer = CoveragePredictionBuffer() if coverage_feedback_enabled else None
        prediction_buffer = (
            PredictionBuffer()
            if (not strict_hrdcr and (distribution_feedback_enabled or collect_sparse_teacher))
            else None
        )
        hrdcr_feedback = (
            HRDCRFeedback(
                lm_head,
                num_heads=min(int(cfg.num_medusa_heads), int(self.medusa_heads.num_heads)),
                proposal_topk=int(cfg.reflex_top_m_feedback),
                target_topk=int(cfg.reflex_target_topk),
                support_cap=int(cfg.reflex_feedback_union_cap),
                temperature=float(cfg.reflex_feedback_temperature),
                distribution_weight=float(cfg.reflex_distribution_weight),
                coverage_weight=float(cfg.reflex_coverage_feedback_weight),
                boundary_width=int(cfg.reflex_boundary_width),
                severity_tv_weight=float(cfg.reflex_severity_tv_weight),
                severity_out_weight=float(cfg.reflex_severity_out_weight),
                severity_min=float(cfg.reflex_severity_min),
                sketch_projection=(
                    reflex_manager.sketch_projection
                    if isinstance(reflex_manager, HRDCRStateManager)
                    else None
                ),
            )
            if (strict_hrdcr and reflex_feedback_enabled and reflex_manager is not None)
            else None
        )
        lm_feedback = (
            LMHeadFeedback(
                lm_head,
                target_topk=int(cfg.reflex_target_topk),
                union_cap=int(cfg.reflex_feedback_union_cap),
                tv_gate_low=float(cfg.reflex_tv_gate_low),
                tv_gate_high=float(cfg.reflex_tv_gate_high),
                horizon_weight_decay=float(cfg.reflex_horizon_weight_decay),
                num_heads=min(int(cfg.num_medusa_heads), int(self.medusa_heads.num_heads)),
                feature_feedback_weight=float(cfg.reflex_feature_feedback_weight),
                feature_agreement_floor=float(cfg.reflex_feature_agreement_floor),
                coverage_feedback_weight=float(cfg.reflex_coverage_feedback_weight),
                feedback_objective=str(cfg.reflex_feedback_objective),
            )
            if (not strict_hrdcr and (reflex_feedback_enabled or collect_sparse_teacher))
            else None
        )
        reflex_stats = ReflexBatchStats(num_heads=min(cfg.num_medusa_heads, self.medusa_heads.num_heads))
        utility_scheduler = (
            self._get_verification_utility_scheduler()
            if reflex_enabled and not reflex_guard_disabled
            else None
        )
        head3_calibrator = self._get_head3_quality_calibrator()
        head3_calibrator_start = (
            head3_calibrator.snapshot() if head3_calibrator is not None else None
        )
        memory_feedback_accumulator = (
            PromptDepthFeedbackAccumulator(
                num_sequences=total_sequences,
                state_dim=state_dim,
                bucket_bounds=prompt_memory_bounds,
                device=device,
                min_weight=float(cfg.persistent_memory_min_weight),
            )
            if bool(cfg.persistent_memory_enabled) and reflex_feedback_enabled and reflex_manager is not None
            else None
        )

        total_acc_length = 0
        total_decoded_steps = 0
        total_accepted_medusa_tokens = 0
        total_proposed_medusa_tokens = 0
        total_correction_tokens = 0
        total_verify_rounds = 0
        active_batch_sum = 0
        tree_node_sum = 0
        tree_sample_count = 0
        total_tree_query_rows = 0
        total_tree_lm_head_rows = 0
        medusa_head_time = 0.0
        tree_verify_time = 0.0
        cache_update_time = 0.0
        kv_extraction_time = 0.0
        recompute_fallback_time = 0.0
        kv_extraction_success_count = 0
        kv_extraction_fallback_count = 0
        oom_count = 0
        accept_hist: dict[int, int] = {}
        accept_by_depth: dict[int, int] = {}
        proposed_by_depth: dict[int, int] = {}
        tree_nodes_by_head = [0 for _ in range(min(cfg.num_medusa_heads, self.medusa_heads.num_heads))]
        tree_budget_unused = 0
        tree_plan_last = {}
        actual_tree_layout_counts: dict[str, int] = {}
        head3_conditional_success = 0
        head3_conditional_opportunities = 0
        adaptive_tree_stats_last = {}
        reflex_feedback_collection_rounds = 0
        persistent_memory_active_sequence_sum = 0
        persistent_memory_rounds = 0
        persistent_memory_strength_max = 0.0
        persistent_memory_strength_sum = 0.0
        persistent_memory_gate_sum = 0.0
        sampled_head_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        sampled_verify_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

        while active_original_indices:
            active_bsz = len(active_original_indices)
            remaining = max_length - logical_lens
            old_logical_lens = logical_lens.clone()
            if reflex_manager is not None:
                reflex_state_start = time.perf_counter()
                if isinstance(reflex_manager, HRDCRStateManager):
                    active_context_keys = None
                    active_fast_state, active_effective_updates = (
                        reflex_manager.get_state_and_effective_updates(active_original_indices)
                    )
                    active_hint_trust = reflex_manager.trust(active_original_indices)
                    active_ratio_scale = reflex_manager.ratio_scale(active_original_indices)
                    active_important_ids, active_boundary_ids = (
                        reflex_manager.boundary_hints(active_original_indices)
                    )
                    active_state_sketch = reflex_manager.sketch(active_fast_state)
                else:
                    active_context_keys = (
                        self._reflex_context_keys(current_hidden)
                        if reflex_manager.context_rank > 0
                        else None
                    )
                    active_fast_state, active_effective_updates = reflex_manager.get_state_and_effective_updates(
                        active_original_indices,
                        context_keys=active_context_keys,
                        apply_hint_trust=not normalized_injection,
                    )
                    active_hint_trust = (
                        reflex_manager.get_hint_trust(active_original_indices)
                        if normalized_injection
                        else None
                    )
                    active_state_sketch = None
                    active_ratio_scale = None
                    active_important_ids = None
                    active_boundary_ids = None
                active_fast_state, memory_step_stats = self._apply_persistent_memory_prior(
                    active_fast_state,
                    active_effective_updates,
                    active_original_indices=active_original_indices,
                    logical_lens=logical_lens,
                    initial_logical_lens=initial_logical_lens,
                    persistent_memory=prompt_memory_tensor,
                    persistent_strength=prompt_memory_strength,
                )
                if memory_step_stats:
                    active_count = int(memory_step_stats.get("persistent_memory_active_sequences", 0) or 0)
                    if active_count > 0:
                        persistent_memory_active_sequence_sum += active_count
                        persistent_memory_rounds += 1
                        persistent_memory_strength_sum += float(memory_step_stats.get("persistent_memory_strength_mean", 0.0) or 0.0)
                        persistent_memory_gate_sum += float(memory_step_stats.get("persistent_memory_gate_mean", 0.0) or 0.0)
                        persistent_memory_strength_max = max(
                            persistent_memory_strength_max,
                            float(memory_step_stats.get("persistent_memory_strength_max", 0.0) or 0.0),
                        )
                self._add_reflex_time(time.perf_counter() - reflex_state_start)
            else:
                active_fast_state = None
                active_effective_updates = None
                active_context_keys = None
                active_hint_trust = None
                active_state_sketch = None
                active_ratio_scale = None
                active_important_ids = None
                active_boundary_ids = None

            pending_rows = [
                row
                for row, sequence_id in enumerate(active_original_indices)
                if int(sequence_id) in pending_root_tokens
            ]
            pending_row_set = set(pending_rows)
            fresh_rows = [row for row in range(active_bsz) if row not in pending_row_set]
            root_tokens = torch.empty((active_bsz,), dtype=torch.long, device=device)
            if fresh_rows:
                fresh_index = torch.as_tensor(fresh_rows, dtype=torch.long, device=device)
                fresh_tokens = sample_from_logits(
                    current_logits.index_select(0, fresh_index),
                    do_sample=cfg.do_sample,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                )
                root_tokens.index_copy_(0, fresh_index, fresh_tokens)
            if pending_rows:
                pending_index = torch.as_tensor(pending_rows, dtype=torch.long, device=device)
                pending_values = torch.as_tensor(
                    [pending_root_tokens.pop(int(active_original_indices[row])) for row in pending_rows],
                    dtype=torch.long,
                    device=device,
                )
                root_tokens.index_copy_(0, pending_index, pending_values)
                total_correction_tokens += len(pending_rows)

            if (
                isinstance(reflex_manager, HRDCRStateManager)
                and bool(cfg.reflex_dynamic_tree_enabled)
                and total_verify_rounds - hrdcr_last_layout_refresh_round
                >= hrdcr_layout_refresh_interval
            ):
                hrdcr_base_nodes, hrdcr_base_layout_name, _ = (
                    reflex_manager.dynamic_tree_layout()
                )
                hrdcr_last_layout_refresh_round = total_verify_rounds
                head3_warm = (
                    reflex_manager.num_heads >= 3
                    and int(
                        reflex_manager.mature_feedback_count[2].detach().cpu()
                    )
                    < int(cfg.sparse_head3_warmup_records)
                )
                head3_exploration_fraction = (
                    float(cfg.sparse_head3_warmup_exploration_fraction)
                    if head3_warm
                    else float(cfg.sparse_head3_exploration_fraction)
                )
                head3_exploration_period = max(
                    1,
                    int(
                        round(
                            1.0
                            / max(head3_exploration_fraction, 1e-9)
                        )
                    ),
                )

            use_medusa_tree = int(generation_step) >= int(cfg.enable_medusa_spec_after)
            if use_medusa_tree:
                sparse_nodes = list(hrdcr_base_nodes)
                layout_name = str(hrdcr_base_layout_name)
                force_head3_exploration = False
                if (
                    isinstance(reflex_manager, HRDCRStateManager)
                    and len(sparse_nodes) >= 3
                    and sparse_nodes[2] == 0
                    and head3_exploration_fraction > 0.0
                    and total_verify_rounds % head3_exploration_period == 0
                ):
                    # Exploration spends the same ten-node budget; it only
                    # reallocates one shallow node to gather causal Head-3 evidence.
                    sparse_nodes = [4, 4, 1]
                    layout_name = "head3_exploration"
                    force_head3_exploration = True
                plan = plan_tree(
                    active_batch_size=active_bsz,
                    num_medusa_heads=min(cfg.num_medusa_heads, self.medusa_heads.num_heads),
                    tree_mode=cfg.tree_mode,
                    tree_layout=cfg.tree_layout,
                    cpeak_nodes=cfg.cpeak_nodes,
                    min_tree_nodes_per_seq=cfg.min_tree_nodes_per_seq,
                    max_tree_nodes_per_seq=cfg.max_tree_nodes_per_seq,
                    max_tree_depth=cfg.max_tree_depth,
                    fixed_tree_topk_by_depth=list(cfg.fixed_tree_topk_by_depth),
                    adaptive_tree_enabled=bool(cfg.adaptive_tree_enabled),
                    adaptive_min_topk_by_depth=list(cfg.adaptive_min_topk_by_depth),
                    sparse_nodes_by_head=sparse_nodes,
                    sparse_min_head3_nodes=int(cfg.sparse_min_head3_nodes),
                    sparse_head3_min_budget=int(cfg.sparse_head3_min_budget),
                    sparse_branch_score_temperature=float(cfg.sparse_branch_score_temperature),
                    sparse_diversity_penalty=float(cfg.sparse_diversity_penalty),
                )
                head3_gate_result = None
                utility_plan_stats = {}
                if utility_scheduler is not None:
                    plan, utility_plan_stats = utility_scheduler.adapt(plan)
                use_chain = cfg.proposal_mode == "chain" and int(generation_step) >= int(cfg.chain_enable_after)
                sample_cuda_timing = bool(
                    device.type == "cuda"
                    and total_verify_rounds
                    % max(1, int(cfg.metrics_timing_sample_interval))
                    == 0
                )
                head_cuda_start = torch.cuda.Event(enable_timing=True) if sample_cuda_timing else None
                head_cuda_end = torch.cuda.Event(enable_timing=True) if sample_cuda_timing else None
                if head_cuda_start is not None:
                    head_cuda_start.record()
                if use_chain:
                    if statistical_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    head_start = time.time()
                    with torch.no_grad():
                        trees, plan, adaptive_tree_stats = self._build_chain_batch_trees(
                            root_tokens,
                            current_hidden.detach(),
                            plan,
                            lm_head=lm_head,
                            embedding_layer=base.get_input_embeddings(),
                            fast_state=active_fast_state,
                            effective_updates=active_effective_updates,
                            hint_trust=active_hint_trust,
                            ratio_scale=active_ratio_scale,
                            generation_step=generation_step,
                        )
                    if statistical_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    medusa_head_time += time.time() - head_start
                    medusa_logits = []
                    record_logits = adaptive_tree_stats.pop("record_logits", [])
                    record_hidden = []
                    raw_record_hidden = []
                else:
                    if statistical_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    head_start = time.time()
                    with torch.no_grad():
                        proposal_result = self._medusa_logits_for_last_hidden(
                            current_hidden.detach(),
                            lm_head=lm_head,
                            max_heads=plan.active_heads,
                            fast_state=self._scaled_fast_state(active_fast_state, generation_step),
                            effective_updates=active_effective_updates,
                            generation_step=generation_step,
                            hint_trust=active_hint_trust,
                            ratio_scale=active_ratio_scale,
                            important_ids=active_important_ids,
                            boundary_ids=active_boundary_ids,
                            root_tokens=root_tokens,
                            embedding_layer=base.get_input_embeddings(),
                            return_projected=True,
                            return_raw_projected=bool(strict_hrdcr),
                        )
                        if strict_hrdcr:
                            medusa_logits, record_hidden, raw_record_hidden = proposal_result
                        else:
                            medusa_logits, record_hidden = proposal_result
                            raw_record_hidden = record_hidden
                    if statistical_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    medusa_head_time += time.time() - head_start
                    plan, adaptive_tree_stats = self._adapt_plan_from_logits(medusa_logits, plan)
                    if (
                        head3_calibrator is not None
                        and plan.layout == "sparse_asymmetric"
                        and plan.active_heads >= 3
                        and len(medusa_logits) >= 3
                    ):
                        first_logp = (
                            medusa_logits[0].float().amax(dim=-1)
                            - torch.logsumexp(medusa_logits[0].float(), dim=-1)
                        )
                        second_logp = (
                            medusa_logits[1].float().amax(dim=-1)
                            - torch.logsumexp(medusa_logits[1].float(), dim=-1)
                        )
                        eligible = torch.full(
                            (active_bsz,),
                            bool(plan.node_budget_per_seq >= int(cfg.sparse_head3_min_budget)),
                            device=device,
                            dtype=torch.bool,
                        )
                        head3_gate_result = head3_calibrator.select(
                            medusa_logits[2],
                            cumulative_path_score=first_logp + second_logp,
                            eligible=eligible,
                        )
                        if force_head3_exploration:
                            head3_gate_result.gate_mask.fill_(True)
                            head3_gate_result.exploration_mask.fill_(True)
                    trees = build_batch_trees(
                        root_tokens,
                        medusa_logits,
                        plan,
                        head3_gate_mask=(
                            head3_gate_result.gate_mask if head3_gate_result is not None else None
                        ),
                        head3_exploration_mask=(
                            head3_gate_result.exploration_mask if head3_gate_result is not None else None
                        ),
                        head3_quality=(
                            head3_gate_result.quality if head3_gate_result is not None else None
                        ),
                    )
                    record_logits = medusa_logits
                actual_tree_layout_counts[layout_name] = (
                    actual_tree_layout_counts.get(layout_name, 0) + active_bsz
                )
                if head_cuda_end is not None:
                    head_cuda_end.record()
                    sampled_head_events.append((head_cuda_start, head_cuda_end))
                proposal_mode_used = "chain" if use_chain else "medusa"
                if utility_plan_stats:
                    adaptive_tree_stats["verification_utility"] = utility_plan_stats
                adaptive_tree_stats_last = adaptive_tree_stats
            else:
                medusa_logits = []
                record_logits = []
                record_hidden = []
                raw_record_hidden = []
                proposal_mode_used = "target_only"
                adaptive_tree_stats_last = {}
                head3_gate_result = None
                plan = TreePlan(
                    node_budget_per_seq=1,
                    active_heads=0,
                    topk_by_depth=[],
                    actual_nodes=1,
                    mode=cfg.tree_mode,
                    layout=cfg.tree_layout,
                )
                trees = build_batch_trees(root_tokens, medusa_logits, plan)
            active_original_tensor = None
            if (
                (prediction_buffer is not None and record_logits)
                or (hrdcr_prediction_buffer is not None and record_logits)
                or (reflex_aux_buffer is not None and plan.active_heads > 0)
            ):
                active_original_tensor = torch.as_tensor(active_original_indices, dtype=torch.long, device=device)
            feedback_stride = self._dynamic_feedback_stride(
                feedback_stride=int(cfg.reflex_feedback_stride),
                feedback_stride_min=int(cfg.reflex_feedback_stride_min),
                active_batch_size=active_bsz,
                initial_batch_size=total_sequences,
            )
            feedback_round = total_verify_rounds % feedback_stride == 0
            collect_coverage_this_round = (
                coverage_prediction_buffer is not None
                and bool(record_logits)
                and feedback_round
            )
            collect_distribution_this_round = bool(
                distribution_feedback_enabled
                and prediction_buffer is not None
                and record_logits
                and feedback_round
            )
            aux_cache_round = bool(
                reflex_aux_buffer is not None
                and plan.active_heads > 0
                and total_verify_rounds % max(1, int(cfg.reflex_aux_cache_stride)) == 0
            )
            teacher_round = bool(collect_sparse_teacher and aux_cache_round)
            collect_hrdcr_this_round = bool(
                hrdcr_prediction_buffer is not None
                and record_logits
                and (feedback_round or (collect_reflex_aux_cache and aux_cache_round))
            )
            if collect_coverage_this_round or collect_distribution_this_round or collect_hrdcr_this_round:
                reflex_feedback_collection_rounds += 1
            if collect_hrdcr_this_round:
                hrdcr_start = time.perf_counter()
                if active_fast_state is None or active_hint_trust is None or active_state_sketch is None:
                    raise RuntimeError("Strict HRDCR requires per-head state, trust, and proposal-time sketches")
                effective_candidate_ids, effective_candidate_valid = candidate_sets_by_head(
                    trees,
                    plan.active_heads,
                    device=device,
                    widths=list(plan.nodes_by_head or plan.topk_by_depth),
                )
                quality_by_horizon = [
                    torch.zeros((active_bsz,), device=device, dtype=torch.float32)
                    for _ in range(plan.active_heads)
                ]
                if head3_gate_result is not None and plan.active_heads >= 3:
                    quality_by_horizon[2] = head3_gate_result.quality

                probe_rows = None
                probe_head_idx = -1
                raw_candidate_ids: list[torch.Tensor | None] | None = None
                raw_candidate_valid: list[torch.Tensor | None] | None = None
                probe_interval = max(1, int(cfg.reflex_candidate_probe_interval))
                normal_probe_round = bool(
                    total_verify_rounds > 0
                    and total_verify_rounds % probe_interval == 0
                )
                if isinstance(reflex_manager, HRDCRStateManager):
                    probe_head_idx = reflex_manager.select_probe_head(
                        normal_probe_round
                    )
                strict_probe_round = bool(
                    reflex_injection_enabled
                    and proposal_mode_used == "medusa"
                    and 0 <= probe_head_idx < plan.active_heads
                    and probe_head_idx < len(raw_record_hidden)
                )
                if strict_probe_round:
                    correction_rms = (
                        record_hidden[probe_head_idx].float()
                        - raw_record_hidden[probe_head_idx].float()
                    ).square().mean(dim=-1).sqrt()
                    probe_rows = correction_rms.gt(0.0).nonzero(
                        as_tuple=False
                    ).flatten()[: max(1, int(cfg.reflex_counterfactual_max_sequences))]
                if strict_probe_round and probe_rows is not None and probe_rows.numel() > 0:
                    raw_probe_hidden = raw_record_hidden[probe_head_idx].index_select(
                        0, probe_rows
                    )
                    output_layer = self.medusa_heads.heads[probe_head_idx].output
                    raw_probe_logits = (
                        output_layer(raw_probe_hidden)
                        if output_layer is not None
                        else lm_head(
                            raw_probe_hidden.to(dtype=lm_head.weight.dtype)
                        )
                    )
                    raw_tree_logits = [
                        logits.index_select(0, probe_rows)
                        for logits in record_logits[: plan.active_heads]
                    ]
                    raw_tree_logits[probe_head_idx] = raw_probe_logits
                    raw_trees = build_batch_trees(
                        root_tokens.index_select(0, probe_rows),
                        raw_tree_logits,
                        plan,
                        head3_gate_mask=(
                            head3_gate_result.gate_mask.index_select(0, probe_rows)
                            if head3_gate_result is not None
                            else None
                        ),
                        head3_exploration_mask=(
                            head3_gate_result.exploration_mask.index_select(0, probe_rows)
                            if head3_gate_result is not None
                            else None
                        ),
                        head3_quality=(
                            head3_gate_result.quality.index_select(0, probe_rows)
                            if head3_gate_result is not None
                            else None
                        ),
                    )
                    packed_raw_ids, packed_raw_valid = candidate_sets_by_head(
                        raw_trees,
                        plan.active_heads,
                        device=device,
                        widths=list(plan.nodes_by_head or plan.topk_by_depth),
                    )
                    raw_candidate_ids = [None for _ in range(plan.active_heads)]
                    raw_candidate_valid = [None for _ in range(plan.active_heads)]
                    raw_candidate_ids[probe_head_idx] = packed_raw_ids[probe_head_idx]
                    raw_candidate_valid[probe_head_idx] = packed_raw_valid[probe_head_idx]
                hrdcr_prediction_buffer.add_from_logits(
                    sequence_ids=active_original_indices,
                    anchor_positions=old_logical_lens,
                    logits_by_horizon=record_logits[: plan.active_heads],
                    proposal_hidden_by_horizon=record_hidden[: plan.active_heads],
                    candidate_topk_by_horizon=plan.topk_by_depth[: plan.active_heads],
                    anchor_hidden=current_hidden.detach(),
                    fast_states=active_fast_state,
                    trust=active_hint_trust,
                    ratio_scales=active_ratio_scale,
                    state_sketch=active_state_sketch,
                    candidate_ids_by_horizon=effective_candidate_ids,
                    candidate_valid_by_horizon=effective_candidate_valid,
                    quality_by_horizon=quality_by_horizon,
                    probe_rows=probe_rows,
                    probe_head_idx=probe_head_idx,
                    raw_proposal_hidden_by_horizon=raw_record_hidden,
                    raw_candidate_ids_by_horizon=raw_candidate_ids,
                    raw_candidate_valid_by_horizon=raw_candidate_valid,
                )
                self._add_reflex_time(time.perf_counter() - hrdcr_start)
            if collect_coverage_this_round:
                coverage_start = time.perf_counter()
                baseline_logits = None
                probe_interval = max(16, min(32, int(cfg.reflex_candidate_probe_interval)))
                probe_round = bool(
                    reflex_injection_enabled
                    and proposal_mode_used == "medusa"
                    and total_verify_rounds > 0
                    and total_verify_rounds % probe_interval == 0
                )
                if probe_round:
                    baseline_logits = self._medusa_logits_for_last_hidden(
                        current_hidden.detach(),
                        lm_head=lm_head,
                        max_heads=plan.active_heads,
                        fast_state=None,
                        effective_updates=None,
                        generation_step=generation_step,
                        hint_trust=None,
                        root_tokens=root_tokens,
                        embedding_layer=base.get_input_embeddings(),
                    )
                changed, probed = coverage_prediction_buffer.add_from_logits(
                    sequence_ids=active_original_indices,
                    anchor_positions=old_logical_lens,
                    logits_by_horizon=record_logits[: plan.active_heads],
                    candidate_topk_by_horizon=plan.topk_by_depth[: plan.active_heads],
                    baseline_logits_by_horizon=baseline_logits,
                )
                if probe_round:
                    reflex_stats.add_candidate_probe(changed, probed)
                self._add_reflex_time(time.perf_counter() - coverage_start)
            if collect_distribution_this_round or teacher_round:
                legacy_prediction_start = time.perf_counter()
                raw_fast_hints = (
                    reflex_manager.get_raw_horizon_state(
                        active_original_indices,
                        context_keys=active_context_keys,
                    )
                    if (
                        collect_distribution_this_round
                        and reflex_manager is not None
                        and reflex_manager.horizon_resolved
                    )
                    else None
                )
                prediction_buffer.add_from_logits(
                    sequence_ids=active_original_indices,
                    anchor_positions=old_logical_lens,
                    logits_by_horizon=record_logits[: plan.active_heads],
                    top_m=int(cfg.reflex_top_m_feedback),
                    initial_lengths=initial_logical_lens.index_select(0, active_original_tensor),
                    hidden_by_horizon=(
                        record_hidden[: plan.active_heads]
                        if (
                            collect_distribution_this_round
                            and float(cfg.reflex_feature_feedback_weight) > 0.0
                        )
                        else None
                    ),
                    context_keys=(active_context_keys if collect_distribution_this_round else None),
                    fast_hints=raw_fast_hints,
                    candidate_topk_by_horizon=plan.topk_by_depth[: plan.active_heads],
                    probabilities_required=True,
                )
                self._add_reflex_time(time.perf_counter() - legacy_prediction_start)
            if (
                reflex_aux_buffer is not None
                and plan.active_heads > 0
                and aux_cache_round
            ):
                reflex_aux_buffer.add_anchor_predictions(
                    sequence_ids=active_original_indices,
                    anchor_positions=old_logical_lens,
                    initial_lengths=initial_logical_lens.index_select(0, active_original_tensor),
                    hidden_states=current_hidden.detach(),
                    fast_states=(
                        self._scaled_fast_state(active_fast_state, generation_step)
                        if bool(cfg.reflex_aux_store_fast_state)
                        else None
                    ),
                    max_horizon=min(int(plan.active_heads) + 1, self.medusa_heads.num_heads + 1),
                    reflex_scale=float(self._reflex_effective_injection_scale(generation_step)),
                )
            tree_plan_last = {
                "B_cur": active_bsz,
                "node_budget_per_seq": plan.node_budget_per_seq,
                "active_heads": plan.active_heads,
                "topk_by_depth": plan.topk_by_depth,
                "actual_nodes": plan.actual_nodes,
                "proposal_mode": proposal_mode_used,
                "adaptive_tree": adaptive_tree_stats_last,
                "nodes_by_head": list(plan.nodes_by_head),
            }
            active_batch_sum += active_bsz
            tree_node_sum += sum(tree.node_count for tree in trees) / max(active_bsz, 1)
            tree_sample_count += 1
            for tree in trees:
                counts = tree.nodes_by_head
                for head_idx, count in enumerate(counts[: len(tree_nodes_by_head)]):
                    tree_nodes_by_head[head_idx] += int(count)
                tree_budget_unused += max(0, int(plan.node_budget_per_seq) - int(tree.node_count))
                for depth in tree.depths:
                    if int(depth) >= 2:
                        proposed_by_depth[int(depth)] = proposed_by_depth.get(int(depth), 0) + 1

            tree_logits = None
            tree_logit_slots_cpu: list[list[int]] | None = None
            tree_hidden = None
            tree_past_key_values = None
            if plan.active_heads > 0 and max(tree.node_count for tree in trees) > 1:
                try:
                    tree_input_ids, tree_mask, tree_position_ids, _ = build_tree_attention_inputs(
                        trees,
                        full_attention_mask,
                        logical_lens,
                        pad_token_id=pad_token_id,
                        dtype=autocast_dtype(base),
                    )
                    total_tree_query_rows += int(tree_input_ids.shape[0] * tree_input_ids.shape[1])
                    if statistical_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    verify_start = time.time()
                    verify_cuda_start = (
                        torch.cuda.Event(enable_timing=True) if sample_cuda_timing else None
                    )
                    verify_cuda_end = (
                        torch.cuda.Event(enable_timing=True) if sample_cuda_timing else None
                    )
                    if verify_cuda_start is not None:
                        verify_cuda_start.record()
                    tree_out = forward_tree(
                        self.target_model,
                        tree_input_ids,
                        tree_mask,
                        past_key_values,
                        tree_position_ids,
                        clone_past=cfg.clone_tree_cache,
                        compute_logits=not bool(cfg.project_internal_tree_logits_only),
                    )
                    tree_hidden = tree_out["hidden_states"]
                    if cfg.project_internal_tree_logits_only:
                        tree_logits, tree_logit_slots_cpu = self._project_packed_tree_logits(
                            tree_hidden,
                            trees,
                            lm_head,
                        )
                    else:
                        tree_logits = tree_out["logits"]
                    total_tree_lm_head_rows += int(tree_logits.numel() // tree_logits.shape[-1])
                    if verify_cuda_end is not None:
                        verify_cuda_end.record()
                        sampled_verify_events.append((verify_cuda_start, verify_cuda_end))
                    if statistical_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    tree_verify_time += time.time() - verify_start
                    tree_past_key_values = tree_out["past_key_values"]
                    del tree_out
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    oom_count += 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    tree_logits = None
                    tree_logit_slots_cpu = None

            if tree_logits is None:
                raw_accepted_per_row = [[int(token)] for token in root_tokens.detach().cpu().tolist()]
                raw_accepted_nodes_per_row = [[0] for _ in trees]
                raw_parent_nodes = [0 for _ in trees]
                raw_correction_tokens = [None for _ in trees]
            else:
                (
                    raw_accepted_per_row,
                    raw_accepted_nodes_per_row,
                    raw_parent_nodes,
                    raw_correction_tokens,
                ) = exact_accept_paths_batch(
                    trees,
                    tree_logits,
                    do_sample=cfg.do_sample,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                    node_to_logit_cpu=tree_logit_slots_cpu,
                )

            remaining_cpu = remaining.detach().cpu().tolist()
            old_logical_lens_cpu = old_logical_lens.detach().cpu().tolist()
            accepted_per_row: list[list[int]] = []
            accepted_nodes_per_row: list[list[int]] = []
            finished_flags: list[bool] = []
            parent_nodes: list[int] = []
            for row, (raw_tokens, raw_nodes, raw_parent, raw_correction) in enumerate(
                zip(
                    raw_accepted_per_row,
                    raw_accepted_nodes_per_row,
                    raw_parent_nodes,
                    raw_correction_tokens,
                )
            ):
                accepted_tokens = list(raw_tokens)
                accepted_nodes = list(raw_nodes)
                parent = int(raw_parent)
                max_accept = max(1, int(remaining_cpu[row]))
                accepted_tokens = accepted_tokens[:max_accept]
                accepted_nodes = accepted_nodes[: len(accepted_tokens)]
                eos_seen = False
                for eos_pos, token in enumerate(accepted_tokens):
                    if token == eos_token_id:
                        accepted_tokens = accepted_tokens[: eos_pos + 1]
                        accepted_nodes = accepted_nodes[: eos_pos + 1]
                        eos_seen = True
                        break
                accepted_len = len(accepted_tokens)
                finished = eos_seen or int(old_logical_lens_cpu[row]) + accepted_len >= max_length
                if raw_correction is not None and accepted_len == len(raw_tokens) and not finished:
                    pending_root_tokens[int(active_original_indices[row])] = int(raw_correction)
                accept_hist[accepted_len] = accept_hist.get(accepted_len, 0) + 1
                for depth in range(2, accepted_len + 1):
                    accept_by_depth[depth] = accept_by_depth.get(depth, 0) + 1
                total_acc_length += accepted_len
                total_decoded_steps += 1
                total_accepted_medusa_tokens += max(accepted_len - 1, 0)
                total_proposed_medusa_tokens += max(tree.node_count - 1, 0)
                if trace_enabled:
                    sequence_id = int(active_original_indices[row])
                    bucket = trace_bucket(sequence_id, len(generated[sequence_id]))
                    bucket["verify_rounds"] += 1
                    bucket["accepted_length_sum"] += int(accepted_len)
                    bucket["accepted_medusa_tokens"] += max(accepted_len - 1, 0)
                    bucket["proposed_medusa_tokens"] += max(tree.node_count - 1, 0)
                    if active_fast_state is not None:
                        state_rms = float(
                            active_fast_state[row].float().square().mean().sqrt().detach().cpu()
                        )
                        bucket["fast_state_rms_sum"] += state_rms
                        bucket["fast_state_rms_observations"] += 1
                accepted_per_row.append(accepted_tokens)
                accepted_nodes_per_row.append([int(node_idx) for node_idx in accepted_nodes])
                finished_flags.append(finished)
                parent_nodes.append(parent)

            if utility_scheduler is not None:
                utility_scheduler.observe(accepted_per_row, trees)
            if isinstance(reflex_manager, HRDCRStateManager) and reflex_manager.num_heads >= 3:
                # Head-3 is useful only after its Head-1/2 parent path survives.
                # This measures that conditional event from the exact verifier path.
                round_head3_success = 0
                round_head3_opportunities = 0
                for tree, accepted_nodes in zip(trees, accepted_nodes_per_row):
                    if len(accepted_nodes) < 3:
                        continue
                    parent_node = int(accepted_nodes[2])
                    has_head3_child = any(
                        int(tree.depths[child]) == 4
                        for child in tree.children.get(parent_node, [])
                    )
                    if not has_head3_child:
                        continue
                    round_head3_opportunities += 1
                    if len(accepted_nodes) >= 4 and int(tree.depths[accepted_nodes[3]]) == 4:
                        round_head3_success += 1
                if round_head3_opportunities:
                    head3_conditional_success += round_head3_success
                    head3_conditional_opportunities += round_head3_opportunities
                    reflex_manager.observe_path_acceptance(
                        2,
                        accepted=round_head3_success,
                        opportunities=round_head3_opportunities,
                    )
            total_verify_rounds += 1
            max_acc = max(len(tokens) for tokens in accepted_per_row)
            accepted_ids_cpu = torch.full((active_bsz, max_acc), int(pad_token_id), dtype=torch.long)
            valid_ext_cpu = torch.zeros((active_bsz, max_acc), dtype=torch.long)
            position_ids_cpu = torch.zeros((active_bsz, max_acc), dtype=torch.long)
            for row, tokens in enumerate(accepted_per_row):
                accepted_ids_cpu[row, : len(tokens)] = torch.as_tensor(tokens, dtype=torch.long)
                valid_ext_cpu[row, : len(tokens)] = 1
                position_ids_cpu[row, : len(tokens)] = int(old_logical_lens_cpu[row]) + torch.arange(len(tokens))
                original_idx = active_original_indices[row]
                generated[original_idx].extend(tokens)
            accepted_ids = accepted_ids_cpu.to(device=device, non_blocking=True)
            valid_ext = valid_ext_cpu.to(device=device, non_blocking=True)
            position_ids = position_ids_cpu.to(device=device, non_blocking=True)

            if (
                hrdcr_prediction_buffer is not None
                and hrdcr_feedback is not None
                and isinstance(reflex_manager, HRDCRStateManager)
            ):
                for offset in range(1, max_acc + 1):
                    offset_rows = [
                        row for row, tokens in enumerate(accepted_per_row) if len(tokens) >= offset
                    ]
                    if not offset_rows:
                        continue
                    feedback_start = time.perf_counter()
                    row_index = torch.as_tensor(offset_rows, dtype=torch.long, device=device)
                    offset_seq_ids = [int(active_original_indices[row]) for row in offset_rows]
                    sequence_tensor = torch.as_tensor(offset_seq_ids, dtype=torch.long, device=device)
                    target_positions = old_logical_lens.index_select(0, row_index) + offset
                    mature = hrdcr_prediction_buffer.pop_mature(sequence_tensor, target_positions)
                    if mature.count == 0:
                        reflex_manager.decay_token(offset_seq_ids)
                        self._add_reflex_time(time.perf_counter() - feedback_start)
                        continue
                    target_rows: list[torch.Tensor] = []
                    for row in offset_rows:
                        if offset == 1:
                            target_rows.append(current_logits[row])
                        else:
                            if tree_logits is None:
                                raise RuntimeError("HRDCR feedback is missing target verification logits")
                            parent_node = int(accepted_nodes_per_row[row][offset - 2])
                            if tree_logit_slots_cpu is None:
                                target_rows.append(tree_logits[row, parent_node])
                            else:
                                target_rows.append(tree_logits[int(tree_logit_slots_cpu[row][parent_node])])
                    actual = accepted_ids.index_select(0, row_index)[:, offset - 1]
                    strict_feedback = hrdcr_feedback.compute(
                        mature,
                        torch.stack(target_rows, dim=0),
                        actual,
                        collect_auxiliary=bool(collect_reflex_aux_cache),
                    )
                    reflex_manager.observe_mature_feedback(
                        strict_feedback.record_head_indices
                    )
                    if head3_calibrator is not None:
                        head3_records = strict_feedback.record_head_indices.eq(2)
                        head3_calibrator.observe(
                            strict_feedback.record_quality[head3_records],
                            strict_feedback.record_candidate_hit[head3_records],
                            strict_feedback.record_candidate_regret[head3_records],
                        )
                        head3_probes = strict_feedback.probe_head_indices.eq(2)
                        head3_calibrator.observe_mass_gain(
                            strict_feedback.probe_quality[head3_probes],
                            (
                                strict_feedback.probe_effective_mass
                                - strict_feedback.probe_raw_mass
                            )[head3_probes],
                        )
                    reflex_manager.observe_counterfactual(strict_feedback)
                    feedback_rms = reflex_manager.advance_token(
                        offset_seq_ids,
                        strict_feedback.head_feedback,
                        strict_feedback.head_has_feedback,
                        strict_feedback.head_severity,
                        strict_feedback.head_alignment,
                        strict_feedback.head_alignment_observed,
                        strict_feedback.head_important_ids,
                        strict_feedback.head_boundary_ids,
                    )
                    if strict_feedback.auxiliary_records:
                        hrdcr_aux_batches.append(strict_feedback.auxiliary_records)
                        if len(hrdcr_aux_batches) >= 8:
                            # Keep the verifier reservoir bounded without a
                            # token-level host transfer or unbounded Python list.
                            hrdcr_aux_batches[:] = [
                                merge_auxiliary_records(
                                    hrdcr_aux_batches,
                                    int(cfg.reflex_aux_cache_max_records),
                                )
                            ]
                    reflex_stats.add_hrdcr_feedback(strict_feedback)
                    reflex_stats.add_feedback_rms(
                        feedback_rms,
                        strict_feedback.head_has_feedback.any(dim=-1),
                    )
                    self._add_reflex_time(time.perf_counter() - feedback_start)

            if (
                coverage_prediction_buffer is not None
                and lm_feedback is not None
                and reflex_manager is not None
            ):
                # Normal coverage feedback remains tensorized on the proposal
                # device. No proposal logits, logsumexp, or fast hints enter
                # this path.
                for offset in range(1, max_acc + 1):
                    offset_rows = [row for row, tokens in enumerate(accepted_per_row) if len(tokens) >= offset]
                    if not offset_rows:
                        continue
                    coverage_start = time.perf_counter()
                    row_index = torch.as_tensor(offset_rows, dtype=torch.long, device=device)
                    offset_seq_ids = [int(active_original_indices[row]) for row in offset_rows]
                    sequence_ids_tensor = torch.as_tensor(offset_seq_ids, dtype=torch.long, device=device)
                    target_positions = old_logical_lens.index_select(0, row_index) + offset
                    true_tokens_tensor = accepted_ids.index_select(0, row_index)[:, offset - 1]
                    coverage_records = coverage_prediction_buffer.pop_mature_batch(
                        sequence_ids_tensor,
                        target_positions,
                    )
                    sparse = lm_feedback.compute_coverage_tensors(
                        coverage_records,
                        true_tokens_tensor,
                        group_count=len(offset_rows),
                        compute_hidden_feedback=True,
                    )
                    reflex_stats.add_coverage_feedback(sparse)
                    feedback_rms = reflex_manager.advance_token(
                        offset_seq_ids,
                        sparse.feedback,
                        sparse.has_feedback,
                        sparse.effective_mass,
                        head_feedback=sparse.head_feedback,
                        head_has_feedback=sparse.head_has_feedback,
                        head_effective_mass=sparse.head_effective_mass,
                        head_context_keys=sparse.head_context_keys,
                        head_prediction_hint=sparse.head_prediction_hint,
                        head_hint_observed=sparse.head_hint_observed,
                        feedback_present=bool(coverage_records.group_indices.numel()),
                    )
                    if memory_feedback_accumulator is not None and coverage_records.group_indices.numel():
                        depth_values = [
                            int(old_logical_lens_cpu[row])
                            + offset
                            - int(initial_logical_lens_cpu[int(active_original_indices[row])])
                            for row in offset_rows
                        ]
                        memory_feedback_accumulator.add(
                            sequence_ids=offset_seq_ids,
                            depths=depth_values,
                            feedback=self._normalize_reflex_feedback(sparse.feedback),
                            has_feedback=sparse.has_feedback,
                            weights=sparse.effective_mass,
                        )
                    reflex_stats.add_feedback_rms(feedback_rms, sparse.has_feedback)
                    if trace_enabled:
                        active_feedback = sparse.has_feedback.detach().cpu().tolist()
                        for local_row, has_event in enumerate(active_feedback):
                            if not has_event:
                                continue
                            row = offset_rows[local_row]
                            sequence_id = int(active_original_indices[row])
                            generated_position = (
                                int(old_logical_lens_cpu[row])
                                - int(initial_logical_lens[sequence_id].item())
                                + offset
                                - 1
                            )
                            trace_bucket(sequence_id, generated_position)["feedback_events"] += 1
                    self._add_reflex_time(time.perf_counter() - coverage_start)

            aux_teachers: dict[tuple[int, int], dict] = {}
            if prediction_buffer is not None and lm_feedback is not None:
                # States advance once per actual token. Positions accepted in the
                # same verification round are processed in trajectory order.
                for offset in range(1, max_acc + 1):
                    legacy_feedback_start = time.perf_counter()
                    offset_rows = [row for row, tokens in enumerate(accepted_per_row) if len(tokens) >= offset]
                    if not offset_rows:
                        continue
                    offset_seq_ids = [int(active_original_indices[row]) for row in offset_rows]
                    feedback = (
                        torch.zeros(
                            (len(offset_rows), reflex_manager.fast_state_dim),
                            device=device,
                            dtype=torch.float32,
                        )
                        if distribution_feedback_enabled and reflex_manager is not None
                        else None
                    )
                    has_feedback = (
                        torch.zeros((len(offset_rows),), device=device, dtype=torch.bool)
                        if distribution_feedback_enabled and reflex_manager is not None
                        else None
                    )
                    effective_mass = (
                        torch.zeros((len(offset_rows),), device=device, dtype=torch.float32)
                        if distribution_feedback_enabled and reflex_manager is not None
                        else None
                    )
                    horizon_feedback = (
                        torch.zeros(
                            (
                                len(offset_rows),
                                reflex_manager.num_heads,
                                reflex_manager.fast_state_dim,
                            ),
                            device=device,
                            dtype=torch.float32,
                        )
                        if (
                            distribution_feedback_enabled
                            and reflex_manager is not None
                            and reflex_manager.horizon_resolved
                        )
                        else None
                    )
                    horizon_has_feedback = (
                        torch.zeros(
                            (len(offset_rows), reflex_manager.num_heads),
                            device=device,
                            dtype=torch.bool,
                        )
                        if (
                            distribution_feedback_enabled
                            and reflex_manager is not None
                            and reflex_manager.horizon_resolved
                        )
                        else None
                    )
                    horizon_effective_mass = (
                        torch.zeros(
                            (len(offset_rows), reflex_manager.num_heads),
                            device=device,
                            dtype=torch.float32,
                        )
                        if (
                            distribution_feedback_enabled
                            and reflex_manager is not None
                            and reflex_manager.horizon_resolved
                        )
                        else None
                    )
                    horizon_context_keys = (
                        torch.zeros(
                            (
                                len(offset_rows),
                                reflex_manager.num_heads,
                                reflex_manager.context_rank,
                            ),
                            device=device,
                            dtype=torch.float32,
                        )
                        if (
                            distribution_feedback_enabled
                            and reflex_manager is not None
                            and reflex_manager.horizon_resolved
                            and reflex_manager.context_rank > 0
                        )
                        else None
                    )
                    horizon_prediction_hint = (
                        torch.zeros_like(horizon_feedback)
                        if horizon_feedback is not None
                        else None
                    )
                    horizon_hint_observed = (
                        torch.zeros_like(horizon_has_feedback)
                        if horizon_has_feedback is not None
                        else None
                    )
                    record_groups: list[list] = []
                    target_rows: list[torch.Tensor] = []
                    target_hidden_rows: list[torch.Tensor] = []
                    true_tokens: list[int] = []
                    feedback_row_indices: list[int] = []
                    flat_records = []
                    flat_accepted_flags: list[bool] = []

                    for local_row, row in enumerate(offset_rows):
                        seq_id = int(active_original_indices[row])
                        anchor_pos = int(old_logical_lens_cpu[row])
                        token = int(accepted_per_row[row][offset - 1])
                        records = prediction_buffer.pop_mature(seq_id, anchor_pos + offset)
                        if not records:
                            continue
                        if offset == 1:
                            target_row = current_logits[row]
                            target_hidden_row = current_hidden[row]
                        else:
                            if tree_logits is None or tree_hidden is None:
                                raise RuntimeError("Accepted tree token is missing target verification state")
                            parent_node = int(accepted_nodes_per_row[row][offset - 2])
                            target_hidden_row = tree_hidden[row, parent_node]
                            if tree_logit_slots_cpu is None:
                                target_row = tree_logits[row, parent_node]
                            else:
                                target_row = tree_logits[int(tree_logit_slots_cpu[row][parent_node])]
                        record_groups.append(records)
                        target_rows.append(target_row)
                        target_hidden_rows.append(target_hidden_row)
                        true_tokens.append(token)
                        feedback_row_indices.append(local_row)
                        flat_records.extend(records)
                        flat_accepted_flags.extend(
                            record.anchor_pos == anchor_pos and int(record.horizon) <= len(accepted_per_row[row])
                            for record in records
                        )

                    if record_groups:
                        sparse = lm_feedback.compute_batch(
                            record_groups,
                            torch.stack(target_rows, dim=0),
                            true_tokens,
                            compute_hidden_feedback=(
                                distribution_feedback_enabled and reflex_manager is not None
                            ),
                            target_hidden=torch.stack(target_hidden_rows, dim=0),
                            compute_sparse_teacher=collect_sparse_teacher,
                        )
                        if distribution_feedback_enabled and reflex_manager is not None:
                            feedback_index = torch.as_tensor(feedback_row_indices, dtype=torch.long, device=device)
                            feedback.index_copy_(0, feedback_index, sparse.feedback)
                            has_feedback.index_copy_(0, feedback_index, sparse.has_feedback)
                            effective_mass.index_copy_(0, feedback_index, sparse.effective_mass)
                            if horizon_feedback is not None:
                                horizon_feedback.index_copy_(0, feedback_index, sparse.head_feedback)
                                horizon_has_feedback.index_copy_(
                                    0,
                                    feedback_index,
                                    sparse.head_has_feedback,
                                )
                                horizon_effective_mass.index_copy_(
                                    0,
                                    feedback_index,
                                    sparse.head_effective_mass,
                                )
                                horizon_prediction_hint.index_copy_(
                                    0,
                                    feedback_index,
                                    sparse.head_prediction_hint,
                                )
                                horizon_hint_observed.index_copy_(
                                    0,
                                    feedback_index,
                                    sparse.head_hint_observed,
                                )
                                if horizon_context_keys is not None:
                                    horizon_context_keys.index_copy_(
                                        0,
                                        feedback_index,
                                        sparse.head_context_keys,
                                    )
                        if distribution_feedback_enabled:
                            reflex_stats.add_records(
                                flat_records,
                                sparse.record_true_probs,
                                flat_accepted_flags,
                                sparse.record_tv,
                                sparse.record_gates,
                            )
                            reflex_stats.add_feature_alignment(
                                sparse.feature_agreement,
                                sparse.feature_gate,
                            )
                        if trace_enabled and distribution_feedback_enabled:
                            for group_idx, local_row in enumerate(feedback_row_indices):
                                if not bool(sparse.has_feedback[group_idx].item()):
                                    continue
                                row = offset_rows[local_row]
                                sequence_id = int(active_original_indices[row])
                                generated_position = (
                                    int(old_logical_lens_cpu[row])
                                    - int(initial_logical_lens[sequence_id].item())
                                    + offset
                                    - 1
                                )
                                trace_bucket(sequence_id, generated_position)["feedback_events"] += 1
                        if reflex_aux_buffer is not None:
                            for group_idx, local_row in enumerate(feedback_row_indices):
                                row = offset_rows[local_row]
                                seq_id = int(active_original_indices[row])
                                target_pos = int(old_logical_lens_cpu[row]) + offset
                                aux_teachers[(seq_id, target_pos)] = {
                                    "target_top_ids": sparse.target_top_ids[group_idx],
                                    "target_top_logits": sparse.target_top_logits[group_idx],
                                    "target_logsumexp": float(sparse.target_logsumexp[group_idx]),
                                    "proposal_records": record_groups[group_idx],
                                }

                    if distribution_feedback_enabled and reflex_manager is not None:
                        feedback_rms = reflex_manager.advance_token(
                            offset_seq_ids,
                            feedback,
                            has_feedback,
                            effective_mass,
                            head_feedback=horizon_feedback,
                            head_has_feedback=horizon_has_feedback,
                            head_effective_mass=horizon_effective_mass,
                            head_context_keys=horizon_context_keys,
                            head_prediction_hint=horizon_prediction_hint,
                            head_hint_observed=horizon_hint_observed,
                            feedback_present=bool(record_groups),
                        )
                        if memory_feedback_accumulator is not None and bool(record_groups):
                            depth_values = [
                                int(old_logical_lens_cpu[row]) + offset - int(initial_logical_lens_cpu[int(active_original_indices[row])])
                                for row in offset_rows
                            ]
                            memory_feedback_accumulator.add(
                                sequence_ids=offset_seq_ids,
                                depths=depth_values,
                                feedback=self._normalize_reflex_feedback(feedback),
                                has_feedback=has_feedback,
                                weights=effective_mass,
                            )
                        reflex_stats.add_feedback_rms(feedback_rms, has_feedback)
                    self._add_reflex_time(time.perf_counter() - legacy_feedback_start)
            if reflex_aux_buffer is not None:
                for row, tokens in enumerate(accepted_per_row):
                    seq_id = int(active_original_indices[row])
                    anchor_pos = int(old_logical_lens_cpu[row])
                    for offset, token in enumerate(tokens, start=1):
                        reflex_aux_buffer.pop_mature(
                            seq_id,
                            anchor_pos + offset,
                            generated[seq_id],
                            int(token),
                            teacher=aux_teachers.get((seq_id, anchor_pos + offset)),
                        )

            new_attention_mask = torch.cat([full_attention_mask, valid_ext], dim=1)
            use_extract = (
                cfg.cache_update_mode == "extract_path"
                and tree_logits is not None
                and tree_hidden is not None
                and tree_past_key_values is not None
            )
            extracted = False
            if use_extract:
                if statistical_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                extract_start = time.time()
                try:
                    result = extract_accepted_path_kv(
                        past_key_values,
                        tree_past_key_values,
                        accepted_nodes_per_row,
                        causal_lm=self.target_model,
                        compact_in_place=bool(cfg.inplace_kv_compaction),
                    )
                    past_key_values = result.past_key_values
                    row_idx = torch.arange(active_bsz, device=device)
                    last_nodes_cpu = [int(path[-1]) for path in accepted_nodes_per_row]
                    last_node_idx = torch.tensor(
                        last_nodes_cpu,
                        dtype=torch.long,
                        device=device,
                    )
                    current_hidden = tree_hidden[row_idx, last_node_idx, :]
                    if tree_logit_slots_cpu is None:
                        current_logits = tree_logits[row_idx, last_node_idx, :]
                    else:
                        slots = [int(tree_logit_slots_cpu[row][node]) for row, node in enumerate(last_nodes_cpu)]
                        leaf_rows = [row for row, slot in enumerate(slots) if slot < 0]
                        reusable_slots = torch.tensor(
                            [max(0, slot) for slot in slots],
                            dtype=torch.long,
                            device=device,
                        )
                        current_logits = tree_logits.index_select(0, reusable_slots)
                        if leaf_rows:
                            leaf_index = torch.tensor(leaf_rows, dtype=torch.long, device=device)
                            leaf_hidden = current_hidden.index_select(0, leaf_index)
                            with torch.amp.autocast(
                                "cuda" if device.type == "cuda" else device.type,
                                dtype=autocast_dtype(base),
                                enabled=(device.type == "cuda"),
                            ):
                                leaf_logits = lm_head(
                                    leaf_hidden.to(dtype=getattr(lm_head.weight, "dtype", leaf_hidden.dtype))
                                )
                            current_logits.index_copy_(0, leaf_index, leaf_logits)
                            total_tree_lm_head_rows += len(leaf_rows)
                            del leaf_index, leaf_hidden, leaf_logits
                        del reusable_slots
                    extracted = True
                    kv_extraction_success_count += 1
                    del result, row_idx, last_node_idx
                except Exception as exc:
                    kv_extraction_fallback_count += 1
                    if not cfg.allow_recompute_fallback:
                        raise RuntimeError("KV path extraction failed and allow_recompute_fallback=false") from exc
                    warnings.warn(f"KV path extraction failed; falling back to recompute_accepted: {exc}", RuntimeWarning)
                if statistical_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                kv_extraction_time += time.time() - extract_start
                cache_update_time += time.time() - extract_start
            elif cfg.cache_update_mode == "extract_path" and cfg.allow_recompute_fallback and plan.active_heads > 0:
                kv_extraction_fallback_count += 1

            if not extracted:
                if cfg.cache_update_mode == "extract_path" and not cfg.allow_recompute_fallback and plan.active_heads > 0:
                    raise RuntimeError("KV extraction was requested but no tree cache was available")
                if statistical_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                cache_start = time.time()
                cache_out = forward_tokens(
                    self.target_model,
                    accepted_ids,
                    new_attention_mask,
                    past_key_values,
                    position_ids,
                )
                if statistical_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - cache_start
                cache_update_time += elapsed
                if cfg.cache_update_mode == "extract_path" and plan.active_heads > 0:
                    recompute_fallback_time += elapsed
                past_key_values = cache_out["past_key_values"]
                token_hidden = cache_out["hidden_states"]
                last_indices = (valid_ext.sum(dim=-1) - 1).clamp_min(0).view(-1, 1, 1).expand(-1, 1, token_hidden.shape[-1])
                current_hidden = token_hidden.gather(1, last_indices).squeeze(1)
                with torch.amp.autocast(
                    "cuda" if device.type == "cuda" else device.type,
                    dtype=autocast_dtype(base),
                    enabled=(device.type == "cuda"),
                ):
                    current_logits = lm_head(current_hidden.to(dtype=getattr(lm_head.weight, "dtype", current_hidden.dtype)))
                del cache_out, token_hidden, last_indices
            tree_logits = None
            tree_logit_slots_cpu = None
            tree_hidden = None
            tree_past_key_values = None
            full_attention_mask = new_attention_mask
            logical_lens = logical_lens + valid_ext.sum(dim=-1)

            keep_rows = [idx for idx, done in enumerate(finished_flags) if not done]
            if len(keep_rows) != active_bsz:
                done_seq_ids = [active_original_indices[idx] for idx, done in enumerate(finished_flags) if done]
                if prediction_buffer is not None:
                    for seq_id in done_seq_ids:
                        prediction_buffer.clear_sequence(seq_id)
                if hrdcr_prediction_buffer is not None:
                    hrdcr_prediction_buffer.clear_sequences(done_seq_ids)
                if coverage_prediction_buffer is not None:
                    coverage_prediction_buffer.clear_sequences(done_seq_ids)
                if reflex_aux_buffer is not None:
                    for seq_id in done_seq_ids:
                        reflex_aux_buffer.clear_sequence(seq_id)
                if keep_rows:
                    keep = torch.tensor(keep_rows, dtype=torch.long, device=device)
                    past_key_values = select_cache_batch(past_key_values, keep, causal_lm=self.target_model)
                    current_hidden = current_hidden.index_select(0, keep)
                    current_logits = current_logits.index_select(0, keep)
                    full_attention_mask = full_attention_mask.index_select(0, keep)
                    logical_lens = logical_lens.index_select(0, keep)
                    active_original_indices = [active_original_indices[idx] for idx in keep_rows]
                else:
                    active_original_indices = []
                    break

        max_sequence_length = max((len(seq) for seq in generated), default=0)
        sampled_head_time = 0.0
        sampled_verify_time = 0.0
        if sampled_head_events or sampled_verify_events:
            torch.cuda.synchronize()
            sampled_head_time = sum(start.elapsed_time(end) for start, end in sampled_head_events) / 1000.0
            sampled_verify_time = sum(start.elapsed_time(end) for start, end in sampled_verify_events) / 1000.0
        total_time = time.time() - total_start
        accept_rate = total_accepted_medusa_tokens / max(total_proposed_medusa_tokens, 1)
        avg_accept = total_acc_length / max(total_decoded_steps, 1)
        self._update_reflex_degradation_guard(avg_accept, generation_step)
        reflex_metrics = reflex_stats.to_dict()
        reflex_metrics["enabled"] = bool(reflex_enabled)
        reflex_metrics["feedback_enabled"] = bool(reflex_feedback_enabled)
        reflex_metrics["proposal_injection_enabled"] = bool(reflex_injection_enabled)
        reflex_metrics["injection_gate_mode"] = str(cfg.reflex_injection_gate_mode)
        reflex_metrics["proposal_injection_scale"] = float(cfg.reflex_proposal_injection_scale)
        reflex_metrics["proposal_injection_effective_scale"] = float(self._reflex_effective_injection_scale(generation_step))
        reflex_metrics["anchor_conditioning_enabled"] = bool(
            cfg.reflex_anchor_conditioning_enabled
            and getattr(self.medusa_heads, "anchor_conditioner", None) is not None
        )
        reflex_metrics["aal_guard_baseline"] = float(self._reflex_guard_baseline)
        reflex_metrics["aal_guard_disabled_until"] = int(self._reflex_guard_disabled_until)
        reflex_metrics["aal_guard_bad_windows"] = int(self._reflex_guard_bad_windows)
        reflex_metrics["guard_disabled"] = bool(reflex_guard_disabled)
        reflex_metrics["pending_prediction_records"] = (
            (len(prediction_buffer) if prediction_buffer is not None else 0)
            + (len(coverage_prediction_buffer) if coverage_prediction_buffer is not None else 0)
            + (len(hrdcr_prediction_buffer) if hrdcr_prediction_buffer is not None else 0)
        )
        reflex_metrics["feedback_collection_rounds"] = int(reflex_feedback_collection_rounds)
        reflex_metrics["feedback_collection_fraction"] = reflex_feedback_collection_rounds / max(total_verify_rounds, 1)
        reflex_metrics["utility_scheduler"] = (
            utility_scheduler.to_dict() if utility_scheduler is not None else {"enabled": False}
        )
        reflex_metrics["persistent_memory_enabled"] = bool(cfg.persistent_memory_enabled)
        reflex_metrics["persistent_memory_active_sequences"] = int(persistent_memory_active_sequence_sum)
        reflex_metrics["persistent_memory_active_rounds"] = int(persistent_memory_rounds)
        reflex_metrics["persistent_memory_strength_mean"] = persistent_memory_strength_sum / max(persistent_memory_rounds, 1)
        reflex_metrics["persistent_memory_strength_max"] = float(persistent_memory_strength_max)
        reflex_metrics["persistent_memory_gate_mean"] = persistent_memory_gate_sum / max(persistent_memory_rounds, 1)
        reflex_metrics["adaptation_mode"] = str(cfg.reflex_adaptation_mode)
        if isinstance(reflex_manager, HRDCRStateManager):
            reflex_metrics.update(reflex_manager.stats())
            self._hrdcr_safety_state = reflex_manager.safety_state()
        elif reflex_manager is not None:
            reflex_metrics.update(reflex_manager.norm_stats())
        else:
            reflex_metrics.update(
                {
                    "raw_fast_state_rms": 0.0,
                    "hint_trust": 0.0,
                    "fast_state_norm_mean": 0.0,
                    "fast_state_norm_p95": 0.0,
                }
            )
        reflex_metrics.update(self._finalize_reflex_runtime_stats())
        if head3_calibrator is not None:
            reflex_metrics.update(head3_calibrator.summary(head3_calibrator_start))
        head3_metrics = (reflex_metrics.get("per_head") or {}).get("3", {})
        reflex_metrics["head3_mature_records"] = int(head3_metrics.get("mature", 0) or 0)
        reflex_metrics["head3_accept_count"] = int(head3_metrics.get("accepted", 0) or 0)
        reflex_metrics["head3_acceptance_rate"] = float(
            head3_metrics.get("acceptance_rate", 0.0) or 0.0
        )
        reflex_metrics["head3_candidate_regret"] = float(
            head3_metrics.get("candidate_regret", 0.0) or 0.0
        )
        reflex_metrics["head3_restricted_kl"] = float(
            head3_metrics.get("restricted_kl", 0.0) or 0.0
        )
        reflex_metrics["head3_conditional_path_acceptance"] = (
            float(head3_conditional_success)
            / max(int(head3_conditional_opportunities), 1)
        )
        reflex_metrics["head3_conditional_path_success"] = int(
            head3_conditional_success
        )
        reflex_metrics["head3_conditional_path_opportunities"] = int(
            head3_conditional_opportunities
        )
        reflex_metrics["actual_tree_layout_counts"] = dict(
            actual_tree_layout_counts
        )
        motivation_trace = []
        if trace_enabled:
            for key in sorted(trace_buckets):
                trace_row = dict(trace_buckets[key])
                observations = int(trace_row.pop("fast_state_rms_observations"))
                rms_sum = float(trace_row.pop("fast_state_rms_sum"))
                trace_row["average_accept_length"] = float(trace_row["accepted_length_sum"]) / max(
                    int(trace_row["verify_rounds"]), 1
                )
                trace_row["medusa_acceptance_rate"] = float(trace_row["accepted_medusa_tokens"]) / max(
                    int(trace_row["proposed_medusa_tokens"]), 1
                )
                trace_row["fast_state_rms_mean"] = rms_sum / max(observations, 1)
                motivation_trace.append(trace_row)
        return {
            "generated_token_ids": generated,
            "max_sequence_length": max_sequence_length,
            "total_acc_length": int(total_acc_length),
            "average_accept_length": float(avg_accept),
            "accepted_tokens_per_medusa_step": float(avg_accept),
            "total_decoded_token_num": int(total_decoded_steps),
            "total_accepted_draft_tokens": int(total_accepted_medusa_tokens),
            "total_proposed_draft_tokens": int(total_proposed_medusa_tokens),
            "total_accepted_medusa_tokens": int(total_accepted_medusa_tokens),
            "total_proposed_medusa_tokens": int(total_proposed_medusa_tokens),
            "total_correction_tokens": int(total_correction_tokens),
            "correction_token_rate": total_correction_tokens / max(total_decoded_steps, 1),
            "draft_acceptance_rate": float(accept_rate),
            "medusa_acceptance_rate": float(accept_rate),
            "total_verify_rounds": int(total_verify_rounds),
            "average_active_batch_size": active_batch_sum / max(total_verify_rounds, 1),
            "average_tree_nodes_per_seq": tree_node_sum / max(tree_sample_count, 1),
            "tree_active_heads": max(
                (idx + 1 for idx, count in enumerate(tree_nodes_by_head) if count > 0),
                default=0,
            ),
            "tree_nodes_head1": int(tree_nodes_by_head[0]) if tree_nodes_by_head else 0,
            "tree_nodes_head2": int(tree_nodes_by_head[1]) if len(tree_nodes_by_head) > 1 else 0,
            "tree_nodes_head3": int(tree_nodes_by_head[2]) if len(tree_nodes_by_head) > 2 else 0,
            "tree_total_nodes": int(sum(tree_nodes_by_head) + total_decoded_steps),
            "tree_budget_unused": int(tree_budget_unused),
            "tree_query_rows": int(total_tree_query_rows),
            "tree_lm_head_rows": int(total_tree_lm_head_rows),
            "tree_lm_head_row_ratio": total_tree_lm_head_rows / max(total_tree_query_rows, 1),
            "accept_length_histogram": accept_hist,
            "medusa_accept_by_depth": accept_by_depth,
            "medusa_proposed_by_depth": proposed_by_depth,
            "last_tree_plan": tree_plan_last,
            "actual_tree_layout_counts": actual_tree_layout_counts,
            "head3_conditional_path_acceptance": (
                float(head3_conditional_success)
                / max(int(head3_conditional_opportunities), 1)
            ),
            "head3_conditional_path_success": int(
                head3_conditional_success
            ),
            "head3_conditional_path_opportunities": int(
                head3_conditional_opportunities
            ),
            "cache_update_mode": cfg.cache_update_mode,
            "kv_extraction_success_count": int(kv_extraction_success_count),
            "kv_extraction_fallback_count": int(kv_extraction_fallback_count),
            "kv_extraction_time": kv_extraction_time,
            "recompute_fallback_time": recompute_fallback_time,
            "oom_count": int(oom_count),
            "total_time_cost": total_time,
            "prefill_time_cost": prefill_time,
            "target_time_cost": prefill_time + tree_verify_time + cache_update_time,
            "tree_verify_time_cost": tree_verify_time,
            "cache_update_time_cost": cache_update_time,
            "medusa_head_time_cost": medusa_head_time,
            "sampled_medusa_head_cuda_time_s": sampled_head_time,
            "sampled_tree_verify_cuda_time_s": sampled_verify_time,
            "timing_sample_count": len(sampled_head_events),
            "draft_time_cost": medusa_head_time,
            "check_time_cost": 0.0,
            "reflex_metrics": reflex_metrics,
            "reflex_head_metrics": reflex_metrics.get("per_head", {}),
            "reflex_aux_records": (
                merge_auxiliary_records(
                    hrdcr_aux_batches,
                    int(cfg.reflex_aux_cache_max_records),
                )
                if strict_hrdcr
                else (reflex_aux_buffer.to_batch() if reflex_aux_buffer is not None else {})
            ),
            "persistent_memory_feedback": (
                memory_feedback_accumulator.to_batch()
                if memory_feedback_accumulator is not None
                else {}
            ),
            "motivation_trace": motivation_trace,
        }
