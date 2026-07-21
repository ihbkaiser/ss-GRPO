from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass

import torch

from flashgrpo_b200.decoding.acceptance import exact_accept_paths_batch, sample_from_logits
from flashgrpo_b200.decoding.kv_extraction import extract_accepted_path_kv
from flashgrpo_b200.decoding.medusa_tree import (
    CandidateTree,
    TreePlan,
    build_batch_trees,
    dense_node_count,
    fit_topk_to_budget,
    plan_tree,
)
from flashgrpo_b200.decoding.reflex import (
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
    cpeak_nodes: int = 64
    min_tree_nodes_per_seq: int = 1
    max_tree_nodes_per_seq: int = 16
    max_tree_depth: int = 4
    fixed_tree_topk_by_depth: tuple[int, ...] = (4, 3, 2)
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
    reflex_feedback_union_cap: int = 96
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
    reflex_horizon_resolved: bool = False
    reflex_consensus_strength: float = 0.25
    reflex_consensus_floor: float = 0.0
    reflex_head_shrinkage_updates: float = 8.0
    reflex_preconditioner_mix: float = 0.25
    reflex_hint_quality_beta: float = 0.90
    reflex_hint_quality_floor: float = 0.0
    reflex_hint_quality_temperature: float = 0.10
    reflex_hint_cold_start: float = 0.25
    reflex_context_rank: int = 0
    reflex_context_mix: float = 0.5
    reflex_context_min_mass: float = 1e-3
    reflex_context_learning_rate: float = 0.5
    reflex_context_seed: int = 17
    reflex_state_rms_clip: float = 2.0
    reflex_numerical_reset_rms: float = 2.5
    reflex_relative_rms_delta_base: float = 0.01
    reflex_horizon_delta_rule: str = "inverse_sqrt"
    reflex_warmup_effective_updates: float = 16.0
    reflex_magnitude_gate_floor: float = 0.25
    reflex_guard_calibration_rollouts: int = 20
    reflex_guard_aal_drop_fraction: float = 0.05
    reflex_guard_patience: int = 2
    reflex_guard_disable_rollouts: int = 50
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
        self._reflex_guard_baseline = 0.0
        self._reflex_guard_samples = 0
        self._reflex_guard_bad_windows = 0
        self._reflex_guard_disabled_until = -1
        self._verification_utility_scheduler: VerificationUtilityScheduler | None = None
        self._reflex_context_projection: torch.Tensor | None = None

    def reset_reflex_degradation_guard(self) -> None:
        self._reflex_guard_baseline = 0.0
        self._reflex_guard_samples = 0
        self._reflex_guard_bad_windows = 0
        self._reflex_guard_disabled_until = -1

    def reset_verification_utility_scheduler(self) -> None:
        """Forget proposal utility after auxiliary-head parameters change."""

        self._verification_utility_scheduler = None

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
        if not self._reflex_enabled() or not math.isfinite(float(average_accept_length)):
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

    def _apply_reflex_correction(
        self,
        base_hidden: torch.Tensor,
        fast_state: torch.Tensor | None,
        effective_updates: torch.Tensor | None,
        head_idx: int,
        generation_step: int,
    ) -> torch.Tensor:
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
        if cfg.reflex_state_space != "hidden":
            return self.medusa_heads.add_reflex_delta(
                base_hidden,
                fast_state,
                head_idx,
                max_norm=float(cfg.reflex_correction_clip_norm),
                scale=float(scale),
                normalize=bool(cfg.reflex_normalize_correction),
            )

        state = torch.nan_to_num(fast_state.float(), nan=0.0, posinf=0.0, neginf=0.0)
        state_rms = state.square().mean(dim=-1, keepdim=True).sqrt()
        base_rms = torch.nan_to_num(base_hidden.float(), nan=0.0, posinf=0.0, neginf=0.0).square().mean(
            dim=-1, keepdim=True
        ).sqrt()
        if effective_updates is None:
            effective_updates = state_rms.new_zeros((state_rms.shape[0],))
        warmup = max(float(cfg.reflex_warmup_effective_updates), 1e-6)
        warm_gate = 1.0 - torch.exp(-effective_updates.float().view(-1, 1) / warmup)
        magnitude_gate = state_rms / (state_rms + float(cfg.reflex_magnitude_gate_floor))
        if str(cfg.reflex_horizon_delta_rule) == "trust_calibrated":
            # Horizon-resolved hint quality already estimates whether historical
            # verifier gradients transfer to this head. Keep one common trust
            # region instead of penalizing deeper heads twice.
            delta_k = float(cfg.reflex_relative_rms_delta_base)
        else:
            delta_k = float(cfg.reflex_relative_rms_delta_base) / math.sqrt(max(1, int(head_idx) + 1))
        alpha = (
            float(scale)
            * warm_gate
            * magnitude_gate
            * delta_k
            * base_rms
            / state_rms.clamp_min(1e-6)
        )
        correction = alpha * state
        # This clamp is tensor-only and enforces the relative RMS safety cap.
        correction_rms = correction.square().mean(dim=-1, keepdim=True).sqrt()
        cap = 1.01 * float(scale) * delta_k * base_rms
        correction = correction * torch.clamp(cap / correction_rms.clamp_min(1e-6), max=1.0)
        if base_hidden.dim() == 3 and correction.dim() == 2:
            correction = correction.unsqueeze(1)
        return base_hidden + correction.to(device=base_hidden.device, dtype=base_hidden.dtype)

    def _medusa_logits_for_last_hidden(
        self,
        last_hidden: torch.Tensor,
        *,
        lm_head,
        max_heads: int,
        fast_state: torch.Tensor | None,
        effective_updates: torch.Tensor | None,
        generation_step: int,
        root_tokens: torch.Tensor | None = None,
        embedding_layer=None,
        return_projected: bool = False,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
        max_heads = max(0, min(int(max_heads), self.medusa_heads.num_heads))
        if max_heads == 0:
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
            for head_idx in range(max_heads):
                medusa_hidden = self.medusa_heads.heads[head_idx].project_hidden(last_hidden)
                if anchor_embeddings is not None:
                    medusa_hidden = self.medusa_heads.anchor_conditioner(
                        medusa_hidden,
                        anchor_embeddings,
                        head_idx,
                    )
                medusa_hidden = self._apply_reflex_correction(
                    medusa_hidden,
                    fast_state,
                    effective_updates,
                    head_idx,
                    generation_step,
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
                return (logits, projected) if return_projected else logits

            logits_by_head: list[torch.Tensor] = []
            for head_idx, medusa_hidden in enumerate(projected):
                output = self.medusa_heads.heads[head_idx].output
                if output is not None:
                    logits_by_head.append(output(medusa_hidden))
                else:
                    lm_dtype = getattr(lm_head.weight, "dtype", medusa_hidden.dtype)
                    logits_by_head.append(lm_head(medusa_hidden.to(dtype=lm_dtype)))
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
        reflex_injection_enabled = self._reflex_injection_enabled(generation_step)
        reflex_feedback_enabled = reflex_state_enabled and bool(cfg.reflex_feedback_enabled)
        collect_reflex_aux_cache = (
            bool(cfg.reflex_aux_cache_enabled)
            if collect_reflex_aux_cache is None
            else bool(collect_reflex_aux_cache)
        )
        # Auxiliary head refresh is an independent ablation axis. It may cache
        # verified hidden/teacher pairs even when m_t injection is disabled.
        collect_reflex_aux_cache = bool(collect_reflex_aux_cache)
        state_dim = int(current_hidden.shape[-1]) if cfg.reflex_state_space == "hidden" else int(cfg.reflex_fast_state_dim)
        reflex_state_active = bool(
            reflex_state_enabled
            and (
                reflex_feedback_enabled
                or reflex_injection_enabled
                or bool(cfg.persistent_memory_enabled)
                or (collect_reflex_aux_cache and bool(cfg.reflex_aux_store_fast_state))
            )
        )
        reflex_manager = (
            ReflexStateManager(
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
            if collect_reflex_aux_cache
            else None
        )
        collect_sparse_teacher = bool(
            reflex_aux_buffer is not None and cfg.reflex_sparse_teacher_enabled
        )
        prediction_buffer = PredictionBuffer() if (reflex_feedback_enabled or collect_sparse_teacher) else None
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
            if (reflex_feedback_enabled or collect_sparse_teacher)
            else None
        )
        reflex_stats = ReflexBatchStats(num_heads=min(cfg.num_medusa_heads, self.medusa_heads.num_heads))
        utility_scheduler = self._get_verification_utility_scheduler() if reflex_enabled else None
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
        tree_plan_last = {}
        adaptive_tree_stats_last = {}
        reflex_feedback_collection_rounds = 0
        persistent_memory_active_sequence_sum = 0
        persistent_memory_rounds = 0
        persistent_memory_strength_max = 0.0
        persistent_memory_strength_sum = 0.0
        persistent_memory_gate_sum = 0.0

        while active_original_indices:
            active_bsz = len(active_original_indices)
            remaining = max_length - logical_lens
            old_logical_lens = logical_lens.clone()
            if reflex_manager is not None:
                active_context_keys = (
                    self._reflex_context_keys(current_hidden)
                    if reflex_manager.context_rank > 0
                    else None
                )
                active_fast_state, active_effective_updates = reflex_manager.get_state_and_effective_updates(
                    active_original_indices,
                    context_keys=active_context_keys,
                )
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
            else:
                active_fast_state = None
                active_effective_updates = None
                active_context_keys = None

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

            use_medusa_tree = int(generation_step) >= int(cfg.enable_medusa_spec_after)
            if use_medusa_tree:
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
                )
                utility_plan_stats = {}
                if utility_scheduler is not None:
                    plan, utility_plan_stats = utility_scheduler.adapt(plan)
                use_chain = cfg.proposal_mode == "chain" and int(generation_step) >= int(cfg.chain_enable_after)
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
                            generation_step=generation_step,
                        )
                    if statistical_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    medusa_head_time += time.time() - head_start
                    medusa_logits = []
                    record_logits = adaptive_tree_stats.pop("record_logits", [])
                    record_hidden = []
                else:
                    if statistical_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    head_start = time.time()
                    with torch.no_grad():
                        medusa_logits, record_hidden = self._medusa_logits_for_last_hidden(
                            current_hidden.detach(),
                            lm_head=lm_head,
                            max_heads=plan.active_heads,
                            fast_state=self._scaled_fast_state(active_fast_state, generation_step),
                            effective_updates=active_effective_updates,
                            generation_step=generation_step,
                            root_tokens=root_tokens,
                            embedding_layer=base.get_input_embeddings(),
                            return_projected=True,
                        )
                    if statistical_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    medusa_head_time += time.time() - head_start
                    plan, adaptive_tree_stats = self._adapt_plan_from_logits(medusa_logits, plan)
                    trees = build_batch_trees(root_tokens, medusa_logits, plan)
                    record_logits = medusa_logits
                proposal_mode_used = "chain" if use_chain else "medusa"
                if utility_plan_stats:
                    adaptive_tree_stats["verification_utility"] = utility_plan_stats
                adaptive_tree_stats_last = adaptive_tree_stats
            else:
                medusa_logits = []
                record_logits = []
                record_hidden = []
                proposal_mode_used = "target_only"
                adaptive_tree_stats_last = {}
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
            if (prediction_buffer is not None and record_logits) or (reflex_aux_buffer is not None and plan.active_heads > 0):
                active_original_tensor = torch.as_tensor(active_original_indices, dtype=torch.long, device=device)
            feedback_stride = max(
                int(cfg.reflex_feedback_stride_min),
                int(math.ceil(max(1, int(cfg.reflex_feedback_stride)) * active_bsz / max(total_sequences, 1))),
            )
            collect_feedback_this_round = (
                prediction_buffer is not None
                and bool(record_logits)
                and total_verify_rounds % feedback_stride == 0
            )
            if collect_feedback_this_round:
                reflex_feedback_collection_rounds += 1
                raw_fast_hints = (
                    reflex_manager.get_raw_horizon_state(
                        active_original_indices,
                        context_keys=active_context_keys,
                    )
                    if reflex_manager is not None and reflex_manager.horizon_resolved
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
                        if float(cfg.reflex_feature_feedback_weight) > 0.0
                        else None
                    ),
                    context_keys=active_context_keys,
                    fast_hints=raw_fast_hints,
                    candidate_topk_by_horizon=plan.topk_by_depth[: plan.active_heads],
                    probabilities_required=(
                        str(cfg.reflex_feedback_objective) != "coverage"
                        or collect_sparse_teacher
                    ),
                )
            if (
                reflex_aux_buffer is not None
                and plan.active_heads > 0
                and total_verify_rounds % max(1, int(cfg.reflex_aux_cache_stride)) == 0
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
            }
            active_batch_sum += active_bsz
            tree_node_sum += sum(tree.node_count for tree in trees) / max(active_bsz, 1)
            tree_sample_count += 1
            for tree in trees:
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

            aux_teachers: dict[tuple[int, int], dict] = {}
            if prediction_buffer is not None and lm_feedback is not None:
                # States advance once per actual token. Positions accepted in the
                # same verification round are processed in trajectory order.
                for offset in range(1, max_acc + 1):
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
                        if reflex_manager is not None
                        else None
                    )
                    has_feedback = (
                        torch.zeros((len(offset_rows),), device=device, dtype=torch.bool)
                        if reflex_manager is not None
                        else None
                    )
                    effective_mass = (
                        torch.zeros((len(offset_rows),), device=device, dtype=torch.float32)
                        if reflex_manager is not None
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
                        if reflex_manager is not None and reflex_manager.horizon_resolved
                        else None
                    )
                    horizon_has_feedback = (
                        torch.zeros(
                            (len(offset_rows), reflex_manager.num_heads),
                            device=device,
                            dtype=torch.bool,
                        )
                        if reflex_manager is not None and reflex_manager.horizon_resolved
                        else None
                    )
                    horizon_effective_mass = (
                        torch.zeros(
                            (len(offset_rows), reflex_manager.num_heads),
                            device=device,
                            dtype=torch.float32,
                        )
                        if reflex_manager is not None and reflex_manager.horizon_resolved
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
                            reflex_manager is not None
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
                            compute_hidden_feedback=reflex_manager is not None,
                            target_hidden=torch.stack(target_hidden_rows, dim=0),
                            compute_sparse_teacher=collect_sparse_teacher,
                        )
                        if reflex_manager is not None:
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
                        if trace_enabled:
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

                    if reflex_manager is not None:
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
        total_time = time.time() - total_start
        accept_rate = total_accepted_medusa_tokens / max(total_proposed_medusa_tokens, 1)
        avg_accept = total_acc_length / max(total_decoded_steps, 1)
        self._update_reflex_degradation_guard(avg_accept, generation_step)
        reflex_metrics = reflex_stats.to_dict()
        reflex_metrics["enabled"] = bool(reflex_enabled)
        reflex_metrics["feedback_enabled"] = bool(reflex_feedback_enabled)
        reflex_metrics["proposal_injection_enabled"] = bool(reflex_injection_enabled)
        reflex_metrics["proposal_injection_scale"] = float(cfg.reflex_proposal_injection_scale)
        reflex_metrics["proposal_injection_effective_scale"] = float(self._reflex_effective_injection_scale(generation_step))
        reflex_metrics["anchor_conditioning_enabled"] = bool(
            cfg.reflex_anchor_conditioning_enabled
            and getattr(self.medusa_heads, "anchor_conditioner", None) is not None
        )
        reflex_metrics["aal_guard_baseline"] = float(self._reflex_guard_baseline)
        reflex_metrics["aal_guard_disabled_until"] = int(self._reflex_guard_disabled_until)
        reflex_metrics["aal_guard_bad_windows"] = int(self._reflex_guard_bad_windows)
        reflex_metrics["pending_prediction_records"] = len(prediction_buffer) if prediction_buffer is not None else 0
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
        if reflex_manager is not None:
            reflex_metrics.update(reflex_manager.norm_stats())
        else:
            reflex_metrics.update({"fast_state_norm_mean": 0.0, "fast_state_norm_p95": 0.0})
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
            "tree_query_rows": int(total_tree_query_rows),
            "tree_lm_head_rows": int(total_tree_lm_head_rows),
            "tree_lm_head_row_ratio": total_tree_lm_head_rows / max(total_tree_query_rows, 1),
            "accept_length_histogram": accept_hist,
            "medusa_accept_by_depth": accept_by_depth,
            "medusa_proposed_by_depth": proposed_by_depth,
            "last_tree_plan": tree_plan_last,
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
            "draft_time_cost": medusa_head_time,
            "check_time_cost": 0.0,
            "reflex_metrics": reflex_metrics,
            "reflex_head_metrics": reflex_metrics.get("per_head", {}),
            "reflex_aux_records": reflex_aux_buffer.to_batch() if reflex_aux_buffer is not None else {},
            "persistent_memory_feedback": (
                memory_feedback_accumulator.to_batch()
                if memory_feedback_accumulator is not None
                else {}
            ),
            "motivation_trace": motivation_trace,
        }
