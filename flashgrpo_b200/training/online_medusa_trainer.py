from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from flashgrpo_b200.models.qwen_flashgrpo_wrapper import autocast_dtype, unwrap_causal_lm


@dataclass
class OnlineMedusaConfig:
    medusa_lr: float = 5e-4
    medusa_weight_decay: float = 0.0
    medusa_train_every: int = 1
    medusa_update_steps_per_iter: int = 1
    medusa_microbatch_size: int = 1
    medusa_max_tokens_per_update: int = 8192
    medusa_loss_decay: float = 0.8
    medusa_loss_chunk_size: int = 128
    chain_loss_weight: float = 0.0
    chain_loss_max_depth: int = 3
    chain_bootstrap_from_medusa: bool = True
    grad_clip_norm: float = 1.0
    reflex_record_microbatch_size: int = 256
    reflex_correction_clip_norm: float = 1.0
    reflex_normalize_correction: bool = True
    rollback_nonfinite_update: bool = True
    refresh_distill_weight: float = 0.7
    refresh_tv_weight: float = 0.0
    refresh_coverage_weight: float = 0.0
    refresh_hard_token_weight: float = 0.3
    refresh_proximal_weight: float = 0.1
    anchor_conditioning_enabled: bool = False
    acceptance_tv_weight: float = 0.0
    acceptance_kl_weight: float = 0.0
    acceptance_distill_topk: int = 64
    acceptance_temperature: float = 1.0
    acceptance_coverage_weight: float = 0.0
    acceptance_candidate_topk_by_head: tuple[int, ...] = (4, 3, 2)
    acceptance_rank_temperature: float = 0.5
    sparse_support_cap: int = 48
    sparse_kl_weight: float = 0.25
    sparse_coverage_weight: float = 1.0
    sparse_proximal_weight: float = 0.05
    sparse_ranking_margin: float = 0.5
    sparse_min_expected_benefit: float = 1e-3
    sparse_relative_rms_delta: float = 0.02
    sparse_correction_ratio_min: float = 0.005
    sparse_correction_ratio_max: float = 0.020


class OnlineMedusaTrainer:
    def __init__(self, target_model, medusa_heads, optimizer, config: OnlineMedusaConfig):
        self.target_model = target_model
        self.medusa_heads = medusa_heads
        self.optimizer = optimizer
        self.config = config

    def _trainable_param_backup(self) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
        if not bool(self.config.rollback_nonfinite_update):
            return []
        return [(param, param.detach().clone()) for param in self.medusa_heads.parameters() if param.requires_grad]

    @staticmethod
    def _params_are_finite(module: torch.nn.Module) -> bool:
        with torch.no_grad():
            for param in module.parameters():
                if param.requires_grad and not bool(torch.isfinite(param.detach()).all().item()):
                    return False
        return True

    def _restore_backup(self, backup: list[tuple[torch.nn.Parameter, torch.Tensor]]) -> None:
        with torch.no_grad():
            for param, saved in backup:
                param.copy_(saved)
        self.optimizer.zero_grad(set_to_none=True)

    def update(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        head_weights: dict[int | str, float] | list[float] | tuple[float, ...] | None = None,
    ) -> dict:
        cfg = self.config
        if input_ids.numel() == 0:
            return {"medusa_loss": 0.0, "head_update_tokens": 0, "head_update_time": 0.0}
        start_time = time.time()
        device = next(self.medusa_heads.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        loss_mask = loss_mask.to(device) if loss_mask is not None else None
        max_total_tokens = int(cfg.medusa_max_tokens_per_update or 0)
        if max_total_tokens > 0 and int(attention_mask.sum().item()) > max_total_tokens:
            lengths = attention_mask.sum(dim=-1).long()
            # Randomized row subsampling keeps online head learning diverse
            # while bounding the expensive backbone hidden-state pass.
            perm = torch.randperm(input_ids.shape[0], device=device)
            selected = []
            running_tokens = 0
            for idx in perm.tolist():
                row_tokens = int(lengths[idx].item())
                if selected and running_tokens + row_tokens > max_total_tokens:
                    continue
                selected.append(idx)
                running_tokens += row_tokens
                if running_tokens >= max_total_tokens:
                    break
            if selected:
                keep = torch.tensor(sorted(selected), dtype=torch.long, device=device)
                input_ids = input_ids.index_select(0, keep)
                attention_mask = attention_mask.index_select(0, keep)
                loss_mask = loss_mask.index_select(0, keep) if loss_mask is not None else None
        base = unwrap_causal_lm(self.target_model)
        lm_head = base.lm_head
        embedding_layer = base.get_input_embeddings()
        self.medusa_heads.train()

        total_loss = 0.0
        total_tokens = 0
        per_head_sums: dict[str, float] = {}
        updates = 0
        max_rows = max(1, int(cfg.medusa_microbatch_size))
        rows = input_ids.shape[0]
        grad_denom = max(1, (rows + max_rows - 1) // max_rows)
        self.optimizer.zero_grad(set_to_none=True)

        for start in range(0, rows, max_rows):
            end = min(start + max_rows, rows)
            mb_input = input_ids[start:end]
            mb_mask = attention_mask[start:end]
            mb_loss_mask = loss_mask[start:end] if loss_mask is not None else None
            valid_tokens = int(mb_mask.sum().item())
            if cfg.medusa_max_tokens_per_update and valid_tokens > cfg.medusa_max_tokens_per_update:
                keep_len = max(2, int(cfg.medusa_max_tokens_per_update // max(1, end - start)))
                mb_input = mb_input[:, -keep_len:]
                mb_mask = mb_mask[:, -keep_len:]
                mb_loss_mask = mb_loss_mask[:, -keep_len:] if mb_loss_mask is not None else None
            with torch.no_grad():
                device_type = "cuda" if device.type == "cuda" else device.type
                with torch.amp.autocast(device_type, dtype=autocast_dtype(base), enabled=(device.type == "cuda")):
                    outputs = base.model(
                        input_ids=mb_input,
                        attention_mask=mb_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            loss, stats = self.medusa_heads.compute_loss(
                hidden_states.detach(),
                mb_input,
                mb_mask,
                lm_head=lm_head,
                loss_mask=mb_loss_mask,
                chunk_size=cfg.medusa_loss_chunk_size,
                chain_loss_weight=cfg.chain_loss_weight,
                chain_max_depth=cfg.chain_loss_max_depth,
                chain_bootstrap_from_medusa=cfg.chain_bootstrap_from_medusa,
                embedding_layer=embedding_layer,
                head_weights=head_weights,
                anchor_conditioning=bool(cfg.anchor_conditioning_enabled),
                acceptance_tv_weight=float(cfg.acceptance_tv_weight),
                acceptance_kl_weight=float(cfg.acceptance_kl_weight),
                acceptance_distill_topk=int(cfg.acceptance_distill_topk),
                acceptance_temperature=float(cfg.acceptance_temperature),
                acceptance_coverage_weight=float(cfg.acceptance_coverage_weight),
                acceptance_candidate_topk_by_head=tuple(cfg.acceptance_candidate_topk_by_head),
                acceptance_rank_temperature=float(cfg.acceptance_rank_temperature),
            )
            if not torch.isfinite(loss):
                continue
            if not loss.requires_grad:
                continue
            (loss / grad_denom).backward()
            total_loss += float(loss.detach().cpu())
            total_tokens += int(mb_mask.sum().item())
            updates += 1
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    per_head_sums[key] = per_head_sums.get(key, 0.0) + float(value)

        reverted_nonfinite = False
        if updates:
            backup = self._trainable_param_backup()
            torch.nn.utils.clip_grad_norm_(self.medusa_heads.parameters(), cfg.grad_clip_norm)
            self.optimizer.step()
            if backup and not self._params_are_finite(self.medusa_heads):
                self._restore_backup(backup)
                reverted_nonfinite = True
        self.optimizer.zero_grad(set_to_none=True)
        elapsed = time.time() - start_time
        out = {
            "medusa_loss": total_loss / max(updates, 1),
            "head_update_tokens": int(total_tokens),
            "head_update_time": elapsed,
            "head_update_tokens_per_sec": total_tokens / max(elapsed, 1e-9),
            "head_update_steps": int(updates),
            "head_update_reverted_nonfinite": bool(reverted_nonfinite),
        }
        if head_weights is not None:
            for idx in range(len(self.medusa_heads.heads)):
                if isinstance(head_weights, (list, tuple)):
                    value = float(head_weights[idx]) if idx < len(head_weights) else 0.0
                else:
                    value = float(
                        head_weights.get(str(idx + 1))
                        or head_weights.get(idx + 1)
                        or head_weights.get(str(idx))
                        or head_weights.get(idx)
                        or 0.0
                    )
                out[f"aux_weight_head_{idx + 1}"] = value
        for key, value in per_head_sums.items():
            out[key] = value / max(updates, 1)
        return out

    @staticmethod
    def _head_weight(head_weights, head_idx: int) -> float:
        if head_weights is None:
            return 1.0
        if isinstance(head_weights, (list, tuple)):
            return float(head_weights[head_idx]) if head_idx < len(head_weights) else 0.0
        return float(
            head_weights.get(str(head_idx + 1))
            or head_weights.get(head_idx + 1)
            or head_weights.get(str(head_idx))
            or head_weights.get(head_idx)
            or 0.0
        )

    @staticmethod
    def _lm_head_logits(hidden: torch.Tensor, lm_head) -> torch.Tensor:
        weight = lm_head.weight.detach()
        bias = getattr(lm_head, "bias", None)
        bias = bias.detach() if bias is not None else None
        return F.linear(hidden.to(dtype=weight.dtype), weight, bias)

    @staticmethod
    def _sparse_cross_entropy_with_tail(
        new_logits: torch.Tensor,
        support_ids: torch.Tensor,
        support_logits: torch.Tensor,
        support_logsumexp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = support_ids.ge(0)
        safe_ids = support_ids.clamp_min(0)
        new_log_z = torch.logsumexp(new_logits.float(), dim=-1, keepdim=True)
        selected_new_logp = torch.gather(new_logits.float(), -1, safe_ids) - new_log_z
        selected_new_prob = torch.exp(selected_new_logp).masked_fill(~valid, 0.0)
        teacher_prob = torch.exp(
            support_logits.float() - support_logsumexp.float().unsqueeze(-1)
        ).masked_fill(~valid, 0.0)
        teacher_tail = (1.0 - teacher_prob.sum(dim=-1)).clamp(min=1e-8, max=1.0)
        new_tail = (1.0 - selected_new_prob.sum(dim=-1)).clamp(min=1e-8, max=1.0)
        distill = -(
            (teacher_prob * selected_new_logp.masked_fill(~valid, 0.0)).sum(dim=-1)
            + teacher_tail * torch.log(new_tail)
        )
        tv = 0.5 * (
            (teacher_prob - selected_new_prob).abs().sum(dim=-1)
            + (teacher_tail - new_tail).abs()
        )
        return distill, tv

    @staticmethod
    def _sparse_proximal_kl_with_tail(
        new_logits: torch.Tensor,
        old_ids: torch.Tensor,
        old_logits: torch.Tensor,
        old_logsumexp: torch.Tensor,
    ) -> torch.Tensor:
        valid = old_ids.ge(0)
        safe_ids = old_ids.clamp_min(0)
        new_log_z = torch.logsumexp(new_logits.float(), dim=-1, keepdim=True)
        new_logp = torch.gather(new_logits.float(), -1, safe_ids) - new_log_z
        new_prob = torch.exp(new_logp).masked_fill(~valid, 0.0)
        old_logp = old_logits.float() - old_logsumexp.float().unsqueeze(-1)
        old_prob = torch.exp(old_logp).masked_fill(~valid, 0.0)
        old_tail = (1.0 - old_prob.sum(dim=-1)).clamp(min=1e-8, max=1.0)
        new_tail = (1.0 - new_prob.sum(dim=-1)).clamp(min=1e-8, max=1.0)
        return (
            old_prob * (old_logp.masked_fill(~valid, 0.0) - new_logp.masked_fill(~valid, 0.0))
        ).sum(dim=-1) + old_tail * (torch.log(old_tail) - torch.log(new_tail))

    @staticmethod
    def _sparse_candidate_coverage(
        new_logits: torch.Tensor,
        target_ids: torch.Tensor,
        target_logits: torch.Tensor,
        target_logsumexp: torch.Tensor,
        *,
        candidate_topk: int,
        rank_temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Straight-through target mass covered by new proposal candidates."""

        valid = target_ids.ge(0)
        safe_ids = target_ids.clamp_min(0)
        target_prob = torch.exp(
            target_logits.float() - target_logsumexp.float().unsqueeze(-1)
        ).masked_fill(~valid, 0.0)
        proposal = new_logits.float()
        max_k = min(max(1, int(candidate_topk)), int(proposal.shape[-1]))
        top_values, top_ids = torch.topk(proposal, k=max_k, dim=-1)
        support_logits = torch.gather(proposal, -1, safe_ids)
        rank_temperature = max(float(rank_temperature), 1e-4)
        losses = []
        masses = []
        for budget in range(1, max_k + 1):
            cutoff = top_values[:, budget - 1 : budget].detach()
            hard = (
                safe_ids.unsqueeze(-1).eq(top_ids[:, :budget].unsqueeze(-2)).any(dim=-1)
                & valid
            )
            soft = torch.sigmoid((support_logits - cutoff) / rank_temperature) * valid
            membership = hard.float() + soft - soft.detach()
            coverage = (target_prob * membership).sum(dim=-1).clamp(min=0.0, max=1.0)
            losses.append(1.0 - coverage)
            masses.append((target_prob * hard).sum(dim=-1))
        return torch.stack(losses).mean(dim=0), torch.stack(masses).mean(dim=0)

    def _project_with_reflex(
        self,
        hidden: torch.Tensor,
        fast_state: torch.Tensor,
        head_idx: int,
        *,
        update_fast_state_injections: bool,
        scale: torch.Tensor | float = 1.0,
        anchor_token_ids: torch.Tensor | None = None,
        embedding_layer=None,
    ) -> torch.Tensor:
        head = self.medusa_heads.heads[head_idx]
        projected = head.project_hidden(hidden)
        if (
            bool(self.config.anchor_conditioning_enabled)
            and getattr(self.medusa_heads, "anchor_conditioner", None) is not None
            and anchor_token_ids is not None
        ):
            if embedding_layer is None:
                raise ValueError("embedding_layer is required for anchor-conditioned refresh")
            anchor_embeddings = embedding_layer(anchor_token_ids.to(device=projected.device)).detach()
            projected = self.medusa_heads.anchor_conditioner(projected, anchor_embeddings, head_idx)
        if getattr(self.medusa_heads, "reflex_fast_state_dim", 0) <= 0 or fast_state.numel() == 0:
            return projected
        up = self.medusa_heads.reflex_up[head_idx]
        fast = fast_state.to(device=up.weight.device, dtype=up.weight.dtype)
        if update_fast_state_injections:
            delta = up(fast)
        else:
            delta = F.linear(fast, up.weight.detach(), None)
        if bool(self.config.reflex_normalize_correction):
            delta_float = torch.nan_to_num(delta.float(), nan=0.0, posinf=0.0, neginf=0.0)
            rms = delta_float.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            delta = (delta_float / rms).to(dtype=delta.dtype)
        if float(self.config.reflex_correction_clip_norm) > 0:
            norm = torch.nan_to_num(delta.float(), nan=0.0, posinf=0.0, neginf=0.0).norm(dim=-1, keepdim=True).clamp_min(1e-6)
            delta = delta * torch.clamp(float(self.config.reflex_correction_clip_norm) / norm, max=1.0).to(dtype=delta.dtype)
        if torch.is_tensor(scale):
            delta = delta * scale.to(device=delta.device, dtype=delta.dtype).view(-1, 1)
        elif float(scale) != 1.0:
            delta = delta * float(scale)
        return projected + delta.to(device=projected.device, dtype=projected.dtype)

    def _logits_from_hidden(self, hidden: torch.Tensor, head_idx: int, lm_head) -> torch.Tensor:
        output = self.medusa_heads.heads[head_idx].output
        if output is not None:
            return output(hidden)
        return self._lm_head_logits(hidden, lm_head)

    def _chain_logits_from_state(self, state: torch.Tensor, lm_head) -> torch.Tensor:
        return self._lm_head_logits(state, lm_head)

    def update_sparse_online(
        self,
        records: dict[str, torch.Tensor],
        *,
        records_per_update: int = 512,
        max_heads_per_update: int = 2,
        optimizer_steps: int = 1,
        all_heads: bool = False,
        min_records_per_selected_head: int = 64,
        head_sampling_weights: list[float] | tuple[float, ...] | None = None,
    ) -> dict:
        """Update only MEDUSA heads from cached sparse verifier supports."""

        hidden = records.get("hidden") if records else None
        if hidden is None or int(hidden.shape[0]) == 0:
            return {
                "medusa_loss": 0.0,
                "head_update_time": 0.0,
                "head_update_steps": 0,
                "aux_records_used": 0,
                "aux_selected_heads": [],
                "aux_expected_benefit": 0.0,
            }
        start_time = time.time()
        device = next(self.medusa_heads.parameters()).device
        head_indices = records["head_indices"].to(device=device, dtype=torch.long)
        candidate_mass = records["candidate_mass"].to(device=device, dtype=torch.float32)
        candidate_regret = records.get("candidate_regret", 1.0 - candidate_mass).to(
            device=device, dtype=torch.float32
        )
        restricted_kl = records.get("restricted_kl", torch.zeros_like(candidate_mass)).to(
            device=device, dtype=torch.float32
        )
        candidate_hit = records.get(
            "candidate_hit", torch.ones_like(candidate_mass, dtype=torch.bool)
        ).to(device=device, dtype=torch.bool)
        min_per_head = max(1, int(min_records_per_selected_head))
        sampling_weights = list(head_sampling_weights or [])
        available: list[tuple[float, int, int]] = []
        for head_idx in range(len(self.medusa_heads.heads)):
            mask = head_indices.eq(head_idx)
            count = int(mask.sum().detach().cpu())
            if count >= min_per_head:
                weight = float(sampling_weights[head_idx]) if head_idx < len(sampling_weights) else 1.0
                score = (
                    candidate_regret[mask].mean()
                    + 0.25 * restricted_kl[mask].mean()
                    + 0.25 * (~candidate_hit[mask]).float().mean()
                )
                benefit = float((weight * score).detach().cpu())
                available.append((benefit, head_idx, count))
        available.sort(reverse=True)
        if all_heads:
            selected_heads = [head for _, head, _ in available]
        else:
            selected_heads = [
                head
                for benefit, head, _ in available
                if benefit > float(self.config.sparse_min_expected_benefit)
            ][: max(1, int(max_heads_per_update))]
            # The deepest calibrated head gets one slot whenever it has enough
            # records. This prevents shallow-head volume from starving Head 3.
            deepest = max((head for _, head, _ in available), default=-1)
            if deepest >= 2 and deepest not in selected_heads:
                budget = max(1, int(max_heads_per_update))
                selected_heads = (selected_heads[: max(0, budget - 1)] + [deepest])[:budget]
        expected_benefit = max(
            (benefit for benefit, head, _ in available if head in selected_heads), default=0.0
        )
        if not selected_heads:
            return {
                "medusa_loss": 0.0,
                "head_update_time": time.time() - start_time,
                "head_update_steps": 0,
                "aux_records_used": 0,
                "aux_selected_heads": [],
                "aux_expected_benefit": float(expected_benefit),
                "aux_update_reason": "zero_expected_benefit",
            }

        limit = max(1, int(records_per_update))
        if limit < min_per_head:
            return {
                "medusa_loss": 0.0,
                "head_update_time": time.time() - start_time,
                "head_update_steps": 0,
                "aux_records_used": 0,
                "aux_selected_heads": [],
                "aux_expected_benefit": float(expected_benefit),
                "aux_update_reason": "record_budget_below_head_quota",
            }
        if limit < min_per_head * len(selected_heads):
            selected_heads = selected_heads[: max(1, limit // min_per_head)]
        quota_rows: list[torch.Tensor] = []
        remainder_rows: list[torch.Tensor] = []
        priority_all = candidate_regret + 0.25 * restricted_kl + 0.25 * (~candidate_hit).float()
        for head_idx in selected_heads:
            rows = head_indices.eq(head_idx).nonzero(as_tuple=False).flatten()
            weight = float(sampling_weights[head_idx]) if head_idx < len(sampling_weights) else 1.0
            priority = priority_all.index_select(0, rows) * weight
            quota = min(min_per_head, int(rows.numel()))
            chosen_local = torch.topk(priority, k=quota, sorted=False).indices
            chosen = rows.index_select(0, chosen_local)
            quota_rows.append(chosen)
            keep = torch.ones(int(rows.numel()), device=device, dtype=torch.bool)
            keep[chosen_local] = False
            remainder_rows.append(rows[keep])
        selected_rows = torch.cat(quota_rows, dim=0) if quota_rows else torch.empty(0, device=device, dtype=torch.long)
        remaining_budget = max(0, limit - int(selected_rows.numel()))
        remainder = torch.cat(remainder_rows, dim=0) if remainder_rows else torch.empty(0, device=device, dtype=torch.long)
        if remaining_budget > 0 and remainder.numel() > 0:
            weights = torch.ones_like(remainder, dtype=torch.float32)
            for head_idx in selected_heads:
                weight = float(sampling_weights[head_idx]) if head_idx < len(sampling_weights) else 1.0
                weights.masked_fill_(head_indices.index_select(0, remainder).eq(head_idx), weight)
            priority = priority_all.index_select(0, remainder) * weights
            extra = remainder.index_select(
                0, torch.topk(priority, k=min(remaining_budget, int(remainder.numel())), sorted=False).indices
            )
            selected_rows = torch.cat((selected_rows, extra), dim=0)
        selected_rows = selected_rows.sort().values
        selected = {
            key: value.index_select(0, selected_rows.to(device=value.device))
            for key, value in records.items()
        }
        total_records = int(selected_rows.numel())
        if total_records == 0:
            return {
                "medusa_loss": 0.0,
                "head_update_time": time.time() - start_time,
                "head_update_steps": 0,
                "aux_records_used": 0,
                "aux_selected_heads": selected_heads,
                "aux_expected_benefit": float(expected_benefit),
            }

        base = unwrap_causal_lm(self.target_model)
        lm_weight = base.lm_head.weight.detach()
        trainable: list[torch.nn.Parameter] = []
        for head_idx in selected_heads:
            trainable.extend(
                parameter
                for parameter in self.medusa_heads.heads[head_idx].parameters()
                if parameter.requires_grad
            )
        before = [(parameter, parameter.detach().clone()) for parameter in trainable]
        self.medusa_heads.train()
        total_loss = 0.0
        total_microbatches = 0
        performed_steps = 0
        micro = max(1, int(self.config.reflex_record_microbatch_size))

        for _ in range(max(1, int(optimizer_steps))):
            self.optimizer.zero_grad(set_to_none=True)
            step_microbatches = 0
            for head_idx in selected_heads:
                rows = selected["head_indices"].to(device=device, dtype=torch.long).eq(head_idx).nonzero(
                    as_tuple=False
                ).flatten()
                for offset in range(0, int(rows.numel()), micro):
                    index = rows[offset : offset + micro]
                    if index.numel() == 0:
                        continue
                    anchor = selected["hidden"].index_select(
                        0, index.to(device=selected["hidden"].device)
                    ).to(device=device, dtype=next(self.medusa_heads.parameters()).dtype)
                    projected = self.medusa_heads.heads[head_idx].project_hidden(anchor)
                    fast = selected["fast_state"].index_select(
                        0, index.to(device=selected["fast_state"].device)
                    ).to(device=device, dtype=torch.float32)
                    trust = selected["trust"].index_select(
                        0, index.to(device=selected["trust"].device)
                    ).to(device=device, dtype=torch.float32)
                    base_rms = projected.float().square().mean(dim=-1, keepdim=True).sqrt()
                    target_ratio = torch.where(
                        trust.unsqueeze(-1).gt(0.0),
                        (
                            float(self.config.sparse_correction_ratio_min)
                            + (
                                float(self.config.sparse_correction_ratio_max)
                                - float(self.config.sparse_correction_ratio_min)
                            )
                            * trust.unsqueeze(-1)
                        ),
                        torch.zeros_like(trust.unsqueeze(-1)),
                    )
                    correction = (
                        target_ratio
                        * base_rms
                        * (
                            fast
                            / fast.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
                        )
                    )
                    corrected = projected + correction.to(dtype=projected.dtype)

                    support_ids = selected["support_ids"].index_select(
                        0, index.to(device=selected["support_ids"].device)
                    ).to(device=device, dtype=torch.long)
                    support_valid = selected["support_valid"].index_select(
                        0, index.to(device=selected["support_valid"].device)
                    ).to(device=device, dtype=torch.bool)
                    support_weight = lm_weight.index_select(0, support_ids.clamp_min(0).reshape(-1)).view(
                        int(index.numel()), int(support_ids.shape[1]), int(lm_weight.shape[1])
                    )
                    logits = torch.einsum("rsh,rh->rs", support_weight, corrected.to(lm_weight.dtype)).float()
                    target_logits = selected["target_logits"].index_select(
                        0, index.to(device=selected["target_logits"].device)
                    ).to(device=device, dtype=torch.float32)
                    masked_target = target_logits.masked_fill(~support_valid, -torch.inf)
                    masked_logits = logits.masked_fill(~support_valid, -torch.inf)
                    p = torch.softmax(masked_target, dim=-1).masked_fill(~support_valid, 0.0)
                    log_p = torch.log_softmax(masked_target, dim=-1).masked_fill(~support_valid, 0.0)
                    log_q = torch.log_softmax(masked_logits, dim=-1).masked_fill(~support_valid, 0.0)
                    kl = (p * (log_p - log_q)).sum(dim=-1)

                    candidate_ids = selected["candidate_ids"].index_select(
                        0, index.to(device=selected["candidate_ids"].device)
                    ).to(device=device, dtype=torch.long)
                    candidate_valid = selected["candidate_valid"].index_select(
                        0, index.to(device=selected["candidate_valid"].device)
                    ).to(device=device, dtype=torch.bool)
                    in_candidate = (
                        support_ids.unsqueeze(-1).eq(candidate_ids.unsqueeze(1))
                        & support_valid.unsqueeze(-1)
                        & candidate_valid.unsqueeze(1)
                    ).any(dim=-1)
                    candidate_weight = lm_weight.index_select(
                        0, candidate_ids.clamp_min(0).reshape(-1)
                    ).view(int(index.numel()), int(candidate_ids.shape[1]), int(lm_weight.shape[1]))
                    candidate_logits = torch.einsum(
                        "rkh,rh->rk", candidate_weight, corrected.to(lm_weight.dtype)
                    ).float().masked_fill(~candidate_valid, -torch.inf)
                    boundary = candidate_logits.masked_fill(~candidate_valid, torch.inf).min(dim=-1).values
                    ranking = p * (~in_candidate) * F.softplus(
                        float(self.config.sparse_ranking_margin) + boundary.unsqueeze(-1) - logits
                    )
                    coverage = ranking.masked_fill(~support_valid, 0.0).sum(dim=-1)
                    loss = (
                        float(self.config.sparse_kl_weight) * kl.mean()
                        + float(self.config.sparse_coverage_weight) * coverage.mean()
                    )
                    if not torch.isfinite(loss) or not loss.requires_grad:
                        continue
                    (loss / max(1, len(selected_heads))).backward()
                    total_loss += float(loss.detach().cpu())
                    total_microbatches += 1
                    step_microbatches += 1
            if step_microbatches:
                torch.nn.utils.clip_grad_norm_(trainable, self.config.grad_clip_norm)
                self.optimizer.step()
                performed_steps += 1

        # One-step updates have zero gradient from an anchor penalty at phi_before.
        # A proximal post-step projection implements the intended trust region.
        proximal = max(0.0, float(self.config.sparse_proximal_weight))
        shrink = 1.0 / (1.0 + proximal)
        delta_ms = 0.0
        delta_count = 0
        with torch.no_grad():
            for parameter, old in before:
                delta = parameter - old
                parameter.copy_(old + shrink * delta)
                delta_ms += float((parameter - old).float().square().sum().detach().cpu())
                delta_count += parameter.numel()
        self.optimizer.zero_grad(set_to_none=True)
        elapsed = time.time() - start_time
        return {
            "medusa_loss": total_loss / max(total_microbatches, 1),
            "head_update_time": elapsed,
            "head_update_steps": int(performed_steps),
            "aux_optimizer_steps": int(performed_steps),
            "aux_records_used": int(total_records),
            "aux_selected_heads": [int(head + 1) for head in selected_heads],
            "aux_records_used_by_head": {
                str(int(head + 1)): int(
                    selected["head_indices"].to(device=device, dtype=torch.long).eq(head).sum().detach().cpu()
                )
                for head in selected_heads
            },
            "head3_aux_records_used": int(
                selected["head_indices"].to(device=device, dtype=torch.long).eq(2).sum().detach().cpu()
            ),
            "head3_aux_optimizer_steps": int(performed_steps if 2 in selected_heads else 0),
            "aux_expected_benefit": float(expected_benefit),
            "aux_parameter_delta_rms": (delta_ms / max(delta_count, 1)) ** 0.5,
            "refresh_committed": bool(performed_steps > 0),
            "aux_update_reason": "sparse_online" if performed_steps else "no_finite_sparse_loss",
        }

    def update_reflex_records(
        self,
        records: dict,
        *,
        head_weights: dict[int | str, float] | list[float] | tuple[float, ...] | None = None,
        update_fast_state_injections: bool = False,
    ) -> dict:
        cfg = self.config
        hidden_cpu = records.get("hidden") if records else None
        if hidden_cpu is None or int(hidden_cpu.shape[0]) == 0:
            return {"medusa_loss": 0.0, "head_update_tokens": 0, "head_update_time": 0.0, "head_update_steps": 0}
        start_time = time.time()
        device = next(self.medusa_heads.parameters()).device
        hidden_cpu = hidden_cpu.detach()
        total_records = int(hidden_cpu.shape[0])
        max_records = int(cfg.medusa_max_tokens_per_update or 0)
        if max_records > 0 and total_records > max_records:
            keep = torch.randperm(total_records)[:max_records]
            records = {
                key: value.index_select(0, keep) if torch.is_tensor(value) and value.shape[:1] == (total_records,) else value
                for key, value in records.items()
            }
            total_records = max_records

        base = unwrap_causal_lm(self.target_model)
        lm_head = base.lm_head
        embedding_layer = base.get_input_embeddings()
        self.medusa_heads.train()
        self.optimizer.zero_grad(set_to_none=True)

        micro = max(1, int(cfg.reflex_record_microbatch_size or 256))
        grad_denom = max(1, (total_records + micro - 1) // micro)
        total_loss = 0.0
        total_tokens = 0
        updates = 0
        stat_sums: dict[str, float] = {}

        for start in range(0, total_records, micro):
            end = min(start + micro, total_records)
            hidden = records["hidden"][start:end].to(device=device, dtype=next(self.medusa_heads.parameters()).dtype)
            fast = records["fast_state"][start:end].to(device=device, dtype=next(self.medusa_heads.parameters()).dtype)
            labels = records["labels"][start:end].to(device=device).long()
            horizons = records["horizons"][start:end].to(device=device).long()
            scales = records.get("reflex_scale")
            scales = scales[start:end].to(device=device, dtype=torch.float32) if torch.is_tensor(scales) else torch.ones((end - start,), device=device)
            prev_tokens = records["prev_tokens"][start:end].to(device=device).long()
            has_sparse_teacher = records.get("has_sparse_teacher")
            has_sparse_teacher = (
                has_sparse_teacher[start:end].to(device=device).bool()
                if torch.is_tensor(has_sparse_teacher)
                else torch.zeros((end - start,), device=device, dtype=torch.bool)
            )
            target_top_ids = records.get("target_top_ids")
            target_top_logits = records.get("target_top_logits")
            target_logsumexp = records.get("target_logsumexp")
            old_top_ids = records.get("old_top_ids")
            old_top_logits = records.get("old_top_logits")
            old_logsumexp = records.get("old_logsumexp")
            losses = []
            parallel_losses = []
            chain_losses = []
            weight_sum = 0.0

            for head_idx in range(len(self.medusa_heads.heads)):
                horizon = head_idx + 2
                mask = horizons.eq(horizon)
                if not bool(mask.any().item()):
                    continue
                aux_weight = self._head_weight(head_weights, head_idx)
                if aux_weight <= 0.0:
                    continue
                decay = float(cfg.medusa_loss_decay ** head_idx)
                h = hidden[mask]
                f = fast[mask]
                s = scales[mask]
                y = labels[mask]
                anchor_tokens = prev_tokens[mask, 0].clamp_min(0) if prev_tokens.shape[1] > 0 else None
                projected = self._project_with_reflex(
                    h,
                    f,
                    head_idx,
                    update_fast_state_injections=update_fast_state_injections,
                    scale=s,
                    anchor_token_ids=anchor_tokens,
                    embedding_layer=embedding_layer,
                )
                logits = self._logits_from_hidden(projected, head_idx, lm_head).float()
                hard_loss = F.cross_entropy(logits, y, reduction="none")
                selected_teacher = has_sparse_teacher[mask]
                per_record_loss = hard_loss
                if (
                    bool(selected_teacher.any().item())
                    and torch.is_tensor(target_top_ids)
                    and torch.is_tensor(target_top_logits)
                    and torch.is_tensor(target_logsumexp)
                ):
                    target_ids = target_top_ids[start:end].to(device=device).long()[mask]
                    target_values = target_top_logits[start:end].to(device=device).float()[mask]
                    target_lse = target_logsumexp[start:end].to(device=device).float()[mask]
                    distill, tv = self._sparse_cross_entropy_with_tail(
                        logits,
                        target_ids,
                        target_values,
                        target_lse,
                    )
                    candidate_budgets = tuple(cfg.acceptance_candidate_topk_by_head or (4, 3, 2))
                    candidate_k = int(
                        candidate_budgets[min(head_idx, len(candidate_budgets) - 1)]
                        if candidate_budgets
                        else 1
                    )
                    coverage, candidate_mass = self._sparse_candidate_coverage(
                        logits,
                        target_ids,
                        target_values,
                        target_lse,
                        candidate_topk=candidate_k,
                        rank_temperature=float(cfg.acceptance_rank_temperature),
                    )
                    proximal = torch.zeros_like(distill)
                    if (
                        float(cfg.refresh_proximal_weight) > 0.0
                        and torch.is_tensor(old_top_ids)
                        and torch.is_tensor(old_top_logits)
                        and torch.is_tensor(old_logsumexp)
                    ):
                        proximal = self._sparse_proximal_kl_with_tail(
                            logits,
                            old_top_ids[start:end].to(device=device).long()[mask],
                            old_top_logits[start:end].to(device=device).float()[mask],
                            old_logsumexp[start:end].to(device=device).float()[mask],
                        )
                    sparse_loss = (
                        float(cfg.refresh_distill_weight) * distill
                        + float(cfg.refresh_tv_weight) * tv
                        + float(cfg.refresh_coverage_weight) * coverage
                        + float(cfg.refresh_hard_token_weight) * hard_loss
                        + float(cfg.refresh_proximal_weight) * proximal
                    )
                    per_record_loss = torch.where(selected_teacher, sparse_loss, hard_loss)
                    stat_sums[f"head_{head_idx + 1}_distill"] = stat_sums.get(
                        f"head_{head_idx + 1}_distill", 0.0
                    ) + float(distill[selected_teacher].mean().detach().cpu())
                    stat_sums[f"head_{head_idx + 1}_acceptance_tv"] = stat_sums.get(
                        f"head_{head_idx + 1}_acceptance_tv", 0.0
                    ) + float(tv[selected_teacher].mean().detach().cpu())
                    stat_sums[f"head_{head_idx + 1}_candidate_mass"] = stat_sums.get(
                        f"head_{head_idx + 1}_candidate_mass", 0.0
                    ) + float(candidate_mass[selected_teacher].mean().detach().cpu())
                    stat_sums[f"head_{head_idx + 1}_coverage_loss"] = stat_sums.get(
                        f"head_{head_idx + 1}_coverage_loss", 0.0
                    ) + float(coverage[selected_teacher].mean().detach().cpu())
                loss = per_record_loss.mean()
                weighted = float(aux_weight * decay) * loss
                losses.append(weighted)
                parallel_losses.append(weighted)
                weight_sum += float(aux_weight)
                stat_sums[f"head_{head_idx + 1}"] = stat_sums.get(f"head_{head_idx + 1}", 0.0) + float(loss.detach().cpu())
                stat_sums[f"head_{head_idx + 1}_tokens"] = stat_sums.get(f"head_{head_idx + 1}_tokens", 0.0) + int(mask.sum().item())
                stat_sums[f"head_{head_idx + 1}_weight"] = float(aux_weight)

            chain_weight = float(cfg.chain_loss_weight or 0.0)
            if chain_weight > 0.0 and prev_tokens.numel() > 0:
                max_depth = min(int(cfg.chain_loss_max_depth or len(self.medusa_heads.heads)), len(self.medusa_heads.heads))
                for depth_idx in range(max_depth):
                    horizon = depth_idx + 2
                    mask = horizons.eq(horizon)
                    if not bool(mask.any().item()):
                        continue
                    aux_weight = self._head_weight(head_weights, depth_idx)
                    if aux_weight <= 0.0:
                        continue
                    h = hidden[mask]
                    f = fast[mask]
                    s = scales[mask]
                    y = labels[mask]
                    prev = prev_tokens[mask]
                    if cfg.chain_bootstrap_from_medusa and len(self.medusa_heads.heads) > 0:
                        state = self.medusa_heads.heads[0].project_hidden(h)
                        first_prev_index = 1
                    else:
                        state = h
                        first_prev_index = 0
                    for prev_index in range(first_prev_index, max(horizon - 1, first_prev_index)):
                        if prev_index >= prev.shape[1]:
                            break
                        tokens = prev[:, prev_index]
                        valid = tokens.ge(0)
                        if not bool(valid.all().item()):
                            state = state[valid]
                            f = f[valid]
                            s = s[valid]
                            y = y[valid]
                            tokens = tokens[valid]
                            if state.numel() == 0:
                                break
                        state = self.medusa_heads.chain_next_state(state, tokens, embedding_layer)
                    if state.numel() == 0:
                        continue
                    if update_fast_state_injections:
                        delta = self.medusa_heads.reflex_up[depth_idx](
                            f.to(dtype=self.medusa_heads.reflex_up[depth_idx].weight.dtype)
                        )
                    else:
                        delta = F.linear(
                            f.to(dtype=self.medusa_heads.reflex_up[depth_idx].weight.dtype),
                            self.medusa_heads.reflex_up[depth_idx].weight.detach(),
                            None,
                        )
                    if bool(cfg.reflex_normalize_correction):
                        delta_float = torch.nan_to_num(delta.float(), nan=0.0, posinf=0.0, neginf=0.0)
                        rms = delta_float.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
                        delta = (delta_float / rms).to(dtype=delta.dtype)
                    if float(cfg.reflex_correction_clip_norm) > 0:
                        norm = torch.nan_to_num(delta.float(), nan=0.0, posinf=0.0, neginf=0.0).norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        delta = delta * torch.clamp(float(cfg.reflex_correction_clip_norm) / norm, max=1.0).to(dtype=delta.dtype)
                    delta = delta * s.to(device=delta.device, dtype=delta.dtype).view(-1, 1)
                    state = state + delta.to(dtype=state.dtype)
                    logits = self._chain_logits_from_state(state, lm_head).float()
                    loss = F.cross_entropy(logits, y)
                    weighted = float(aux_weight * (cfg.medusa_loss_decay ** depth_idx)) * loss
                    chain_losses.append(weighted)
                    losses.append(chain_weight * weighted)

            if not losses:
                continue
            loss = torch.stack(losses).mean() if weight_sum <= 0 else torch.stack(losses).sum() / max(weight_sum, 1e-6)
            if not torch.isfinite(loss) or not loss.requires_grad:
                continue
            (loss / grad_denom).backward()
            total_loss += float(loss.detach().cpu())
            total_tokens += int(end - start)
            updates += 1
            if parallel_losses:
                stat_sums["parallel_medusa_loss"] = stat_sums.get("parallel_medusa_loss", 0.0) + float(torch.stack(parallel_losses).mean().detach().cpu())
            if chain_losses:
                stat_sums["chain_loss"] = stat_sums.get("chain_loss", 0.0) + float(torch.stack(chain_losses).mean().detach().cpu())
                stat_sums["chain_loss_weight"] = chain_weight

        reverted_nonfinite = False
        if updates:
            backup = self._trainable_param_backup()
            torch.nn.utils.clip_grad_norm_(self.medusa_heads.parameters(), cfg.grad_clip_norm)
            self.optimizer.step()
            if backup and not self._params_are_finite(self.medusa_heads):
                self._restore_backup(backup)
                reverted_nonfinite = True
        self.optimizer.zero_grad(set_to_none=True)
        elapsed = time.time() - start_time
        out = {
            "medusa_loss": total_loss / max(updates, 1),
            "head_update_tokens": int(total_tokens),
            "head_update_time": elapsed,
            "head_update_tokens_per_sec": total_tokens / max(elapsed, 1e-9),
            "head_update_steps": int(updates),
            "reflex_cached_records": int(total_records),
            "reflex_cached_update": True,
            "head_update_reverted_nonfinite": bool(reverted_nonfinite),
        }
        for key, value in stat_sums.items():
            if key.endswith("_tokens") or key.endswith("_weight") or key == "chain_loss_weight":
                out[key] = value
            else:
                out[key] = value / max(updates, 1)
        return out

    @torch.no_grad()
    def evaluate_reflex_records(
        self,
        records: dict,
        *,
        head_weights: dict[int | str, float] | list[float] | tuple[float, ...] | None = None,
    ) -> dict[str, float]:
        hidden_cpu = records.get("hidden") if records else None
        if hidden_cpu is None or int(hidden_cpu.shape[0]) == 0:
            return {
                "validation_ce": float("inf"),
                "validation_records": 0,
                "validation_sparse_tv": float("inf"),
                "validation_sparse_records": 0,
                "validation_candidate_mass": float("-inf"),
                "validation_candidate_records": 0,
            }
        device = next(self.medusa_heads.parameters()).device
        base = unwrap_causal_lm(self.target_model)
        lm_head = base.lm_head
        embedding_layer = base.get_input_embeddings()
        micro = max(1, int(self.config.reflex_record_microbatch_size or 64))
        loss_sum = 0.0
        count = 0
        tv_sum = 0.0
        tv_count = 0
        candidate_mass_sum = 0.0
        candidate_mass_count = 0
        was_training = self.medusa_heads.training
        self.medusa_heads.eval()
        for start in range(0, int(hidden_cpu.shape[0]), micro):
            end = min(start + micro, int(hidden_cpu.shape[0]))
            hidden = records["hidden"][start:end].to(
                device=device,
                dtype=next(self.medusa_heads.parameters()).dtype,
            )
            labels = records["labels"][start:end].to(device=device).long()
            horizons = records["horizons"][start:end].to(device=device).long()
            prev_tokens = records["prev_tokens"][start:end].to(device=device).long()
            has_sparse_teacher = records.get("has_sparse_teacher")
            has_sparse_teacher = (
                has_sparse_teacher[start:end].to(device=device).bool()
                if torch.is_tensor(has_sparse_teacher)
                else torch.zeros((end - start,), device=device, dtype=torch.bool)
            )
            for head_idx, head in enumerate(self.medusa_heads.heads):
                mask = horizons.eq(head_idx + 2)
                if not bool(mask.any().item()) or self._head_weight(head_weights, head_idx) <= 0.0:
                    continue
                projected = head.project_hidden(hidden[mask])
                if (
                    bool(self.config.anchor_conditioning_enabled)
                    and getattr(self.medusa_heads, "anchor_conditioner", None) is not None
                    and prev_tokens.shape[1] > 0
                ):
                    anchor_tokens = prev_tokens[mask, 0].clamp_min(0)
                    anchor_embeddings = embedding_layer(anchor_tokens).detach()
                    projected = self.medusa_heads.anchor_conditioner(projected, anchor_embeddings, head_idx)
                logits = self._logits_from_hidden(projected, head_idx, lm_head).float()
                loss_sum += float(F.cross_entropy(logits, labels[mask], reduction="sum").cpu())
                count += int(mask.sum().item())
                selected_teacher = has_sparse_teacher[mask]
                if bool(selected_teacher.any().item()) and torch.is_tensor(records.get("target_top_ids")):
                    target_ids = records["target_top_ids"][start:end].to(device=device).long()[mask]
                    target_values = records["target_top_logits"][start:end].to(device=device).float()[mask]
                    target_lse = records["target_logsumexp"][start:end].to(device=device).float()[mask]
                    _, tv = self._sparse_cross_entropy_with_tail(
                        logits,
                        target_ids,
                        target_values,
                        target_lse,
                    )
                    tv_sum += float(tv[selected_teacher].sum().cpu())
                    tv_count += int(selected_teacher.sum().item())
                    candidate_budgets = tuple(self.config.acceptance_candidate_topk_by_head or (4, 3, 2))
                    candidate_k = int(
                        candidate_budgets[min(head_idx, len(candidate_budgets) - 1)]
                        if candidate_budgets
                        else 1
                    )
                    _, candidate_mass = self._sparse_candidate_coverage(
                        logits,
                        target_ids,
                        target_values,
                        target_lse,
                        candidate_topk=candidate_k,
                        rank_temperature=float(self.config.acceptance_rank_temperature),
                    )
                    candidate_mass_sum += float(candidate_mass[selected_teacher].sum().cpu())
                    candidate_mass_count += int(selected_teacher.sum().item())
        self.medusa_heads.train(was_training)
        return {
            "validation_ce": loss_sum / max(count, 1),
            "validation_records": int(count),
            "validation_sparse_tv": tv_sum / max(tv_count, 1) if tv_count else float("inf"),
            "validation_sparse_records": int(tv_count),
            "validation_candidate_mass": (
                candidate_mass_sum / max(candidate_mass_count, 1)
                if candidate_mass_count
                else float("-inf")
            ),
            "validation_candidate_records": int(candidate_mass_count),
        }
