from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _rms(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return value.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)


def _masked_softmax(logits: torch.Tensor, valid: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled = logits.float() / max(float(temperature), 1e-6)
    scaled = scaled.masked_fill(~valid, -torch.inf)
    return torch.softmax(scaled, dim=-1).masked_fill(~valid, 0.0)


@dataclass(slots=True)
class HRDCRPredictionBatch:
    sequence_ids: torch.Tensor
    target_positions: torch.Tensor
    head_indices: torch.Tensor
    proposal_top_ids: torch.Tensor
    proposal_hidden: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_valid: torch.Tensor
    state_sketch: torch.Tensor
    anchor_hidden: torch.Tensor
    fast_state: torch.Tensor
    trust: torch.Tensor
    quality: torch.Tensor
    probe_mask: torch.Tensor
    raw_proposal_hidden: torch.Tensor
    raw_candidate_ids: torch.Tensor
    raw_candidate_valid: torch.Tensor


@dataclass(slots=True)
class HRDCRMatureRecords:
    group_indices: torch.Tensor
    head_indices: torch.Tensor
    proposal_top_ids: torch.Tensor
    proposal_hidden: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_valid: torch.Tensor
    state_sketch: torch.Tensor
    anchor_hidden: torch.Tensor
    fast_state: torch.Tensor
    trust: torch.Tensor
    quality: torch.Tensor
    probe_mask: torch.Tensor
    raw_proposal_hidden: torch.Tensor
    raw_candidate_ids: torch.Tensor
    raw_candidate_valid: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.group_indices.numel())


@dataclass(slots=True)
class HRDCRFeedbackBatch:
    head_feedback: torch.Tensor
    head_has_feedback: torch.Tensor
    head_severity: torch.Tensor
    head_alignment: torch.Tensor
    head_alignment_observed: torch.Tensor
    record_head_indices: torch.Tensor
    record_true_probs: torch.Tensor
    record_tv: torch.Tensor
    record_severity: torch.Tensor
    record_candidate_mass: torch.Tensor
    record_candidate_hit: torch.Tensor
    record_candidate_regret: torch.Tensor
    record_restricted_kl: torch.Tensor
    record_quality: torch.Tensor
    probe_head_indices: torch.Tensor
    probe_quality: torch.Tensor
    probe_changed: torch.Tensor
    probe_raw_mass: torch.Tensor
    probe_effective_mass: torch.Tensor
    probe_raw_kl: torch.Tensor
    probe_effective_kl: torch.Tensor
    probe_wins: torch.Tensor
    probe_losses: torch.Tensor
    auxiliary_records: dict[str, torch.Tensor]


class HRDCRPredictionBuffer:
    """GPU-resident sparse proposals awaiting exact verifier feedback."""

    def __init__(self, *, proposal_topk: int, max_records: int = 8192):
        self.proposal_topk = max(1, int(proposal_topk))
        self.max_records = max(1, int(max_records))
        self._batches: list[HRDCRPredictionBatch] = []
        self._record_count = 0

    @torch.no_grad()
    def add_from_logits(
        self,
        *,
        sequence_ids: list[int],
        anchor_positions: torch.Tensor,
        logits_by_horizon: list[torch.Tensor],
        proposal_hidden_by_horizon: list[torch.Tensor],
        candidate_topk_by_horizon: list[int],
        anchor_hidden: torch.Tensor,
        fast_states: torch.Tensor,
        trust: torch.Tensor,
        state_sketch: torch.Tensor,
        candidate_ids_by_horizon: list[torch.Tensor] | None = None,
        candidate_valid_by_horizon: list[torch.Tensor] | None = None,
        quality_by_horizon: list[torch.Tensor] | None = None,
        probe_rows: torch.Tensor | None = None,
        raw_proposal_hidden_by_horizon: list[torch.Tensor] | None = None,
        raw_candidate_ids_by_horizon: list[torch.Tensor] | None = None,
        raw_candidate_valid_by_horizon: list[torch.Tensor] | None = None,
    ) -> None:
        if not sequence_ids or not logits_by_horizon:
            return
        device = logits_by_horizon[0].device
        batch_size = len(sequence_ids)
        sequence = torch.as_tensor(sequence_ids, device=device, dtype=torch.long)
        anchors = anchor_positions.detach().to(device=device, dtype=torch.long)
        if fast_states.dim() != 3 or state_sketch.dim() != 3:
            raise ValueError("HRDCR fast states and sketches must be [batch, heads, width]")
        head_count = min(
            len(logits_by_horizon),
            len(proposal_hidden_by_horizon),
            len(candidate_topk_by_horizon),
            int(fast_states.shape[1]),
            int(state_sketch.shape[1]),
        )
        for head_idx in range(head_count):
            logits = logits_by_horizon[head_idx].detach()
            top_l = min(self.proposal_topk, int(logits.shape[-1]))
            top_ids = torch.topk(logits, k=top_l, dim=-1).indices
            candidate_k = min(max(1, int(candidate_topk_by_horizon[head_idx])), top_l)
            if candidate_ids_by_horizon is not None and head_idx < len(candidate_ids_by_horizon):
                candidates = candidate_ids_by_horizon[head_idx].detach().to(
                    device=device, dtype=torch.long
                )
                candidate_valid = (
                    candidate_valid_by_horizon[head_idx].detach().to(device=device, dtype=torch.bool)
                    if candidate_valid_by_horizon is not None
                    and head_idx < len(candidate_valid_by_horizon)
                    else candidates.ge(0)
                )
            else:
                candidates = top_ids[:, :candidate_k].contiguous()
                candidate_valid = torch.ones_like(candidates, dtype=torch.bool)
            quality = (
                quality_by_horizon[head_idx].detach().to(device=device, dtype=torch.float32)
                if quality_by_horizon is not None and head_idx < len(quality_by_horizon)
                else torch.zeros(batch_size, device=device, dtype=torch.float32)
            )
            row_probe = torch.zeros(batch_size, device=device, dtype=torch.bool)
            raw_hidden = proposal_hidden_by_horizon[head_idx].detach().clone()
            raw_candidates = candidates.clone()
            raw_valid = candidate_valid.clone()
            if probe_rows is not None and probe_rows.numel() > 0:
                probe = probe_rows.to(device=device, dtype=torch.long)
                row_probe.index_fill_(0, probe, True)
                if raw_proposal_hidden_by_horizon is not None and head_idx < len(raw_proposal_hidden_by_horizon):
                    raw_hidden.index_copy_(
                        0,
                        probe,
                        raw_proposal_hidden_by_horizon[head_idx].detach().to(
                            device=device, dtype=raw_hidden.dtype
                        ),
                    )
                if raw_candidate_ids_by_horizon is not None and head_idx < len(raw_candidate_ids_by_horizon):
                    raw_ids = raw_candidate_ids_by_horizon[head_idx].detach().to(device=device, dtype=torch.long)
                    raw_ids = self._pad(raw_ids, int(candidates.shape[-1]), -1)
                    raw_candidates.index_copy_(0, probe, raw_ids)
                    if raw_candidate_valid_by_horizon is not None and head_idx < len(raw_candidate_valid_by_horizon):
                        raw_mask = raw_candidate_valid_by_horizon[head_idx].detach().to(
                            device=device, dtype=torch.bool
                        )
                        raw_mask = self._pad(raw_mask, int(candidates.shape[-1]), False)
                    else:
                        raw_mask = raw_ids.ge(0)
                    raw_valid.index_copy_(0, probe, raw_mask)
            item = HRDCRPredictionBatch(
                    sequence_ids=sequence.clone(),
                    target_positions=anchors + head_idx + 2,
                    head_indices=torch.full(
                        (batch_size,), head_idx, device=device, dtype=torch.long
                    ),
                    proposal_top_ids=top_ids.to(dtype=torch.int32),
                    proposal_hidden=proposal_hidden_by_horizon[head_idx].detach().to(
                        dtype=torch.bfloat16
                    ),
                    candidate_ids=candidates.to(dtype=torch.int32),
                    candidate_valid=candidate_valid,
                    state_sketch=state_sketch[:, head_idx].detach().to(dtype=torch.float16),
                    anchor_hidden=anchor_hidden.detach().to(dtype=torch.bfloat16),
                    fast_state=fast_states[:, head_idx].detach().to(dtype=torch.bfloat16),
                    trust=trust[:, head_idx].detach().to(dtype=torch.float16),
                    quality=quality.to(dtype=torch.float16),
                    probe_mask=row_probe,
                    raw_proposal_hidden=raw_hidden.to(dtype=torch.bfloat16),
                    raw_candidate_ids=raw_candidates.to(dtype=torch.int32),
                    raw_candidate_valid=raw_valid,
                )
            keep = candidate_valid.any(dim=-1).nonzero(as_tuple=False).flatten()
            item = HRDCRPredictionBatch(
                **{
                    field: getattr(item, field).index_select(0, keep)
                    for field in HRDCRPredictionBatch.__dataclass_fields__
                }
            )
            if keep.numel() > 0:
                self._batches.append(item)
                self._record_count += int(keep.numel())
        self._trim_oldest()

    def _trim_oldest(self) -> None:
        while self._record_count > self.max_records and self._batches:
            dropped = self._batches.pop(0)
            self._record_count -= int(dropped.sequence_ids.numel())

    @staticmethod
    def _pad(values: torch.Tensor, width: int, fill: int | bool) -> torch.Tensor:
        current = int(values.shape[-1])
        if current == width:
            return values
        if current > width:
            return values[:, :width]
        output = torch.full(
            (int(values.shape[0]), width), fill, device=values.device, dtype=values.dtype
        )
        output[:, :current].copy_(values)
        return output

    @torch.no_grad()
    def pop_mature(
        self,
        sequence_ids: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> HRDCRMatureRecords:
        if tuple(sequence_ids.shape) != tuple(target_positions.shape):
            raise ValueError("HRDCR sequence IDs and target positions must align")
        device = sequence_ids.device
        max_candidates = max(
            (
                max(int(batch.candidate_ids.shape[-1]), int(batch.raw_candidate_ids.shape[-1]))
                for batch in self._batches
            ),
            default=0,
        )
        selected: list[tuple[HRDCRPredictionBatch, torch.Tensor, torch.Tensor]] = []
        retained: list[HRDCRPredictionBatch] = []
        for batch in self._batches:
            seq = sequence_ids.to(device=batch.sequence_ids.device, dtype=torch.long)
            pos = target_positions.to(device=batch.target_positions.device, dtype=torch.long)
            matches = batch.sequence_ids.unsqueeze(1).eq(seq.unsqueeze(0)) & batch.target_positions.unsqueeze(
                1
            ).eq(pos.unsqueeze(0))
            mature = matches.any(dim=1)
            if bool(mature.any().item()):
                selected.append((batch, mature, matches.long().argmax(dim=1)[mature]))
                self._record_count -= int(mature.sum().item())
            keep = ~mature
            if bool(keep.any().item()):
                retained.append(
                    HRDCRPredictionBatch(
                        **{
                            field: getattr(batch, field)[keep]
                            for field in HRDCRPredictionBatch.__dataclass_fields__
                        }
                    )
                )
        self._batches = retained
        if not selected:
            return self._empty(device, max_candidates)

        def merge(name: str) -> torch.Tensor:
            rows = [getattr(batch, name)[mask].to(device=device) for batch, mask, _ in selected]
            if name in {"candidate_ids", "candidate_valid", "raw_candidate_ids", "raw_candidate_valid"}:
                fill = -1 if name.endswith("ids") else False
                rows = [self._pad(row, max_candidates, fill) for row in rows]
            return torch.cat(rows, dim=0)

        return HRDCRMatureRecords(
            group_indices=torch.cat([group.to(device=device) for _, _, group in selected]),
            head_indices=merge("head_indices"),
            proposal_top_ids=merge("proposal_top_ids"),
            proposal_hidden=merge("proposal_hidden"),
            candidate_ids=merge("candidate_ids"),
            candidate_valid=merge("candidate_valid"),
            state_sketch=merge("state_sketch"),
            anchor_hidden=merge("anchor_hidden"),
            fast_state=merge("fast_state"),
            trust=merge("trust"),
            quality=merge("quality"),
            probe_mask=merge("probe_mask"),
            raw_proposal_hidden=merge("raw_proposal_hidden"),
            raw_candidate_ids=merge("raw_candidate_ids"),
            raw_candidate_valid=merge("raw_candidate_valid"),
        )

    def _empty(self, device: torch.device, candidate_width: int) -> HRDCRMatureRecords:
        return HRDCRMatureRecords(
            group_indices=torch.empty((0,), device=device, dtype=torch.long),
            head_indices=torch.empty((0,), device=device, dtype=torch.long),
            proposal_top_ids=torch.empty(
                (0, self.proposal_topk), device=device, dtype=torch.int32
            ),
            proposal_hidden=torch.empty((0, 0), device=device, dtype=torch.bfloat16),
            candidate_ids=torch.empty((0, candidate_width), device=device, dtype=torch.int32),
            candidate_valid=torch.empty((0, candidate_width), device=device, dtype=torch.bool),
            state_sketch=torch.empty((0, 0), device=device, dtype=torch.float16),
            anchor_hidden=torch.empty((0, 0), device=device, dtype=torch.bfloat16),
            fast_state=torch.empty((0, 0), device=device, dtype=torch.bfloat16),
            trust=torch.empty((0,), device=device, dtype=torch.float16),
            quality=torch.empty((0,), device=device, dtype=torch.float16),
            probe_mask=torch.empty((0,), device=device, dtype=torch.bool),
            raw_proposal_hidden=torch.empty((0, 0), device=device, dtype=torch.bfloat16),
            raw_candidate_ids=torch.empty((0, candidate_width), device=device, dtype=torch.int32),
            raw_candidate_valid=torch.empty((0, candidate_width), device=device, dtype=torch.bool),
        )

    def clear_sequences(self, sequence_ids: list[int]) -> None:
        if not sequence_ids:
            return
        retained: list[HRDCRPredictionBatch] = []
        for batch in self._batches:
            finished = torch.as_tensor(sequence_ids, device=batch.sequence_ids.device, dtype=torch.long)
            keep = ~batch.sequence_ids.unsqueeze(1).eq(finished.unsqueeze(0)).any(dim=1)
            self._record_count -= int((~keep).sum().item())
            if bool(keep.any().item()):
                retained.append(
                    HRDCRPredictionBatch(
                        **{
                            field: getattr(batch, field)[keep]
                            for field in HRDCRPredictionBatch.__dataclass_fields__
                        }
                    )
                )
        self._batches = retained

    def __len__(self) -> int:
        return int(self._record_count)


class HRDCRFeedback:
    """Restricted-KL and candidate-coverage verifier feedback."""

    def __init__(
        self,
        lm_head,
        *,
        num_heads: int,
        proposal_topk: int = 16,
        target_topk: int = 16,
        support_cap: int = 48,
        temperature: float = 1.0,
        distribution_weight: float = 0.25,
        coverage_weight: float = 1.0,
        boundary_width: int = 2,
        severity_tv_weight: float = 0.5,
        severity_out_weight: float = 0.5,
        severity_min: float = 0.02,
        sketch_projection: torch.Tensor | None = None,
        eps: float = 1e-6,
    ):
        if proposal_topk + target_topk + 1 > support_cap:
            raise ValueError("HRDCR support_cap must fit proposal_topk + target_topk + actual token")
        self.lm_head = lm_head
        self.num_heads = int(num_heads)
        self.proposal_topk = int(proposal_topk)
        self.target_topk = int(target_topk)
        self.support_cap = int(support_cap)
        self.temperature = float(temperature)
        self.distribution_weight = float(distribution_weight)
        self.coverage_weight = float(coverage_weight)
        self.boundary_width = max(1, int(boundary_width))
        self.severity_tv_weight = float(severity_tv_weight)
        self.severity_out_weight = float(severity_out_weight)
        self.severity_min = max(0.0, float(severity_min))
        self.sketch_projection = sketch_projection
        self.eps = float(eps)

    @torch.no_grad()
    def compute(
        self,
        records: HRDCRMatureRecords,
        target_logits: torch.Tensor,
        actual_tokens: torch.Tensor,
        *,
        collect_auxiliary: bool,
    ) -> HRDCRFeedbackBatch:
        weight = self.lm_head.weight.detach()
        device = weight.device
        group_count = int(actual_tokens.numel())
        hidden = int(weight.shape[-1])
        head_feedback = torch.zeros(
            (group_count, self.num_heads, hidden), device=device, dtype=torch.float32
        )
        head_has = torch.zeros((group_count, self.num_heads), device=device, dtype=torch.bool)
        head_severity = torch.zeros((group_count, self.num_heads), device=device, dtype=torch.float32)
        head_alignment = torch.zeros_like(head_severity)
        head_alignment_observed = torch.zeros_like(head_has)
        if records.count == 0:
            empty = torch.empty((0,), device=device)
            return HRDCRFeedbackBatch(
                head_feedback=head_feedback,
                head_has_feedback=head_has,
                head_severity=head_severity,
                head_alignment=head_alignment,
                head_alignment_observed=head_alignment_observed,
                record_head_indices=torch.empty((0,), device=device, dtype=torch.long),
                record_true_probs=empty,
                record_tv=empty,
                record_severity=empty,
                record_candidate_mass=empty,
                record_candidate_hit=torch.empty((0,), device=device, dtype=torch.bool),
                record_candidate_regret=empty,
                record_restricted_kl=empty,
                record_quality=empty,
                probe_head_indices=torch.empty((0,), device=device, dtype=torch.long),
                probe_quality=empty,
                probe_changed=torch.empty((0,), device=device, dtype=torch.bool),
                probe_raw_mass=empty,
                probe_effective_mass=empty,
                probe_raw_kl=empty,
                probe_effective_kl=empty,
                probe_wins=torch.empty((0,), device=device, dtype=torch.bool),
                probe_losses=torch.empty((0,), device=device, dtype=torch.bool),
                auxiliary_records={},
            )

        groups = records.group_indices.to(device=device, dtype=torch.long)
        heads = records.head_indices.to(device=device, dtype=torch.long)
        actual = actual_tokens.to(device=device, dtype=torch.long).index_select(0, groups)
        target = target_logits.detach().to(device=device)
        target_ids = torch.topk(target, k=min(self.target_topk, int(target.shape[-1])), dim=-1).indices
        target_ids = target_ids.index_select(0, groups)
        proposal_ids = records.proposal_top_ids.to(device=device, dtype=torch.long)
        raw_candidate_ids = records.raw_candidate_ids.to(device=device, dtype=torch.long)
        raw_candidate_valid = records.raw_candidate_valid.to(device=device, dtype=torch.bool)
        sentinel = int(weight.shape[0])
        raw_candidate_support = torch.where(
            raw_candidate_valid, raw_candidate_ids, torch.full_like(raw_candidate_ids, sentinel)
        )
        support_raw = torch.cat(
            (proposal_ids, target_ids, actual.unsqueeze(-1), raw_candidate_support), dim=-1
        )
        support_sorted = support_raw.sort(dim=-1).values
        support_unique = support_sorted.lt(sentinel)
        support_unique[:, 1:] &= support_sorted[:, 1:] != support_sorted[:, :-1]
        support_dedup = torch.where(
            support_unique, support_sorted, torch.full_like(support_sorted, sentinel)
        ).sort(dim=-1).values
        support_ids = support_dedup[:, : self.support_cap]
        support_valid = support_ids.lt(sentinel)
        safe_support = support_ids.clamp(0, int(weight.shape[0]) - 1)

        selected_weight = weight.index_select(0, safe_support.reshape(-1)).view(
            records.count, int(safe_support.shape[1]), hidden
        )
        proposal_hidden = records.proposal_hidden.to(device=device, dtype=weight.dtype)
        proposal_logits = torch.einsum("rsh,rh->rs", selected_weight, proposal_hidden)
        target_rows = target.index_select(0, groups)
        target_support_logits = torch.gather(target_rows, -1, safe_support)
        p = _masked_softmax(target_support_logits, support_valid, self.temperature)
        q = _masked_softmax(proposal_logits, support_valid, self.temperature)

        residual = (p - q).masked_fill(~support_valid, 0.0)
        restricted_kl = (
            p
            * (
                torch.log(p.clamp_min(self.eps))
                - torch.log(q.clamp_min(self.eps))
            )
        ).masked_fill(~support_valid, 0.0).sum(dim=-1)
        distribution = torch.einsum("rs,rsh->rh", residual, selected_weight.float())

        candidate_ids = records.candidate_ids.to(device=device, dtype=torch.long)
        candidate_valid = records.candidate_valid.to(device=device, dtype=torch.bool)
        in_candidates = (
            support_ids.unsqueeze(-1).eq(candidate_ids.unsqueeze(1))
            & support_valid.unsqueeze(-1)
            & candidate_valid.unsqueeze(1)
        ).any(dim=-1)
        target_rows = target.index_select(0, groups).float() / max(self.temperature, self.eps)
        target_log_z = torch.logsumexp(target_rows, dim=-1)
        safe_candidates = candidate_ids.clamp(0, int(weight.shape[0]) - 1)
        candidate_prob = torch.exp(
            torch.gather(target_rows, -1, safe_candidates) - target_log_z.unsqueeze(-1)
        )
        duplicate = candidate_ids.unsqueeze(2).eq(candidate_ids.unsqueeze(1))
        prior = torch.tril(
            torch.ones(
                (int(candidate_ids.shape[1]), int(candidate_ids.shape[1])),
                device=device,
                dtype=torch.bool,
            ),
            diagonal=-1,
        )
        duplicate = (duplicate & prior.unsqueeze(0)).any(dim=-1)
        candidate_unique = candidate_valid & ~duplicate
        candidate_mass = (candidate_prob * candidate_unique).sum(dim=-1).clamp(0.0, 1.0)
        candidate_k = candidate_unique.sum(dim=-1)
        max_k = max(1, int(candidate_ids.shape[1]))
        target_top_values = torch.topk(target_rows, k=max_k, dim=-1).values
        target_top_prob = torch.exp(target_top_values - target_log_z.unsqueeze(-1))
        target_k_mask = torch.arange(max_k, device=device).unsqueeze(0).lt(candidate_k.unsqueeze(-1))
        optimal_mass = (target_top_prob * target_k_mask).sum(dim=-1)
        candidate_regret = (optimal_mass - candidate_mass).clamp_min(0.0)
        candidate_hit = (
            candidate_ids.eq(actual.unsqueeze(-1)) & candidate_valid
        ).any(dim=-1)
        out_mass = 1.0 - candidate_mass

        target_mode_in_candidates = (
            target_ids.unsqueeze(-1).eq(candidate_ids.unsqueeze(1))
            & candidate_valid.unsqueeze(1)
        ).any(dim=-1)
        omitted_target = ~target_mode_in_candidates
        target_mode_prob = (
            p.unsqueeze(1)
            * target_ids.unsqueeze(-1).eq(support_ids.unsqueeze(1))
            * support_valid.unsqueeze(1)
        ).sum(dim=-1)
        omitted_prob = target_mode_prob * omitted_target
        target_mode_weight = weight.index_select(0, target_ids.reshape(-1)).view(
            records.count, int(target_ids.shape[1]), hidden
        )
        omitted_embedding = torch.einsum(
            "rm,rmh->rh", omitted_prob, target_mode_weight.float()
        )

        boundary_width = min(self.boundary_width, int(candidate_ids.shape[-1]))
        boundary_start = (candidate_valid.sum(dim=-1) - boundary_width).clamp_min(0)
        boundary_offsets = torch.arange(boundary_width, device=device).unsqueeze(0)
        boundary_index = (boundary_start.unsqueeze(-1) + boundary_offsets).clamp_max(
            max(int(candidate_ids.shape[-1]) - 1, 0)
        )
        boundary_ids = torch.gather(candidate_ids, 1, boundary_index)
        boundary_valid = torch.gather(candidate_valid, 1, boundary_index)
        boundary_weight = weight.index_select(0, boundary_ids.clamp_min(0).reshape(-1)).view(
            records.count, boundary_width, hidden
        )
        boundary_logits = torch.einsum(
            "rbh,rh->rb", boundary_weight.float(), proposal_hidden.float()
        ).masked_fill(~boundary_valid, -torch.inf)
        boundary_beta = torch.softmax(boundary_logits, dim=-1).masked_fill(~boundary_valid, 0.0)
        boundary_embedding = torch.einsum(
            "rb,rbh->rh", boundary_beta, boundary_weight.float()
        )
        omitted_mass = omitted_prob.sum(dim=-1)
        coverage = omitted_embedding - omitted_mass.unsqueeze(-1) * boundary_embedding

        correction = self.distribution_weight * distribution + self.coverage_weight * coverage
        tv = 0.5 * residual.abs().sum(dim=-1)
        severity = (
            self.severity_tv_weight * tv + self.severity_out_weight * out_mass
        ).clamp(0.0, 1.0)
        valid_feedback = severity.ge(self.severity_min) & correction.square().mean(dim=-1).gt(0.0)
        correction = correction.masked_fill(~valid_feedback.unsqueeze(-1), 0.0)

        flat = groups * self.num_heads + heads
        counts = torch.zeros((group_count * self.num_heads,), device=device, dtype=torch.float32)
        counts.index_add_(0, flat, valid_feedback.float())
        head_feedback.view(group_count * self.num_heads, hidden).index_add_(0, flat, correction)
        head_feedback.view(group_count * self.num_heads, hidden).div_(
            counts.clamp_min(1.0).unsqueeze(-1)
        )
        severity_sum = torch.zeros_like(counts)
        severity_sum.index_add_(0, flat, severity * valid_feedback)
        head_severity.copy_((severity_sum / counts.clamp_min(1.0)).view(group_count, self.num_heads))
        head_has.copy_(counts.view(group_count, self.num_heads).gt(0.0))

        if self.sketch_projection is not None and records.state_sketch.numel() > 0:
            projection = self.sketch_projection.to(device=device, dtype=torch.float32)
            normalized_error = correction / _rms(correction, self.eps)
            error_sketch = normalized_error @ projection
            error_sketch = error_sketch / error_sketch.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            state_sketch = records.state_sketch.to(device=device, dtype=torch.float32)
            state_norm = state_sketch.norm(dim=-1)
            error_norm = error_sketch.norm(dim=-1)
            alignment_valid = valid_feedback & state_norm.gt(self.eps) & error_norm.gt(self.eps)
            alignment = F.cosine_similarity(state_sketch, error_sketch, dim=-1).clamp(-1.0, 1.0)
            alignment_sum = torch.zeros_like(counts)
            alignment_count = torch.zeros_like(counts)
            alignment_sum.index_add_(0, flat, alignment * alignment_valid)
            alignment_count.index_add_(0, flat, alignment_valid.float())
            head_alignment.copy_(
                (alignment_sum / alignment_count.clamp_min(1.0)).view(group_count, self.num_heads)
            )
            head_alignment_observed.copy_(alignment_count.view(group_count, self.num_heads).gt(0.0))

        actual_prob = torch.exp(
            torch.gather(target_rows, -1, actual.unsqueeze(-1)).squeeze(-1) - target_log_z
        ).clamp_min(self.eps)

        probe_mask = records.probe_mask.to(device=device, dtype=torch.bool)
        raw_hidden = records.raw_proposal_hidden.to(device=device, dtype=weight.dtype)
        raw_logits = torch.einsum("rsh,rh->rs", selected_weight, raw_hidden)
        raw_q = _masked_softmax(raw_logits, support_valid, self.temperature)
        raw_kl = (
            p
            * (
                torch.log(p.clamp_min(self.eps))
                - torch.log(raw_q.clamp_min(self.eps))
            )
        ).masked_fill(~support_valid, 0.0).sum(dim=-1)
        safe_raw_candidates = raw_candidate_ids.clamp(0, int(weight.shape[0]) - 1)
        raw_candidate_prob = torch.exp(
            torch.gather(target_rows, -1, safe_raw_candidates) - target_log_z.unsqueeze(-1)
        )
        raw_duplicate = raw_candidate_ids.unsqueeze(2).eq(raw_candidate_ids.unsqueeze(1))
        raw_duplicate = (raw_duplicate & prior.unsqueeze(0)).any(dim=-1)
        raw_unique = raw_candidate_valid & ~raw_duplicate
        raw_mass = (raw_candidate_prob * raw_unique).sum(dim=-1).clamp(0.0, 1.0)
        raw_hit = (raw_candidate_ids.eq(actual.unsqueeze(-1)) & raw_candidate_valid).any(dim=-1)
        set_overlap = (
            candidate_ids.unsqueeze(-1).eq(raw_candidate_ids.unsqueeze(1))
            & candidate_valid.unsqueeze(-1)
            & raw_candidate_valid.unsqueeze(1)
        )
        effective_subset = (~candidate_valid | set_overlap.any(dim=-1)).all(dim=-1)
        raw_subset = (~raw_candidate_valid | set_overlap.any(dim=1)).all(dim=-1)
        changed = ~(effective_subset & raw_subset)
        probe_rows = probe_mask.nonzero(as_tuple=False).flatten()
        auxiliary: dict[str, torch.Tensor] = {}
        if collect_auxiliary:
            auxiliary = {
                "hidden": records.anchor_hidden.detach(),
                "head_indices": heads.detach(),
                "support_ids": support_ids.to(dtype=torch.int32).detach(),
                "support_valid": support_valid.detach(),
                "target_logits": target_support_logits.to(dtype=torch.float16).detach(),
                "proposal_logits": proposal_logits.to(dtype=torch.float16).detach(),
                "candidate_ids": candidate_ids.to(dtype=torch.int32).detach(),
                "candidate_valid": candidate_valid.detach(),
                "candidate_mass": candidate_mass.detach(),
                "candidate_regret": candidate_regret.detach(),
                "restricted_kl": restricted_kl.detach(),
                "candidate_hit": candidate_hit.detach(),
                "actual_tokens": actual.detach(),
                "fast_state": records.fast_state.detach(),
                "trust": records.trust.detach(),
                "quality": records.quality.detach(),
            }
        return HRDCRFeedbackBatch(
            head_feedback=head_feedback,
            head_has_feedback=head_has,
            head_severity=head_severity,
            head_alignment=head_alignment,
            head_alignment_observed=head_alignment_observed,
            record_head_indices=heads,
            record_true_probs=actual_prob,
            record_tv=tv,
            record_severity=severity,
            record_candidate_mass=candidate_mass,
            record_candidate_hit=candidate_hit,
            record_candidate_regret=candidate_regret,
            record_restricted_kl=restricted_kl,
            record_quality=records.quality.to(device=device, dtype=torch.float32),
            probe_head_indices=heads.index_select(0, probe_rows),
            probe_quality=records.quality.to(device=device, dtype=torch.float32).index_select(0, probe_rows),
            probe_changed=changed.index_select(0, probe_rows),
            probe_raw_mass=raw_mass.index_select(0, probe_rows),
            probe_effective_mass=candidate_mass.index_select(0, probe_rows),
            probe_raw_kl=raw_kl.index_select(0, probe_rows),
            probe_effective_kl=restricted_kl.index_select(0, probe_rows),
            probe_wins=(candidate_hit & ~raw_hit).index_select(0, probe_rows),
            probe_losses=(raw_hit & ~candidate_hit).index_select(0, probe_rows),
            auxiliary_records=auxiliary,
        )


class HRDCRStateManager:
    """Per-sequence, per-horizon delayed-credit memory without shared state."""

    def __init__(
        self,
        num_sequences: int,
        num_heads: int,
        hidden_size: int,
        *,
        device: torch.device,
        half_life_tokens: float = 32.0,
        alignment_beta: float = 0.9,
        trust_n0: float = 4.0,
        sketch_rank: int = 24,
        sketch_seed: int = 29,
        min_effective_updates: float = 4.0,
        min_alignment_count: float = 4.0,
        min_state_rms: float = 0.005,
        state_reference_rms: float = 0.03,
        alignment_floor: float = 0.0,
        alignment_full: float = 0.10,
        alignment_lcb_z: float = 1.0,
        safety_min_probe_count: int = 128,
        safety_bad_probe_patience: int = 3,
        safety_ratio_decay: float = 0.5,
        safety_reenable_probe_interval: int = 256,
        eps: float = 1e-6,
    ):
        self.num_heads = int(num_heads)
        self.hidden_size = int(hidden_size)
        self.rho = float(2.0 ** (-1.0 / max(float(half_life_tokens), 1e-6)))
        self.alignment_beta = min(0.9999, max(0.0, float(alignment_beta)))
        self.trust_n0 = max(float(trust_n0), 1e-6)
        self.min_effective_updates = max(0.0, float(min_effective_updates))
        self.min_alignment_count = max(0.0, float(min_alignment_count))
        self.min_state_rms = max(0.0, float(min_state_rms))
        self.state_reference_rms = max(float(state_reference_rms), eps)
        self.alignment_floor = float(alignment_floor)
        self.alignment_full = max(float(alignment_full), self.alignment_floor + eps)
        self.alignment_lcb_z = max(0.0, float(alignment_lcb_z))
        self.safety_min_probe_count = max(1, int(safety_min_probe_count))
        self.safety_bad_probe_patience = max(1, int(safety_bad_probe_patience))
        self.safety_ratio_decay = min(1.0, max(0.0, float(safety_ratio_decay)))
        self.safety_reenable_probe_interval = max(1, int(safety_reenable_probe_interval))
        self.eps = float(eps)
        self.states = torch.zeros(
            (int(num_sequences), self.num_heads, self.hidden_size),
            device=device,
            dtype=torch.float32,
        )
        self.effective_updates = torch.zeros(
            (int(num_sequences), self.num_heads), device=device, dtype=torch.float32
        )
        self.alignment_ema = torch.zeros_like(self.effective_updates)
        self.alignment_count = torch.zeros_like(self.effective_updates)
        self.alignment_m2 = torch.zeros_like(self.effective_updates)
        self.safety_ratio = torch.ones((self.num_heads,), device=device, dtype=torch.float32)
        self.safety_probe_count = torch.zeros((self.num_heads,), device=device, dtype=torch.long)
        self.safety_mass_gain_ema = torch.zeros((self.num_heads,), device=device, dtype=torch.float32)
        self.safety_net_win_ema = torch.zeros((self.num_heads,), device=device, dtype=torch.float32)
        self.safety_bad_windows = torch.zeros((self.num_heads,), device=device, dtype=torch.long)
        self.safety_query_count = 0
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(sketch_seed))
        projection = torch.randn(
            (self.hidden_size, max(1, int(sketch_rank))), generator=generator, dtype=torch.float32
        )
        projection = projection / projection.norm(dim=0, keepdim=True).clamp_min(self.eps)
        self.sketch_projection = projection.to(device=device)
        self.numerical_reset_count = torch.zeros((), device=device, dtype=torch.long)

    def _ids(self, sequence_ids: list[int]) -> torch.Tensor:
        return torch.as_tensor(sequence_ids, device=self.states.device, dtype=torch.long)

    def trust(self, sequence_ids: list[int]) -> torch.Tensor:
        ids = self._ids(sequence_ids)
        state = self.states.index_select(0, ids)
        state_rms = _rms(state, self.eps).squeeze(-1)
        quality = self.alignment_ema.index_select(0, ids)
        count = self.alignment_count.index_select(0, ids)
        m2 = self.alignment_m2.index_select(0, ids)
        variance = m2 / (count - 1.0).clamp_min(1.0)
        stderr = torch.sqrt(variance.clamp_min(0.0) / count.clamp_min(1.0))
        lcb = quality - self.alignment_lcb_z * stderr
        align_gate = ((lcb - self.alignment_floor) / (
            self.alignment_full - self.alignment_floor
        )).clamp(0.0, 1.0)
        state_gate = (state_rms / self.state_reference_rms).clamp(0.0, 1.0)
        eligible = (
            self.effective_updates.index_select(0, ids).ge(self.min_effective_updates)
            & count.ge(self.min_alignment_count)
            & state_rms.ge(self.min_state_rms)
            & lcb.gt(0.0)
            & torch.isfinite(state).all(dim=-1)
        )
        confidence = torch.sqrt(align_gate * state_gate).masked_fill(~eligible, 0.0)
        return confidence

    def ratio_scale(self, sequence_ids: list[int]) -> torch.Tensor:
        batch = len(sequence_ids)
        self.safety_query_count += 1
        recovery_probe = self.safety_query_count % self.safety_reenable_probe_interval == 1
        ratio = self.safety_ratio
        if recovery_probe:
            ratio = torch.where(ratio.eq(0.0), torch.ones_like(ratio), ratio)
        return ratio.view(1, -1).expand(batch, -1)

    def get_state_and_effective_updates(
        self, sequence_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self._ids(sequence_ids)
        return self.states.index_select(0, ids), self.effective_updates.index_select(0, ids)

    def sketch(self, state: torch.Tensor) -> torch.Tensor:
        normalized = state.float() / _rms(state, self.eps)
        sketch = normalized @ self.sketch_projection
        return sketch / sketch.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    @torch.no_grad()
    def decay_token(self, sequence_ids: list[int]) -> None:
        ids = self._ids(sequence_ids)
        if ids.numel() == 0:
            return
        self.states.index_copy_(0, ids, float(self.rho) * self.states.index_select(0, ids))

    @torch.no_grad()
    def advance_token(
        self,
        sequence_ids: list[int],
        head_feedback: torch.Tensor,
        head_has_feedback: torch.Tensor,
        head_severity: torch.Tensor,
        head_alignment: torch.Tensor,
        head_alignment_observed: torch.Tensor,
    ) -> torch.Tensor:
        ids = self._ids(sequence_ids)
        expected = (int(ids.numel()), self.num_heads, self.hidden_size)
        if tuple(head_feedback.shape) != expected:
            raise ValueError(f"HRDCR head feedback must be {expected}")
        current = self.states.index_select(0, ids)
        updated = float(self.rho) * current
        feedback = torch.nan_to_num(
            head_feedback.to(device=self.states.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        valid = head_has_feedback.to(device=self.states.device, dtype=torch.bool)
        raw_rms = _rms(feedback, self.eps).squeeze(-1)
        valid &= raw_rms.gt(self.eps)
        normalized = feedback / raw_rms.clamp_min(self.eps).unsqueeze(-1)
        severity = torch.nan_to_num(
            head_severity.to(device=self.states.device, dtype=torch.float32), nan=0.0
        ).clamp(0.0, 1.0)
        updated += (1.0 - float(self.rho)) * severity.unsqueeze(-1) * normalized * valid.unsqueeze(-1)
        state_rms = _rms(updated, self.eps)
        updated = updated / torch.maximum(torch.ones_like(state_rms), state_rms)
        finite = torch.isfinite(updated).all(dim=-1)
        updated = updated.masked_fill(~finite.unsqueeze(-1), 0.0)
        self.numerical_reset_count.add_((~finite).sum())
        self.states.index_copy_(0, ids, updated)
        self.effective_updates.index_add_(0, ids, severity * valid)

        observed = head_alignment_observed.to(device=self.states.device, dtype=torch.bool)
        rows, heads = observed.nonzero(as_tuple=True)
        global_ids = ids.index_select(0, rows)
        old = self.alignment_ema[global_ids, heads]
        count = self.alignment_count[global_ids, heads]
        old_m2 = self.alignment_m2[global_ids, heads]
        alignment = head_alignment.to(device=self.states.device, dtype=torch.float32)[rows, heads]
        new_count = count + 1.0
        exact_mean = old + (alignment - old) / new_count.clamp_min(1.0)
        new_mean = exact_mean.clamp(-1.0, 1.0)
        new_m2 = old_m2 + (alignment - old) * (alignment - exact_mean)
        self.alignment_ema[global_ids, heads] = new_mean
        self.alignment_m2[global_ids, heads] = new_m2.clamp_min(0.0)
        self.alignment_count[global_ids, heads] = new_count
        return raw_rms.masked_fill(~valid, 0.0).mean(dim=-1)

    @torch.no_grad()
    def observe_counterfactual(self, feedback: HRDCRFeedbackBatch) -> None:
        heads = feedback.probe_head_indices.to(device=self.states.device, dtype=torch.long)
        valid = heads.ge(0) & heads.lt(self.num_heads)
        heads = heads[valid]
        if heads.numel() == 0:
            return
        mass_gain = (
            feedback.probe_effective_mass - feedback.probe_raw_mass
        ).to(device=self.states.device, dtype=torch.float32)[valid]
        net_win = (
            feedback.probe_wins.float() - feedback.probe_losses.float()
        ).to(device=self.states.device)[valid]
        counts = torch.bincount(heads, minlength=self.num_heads)[: self.num_heads]
        gain_sum = torch.zeros((self.num_heads,), device=self.states.device)
        win_sum = torch.zeros_like(gain_sum)
        gain_sum.index_add_(0, heads, mass_gain)
        win_sum.index_add_(0, heads, net_win)
        observed = counts.gt(0)
        gain = gain_sum / counts.clamp_min(1)
        wins = win_sum / counts.clamp_min(1)
        beta = 0.9
        self.safety_mass_gain_ema.copy_(
            torch.where(observed, beta * self.safety_mass_gain_ema + (1.0 - beta) * gain, self.safety_mass_gain_ema)
        )
        self.safety_net_win_ema.copy_(
            torch.where(observed, beta * self.safety_net_win_ema + (1.0 - beta) * wins, self.safety_net_win_ema)
        )
        self.safety_probe_count.add_(counts)
        ready = self.safety_probe_count.ge(self.safety_min_probe_count)
        bad = ready & observed & (
            self.safety_mass_gain_ema.lt(0.0) | self.safety_net_win_ema.lt(0.0)
        )
        self.safety_bad_windows.copy_(torch.where(bad, self.safety_bad_windows + 1, torch.zeros_like(self.safety_bad_windows)))
        decay = bad & self.safety_bad_windows.ge(self.safety_bad_probe_patience)
        self.safety_ratio.copy_(
            torch.where(decay, self.safety_ratio * self.safety_ratio_decay, self.safety_ratio)
        )
        self.safety_ratio.masked_fill_(self.safety_ratio.lt(0.125), 0.0)
        recovery = ready & observed & self.safety_ratio.eq(0.0) & (
            self.safety_mass_gain_ema.ge(0.0) & self.safety_net_win_ema.ge(0.0)
        )
        self.safety_ratio.copy_(
            torch.where(recovery, torch.full_like(self.safety_ratio, self.safety_ratio_decay), self.safety_ratio)
        )
        self.safety_bad_windows.masked_fill_(decay, 0)

    def reset(self, sequence_ids: list[int]) -> None:
        ids = self._ids(sequence_ids)
        if ids.numel() == 0:
            return
        self.states.index_fill_(0, ids, 0.0)
        self.effective_updates.index_fill_(0, ids, 0.0)
        self.alignment_ema.index_fill_(0, ids, 0.0)
        self.alignment_count.index_fill_(0, ids, 0.0)
        self.alignment_m2.index_fill_(0, ids, 0.0)

    def load_safety_state(self, state: dict[str, torch.Tensor | int] | None) -> None:
        if not state:
            return
        for name in (
            "safety_ratio",
            "safety_probe_count",
            "safety_mass_gain_ema",
            "safety_net_win_ema",
            "safety_bad_windows",
        ):
            value = state.get(name)
            current = getattr(self, name)
            if torch.is_tensor(value) and tuple(value.shape) == tuple(current.shape):
                current.copy_(value.to(device=current.device, dtype=current.dtype))
        self.safety_query_count = int(state.get("safety_query_count", 0))

    def safety_state(self) -> dict[str, torch.Tensor | int]:
        return {
            "safety_ratio": self.safety_ratio.detach().clone(),
            "safety_probe_count": self.safety_probe_count.detach().clone(),
            "safety_mass_gain_ema": self.safety_mass_gain_ema.detach().clone(),
            "safety_net_win_ema": self.safety_net_win_ema.detach().clone(),
            "safety_bad_windows": self.safety_bad_windows.detach().clone(),
            "safety_query_count": int(self.safety_query_count),
        }

    def stats(self) -> dict:
        rms = _rms(self.states, self.eps).squeeze(-1)
        all_ids = list(range(int(self.states.shape[0])))
        trust = self.trust(all_ids)
        return {
            "strict_horizon_pipeline": True,
            "horizon_resolved": True,
            "fast_state_rms_mean": float(rms.mean().detach().cpu()),
            "fast_state_rms_p95": float(torch.quantile(rms.flatten(), 0.95).detach().cpu()),
            "raw_fast_state_rms": float(rms.mean().detach().cpu()),
            "hint_trust": float(trust.mean().detach().cpu()),
            "hint_trust_mean": float(trust.mean().detach().cpu()),
            "head_fast_state_rms_mean": [float(x) for x in rms.mean(dim=0).detach().cpu()],
            "head_hint_trust_mean": [float(x) for x in trust.mean(dim=0).detach().cpu()],
            "head_safety_ratio": [float(x) for x in self.safety_ratio.detach().cpu()],
            "head_probe_count": [int(x) for x in self.safety_probe_count.detach().cpu()],
            "head_candidate_mass_gain_ema": [
                float(x) for x in self.safety_mass_gain_ema.detach().cpu()
            ],
            "head_reflex_net_wins_ema": [
                float(x) for x in self.safety_net_win_ema.detach().cpu()
            ],
            "head_effective_updates_mean": [
                float(x) for x in self.effective_updates.mean(dim=0).detach().cpu()
            ],
            "numerical_reset_count": int(self.numerical_reset_count.detach().cpu()),
        }


def merge_auxiliary_records(
    batches: list[dict[str, torch.Tensor]], max_records: int
) -> dict[str, torch.Tensor]:
    batches = [batch for batch in batches if batch and batch.get("hidden") is not None]
    if not batches:
        return {}

    def pad_width(value: torch.Tensor, width: int, fill_value: int | float | bool) -> torch.Tensor:
        current = int(value.shape[1])
        if current == width:
            return value
        if current > width:
            return value[:, :width]
        padding = torch.full(
            (int(value.shape[0]), width - current),
            fill_value,
            device=value.device,
            dtype=value.dtype,
        )
        return torch.cat((value, padding), dim=1)

    candidate_width = max(int(batch["candidate_ids"].shape[1]) for batch in batches)
    support_width = max(int(batch["support_ids"].shape[1]) for batch in batches)
    normalized: list[dict[str, torch.Tensor]] = []
    for batch in batches:
        row = dict(batch)
        row["candidate_ids"] = pad_width(row["candidate_ids"], candidate_width, -1)
        row["candidate_valid"] = pad_width(row["candidate_valid"], candidate_width, False)
        row["support_ids"] = pad_width(row["support_ids"], support_width, -1)
        row["support_valid"] = pad_width(row["support_valid"], support_width, False)
        row["target_logits"] = pad_width(row["target_logits"], support_width, 0.0)
        row["proposal_logits"] = pad_width(row["proposal_logits"], support_width, 0.0)
        normalized.append(row)
    merged = {
        key: torch.cat([batch[key] for batch in normalized], dim=0)
        for key in normalized[0]
    }
    limit = max(0, int(max_records))
    if limit and int(merged["hidden"].shape[0]) > limit:
        total = int(merged["hidden"].shape[0])
        head_indices = merged.get("head_indices")
        if torch.is_tensor(head_indices) and head_indices.numel() > 0:
            head_device = head_indices.device
            available_heads = [int(x) for x in torch.unique(head_indices).detach().cpu().tolist()]
            quota = max(1, limit // max(len(available_heads), 1))
            selected_parts: list[torch.Tensor] = []
            selected_mask = torch.zeros(total, device=head_device, dtype=torch.bool)
            for head_idx in available_heads:
                rows = head_indices.eq(head_idx).nonzero(as_tuple=False).flatten()
                rows = rows[-min(quota, int(rows.numel())) :]
                selected_parts.append(rows)
                selected_mask.index_fill_(0, rows, True)
            selected = torch.cat(selected_parts, dim=0) if selected_parts else torch.empty(0, device=head_device, dtype=torch.long)
            remaining = max(0, limit - int(selected.numel()))
            if remaining:
                recent = (~selected_mask).nonzero(as_tuple=False).flatten()[-remaining:]
                selected = torch.cat((selected, recent), dim=0)
            keep = selected.sort().values.to(device=merged["hidden"].device)
        else:
            keep = torch.arange(total - limit, total, device=merged["hidden"].device)
        merged = {key: value.index_select(0, keep) for key, value in merged.items()}
    return merged
