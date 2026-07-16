from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


class ChainDraftCell(nn.Module):
    """Lightweight recurrent proposal state used by optional Chain-MEDUSA."""

    def __init__(
        self,
        hidden_size: int,
        *,
        bottleneck_ratio: int = 8,
        gate_init: float = -3.0,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        hidden_size = int(hidden_size)
        bottleneck = max(64, hidden_size // max(1, int(bottleneck_ratio)))
        self.state_norm = nn.LayerNorm(hidden_size, dtype=dtype)
        self.token_norm = nn.LayerNorm(hidden_size, dtype=dtype)
        self.state_proj = nn.Linear(hidden_size, bottleneck, bias=False, dtype=dtype)
        self.token_proj = nn.Linear(hidden_size, bottleneck, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(bottleneck, hidden_size, bias=True, dtype=dtype)
        self.act = nn.SiLU()
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=dtype or torch.float32))
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, state: torch.Tensor, token_embedding: torch.Tensor) -> torch.Tensor:
        cell_dtype = self.state_norm.weight.dtype
        state_for_cell = state.to(dtype=cell_dtype)
        token_for_cell = token_embedding.to(dtype=cell_dtype)
        mixed = self.state_proj(self.state_norm(state_for_cell)) + self.token_proj(self.token_norm(token_for_cell))
        delta = self.out_proj(self.act(mixed))
        return state_for_cell + torch.sigmoid(self.gate) * delta


class MedusaPredictionHead(nn.Module):
    """A small residual transform that predicts one future token."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        dtype: torch.dtype | None = None,
        tie_lm_head: bool = True,
        lm_head: nn.Module | None = None,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.tie_lm_head = bool(tie_lm_head)
        self.fc = nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(hidden_size, dtype=dtype)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        if self.tie_lm_head:
            self.output = None
        else:
            self.output = nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype)
            if lm_head is not None and hasattr(lm_head, "weight") and lm_head.weight.shape == self.output.weight.shape:
                with torch.no_grad():
                    self.output.weight.copy_(lm_head.weight.detach().to(dtype=self.output.weight.dtype))

    def project_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        head_dtype = self.norm.weight.dtype
        hidden_for_head = hidden_states.to(dtype=head_dtype)
        residual = self.act(self.fc(self.norm(hidden_for_head)))
        return hidden_for_head + residual

    def forward(self, hidden_states: torch.Tensor, lm_head: nn.Module | None = None) -> torch.Tensor:
        medusa_hidden = self.project_hidden(hidden_states)
        if self.output is not None:
            return self.output(medusa_hidden)
        if lm_head is None:
            raise ValueError("lm_head is required when tie_lm_head=True")
        lm_dtype = getattr(lm_head.weight, "dtype", medusa_hidden.dtype)
        return lm_head(medusa_hidden.to(dtype=lm_dtype))


class MedusaHeads(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 3,
        *,
        dtype: torch.dtype | None = None,
        tie_lm_head: bool = True,
        lm_head: nn.Module | None = None,
        medusa_loss_decay: float = 0.8,
        chain_bottleneck_ratio: int = 8,
        chain_gate_init: float = -3.0,
        reflex_fast_state_dim: int = 0,
        reflex_init_scale: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.num_heads = int(num_heads)
        self.tie_lm_head = bool(tie_lm_head)
        self.medusa_loss_decay = float(medusa_loss_decay)
        self.chain_bottleneck_ratio = int(chain_bottleneck_ratio)
        self.chain_gate_init = float(chain_gate_init)
        self.reflex_fast_state_dim = int(reflex_fast_state_dim or 0)
        self.reflex_init_scale = float(reflex_init_scale)
        self.heads = nn.ModuleList(
            [
                MedusaPredictionHead(
                    hidden_size,
                    vocab_size,
                    dtype=dtype,
                    tie_lm_head=tie_lm_head,
                    lm_head=lm_head,
                )
                for _ in range(num_heads)
            ]
        )
        self.chain_cell = ChainDraftCell(
            hidden_size,
            bottleneck_ratio=chain_bottleneck_ratio,
            gate_init=chain_gate_init,
            dtype=dtype,
        )
        if self.reflex_fast_state_dim > 0:
            self.reflex_down = nn.Linear(hidden_size, self.reflex_fast_state_dim, bias=False, dtype=dtype)
            self.reflex_up = nn.ModuleList(
                [nn.Linear(self.reflex_fast_state_dim, hidden_size, bias=False, dtype=dtype) for _ in range(num_heads)]
            )
            self._init_reflex_parameters()
        else:
            self.reflex_down = None
            self.reflex_up = nn.ModuleList()

    def _init_reflex_parameters(self) -> None:
        if self.reflex_down is None:
            return
        nn.init.normal_(self.reflex_down.weight, mean=0.0, std=1.0 / max(self.hidden_size, 1) ** 0.5)
        with torch.no_grad():
            init_scale = float(self.reflex_init_scale)
            for up in self.reflex_up:
                if init_scale <= 0.0:
                    up.weight.zero_()
                else:
                    tied = self.reflex_down.weight.detach().t().contiguous() * init_scale
                    up.weight.copy_(tied.to(dtype=up.weight.dtype))

    def reflex_delta(
        self,
        fast_state: torch.Tensor | None,
        head_idx: int,
        *,
        max_norm: float = 0.0,
        scale: float = 1.0,
        normalize: bool = True,
    ) -> torch.Tensor | None:
        if fast_state is None or self.reflex_fast_state_dim <= 0 or not self.reflex_up:
            return None
        head_idx = max(0, min(int(head_idx), len(self.reflex_up) - 1))
        up = self.reflex_up[head_idx]
        delta = up(fast_state.to(device=up.weight.device, dtype=up.weight.dtype))
        if normalize:
            delta_float = torch.nan_to_num(delta.float(), nan=0.0, posinf=0.0, neginf=0.0)
            rms = delta_float.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            delta = (delta_float / rms).to(dtype=delta.dtype)
        if max_norm and max_norm > 0:
            norm = torch.nan_to_num(delta.float(), nan=0.0, posinf=0.0, neginf=0.0).norm(dim=-1, keepdim=True).clamp_min(1e-6)
            norm_scale = torch.clamp(float(max_norm) / norm, max=1.0).to(dtype=delta.dtype)
            delta = delta * norm_scale
        if float(scale) != 1.0:
            delta = delta * float(scale)
        return delta

    def add_reflex_delta(
        self,
        hidden_states: torch.Tensor,
        fast_state: torch.Tensor | None,
        head_idx: int,
        *,
        max_norm: float = 0.0,
        scale: float = 1.0,
        normalize: bool = True,
    ) -> torch.Tensor:
        delta = self.reflex_delta(fast_state, head_idx, max_norm=max_norm, scale=scale, normalize=normalize)
        if delta is None:
            return hidden_states
        if hidden_states.dim() == 3 and delta.dim() == 2:
            delta = delta.unsqueeze(1)
        return hidden_states + delta.to(device=hidden_states.device, dtype=hidden_states.dtype)

    def feedback_to_fast_state(self, hidden_feedback: torch.Tensor) -> torch.Tensor:
        if self.reflex_down is None:
            raise RuntimeError("Reflex feedback projection is not enabled")
        return self.reflex_down(hidden_feedback.to(device=self.reflex_down.weight.device, dtype=self.reflex_down.weight.dtype))

    def project_hidden_for_head(
        self,
        hidden_states: torch.Tensor,
        head_idx: int,
        fast_state: torch.Tensor | None = None,
        reflex_clip_norm: float = 0.0,
        reflex_scale: float = 1.0,
        reflex_normalize: bool = True,
    ) -> torch.Tensor:
        head_idx = max(0, min(int(head_idx), len(self.heads) - 1))
        projected = self.heads[head_idx].project_hidden(hidden_states)
        return self.add_reflex_delta(
            projected,
            fast_state,
            head_idx,
            max_norm=reflex_clip_norm,
            scale=reflex_scale,
            normalize=reflex_normalize,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module | None = None,
        *,
        max_heads: int | None = None,
    ) -> list[torch.Tensor]:
        if max_heads is None:
            heads = self.heads
        else:
            max_heads = max(0, min(int(max_heads), len(self.heads)))
            if max_heads == 0:
                return []
            heads = self.heads[:max_heads]
        return [head(hidden_states, lm_head=lm_head) for head in heads]

    def logits_for_last_hidden(
        self,
        last_hidden: torch.Tensor,
        lm_head: nn.Module | None = None,
        *,
        max_heads: int | None = None,
        fast_state: torch.Tensor | None = None,
        reflex_clip_norm: float = 0.0,
        reflex_scale: float = 1.0,
        reflex_normalize: bool = True,
    ) -> list[torch.Tensor]:
        if last_hidden.dim() == 2:
            last_hidden = last_hidden.unsqueeze(1)
        if max_heads is None:
            max_heads = len(self.heads)
        max_heads = max(0, min(int(max_heads), len(self.heads)))
        if max_heads == 0:
            return []
        out = []
        for head_idx in range(max_heads):
            medusa_hidden = self.project_hidden_for_head(
                last_hidden,
                head_idx,
                fast_state=fast_state,
                reflex_clip_norm=reflex_clip_norm,
                reflex_scale=reflex_scale,
                reflex_normalize=reflex_normalize,
            )
            if self.heads[head_idx].output is not None:
                logits = self.heads[head_idx].output(medusa_hidden)
            else:
                if lm_head is None:
                    raise ValueError("lm_head is required when tie_lm_head=True")
                lm_dtype = getattr(lm_head.weight, "dtype", medusa_hidden.dtype)
                logits = lm_head(medusa_hidden.to(dtype=lm_dtype))
            out.append(logits[:, -1, :])
        return out

    def chain_next_state(
        self,
        state: torch.Tensor,
        token_ids: torch.Tensor,
        embedding_layer: nn.Module,
    ) -> torch.Tensor:
        token_embedding = embedding_layer(token_ids.to(device=state.device)).detach()
        return self.chain_cell(state, token_embedding)

    def chain_logits_from_state(self, state: torch.Tensor, lm_head: nn.Module) -> torch.Tensor:
        lm_dtype = getattr(lm_head.weight, "dtype", state.dtype)
        return lm_head(state.to(dtype=lm_dtype))

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        lm_head: nn.Module | None = None,
        loss_mask: torch.Tensor | None = None,
        ignore_index: int = -100,
        chunk_size: int = 128,
        chain_loss_weight: float = 0.0,
        chain_max_depth: int | None = None,
        chain_bootstrap_from_medusa: bool = True,
        embedding_layer: nn.Module | None = None,
        head_weights: dict[int | str, float] | list[float] | tuple[float, ...] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if hidden_states.shape[:2] != input_ids.shape:
            raise ValueError("hidden_states and input_ids must share [B, T]")
        total_loss = hidden_states.new_zeros(())
        per_head: dict[str, float] = {}
        valid_weight_sum = 0.0
        seq_len = input_ids.shape[1]

        def aux_head_weight(head_idx: int) -> float:
            if head_weights is None:
                return 1.0
            if isinstance(head_weights, (list, tuple)):
                return float(head_weights[head_idx]) if head_idx < len(head_weights) else 0.0
            raw = head_weights.get(str(head_idx + 1))
            if raw is None:
                raw = head_weights.get(head_idx + 1)
            if raw is None:
                raw = head_weights.get(str(head_idx))
            if raw is None:
                raw = head_weights.get(head_idx)
            return float(raw or 0.0)

        for head_idx, head in enumerate(self.heads):
            # h_t already predicts y1=t+1 through the target LM head. MEDUSA
            # head 1 proposes y2=t+2 under the forced target root y1.
            shift = head_idx + 2
            if seq_len <= shift:
                continue
            aux_weight = aux_head_weight(head_idx)
            if aux_weight <= 0.0:
                per_head[f"head_{head_idx + 1}_weight"] = 0.0
                continue
            cur_losses = []
            cur_valid = 0
            weight = self.medusa_loss_decay ** head_idx
            for start in range(0, seq_len - shift, chunk_size):
                end = min(start + chunk_size, seq_len - shift)
                hidden = hidden_states[:, start:end, :]
                labels = input_ids[:, start + shift : end + shift].to(hidden.device)
                valid = attention_mask[:, start + shift : end + shift].bool().to(hidden.device)
                if loss_mask is not None:
                    valid = valid & loss_mask[:, start + shift : end + shift].bool().to(hidden.device)
                labels = labels.masked_fill(~valid, ignore_index)
                if not bool(valid.any().item()):
                    continue
                logits = head(hidden, lm_head=lm_head).float()
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=ignore_index,
                )
                cur_losses.append(loss)
                cur_valid += int(valid.sum().item())
                del logits
            if cur_losses:
                head_loss = torch.stack(cur_losses).mean()
                total_loss = total_loss + float(aux_weight * weight) * head_loss
                per_head[f"head_{head_idx + 1}"] = float(head_loss.detach().cpu())
                per_head[f"head_{head_idx + 1}_tokens"] = cur_valid
                per_head[f"head_{head_idx + 1}_weight"] = float(aux_weight)
                valid_weight_sum += float(aux_weight)
        if valid_weight_sum > 0:
            total_loss = total_loss / valid_weight_sum
            per_head["parallel_medusa_loss"] = float(total_loss.detach().cpu())
        else:
            total_loss = hidden_states.sum() * 0.0

        chain_weight = float(chain_loss_weight or 0.0)
        chain_losses = []
        if chain_weight > 0.0 and embedding_layer is not None and lm_head is not None:
            max_depth = min(int(chain_max_depth or self.num_heads), self.num_heads)
            for depth_idx in range(max_depth):
                shift = depth_idx + 2
                if chain_bootstrap_from_medusa and shift <= 2:
                    continue
                if seq_len <= shift:
                    continue
                aux_weight = aux_head_weight(depth_idx)
                if aux_weight <= 0.0:
                    continue
                weight = self.medusa_loss_decay ** depth_idx
                for start in range(0, seq_len - shift, chunk_size):
                    end = min(start + chunk_size, seq_len - shift)
                    if chain_bootstrap_from_medusa and len(self.heads) > 0:
                        state = self.heads[0].project_hidden(hidden_states[:, start:end, :].detach())
                        first_prev_offset = 2
                    else:
                        state = hidden_states[:, start:end, :].detach()
                        first_prev_offset = 1
                    labels = input_ids[:, start + shift : end + shift].to(state.device)
                    valid = attention_mask[:, start + shift : end + shift].bool().to(state.device)
                    if loss_mask is not None:
                        valid = valid & loss_mask[:, start + shift : end + shift].bool().to(state.device)
                    labels = labels.masked_fill(~valid, ignore_index)
                    if not bool(valid.any().item()):
                        continue
                    for prev_offset in range(first_prev_offset, shift):
                        prev_tokens = input_ids[:, start + prev_offset : end + prev_offset].to(state.device)
                        state = self.chain_next_state(state, prev_tokens, embedding_layer)
                    logits = self.chain_logits_from_state(state, lm_head).float()
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        labels.reshape(-1),
                        ignore_index=ignore_index,
                    )
                    chain_losses.append(float(aux_weight * weight) * loss)
                    del logits, state
        if chain_losses:
            chain_loss = torch.stack(chain_losses).mean()
            total_loss = total_loss + chain_weight * chain_loss
            per_head["chain_loss"] = float(chain_loss.detach().cpu())
            per_head["chain_loss_weight"] = chain_weight
        if valid_weight_sum == 0 and not chain_losses:
            return hidden_states.sum() * 0.0, {"medusa_loss": 0.0}
        per_head["medusa_loss"] = float(total_loss.detach().cpu())
        return total_loss, per_head

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        config = {
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_heads": self.num_heads,
            "tie_lm_head": self.tie_lm_head,
            "medusa_loss_decay": self.medusa_loss_decay,
            "chain_bottleneck_ratio": self.chain_bottleneck_ratio,
            "chain_gate_init": self.chain_gate_init,
            "reflex_fast_state_dim": self.reflex_fast_state_dim,
            "reflex_init_scale": self.reflex_init_scale,
        }
        (save_directory / "medusa_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (save_directory / "medusa_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        torch.save(self.state_dict(), save_directory / "medusa_heads.pt")

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        lm_head: nn.Module | None = None,
        chain_bottleneck_ratio: int | None = None,
        chain_gate_init: float | None = None,
        reflex_fast_state_dim: int | None = None,
        reflex_init_scale: float | None = None,
    ) -> "MedusaHeads":
        load_directory = Path(load_directory)
        config = json.loads((load_directory / "medusa_config.json").read_text(encoding="utf-8"))
        model = cls(
            config["hidden_size"],
            config["vocab_size"],
            config["num_heads"],
            dtype=dtype,
            tie_lm_head=config.get("tie_lm_head", True),
            lm_head=lm_head,
            medusa_loss_decay=config.get("medusa_loss_decay", 0.8),
            chain_bottleneck_ratio=chain_bottleneck_ratio if chain_bottleneck_ratio is not None else config.get("chain_bottleneck_ratio", 8),
            chain_gate_init=chain_gate_init if chain_gate_init is not None else config.get("chain_gate_init", -3.0),
            reflex_fast_state_dim=reflex_fast_state_dim if reflex_fast_state_dim is not None else config.get("reflex_fast_state_dim", 0),
            reflex_init_scale=reflex_init_scale if reflex_init_scale is not None else config.get("reflex_init_scale", 0.0),
        )
        state = torch.load(load_directory / "medusa_heads.pt", map_location=map_location)
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in current and tuple(current[key].shape) == tuple(value.shape)
        }
        model.load_state_dict(compatible, strict=False)
        model._loaded_state_keys = set(state.keys())
        model._loaded_compatible_keys = set(compatible.keys())
        return model
