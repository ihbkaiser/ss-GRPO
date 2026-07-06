from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


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
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.num_heads = int(num_heads)
        self.tie_lm_head = bool(tie_lm_head)
        self.medusa_loss_decay = float(medusa_loss_decay)
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
    ) -> list[torch.Tensor]:
        if last_hidden.dim() == 2:
            last_hidden = last_hidden.unsqueeze(1)
        logits = self.forward(last_hidden, lm_head=lm_head, max_heads=max_heads)
        return [item[:, -1, :] for item in logits]

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
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if hidden_states.shape[:2] != input_ids.shape:
            raise ValueError("hidden_states and input_ids must share [B, T]")
        total_loss = hidden_states.new_zeros(())
        per_head: dict[str, float] = {}
        valid_heads = 0
        seq_len = input_ids.shape[1]
        for head_idx, head in enumerate(self.heads):
            # h_t already predicts y1=t+1 through the target LM head. MEDUSA
            # head 1 proposes y2=t+2 under the forced target root y1.
            shift = head_idx + 2
            if seq_len <= shift:
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
                total_loss = total_loss + float(weight) * head_loss
                per_head[f"head_{head_idx + 1}"] = float(head_loss.detach().cpu())
                per_head[f"head_{head_idx + 1}_tokens"] = cur_valid
                valid_heads += 1
        if valid_heads == 0:
            return hidden_states.sum() * 0.0, {"medusa_loss": 0.0}
        total_loss = total_loss / valid_heads
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
        )
        state = torch.load(load_directory / "medusa_heads.pt", map_location=map_location)
        model.load_state_dict(state)
        return model
