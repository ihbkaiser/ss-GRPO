from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from flashgrpo.models.qwen_flashgrpo_wrapper import autocast_dtype, unwrap_causal_lm


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
    grad_clip_norm: float = 1.0


class OnlineMedusaTrainer:
    def __init__(self, target_model, medusa_heads, optimizer, config: OnlineMedusaConfig):
        self.target_model = target_model
        self.medusa_heads = medusa_heads
        self.optimizer = optimizer
        self.config = config

    def update(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
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
            )
            if not torch.isfinite(loss):
                continue
            (loss / grad_denom).backward()
            total_loss += float(loss.detach().cpu())
            total_tokens += int(mb_mask.sum().item())
            updates += 1
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    per_head_sums[key] = per_head_sums.get(key, 0.0) + float(value)

        if updates:
            torch.nn.utils.clip_grad_norm_(self.medusa_heads.parameters(), cfg.grad_clip_norm)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        elapsed = time.time() - start_time
        out = {
            "medusa_loss": total_loss / max(updates, 1),
            "head_update_tokens": int(total_tokens),
            "head_update_time": elapsed,
            "head_update_tokens_per_sec": total_tokens / max(elapsed, 1e-9),
            "head_update_steps": int(updates),
        }
        for key, value in per_head_sums.items():
            out[key] = value / max(updates, 1)
        return out
