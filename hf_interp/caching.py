from dataclasses import dataclass
from typing import List

import torch
from jaxtyping import Float

from hf_interp.config import HookedTransformerConfig


@dataclass
class HookedTransformerKeyValueCacheEntry:
    past_keys: Float[torch.Tensor, "batch pos_so_far n_heads d_head"]
    past_values: Float[torch.Tensor, "batch pos_so_far n_heads d_head"]

    @classmethod
    def init_cache_entry(
        cls,
        cfg: HookedTransformerConfig,
        device: torch.device,
        batch_size: int = 1,
    ):
        return cls(
            past_keys=torch.empty(
                (batch_size, 0, cfg.n_heads, cfg.d_head), device=device
            ),
            past_values=torch.empty(
                (batch_size, 0, cfg.n_heads, cfg.d_head), device=device
            ),
        )

    def append(
        self,
        new_keys: Float[torch.Tensor, "batch new_tokens n_heads d_head"],
        new_values: Float[torch.Tensor, "batch new_tokens n_heads d_head"],
    ):
        updated_keys: Float[
            torch.Tensor, "batch pos_so_far_plus_new_tokens n_heads d_head"
        ] = torch.cat([self.past_keys, new_keys], dim=1)
        updated_values: Float[
            torch.Tensor, "batch pos_so_far_plus_new_tokens n_heads d_head"
        ] = torch.cat([self.past_values, new_values], dim=1)
        self.past_keys = updated_keys
        self.past_values = updated_values
        return updated_keys, updated_values
