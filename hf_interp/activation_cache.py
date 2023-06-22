# copied from Neel Nanda's TransformerLens.

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch

import hf_interp.utils as utils


class ActivationCache:
    """
    A wrapper around a dictionary of cached activations from a model run, with a variety of helper functions. In general, any utility which is specifically about editing/processing activations should be a method here, while any utility which is more general should be a function in utils.py, and any utility which is specifically about model weights should be in HookedTransformer.py or components.py.

    NOTE: This is designed to be used with the HookedTransformer class, and will not work with other models. It's also designed to be used with all activations of HookedTransformer being cached, and some internal methods will break without that.

    WARNING: The biggest footgun and source of bugs in this code will be keeping track of indexes, dimensions, and the numbers of each. There are several kinds of activations:

    Internal attn head vectors: q, k, v, z. Shape [batch, pos, head_index, d_head]
    Internal attn pattern style results: pattern (post softmax), attn_scores (pre-softmax). Shape [batch, head_index, query_pos, key_pos]
    Attn head results: result. Shape [batch, pos, head_index, d_model]
    Internal MLP vectors: pre, post, mid (only used for solu_ln - the part between activation + layernorm). Shape [batch, pos, d_mlp]
    Residual stream vectors: resid_pre, resid_mid, resid_post, attn_out, mlp_out, embed, pos_embed, normalized (output of each LN or LNPre). Shape [batch, pos, d_model]
    LayerNorm Scale: scale. Shape [batch, pos, 1]

    Sometimes the batch dimension will be missing because we applied remove_batch_dim (used when batch_size=1), and we need functions to be robust to that. I THINK I've got everything working, but could easily be wrong!

    Type-Annotations key:
    layers_covered is the number of layers queried in functions that stack the residual stream.
    batch_and_pos_dims is the set of dimensions from batch and pos - by default this is ["batch", "pos"], but is only ["pos"] if we've removed the batch dimension and is [()] if we've removed batch dimension and are applying a pos slice which indexes a specific position.
    """

    def __init__(
        self, cache_dict: Dict[str, torch.Tensor], model, has_batch_dim: bool = True
    ):
        self.cache_dict = cache_dict
        self.model = model
        self.has_batch_dim = has_batch_dim
        self.has_embed = "hook_embed" in self.cache_dict
        self.has_pos_embed = "hook_pos_embed" in self.cache_dict

    def remove_batch_dim(self) -> ActivationCache:
        if self.has_batch_dim:
            for key in self.cache_dict:
                assert (
                    self.cache_dict[key].size(0) == 1
                ), f"Cannot remove batch dimension from cache with batch size > 1, for key {key} with shape {self.cache_dict[key].shape}"
                self.cache_dict[key] = self.cache_dict[key][0]
            self.has_batch_dim = False
        else:
            logging.warning(
                "Tried removing batch dimension after already having removed it."
            )
        return self

    def __repr__(self):
        return f"ActivationCache with keys {list(self.cache_dict.keys())}"

    def __getitem__(self, key) -> torch.Tensor:
        """
        This allows us to treat the activation cache as a dictionary, and do cache["key"] to it. We add bonus functionality to take in shorthand names or tuples - see utils.get_act_name for the full syntax and examples.

        Dimension order is (get_act_name, layer_index, layer_type), where layer_type is either "attn" or "mlp" or "ln1" or "ln2" or "ln_final", get_act_name is the name of the hook (without the hook_ prefix).
        """
        if key in self.cache_dict:
            return self.cache_dict[key]
        elif type(key) == str:
            return self.cache_dict[utils.get_act_name(key)]
        else:
            if len(key) > 1 and key[1] is not None:
                if key[1] < 0:
                    # Supports negative indexing on the layer dimension
                    key = (key[0], self.model.cfg.n_layers + key[1], *key[2:])
            return self.cache_dict[utils.get_act_name(*key)]

    def __len__(self):
        return len(self.cache_dict)

    def to(self, device: Union[str, torch.device]) -> ActivationCache:
        """
        Moves the cache to a device - mostly useful for moving it to CPU after model computation finishes to save GPU memory.

        Note that some methods will break unless the model is also moved to the same device, eg compute_head_results
        """
        self.cache_dict = {
            key: value.to(device) for key, value in self.cache_dict.items()
        }
        return self

    def keys(self):
        return self.cache_dict.keys()

    def values(self):
        return self.cache_dict.values()

    def items(self):
        return self.cache_dict.items()

    def __iter__(self):
        return self.cache_dict.__iter__()

    def __len__(self):
        return len(self.cache_dict)
