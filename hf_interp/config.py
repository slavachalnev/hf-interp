from typing import Optional, List
import logging
import random
import numpy as np
import torch
from transformers import PretrainedConfig


SUPPORTED_ACTIVATIONS = ["relu", "gelu", "silu", "gelu_new", "solu_ln", "gelu_fast"]


class HookedTransformerConfig(PretrainedConfig):
    model_type = "hooked_transformer"

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_ctx: int,
        d_head: int,
        model_name: str = "custom",
        n_heads: int = -1,
        d_mlp: Optional[int] = None,
        act_fn: Optional[str] = None,
        d_vocab: int = -1,
        eps: float = 1e-5,
        use_attn_result: bool = False,
        use_attn_scale: bool = True,
        use_split_qkv_input: bool = False,
        use_local_attn: bool = False,
        original_architecture: Optional[str] = None,
        from_checkpoint: bool = False,  ###
        checkpoint_index: Optional[int] = None,  ###
        checkpoint_label_type: Optional[str] = None,  ###
        checkpoint_value: Optional[int] = None,  ###
        tokenizer_name: Optional[str] = None,  ###
        window_size: Optional[int] = None,
        attn_types: Optional[List] = None,
        init_mode: str = "gpt2",  ###
        normalization_type: Optional[str] = "LN",
        # device: Optional[str] = None,  ###
        # n_devices: int = 1,  ###
        attention_dir: str = "causal",
        attn_only: bool = False,
        seed: Optional[int] = None,
        initializer_range: float = -1.0,  ###
        # init_weights: bool = True,  ###
        scale_attn_by_inverse_layer_idx: bool = False,
        positional_embedding_type: str = "standard",
        final_rms: bool = False,
        d_vocab_out: int = -1,
        parallel_attn_mlp: bool = False,
        rotary_dim: Optional[int] = None,
        n_params: Optional[int] = None,
        use_hook_tokens: bool = False,
        gated_mlp: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_layers = n_layers
        self.d_model = d_model
        self.n_ctx = n_ctx
        self.d_head = d_head
        self.model_name = model_name
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.act_fn = act_fn
        self.d_vocab = d_vocab
        self.eps = eps
        self.use_attn_result = use_attn_result
        self.use_attn_scale = use_attn_scale
        self.use_split_qkv_input = use_split_qkv_input
        self.use_local_attn = use_local_attn
        self.original_architecture = original_architecture
        self.from_checkpoint = from_checkpoint  ###
        self.checkpoint_index = checkpoint_index  ###
        self.checkpoint_label_type = checkpoint_label_type  ###
        self.checkpoint_value = checkpoint_value  ###
        self.tokenizer_name = tokenizer_name  ###
        self.window_size = window_size
        self.attn_types = attn_types
        self.init_mode = init_mode  ###
        self.normalization_type = normalization_type
        # self.device = device  ###
        # self.n_devices = n_devices  ###
        self.attention_dir = attention_dir
        self.attn_only = attn_only
        self.seed = seed
        self.initializer_range = initializer_range  ###
        # self.init_weights = init_weights  ###
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.positional_embedding_type = positional_embedding_type
        self.final_rms = final_rms
        self.d_vocab_out = d_vocab_out
        self.parallel_attn_mlp = parallel_attn_mlp
        self.rotary_dim = rotary_dim
        self.n_params = n_params
        self.use_hook_tokens = use_hook_tokens
        self.gated_mlp = gated_mlp

        self.post_init()
    

    def post_init(self):
        if self.n_heads == -1:
            self.n_heads = self.d_model // self.d_head

            if not self.d_model % (self.d_head) == 0:
                logging.warning(
                    f"d_model {self.d_model} is not divisible by d_head {self.d_head}. n_heads was inferred to be {self.n_heads}, rounding down the ratio."
                )

        if self.seed is not None:
            self.set_seed_everywhere(self.seed)
        if self.use_local_attn:
            assert (
                self.window_size is not None
            ), "window_size must be specified for local attention"
            assert (
                self.attn_types is not None
            ), "attn_types must be specified for local attention"
        if not self.attn_only:
            if self.d_mlp is None:
                # For some reason everyone hard codes in this hyper-parameter!
                self.d_mlp = self.d_model * 4
            assert (
                self.act_fn is not None
            ), "act_fn must be specified for non-attn-only models"
            assert (
                self.act_fn in SUPPORTED_ACTIVATIONS
            ), f"act_fn={self.act_fn} must be one of {SUPPORTED_ACTIVATIONS}"
        if self.initializer_range < 0:
            # Roughly copy the GPT-2 value, but proportional to sqrt(1/d_model)
            self.initializer_range = 0.8 / np.sqrt(self.d_model)

        if self.d_vocab_out == -1:
            # d_vocab_out defaults to d_vocab, unless there's an algorithmic task
            # If d_vocab is not set, it'll be inferred from tokenizer_name or from a tokenizer explicitly passed to HookedTransformer initialisation.
            self.d_vocab_out = self.d_vocab

        if self.positional_embedding_type == "rotary" and self.rotary_dim is None:
            self.rotary_dim = self.d_head

        # The number of parameters in attention layers (ignoring biases and layer norm). 4 because W_Q, W_K, W_V and W_O
        self.n_params = self.n_layers * (
            (self.d_model * self.d_head * self.n_heads * 4)
        )
        if not self.attn_only:
            # Number of parameters in MLP layers (ignoring biases and layer norm). 2 because W_in and W_out
            self.n_params += self.n_layers * self.d_model * self.d_mlp * 2

    @classmethod
    def from_dict(cls, config_dict):
        # TODO: inherit from PretrainedConfig.from_dict
        config = cls(**config_dict)
        return config
    
    def set_seed_everywhere(self, seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

