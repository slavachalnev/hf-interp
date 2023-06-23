from typing import Optional, List

from transformers import PretrainedConfig


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
