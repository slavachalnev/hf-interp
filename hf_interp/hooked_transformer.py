# Mostly copied from Neel Nanda's TransformerLens.

import logging
from functools import lru_cache
import tempfile
from typing import Dict, List, NamedTuple, Optional, Tuple, Union, overload

import einops
import numpy as np
import torch
import torch.nn as nn
import tqdm.auto as tqdm
from fancy_einsum import einsum
from jaxtyping import Float, Int
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import accelerate
from accelerate.hooks import remove_hook_from_module
from typeguard import typeguard_ignore
from typing_extensions import Literal

# import transformer_lens.loading_from_pretrained as loading
import hf_interp.utils as utils
from hf_interp.config import HookedTransformerConfig
from hf_interp.activation_cache import ActivationCache
from hf_interp.components import (
    Embed,
    LayerNorm,
    LayerNormPre,
    PosEmbed,
    RMSNorm,
    RMSNormPre,
    TransformerBlock,
    Unembed,
)
from hf_interp.factored_matrix import FactoredMatrix
from hf_interp.hooks import HookedRootModule, HookPoint

import hf_interp.load as loading

# Note - activation cache is used with run_with_cache, past_key_value_caching is used for generation.
from hf_interp.kv_caching import HookedTransformerKeyValueCache


SingleLoss = Float[torch.Tensor, ""]  # Type alias for a single element tensor
LossPerToken = Float[torch.Tensor, "batch pos-1"]
Loss = Union[SingleLoss, LossPerToken]


# Named tuple object for if we want to output both logits and loss
class Output(NamedTuple):
    logits: Float[torch.Tensor, "batch pos d_vocab"]
    loss: Loss


class HookedTransformer(HookedRootModule):
    """
    This class implements a full Transformer using the components in ./components.py, with
    HookPoints on every interesting activation. It inherits from HookedRootModule.

    It can have a pretrained Transformer's weights automatically loaded in via the HookedTransformer.from_pretrained
    class method. It can also be instantiated with randomly initialized weights via __init__ and being passed a dict or
    HookedTransformerConfig object.
    """

    # TODO: Add generate method.

    # TODO: Figure out why this class attribute is ignored when loading.
    _keep_in_fp32_modules: List[str] = ["LayerNorm", "LayerNormPre", "RMSNorm", "RMSNormPre"]

    def __init__(
        self,
        config,
        tokenizer=None,
    ):
        """
        Model initialization. Note that if you want to load the model from pretrained weights, you should use the
        HookedTransformer.from_pretrained() class method instead of this one.

        config Union[HookedTransformerConfig, Dict]: The config to use for the
            model.
        tokenizer (*optional): The tokenizer to use for the model. If not
            provided, it is inferred from config.tokenizer_name or initialized to None.
            If None, then the model cannot be passed strings, and d_vocab must be explicitly set.
        """
        super().__init__(config=config)

        if tokenizer is not None:
            self.set_tokenizer(tokenizer)
        elif self.config.tokenizer_name is not None:
            # If we have a tokenizer name, we can load it from HuggingFace
            if "llama" in self.config.tokenizer_name:
                # llama tokenizer requires special handling
                print("Warning: LLaMA tokenizer not loaded. Please load manually.")
            else:
                self.set_tokenizer(
                    AutoTokenizer.from_pretrained(self.config.tokenizer_name)
                )
        else:
            # If no tokenizer name is provided, we assume we're training on an algorithmic task and will pass in tokens
            # directly. In this case, we don't need a tokenizer.
            assert (
                self.config.d_vocab != -1
            ), "Must provide a tokenizer if d_vocab is not provided"
            self.tokenizer = None

        self.embed = Embed(self.config)
        self.hook_embed = HookPoint()  # [batch, pos, d_model]

        if self.config.positional_embedding_type != "rotary":
            self.pos_embed = PosEmbed(self.config)
            self.hook_pos_embed = HookPoint()  # [batch, pos, d__dictmodel]

        if self.config.use_hook_tokens:
            self.hook_tokens = HookPoint()  # [batch, pos]

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(self.config, block_index)
                for block_index in range(self.config.n_layers)
            ]
        )

        if self.config.normalization_type == "RMS":
            self.ln_final = RMSNorm(self.config)
        elif self.config.normalization_type == "RMSPre":
            self.ln_final = RMSNormPre(self.config)
        elif self.config.normalization_type == "LN":
            if self.config.final_rms:
                self.ln_final = RMSNorm(self.config)
            else:
                self.ln_final = LayerNorm(self.config)
        elif self.config.normalization_type == "LNPre":
            # We've folded in LayerNorm weights, so just need the center + scale parts
            if self.config.final_rms:
                self.ln_final = RMSNormPre(self.config)
            else:
                self.ln_final = LayerNormPre(self.config)
        elif self.config.normalization_type is None:
            # If it's None, don't create either layer
            pass
        else:
            logging.warning(
                f"Invalid normalization_type passed in {self.config.normalization_type}"
            )
        self.unembed = Unembed(self.config)

        if self.config.init_weights:
            self.apply(self.init_weights)
        
        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        self.setup()

    def forward(
        self,
        input: Union[str, List[str], Int[torch.Tensor, "batch pos"]],
        return_type: Optional[str] = "logits",
        loss_per_token: bool = False,
        prepend_bos: bool = True,
        stop_at_layer: Optional[int] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> Union[
        None,
        Float[torch.Tensor, "batch pos d_vocab"],
        Loss,
        Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
    ]:
        """Input is either a batch of tokens ([batch, pos]) or a text string, a string is automatically tokenized to a
        batch of a single element. The prepend_bos flag only applies when inputting a text string.

        return_type Optional[str]: The type of output to return. Can be one of: None (return nothing, don't calculate
            logits), 'logits' (return logits), 'loss' (return cross-entropy loss), 'both' (return logits and loss)
        loss_per_token bool: Whether to return the (next token prediction) loss per token (True) or average (False).
            Average loss is a scalar (averaged over position *and* batch), per-token loss is a tensor ([batch, position-1])
            - position-1 because we're predicting the next token, and there's no specified next token for the final
            token. Defaults to False.
        prepend_bos bool: Whether to prepend the BOS token to the input. Only applies when input is a string. Defaults
            to True (unlike to_tokens) - even for models not explicitly trained with this, heads often use the first
            position as a resting position and accordingly lose information from the first token, so this empirically
            seems to give better results.
        stop_at_layer Optional[int]: If not None, stop the forward pass at the specified layer. Exclusive - ie,
        stop_at_layer = 0 will only run the embedding layer, stop_at_layer = 1 will run the embedding layer and the
        first transformer block, etc. Supports negative indexing. Useful for analysis of intermediate layers, eg finding
        neuron activations in layer 3 of a 24 layer model. Defaults to None (run the full model).

        Note that loss is the standard "predict the next token" cross-entropy loss for GPT-2 style language models -
        if you want a custom loss function, the recommended behaviour is returning the logits and then applying your
        custom loss function.
        """
        if type(input) == str or type(input) == list:
            # If text, convert to tokens (batch_size=1)
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if passing a string to the model"
            # This is only intended to support passing in a single string
            tokens = self.to_tokens(input, prepend_bos=prepend_bos)
        else:
            tokens = input
        if len(tokens.shape) == 1:
            # If tokens are a rank 1 tensor, add a dummy batch dimension to avoid things breaking.
            tokens = tokens[None]

        # If we're doing caching, then we reuse keys and values from previous runs, as that's the only
        # way that past activations will affect the final logits. The cache contains those so we don't
        # need to recompute them. This is useful for generating text. As we have absolute positional
        # encodings, to implement this we have a `pos_offset` variable, defaulting to zero, which says
        # to offset which positional encodings are used (cached keys and values were calculated with
        # their own positional encodings).
        if past_kv_cache is None:
            pos_offset = 0
        else:
            batch_size, ctx_length = tokens.shape
            (
                cached_batch_size,
                cache_ctx_length,
                num_heads_in_cache,
                d_head_in_cache,
            ) = past_kv_cache[0].past_keys.shape
            assert cached_batch_size == batch_size
            assert num_heads_in_cache == self.config.n_heads
            assert d_head_in_cache == self.config.d_head
            # If we want to generate from the empty string, we'd pass in an empty cache, so we need to handle that case
            assert (
                cache_ctx_length == 0 or ctx_length == 1
            ), "Pass in one token at a time after loading cache"
            pos_offset = cache_ctx_length
        if self.config.use_hook_tokens:
            tokens = self.hook_tokens(tokens)
        embed = self.hook_embed(self.embed(tokens))  # [batch, pos, d_model]
        if self.config.positional_embedding_type == "standard":
            pos_embed = self.hook_pos_embed(
                self.pos_embed(tokens, pos_offset)
            )  # [batch, pos, d_model]
            residual = embed + pos_embed  # [batch, pos, d_model]
            shortformer_pos_embed = None
        elif self.config.positional_embedding_type == "shortformer":
            # If we're using shortformer style attention, we don't add the positional embedding to the residual stream.
            # See HookedTransformerConfig for details
            pos_embed = self.hook_pos_embed(
                self.pos_embed(tokens, pos_offset)
            )  # [batch, pos, d_model]
            residual = embed
            shortformer_pos_embed = pos_embed
        elif self.config.positional_embedding_type == "rotary":
            # Rotary doesn't use positional embeddings, instead they're applied when dot producting keys and queries.
            # See HookedTransformerConfig for details
            residual = embed
            shortformer_pos_embed = None
        else:
            raise ValueError(
                f"Invalid positional_embedding_type passed in {self.config.positional_embedding_type}"
            )

        if stop_at_layer is None:
            # We iterate through every block by default
            transformer_block_list = self.blocks
        else:
            # If we explicitly want to stop at a layer, we only iterate through the blocks up to that layer. Note that
            # this is exclusive, eg stop_at_layer==0 means to only run the embed, stop_at_layer==-1 means to run every
            # layer *apart* from the final one, etc.
            transformer_block_list = self.blocks[:stop_at_layer]  # type: ignore

        for i, block in enumerate(transformer_block_list):  # type: ignore
            # Note that each block includes skip connections, so we don't need
            # residual + block(residual)
            # If we're using multiple GPUs, we need to send the residual and shortformer_pos_embed to the correct GPU

            residual = block(
                residual,
                past_kv_cache_entry=past_kv_cache[i]
                if past_kv_cache is not None
                else None,  # Cache contains a list of HookedTransformerKeyValueCache objects, one for each block
                shortformer_pos_embed=shortformer_pos_embed,
            )  # [batch, pos, d_model]

        if stop_at_layer is not None:
            # When we stop at an early layer, we end here rather than doing further computation
            return None

        if self.config.normalization_type is not None:
            residual = self.ln_final(residual)  # [batch, pos, d_model]
        if return_type is None:
            return None
        else:
            logits = self.unembed(residual)  # [batch, pos, d_vocab]
            if return_type == "logits":
                return logits
            else:
                loss = self.loss_fn(logits, tokens, per_token=loss_per_token)

                if return_type == "loss":
                    return loss
                elif return_type == "both":
                    return Output(logits, loss)
                else:
                    logging.warning(f"Invalid return_type passed in: {return_type}")
                    return None

    def loss_fn(
        self,
        logits: Float[torch.Tensor, "batch pos d_vocab"],
        tokens: Int[torch.Tensor, "batch pos"],
        per_token: bool = False,
    ):
        # TODO: Just call utils.lm_cross_entropy_loss directly

        """
        Wrapper around utils.lm_cross_entropy_loss, used in forward() with return_type=="loss" or "both".
        """
        if tokens.device != logits.device:
            tokens = tokens.to(logits.device)
        return utils.lm_cross_entropy_loss(logits, tokens, per_token)

    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[True] = True, **kwargs
    ) -> Tuple[Output, ActivationCache]:
        ...

    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[False] = False, **kwargs
    ) -> Tuple[Output, Dict[str, torch.Tensor]]:
        ...

    def run_with_cache(
        self, *model_args, return_cache_object=True, remove_batch_dim=False, **kwargs
    ) -> Tuple[
        Union[
            None,
            Float[torch.Tensor, "batch pos d_vocab"],
            Loss,
            Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
        ],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """
        Wrapper around run_with_cache in HookedRootModule. If return_cache_object is True, this will return an
        ActivationCache object, with a bunch of useful HookedTransformer specific methods, otherwise it will return a
        dictionary of activations as in HookedRootModule.
        """
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        if return_cache_object:
            cache = ActivationCache(
                cache_dict, self, has_batch_dim=not remove_batch_dim
            )
            return out, cache
        else:
            return out, cache_dict
    
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        if device is not None:
            self.tie_weights()

            if device == 'cpu':
                device_map = {"": 'cpu'}
            elif device == 'mps':
                device_map = {"": 'mps'}
            else:
                device_map = accelerate.infer_auto_device_map(self, no_split_module_classes=self._no_split_modules())
            
            super().to('cpu')
            remove_hook_from_module(self, recurse=True)

            return accelerate.dispatch_model(self, device_map=device_map)

        if dtype is not None:
            return super().to(*args, **kwargs)
    
    @classmethod
    def from_pretrained(
        cls,
        model_name,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        checkpoint_index = None,
        checkpoint_value = None,
        hf_model = None,
        device_map = 'auto',
        device = None,
        **kwargs
    ):
        """Loads a pretrained model from HuggingFace, and converts it to a HookedTransformer."""

        # TODO: check device properly
        if device == 'cpu':
            device_map = {"": 'cpu'}
        elif device == 'mps':
            device_map = {"": 'mps'}

        # Get the model name used in HuggingFace, rather than the alias.
        official_model_name = loading.get_official_model_name(model_name)

        # Load the config into an HookedTransformerConfig object. If loading from a
        # checkpoint, the config object will contain the information about the
        # checkpoint
        config = loading.get_pretrained_model_config(
            official_model_name,
            checkpoint_index=checkpoint_index,
            checkpoint_value=checkpoint_value,
            fold_ln=fold_ln,
            **kwargs,
        )

        if config.positional_embedding_type == "shortformer":
            if fold_ln:
                logging.warning(
                    "You tried to specify fold_ln=True for a shortformer model, but this can't be done! Setting fold_"
                    "ln=False instead."
                )
                fold_ln = False
            if center_unembed:
                logging.warning(
                    "You tried to specify center_unembed=True for a shortformer model, but this can't be done! "
                    "Setting center_unembed=False instead."
                )
                center_unembed = False
            if center_writing_weights:
                logging.warning(
                    "You tried to specify center_writing_weights=True for a shortformer model, but this can't be done! "
                    "Setting center_writing_weights=False instead."
                )
                center_writing_weights = False

        # Get the state dict of the model (ie a mapping of parameter names to tensors), processed to match the
        # HookedTransformer parameter names.
        state_dict = loading.get_pretrained_state_dict(
            official_model_name, config, hf_model,
            # **kwargs # TODO: check if passing kwargs is needed. It breaks float16.
        )

        state_dict = cls.add_missing_keys(state_dict)

        if fold_ln:
            state_dict = cls.fold_layer_norm(state_dict, config)
        if center_writing_weights:
            state_dict = cls.center_writing_weights(state_dict, config)
        if center_unembed:
            state_dict = cls.center_unembed(state_dict, config)

        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            # Save the state dict to a temporary file
            torch.save(state_dict, tmp.name)

            del state_dict
            
            # Load the model from the temporary file
            model = super().from_pretrained(
                tmp.name,
                config=config,
                low_cpu_mem_usage=True,
                device_map=device_map,
                **kwargs
            )

        return model
    
    @staticmethod
    def add_missing_keys(state_dict: Dict[str, torch.Tensor]):
        # make sure unembed.b_U is in state_dict
        if "unembed.b_U" not in state_dict:
            state_dict["unembed.b_U"] = torch.zeros(state_dict["unembed.W_U"].shape[1])

        return state_dict
    
    @classmethod
    def _no_split_modules(cls):
        """Returns a list of modules that should not be split into multiple GPUs."""
        return ["Embed", "Unembed", "PosEmbed", "TransformerBlock"]
    
    @staticmethod
    def fold_layer_norm(state_dict: Dict[str, torch.Tensor], config: HookedTransformerConfig):
        """Takes in a state dict from a pretrained model, formatted to be consistent with HookedTransformer but with
        LayerNorm weights and biases. Folds these into the neighbouring weights. See further_comments.md for more details

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of pretrained model
        """
        for l in range(config.n_layers):
            # Fold ln1 into attention - it's important to fold biases first,
            # since biases depend on weights but not vice versa
            # The various indexing is just to broadcast ln.b and ln.w along every axis other than d_model.
            # Each weight matrix right multiplies.
            # To fold in the bias, we use the W_ matrix to map it to the hidden space of the layer,
            # so we need to sum along axis -2, which is the residual stream space axis.
            state_dict[f"blocks.{l}.attn.b_Q"] = state_dict[f"blocks.{l}.attn.b_Q"] + (
                state_dict[f"blocks.{l}.attn.W_Q"]
                * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
            ).sum(-2)
            state_dict[f"blocks.{l}.attn.b_K"] = state_dict[f"blocks.{l}.attn.b_K"] + (
                state_dict[f"blocks.{l}.attn.W_K"]
                * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
            ).sum(-2)
            state_dict[f"blocks.{l}.attn.b_V"] = state_dict[f"blocks.{l}.attn.b_V"] + (
                state_dict[f"blocks.{l}.attn.W_V"]
                * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
            ).sum(-2)

            state_dict[f"blocks.{l}.attn.W_Q"] = (
                state_dict[f"blocks.{l}.attn.W_Q"]
                * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )
            state_dict[f"blocks.{l}.attn.W_K"] = (
                state_dict[f"blocks.{l}.attn.W_K"]
                * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )
            state_dict[f"blocks.{l}.attn.W_V"] = (
                state_dict[f"blocks.{l}.attn.W_V"]
                * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )

            # Finally, we center the weights reading from the residual stream. The output of the first
            # part of the LayerNorm is mean 0 and standard deviation 1, so the mean of any input vector
            # of the matrix doesn't matter and can be set to zero.
            # Equivalently, the output of LayerNormPre is orthogonal to the vector of all 1s (because
            # dotting with that gets the sum), so we can remove the component of the matrix parallel to this.
            state_dict[f"blocks.{l}.attn.W_Q"] -= einops.reduce(
                state_dict[f"blocks.{l}.attn.W_Q"],
                "head_index d_model d_head -> head_index 1 d_head",
                "mean",
            )
            state_dict[f"blocks.{l}.attn.W_K"] -= einops.reduce(
                state_dict[f"blocks.{l}.attn.W_K"],
                "head_index d_model d_head -> head_index 1 d_head",
                "mean",
            )
            state_dict[f"blocks.{l}.attn.W_V"] -= einops.reduce(
                state_dict[f"blocks.{l}.attn.W_V"],
                "head_index d_model d_head -> head_index 1 d_head",
                "mean",
            )

            del (
                state_dict[f"blocks.{l}.ln1.w"],
                state_dict[f"blocks.{l}.ln1.b"],
            )

            # Fold ln2 into MLP
            if not config.attn_only:
                state_dict[f"blocks.{l}.mlp.b_in"] = state_dict[
                    f"blocks.{l}.mlp.b_in"
                ] + (
                    state_dict[f"blocks.{l}.mlp.W_in"]
                    * state_dict[f"blocks.{l}.ln2.b"][:, None]
                ).sum(
                    -2
                )
                state_dict[f"blocks.{l}.mlp.W_in"] = (
                    state_dict[f"blocks.{l}.mlp.W_in"]
                    * state_dict[f"blocks.{l}.ln2.w"][:, None]
                )

                # Center the weights that read in from the LayerNormPre
                state_dict[f"blocks.{l}.mlp.W_in"] -= einops.reduce(
                    state_dict[f"blocks.{l}.mlp.W_in"],
                    "d_model d_mlp -> 1 d_mlp",
                    "mean",
                )

                del state_dict[f"blocks.{l}.ln2.w"], state_dict[f"blocks.{l}.ln2.b"]

                if config.act_fn.startswith("solu"):
                    # Fold ln3 into activation
                    state_dict[f"blocks.{l}.mlp.b_out"] = state_dict[
                        f"blocks.{l}.mlp.b_out"
                    ] + (
                        state_dict[f"blocks.{l}.mlp.W_out"]
                        * state_dict[f"blocks.{l}.mlp.ln.b"][:, None]
                    ).sum(
                        -2
                    )
                    state_dict[f"blocks.{l}.mlp.W_out"] = (
                        state_dict[f"blocks.{l}.mlp.W_out"]
                        * state_dict[f"blocks.{l}.mlp.ln.w"][:, None]
                    )

                    # Center the weights that read in from the LayerNormPre
                    state_dict[f"blocks.{l}.mlp.W_out"] -= einops.reduce(
                        state_dict[f"blocks.{l}.mlp.W_out"],
                        "d_mlp d_model -> 1 d_model",
                        "mean",
                    )
                    del (
                        state_dict[f"blocks.{l}.mlp.ln.w"],
                        state_dict[f"blocks.{l}.mlp.ln.b"],
                    )
        # Fold ln_final into Unembed
        if not config.final_rms:
            # Dumb bug from my old SoLU training code, some models have RMSNorm instead of LayerNorm pre unembed.
            state_dict[f"unembed.b_U"] = state_dict[f"unembed.b_U"] + (
                state_dict[f"unembed.W_U"] * state_dict[f"ln_final.b"][:, None]
            ).sum(dim=-2)
            del state_dict[f"ln_final.b"]
        state_dict[f"unembed.W_U"] = (
            state_dict[f"unembed.W_U"] * state_dict[f"ln_final.w"][:, None]
        )

        # Center the weights that read in from the LayerNormPre
        state_dict[f"unembed.W_U"] -= einops.reduce(
            state_dict[f"unembed.W_U"], "d_model d_vocab -> 1 d_vocab", "mean"
        )

        del state_dict[f"ln_final.w"]
        return state_dict

    @staticmethod
    def center_writing_weights(state_dict: Dict[str, torch.Tensor], config: HookedTransformerConfig):
        """Centers the weights of the model that write to the residual stream - W_out, W_E, W_pos and W_out. This is
        done by subtracting the mean of the weights from the weights themselves. This is done in-place. See
        fold_layer_norm for more details."""
        state_dict["embed.W_E"] = state_dict["embed.W_E"] - state_dict[
            "embed.W_E"
        ].mean(-1, keepdim=True)
        if config.positional_embedding_type != "rotary":
            state_dict["pos_embed.W_pos"] = state_dict["pos_embed.W_pos"] - state_dict[
                "pos_embed.W_pos"
            ].mean(-1, keepdim=True)
        for l in range(config.n_layers):
            state_dict[f"blocks.{l}.attn.W_O"] = state_dict[
                f"blocks.{l}.attn.W_O"
            ] - state_dict[f"blocks.{l}.attn.W_O"].mean(
                -1, keepdim=True
            )  # W_O is [head_index, d_model, d_head]
            state_dict[f"blocks.{l}.attn.b_O"] = (
                state_dict[f"blocks.{l}.attn.b_O"]
                - state_dict[f"blocks.{l}.attn.b_O"].mean()
            )  # b_O is [d_model]
            if not config.attn_only:
                state_dict[f"blocks.{l}.mlp.W_out"] = state_dict[
                    f"blocks.{l}.mlp.W_out"
                ] - state_dict[f"blocks.{l}.mlp.W_out"].mean(-1, keepdim=True)
                state_dict[f"blocks.{l}.mlp.b_out"] = (
                    state_dict[f"blocks.{l}.mlp.b_out"]
                    - state_dict[f"blocks.{l}.mlp.b_out"].mean()
                )
        return state_dict

    @staticmethod
    def center_unembed(state_dict: Dict[str, torch.Tensor], config=None):
        """Centers the unembedding weights W_U. This is done by subtracting the mean of the weights from the weights
        themselves. This is done in-place. As softmax is translation invariant, this changes the logits but not the
        log probs, and makes the model logits (slightly) more interpretable - when trying to understand how components
        contribute to the logits, we'll be less misled by components that just add something to every logit.
        """
        state_dict["unembed.W_U"] = state_dict["unembed.W_U"] - state_dict[
            "unembed.W_U"
        ].mean(-1, keepdim=True)
        state_dict["unembed.b_U"] = (
            state_dict["unembed.b_U"] - state_dict["unembed.b_U"].mean()
        )
        return state_dict
    
    @classmethod
    def from_pretrained_no_processing(
        cls,
        model_name: str,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        **from_pretrained_kwargs,
    ):
        """Wrapper for from_pretrained with all boolean flags related to simplifying the model set to False. Refer to
        from_pretrained for details."""
        return cls.from_pretrained(
            model_name,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            **from_pretrained_kwargs,
        )

    def set_tokenizer(self, tokenizer):
        """
        Sets the tokenizer to use for this model.
        tokenizer (PreTrainedTokenizer): a pretrained HuggingFace tokenizer
        """
        assert isinstance(
            tokenizer, PreTrainedTokenizerBase
        ), f"{type(tokenizer)} is not a supported tokenizer, please use PreTrainedTokenizer or PreTrainedTokenizerFast"
        self.tokenizer = tokenizer

        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "<|endoftext|>"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

        # Infer vocab size from tokenizer
        if self.config.d_vocab == -1:
            self.config.d_vocab = max(self.tokenizer.vocab.values()) + 1
        if self.config.d_vocab_out == -1:
            self.config.d_vocab_out = self.config.d_vocab

    def to_tokens(
        self,
        input: Union[str, List[str]],
        prepend_bos: bool = True,
        truncate: bool = True,
    ) -> Int[torch.Tensor, "batch pos"]:
        """
        Converts a string to a tensor of tokens. If prepend_bos is True, prepends the BOS token to the input - this is
        recommended when creating a sequence of tokens to be input to a model.

        Args:
            input (Union[str, List[str]]). The input to tokenize
            prepend_bos (bool): Whether to prepend a beginning of sequence token. Defaults to True
            Defaults to True
            truncate (bool): If the output tokens are too long, whether to truncate the output tokens to the model's
            max context window. Does nothing for shorter inputs. Defaults to True.

        Gotcha: prepend_bos prepends a beginning of string token. This is a recommended default when inputting a prompt
        to the model as the first token is often treated weirdly, but should only be done at the START of the prompt.
        Make sure to turn it off if you're looking at the tokenization of part of the prompt!
        (Note: some models eg GPT-2 were not trained with a BOS token, others (OPT and my models) were)

        Gotcha2: Tokenization of a string depends on whether there is a preceding space and whether the first letter is
        capitalized. It's easy to shoot yourself in the foot here if you're not careful!
        """
        assert self.tokenizer is not None, "Cannot use to_tokens without a tokenizer"
        if prepend_bos:
            if isinstance(input, str):
                input = self.tokenizer.bos_token + input
            else:
                input = [self.tokenizer.bos_token + string for string in input]
        tokens = self.tokenizer(
            input,
            return_tensors="pt",
            padding=True,
            truncation=truncate,
            max_length=self.config.n_ctx if truncate else None,
            add_special_tokens=False
            if self.tokenizer.name_or_path.startswith("facebook/opt")
            else True,  # As we manually add the BOS token
        )["input_ids"]
        return tokens
    
    def to_single_token(self, string):
        """Maps a string that makes up a single token to the id for that token. Raises an error for strings that are
        not a single token! If uncertain use to_tokens"""

        # We use the to_tokens method, do not append a BOS token
        token = self.to_tokens(string, prepend_bos=False).squeeze()
        # If token shape is non-empty, raise error
        assert not token.shape, f"Input string: {string} is not a single token!"
        return token.item()

    def tokens_to_residual_directions(
        self,
        tokens: Union[
            str,
            int,
            Int[torch.Tensor, ""],
            Int[torch.Tensor, "pos"],
            Int[torch.Tensor, "batch pos"],
        ],
    ) -> Union[
        Float[torch.Tensor, "d_model"],
        Float[torch.Tensor, "pos d_model"],
        Float[torch.Tensor, "batch pos d_model"],
    ]:
        """Maps tokens to a tensor with the unembedding vector for those tokens, ie the vector in the residual stream
        that we dot with to the get the logit for that token.

        WARNING: If you use this without folding in LayerNorm, the results will be misleading and may be incorrect, as
        the LN weights change the unembed map. This is done automatically with the fold_ln flag on from_pretrained

        WARNING 2: LayerNorm scaling will scale up or down the effective direction in the residual stream for each
        output token on any given input token position. ActivationCache.apply_ln_to_stack will apply the appropriate
        scaling to these directions.

        Args:
            tokens (Union[str, int, torch.Tensor]): The token(s). If a single token, can be a single element tensor, an
                integer, or string. If string, will be mapped to a single token using to_single_token, and an error
                raised if it's multiple tokens. The method also works for a batch of input tokens

        Returns:
            residual_direction torch.Tensor: The unembedding vector for the token(s), a stack of [d_model] tensor.
        """
        if isinstance(tokens, torch.Tensor) and tokens.numel() > 1:
            # If the tokens are a tensor, and have more than one element, assume they are a batch of tokens
            residual_directions = self.W_U[:, tokens]
            residual_directions = einops.rearrange(
                residual_directions, "d_model ... -> ... d_model"
            )
            return residual_directions
        else:
            # Otherwise there is a single token
            if isinstance(tokens, str):
                token = self.to_single_token(tokens)
            elif isinstance(tokens, int):
                token = tokens
            elif isinstance(tokens, torch.Tensor) and tokens.numel() == 1:
                token = tokens.item()
            else:
                raise ValueError(f"Invalid token type: {type(tokens)}")
            residual_direction = self.W_U[:, token]
            return residual_direction

    def init_weights(self, module):
        """
        Initialize weights matrices with a normal of std=initializer_range (default=0.02). This roughly follows the
        GPT-2 paper's scheme (but with truncation, and not halving the std for W_pos).

        LayerNorm weights are already initialized to 1.0, and all biases are initialized to 0.0 (including LayerNorm),
        so this just initializes weight matrices.

        Weight matrices are set to empty by default (to save space + compute, since they're the bulk of the parameters),
        so it is important to call this if you are not loading in pretrained weights! Note that this function assumes that weight names being with W_

        Set seed here to ensure determinism.

        This does NOT follow the PyTorch scheme, which as far as I can tell is super out of date but no one has gotten
        round to updating it?
        https://github.com/pytorch/pytorch/issues/18182

        PyTorch Transformers are especially bad - TransformerEncoder initializes all layers to the exact same weights?!
        https://github.com/pytorch/pytorch/issues/72253

        The best paper I've found on transformer initialization is the muP paper, but haven't integrated those ideas yet:
        https://arxiv.org/abs/2203.03466
        """

        for name, param in module.named_parameters():
            if "W_" in name:
                nn.init.normal_(param, std=self.config.initializer_range)

    def all_head_labels(self):
        return [
            f"L{l}H{h}"
            for l in range(self.config.n_layers)
            for h in range(self.config.n_heads)
        ]
    
    # Give access to all weights as properties.
    @property
    @typeguard_ignore
    def W_U(self) -> Float[torch.Tensor, "d_model d_vocab"]:
        """
        Convenience to get the unembedding matrix (ie the linear map from the final residual stream to the output logits)
        """
        return self.unembed.W_U

    @property
    @typeguard_ignore
    def b_U(self) -> Float[torch.Tensor, "d_vocab"]:
        return self.unembed.b_U

    @property
    @typeguard_ignore
    def W_E(self) -> Float[torch.Tensor, "d_vocab d_model"]:
        """
        Convenience to get the embedding matrix
        """
        return self.embed.W_E

    @property
    @typeguard_ignore
    def W_pos(self) -> Float[torch.Tensor, "n_ctx d_model"]:
        """
        Convenience function to get the positional embedding. Only works on models with absolute positional embeddings!
        """
        return self.pos_embed.W_pos

    @property
    @typeguard_ignore
    def W_E_pos(self) -> Float[torch.Tensor, "d_vocab+n_ctx d_model"]:
        """
        Concatenated W_E and W_pos. Used as a full (overcomplete) basis of the input space, useful for full QK and full OV circuits.
        """
        return torch.cat([self.W_E, self.W_pos], dim=0)
