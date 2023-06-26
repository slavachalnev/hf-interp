from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
from jaxtyping import Float, Int

import inspect
import re
import json

import numpy as np
import einops
import torch
import torch.nn.functional as F

from datasets import Dataset
import transformers
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download


CACHE_DIR = transformers.TRANSFORMERS_CACHE


def gelu_new(
    input: Float[torch.Tensor, "batch pos d_mlp"]
) -> Float[torch.Tensor, "batch pos d_mlp"]:
    # Implementation of GeLU used by GPT2 - subtly different from PyTorch's
    tanh = torch.tanh(np.sqrt(2.0 / np.pi) * (input + 0.044715 * torch.pow(input, 3.0)))
    return 0.5 * input * (1.0 + tanh)


def gelu_fast(
    input: Float[torch.Tensor, "batch pos d_mlp"]
) -> Float[torch.Tensor, "batch pos d_mlp"]:
    tanh = torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input))
    return 0.5 * input * (1.0 + tanh)


def solu(
    input: Float[torch.Tensor, "batch pos d_mlp"]
) -> Float[torch.Tensor, "batch pos d_mlp"]:
    """
    SoLU activation function as described by
    https://transformer-circuits.pub/2022/solu/index.html.

    LayerNorm implemented by the MLP class.
    """
    return input * F.softmax(input, dim=-1)


def transpose(tensor: Float[torch.Tensor, "... a b"]) -> Float[torch.Tensor, "... b a"]:
    """
    Utility to swap the last two dimensions of a tensor, regardless of the number of leading dimensions
    """
    return tensor.transpose(-1, -2)


def get_act_name(
    name: str,
    layer: Optional[int] = None,
    layer_type: Optional[str] = None,
):
    """
    Helper function to convert shorthand to an activation name. Pretty hacky, intended to be useful for short feedback
    loop hacking stuff together, more so than writing good, readable code. But it is deterministic!

    Returns a name corresponding to an activation point in a TransformerLens model.

    Args:
         name (str): Takes in the name of the activation. This can be used to specify any activation name by itself.
         The code assumes the first sequence of digits passed to it (if any) is the layer number, and anything after
         that is the layer type.

         Given only a word and number, it leaves layer_type as is.
         Given only a word, it leaves layer and layer_type as is.

         Examples:
             get_act_name('embed') = get_act_name('embed', None, None)
             get_act_name('k6') = get_act_name('k', 6, None)
             get_act_name('scale4ln1') = get_act_name('scale', 4, 'ln1')

         layer (int, optional): Takes in the layer number. Used for activations that appear in every block.

         layer_type (string, optional): Used to distinguish between activations that appear multiple times in one block.

    Full Examples:

    get_act_name('k', 6, 'a')=='blocks.6.attn.hook_k'
    get_act_name('pre', 2)=='blocks.2.mlp.hook_pre'
    get_act_name('embed')=='hook_embed'
    get_act_name('normalized', 27, 'ln2')=='blocks.27.ln2.hook_normalized'
    get_act_name('k6')=='blocks.6.attn.hook_k'
    get_act_name('scale4ln1')=='blocks.4.ln1.hook_scale'
    get_act_name('pre5')=='blocks.5.mlp.hook_pre'
    """
    if (
        ("." in name or name.startswith("hook_"))
        and layer is None
        and layer_type is None
    ):
        # If this was called on a full name, just return it
        return name
    match = re.match(r"([a-z]+)(\d+)([a-z]?.*)", name)
    if match is not None:
        name, layer, layer_type = match.groups(0)

    layer_type_alias = {
        "a": "attn",
        "m": "mlp",
        "b": "",
        "block": "",
        "blocks": "",
        "attention": "attn",
    }

    act_name_alias = {
        "attn": "pattern",
        "attn_logits": "attn_scores",
        "key": "k",
        "query": "q",
        "value": "v",
        "mlp_pre": "pre",
        "mlp_mid": "mid",
        "mlp_post": "post",
    }

    layer_norm_names = ["scale", "normalized"]

    if name in act_name_alias:
        name = act_name_alias[name]

    full_act_name = ""
    if layer is not None:
        full_act_name += f"blocks.{layer}."
    if name in [
        "k",
        "v",
        "q",
        "z",
        "rot_k",
        "rot_q",
        "result",
        "pattern",
        "attn_scores",
    ]:
        layer_type = "attn"
    elif name in ["pre", "post", "mid"]:
        layer_type = "mlp"
    elif layer_type in layer_type_alias:
        layer_type = layer_type_alias[layer_type]

    if layer_type:
        full_act_name += f"{layer_type}."
    full_act_name += f"hook_{name}"

    if name in layer_norm_names and layer is None:
        full_act_name = f"ln_final.{full_act_name}"
    return full_act_name


def lm_cross_entropy_loss(
    logits: Float[torch.Tensor, "batch pos d_vocab"],
    tokens: Int[torch.Tensor, "batch pos"],
    per_token: bool = False,
) -> Union[Float[torch.Tensor, ""], Float[torch.Tensor, "batch pos"]]:
    """Cross entropy loss for the language model, gives the loss for predicting the NEXT token.

    Args:
        logits (torch.Tensor): Logits. Shape [batch, pos, d_vocab]
        tokens (torch.Tensor[int64]): Input tokens. Shape [batch, pos]
        per_token (bool, optional): Whether to return the log probs predicted for the correct token, 
            or the loss (ie mean of the predicted log probs). Note that the returned array has shape 
            [batch, seq-1] as we cannot predict the first token (alternately, we ignore the final 
            logit). Defaults to False.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Use torch.gather to find the log probs of the correct tokens
    # Offsets needed because we're predicting the NEXT token (this means the final logit is meaningless)
    # None and [..., 0] needed because the tensor used in gather must have the same rank.
    predicted_log_probs = log_probs[..., :-1, :].gather(
        dim=-1, index=tokens[..., 1:, None]
    )[..., 0]
    if per_token:
        return -predicted_log_probs
    else:
        return -predicted_log_probs.mean()


def select_compatible_kwargs(
    kwargs_dict: Dict[str, Any], callable: Callable
) -> Dict[str, Any]:
    """Return a dict with the elements kwargs_dict that are parameters of callable"""
    return {
        k: v
        for k, v in kwargs_dict.items()
        if k in inspect.getfullargspec(callable).args
    }


def download_file_from_hf(
    repo_name,
    file_name,
    subfolder=".",
    cache_dir=CACHE_DIR,
    force_is_torch=False,
    **kwargs,
):
    """
    Helper function to download files from the HuggingFace Hub, from subfolder/file_name in repo_name, saving locally to cache_dir and returning the loaded file (if a json or Torch object) and the file path otherwise.

    If it's a Torch file without the ".pth" extension, set force_is_torch=True to load it as a Torch object.
    """
    file_path = hf_hub_download(
        repo_id=repo_name,
        filename=file_name,
        subfolder=subfolder,
        cache_dir=cache_dir,
        **select_compatible_kwargs(kwargs, hf_hub_download),
    )

    map_location = torch.device("cpu")

    if file_path.endswith(".pth") or force_is_torch:
        return torch.load(file_path, map_location=map_location)
    elif file_path.endswith(".json"):
        return json.load(open(file_path, "r"))
    else:
        print("File type not supported:", file_path.split(".")[-1])
        return file_path
    

def keep_single_column(dataset: Dataset, col_name: str):
    """
    Acts on a HuggingFace dataset to delete all columns apart from a single column name - useful when we want to tokenize and mix together different strings
    """
    for key in dataset.features:
        if key != col_name:
            dataset = dataset.remove_columns(key)
    return dataset


def tokenize_and_concatenate(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    streaming: bool = False,
    max_length: int = 1024,
    column_name: str = "text",
    add_bos_token: bool = True,
    num_proc: int = 10,
) -> Dataset:
    """Helper function to tokenizer and concatenate a dataset of text. This converts the text to tokens, concatenates them (separated by EOS tokens) and then reshapes them into a 2D array of shape (____, sequence_length), dropping the last batch. Tokenizers are much faster if parallelised, so we chop the string into 20, feed it into the tokenizer, in parallel with padding, then remove padding at the end.

    This tokenization is useful for training language models, as it allows us to efficiently train on a large corpus of text of varying lengths (without, eg, a lot of truncation or padding). Further, for models with absolute positional encodings, this avoids privileging early tokens (eg, news articles often begin with CNN, and models may learn to use early positional encodings to predict these)

    Args:
        dataset (Dataset): The dataset to tokenize, assumed to be a HuggingFace text dataset.
        tokenizer (AutoTokenizer): The tokenizer. Assumed to have a bos_token_id and an eos_token_id.
        streaming (bool, optional): Whether the dataset is being streamed. If True, avoids using parallelism. Defaults to False.
        max_length (int, optional): The length of the context window of the sequence. Defaults to 1024.
        column_name (str, optional): The name of the text column in the dataset. Defaults to 'text'.
        add_bos_token (bool, optional): . Defaults to True.

    Returns:
        Dataset: Returns the tokenized dataset, as a dataset of tensors, with a single column called "tokens"

    Note: There is a bug when inputting very small datasets (eg, <1 batch per process) where it just outputs nothing. I'm not super sure why
    """
    dataset = keep_single_column(dataset, column_name)
    if tokenizer.pad_token is None:
        # We add a padding token, purely to implement the tokenizer. This will be removed before inputting tokens to the model, so we do not need to increment d_vocab in the model.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    # Define the length to chop things up into - leaving space for a bos_token if required
    if add_bos_token:
        seq_len = max_length - 1
    else:
        seq_len = max_length

    def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        text = examples[column_name]
        # Concatenate it all into an enormous string, separated by eos_tokens
        full_text = tokenizer.eos_token.join(text)
        # Divide into 20 chunks of ~ equal length
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [
            full_text[i * chunk_length : (i + 1) * chunk_length]
            for i in range(num_chunks)
        ]
        # Tokenize the chunks in parallel. Uses NumPy because HuggingFace map doesn't want tensors returned
        tokens = tokenizer(chunks, return_tensors="np", padding=True)[
            "input_ids"
        ].flatten()
        # Drop padding tokens
        tokens = tokens[tokens != tokenizer.pad_token_id]
        num_tokens = len(tokens)
        num_batches = num_tokens // (seq_len)
        # Drop the final tokens if not enough to make a full sequence
        tokens = tokens[: seq_len * num_batches]
        tokens = einops.rearrange(
            tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
        )
        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"tokens": tokens}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=(num_proc if not streaming else None),
        remove_columns=[column_name],
    )
    tokenized_dataset.set_format(type="torch", columns=["tokens"])
    return tokenized_dataset
