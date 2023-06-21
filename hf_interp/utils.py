from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int


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
