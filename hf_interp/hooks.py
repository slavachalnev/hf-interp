# Mostly copied from Neel Nanda's TransformerLens.

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from transformers import PreTrainedModel

import torch.nn as nn
import torch.utils.hooks as hooks


@dataclass
class LensHandle:
    """
    A dataclass that holds information about a PyTorch hook.

    Attributes:
        hook (hooks.RemovableHandle): Reference to the hook's RemovableHandle.
        context_level (Optional[int], optional): Context level associated with the hooks context
            manager for the given hook. Defaults to None.
    """

    hook: hooks.RemovableHandle
    context_level: Optional[int] = None


# Define type aliases
NamesFilter = Optional[Union[Callable[[str], bool], Sequence[str]]]


class HookPoint(nn.Module):
    """
    A helper class to access intermediate activations in a PyTorch model (inspired by Garcon).

    HookPoint is a dummy module that acts as an identity function by default. By wrapping any
    intermediate activation in a HookPoint, it provides a convenient way to add PyTorch hooks.
    """

    def __init__(self):
        super().__init__()
        self.fwd_hooks: List[LensHandle] = []
        self.bwd_hooks: List[LensHandle] = []

        # A variable giving the hook's name (from the perspective of the root
        # module) - this is set by the root module at setup.
        self.name = None

    def add_hook(
        self, hook, dir="fwd", level=None, prepend=False
    ) -> None:
        """
        Hook format is fn(activation, hook_name)
        Change it into PyTorch hook format (this includes input and output,
        which are the same for a HookPoint)
        If prepend is True, add this hook before all other hooks
        """
        if dir == "fwd":

            def full_hook(module, module_input, module_output):
                return hook(module_output, hook=self)

            full_hook.__name__ = (
                hook.__repr__()
            )  # annotate the `full_hook` with the string representation of the `hook` function

            handle = self.register_forward_hook(full_hook)
            handle = LensHandle(handle, level)

            if prepend:
                # we could just pass this as an argument in PyTorch 2.0, but for now we manually do this...
                self._forward_hooks.move_to_end(handle.hook.id, last=False)
                self.fwd_hooks.insert(0, handle)

            else:
                self.fwd_hooks.append(handle)

        elif dir == "bwd":
            # For a backwards hook, module_output is a tuple of (grad,) - I don't know why.

            def full_hook(module, module_input, module_output):
                return hook(module_output[0], hook=self)

            full_hook.__name__ = (
                hook.__repr__()
            )  # annotate the `full_hook` with the string representation of the `hook` function

            handle = self.register_full_backward_hook(full_hook)
            handle = LensHandle(handle, level)

            if prepend:
                # we could just pass this as an argument in PyTorch 2.0, but for now we manually do this...
                self._backward_hooks.move_to_end(handle.hook.id, last=False)
                self.bwd_hooks.insert(0, handle)
            else:
                self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")
    
    def remove_hooks(self, level=None) -> None:

        def _remove_hooks(handles: List[LensHandle]) -> List[LensHandle]:
            output_handles = []
            for handle in handles:
                if level is None or handle.context_level == level:
                    handle.hook.remove()
                else:
                    output_handles.append(handle)
            return output_handles

        self.fwd_hooks = _remove_hooks(self.fwd_hooks)
        self.bwd_hooks = _remove_hooks(self.bwd_hooks)

    def forward(self, x):
        return x

    def layer(self):
        # Returns the layer index if the name has the form 'blocks.{layer}.{...}'
        # Helper function that's mainly useful on HookedTransformer
        # If it doesn't have this form, raises an error -
        split_name = self.name.split(".")
        return int(split_name[1])


class HookedRootModule(PreTrainedModel):
    """
    A class building on transformers.PreTrainedModel to interface nicely with HookPoints
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_caching = False
        self.context_level = 0

    def setup(self):
        """
        Sets up model.

        This function must be called in the model's `__init__` method AFTER defining all layers. It
        adds a parameter to each module containing its name, and builds a dictionary mapping module
        names to the module instances. It also initializes a hook dictionary for modules of type
        "HookPoint".
        """
        self.mod_dict = {}
        self.hook_dict: Dict[str, HookPoint] = {}
        for name, module in self.named_modules():
            if name == "":
                continue
            module.name = name
            self.mod_dict[name] = module
            if "HookPoint" in str(type(module)):
                self.hook_dict[name] = module
    
    def hook_points(self):
        return self.hook_dict.values()

    def reset_hooks(self, level=None):
        for hp in self.hook_points():
            hp.remove_hooks(level=level)

    @contextmanager
    def hooks(
        self,
        fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
    ):
        """
        A context manager for adding temporary hooks to the model.

        Args:
            fwd_hooks: List[Tuple[name, hook]], where name is either the name of a hook point or a
                Boolean function on hook names and hook is the function to add to that hook point.
            bwd_hooks: Same as fwd_hooks, but for the backward pass.

        Example:
        --------
        .. code-block:: python

            >>> with model.hooks(fwd_hooks=my_hooks):
            >>>     hooked_loss = model(text, return_type="loss")
        """
        try:
            self.context_level += 1

            for name, hook in fwd_hooks:
                if type(name) == str:
                    self.mod_dict[name].add_hook(
                        hook, dir="fwd", level=self.context_level
                    )
                else:
                    # Otherwise, name is a Boolean function on names
                    for hook_name, hp in self.hook_dict.items():
                        if name(hook_name):
                            hp.add_hook(hook, dir="fwd", level=self.context_level)
            for name, hook in bwd_hooks:
                if type(name) == str:
                    self.mod_dict[name].add_hook(
                        hook, dir="bwd", level=self.context_level
                    )
                else:
                    # Otherwise, name is a Boolean function on names
                    for hook_name, hp in self.hook_dict:
                        if name(hook_name):
                            hp.add_hook(hook, dir="bwd", level=self.context_level)
            yield self
        finally:
            self.reset_hooks(level=self.context_level)
            self.context_level -= 1

    def run_with_hooks(
        self,
        *model_args,
        fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        **model_kwargs,
    ):
        """
        Runs the model with specified forward and backward hooks.

        Args:
            fwd_hooks (List[Tuple[Union[str, Callable], Callable]]): A list of (name, hook), where name is
                either the name of a hook point or a boolean function on hook names, and hook is the
                function to add to that hook point. Hooks with names that evaluate to True are added
                respectively.
            bwd_hooks (List[Tuple[Union[str, Callable], Callable]]): Same as fwd_hooks, but for the
                backward pass.
            *model_args: Positional arguments for the model.
            **model_kwargs: Keyword arguments for the model.
        """
        if len(bwd_hooks) > 0:
            logging.warning(
                "WARNING: Hooks will be reset at the end of run_with_hooks. This removes the hooks "
                "before a backward pass can occur. Use hooks context manager to avoid this."
            )

        with self.hooks(fwd_hooks, bwd_hooks) as hooked_model:
            return hooked_model.forward(*model_args, **model_kwargs)

    def run_with_cache(
        self,
        *model_args,
        names_filter: NamesFilter = None,
        device=None,
        remove_batch_dim=False,
        incl_bwd=False,
        **model_kwargs,
    ):
        """
        Runs the model and returns the model output and a Cache object.

        Args:
            *model_args: Positional arguments for the model.
            names_filter (NamesFilter, optional): A filter for which activations to cache. Accepts None, str,
                list of str, or a function that takes a string and returns a bool. Defaults to None, which
                means cache everything.
            device (str or torch.Device, optional): The device to cache activations on. Defaults to the
                model device. WARNING: Setting a different device than the one used by the model leads to
                significant performance degradation.
            remove_batch_dim (bool, optional): If True, removes the batch dimension when caching. Only
                makes sense with batch_size=1 inputs. Defaults to False.
            incl_bwd (bool, optional): If True, calls backward on the model output and caches gradients
                as well. Assumes that the model outputs a scalar (e.g., return_type="loss"). Custom loss
                functions are not supported. Defaults to False.
            **model_kwargs: Keyword arguments for the model.

        Returns:
            tuple: A tuple containing the model output and a Cache object.

        """
        cache_dict, fwd, bwd = self.get_caching_hooks(
            names_filter, incl_bwd, device, remove_batch_dim=remove_batch_dim
        )

        with self.hooks(fwd_hooks=fwd, bwd_hooks=bwd):
            model_out = self(*model_args, **model_kwargs)
            if incl_bwd:
                model_out.backward()

        return model_out, cache_dict

    def get_caching_hooks(
        self,
        names_filter: NamesFilter = None,
        incl_bwd: bool = False,
        device=None,
        remove_batch_dim: bool = False,
        cache: Optional[dict] = None,
    ) -> Tuple[dict, list, list]:
        """Creates hooks to cache activations. Note: It does not add the hooks to the model.

        Args:
            names_filter (NamesFilter, optional): Which activations to cache. Can be a list of strings (hook names) or a filter function mapping hook names to booleans. Defaults to lambda name: True.
            incl_bwd (bool, optional): Whether to also do backwards hooks. Defaults to False.
            device (_type_, optional): The device to store on. Keeps on the same device as the layer if None.
            remove_batch_dim (bool, optional): Whether to remove the batch dimension (only works for batch_size==1). Defaults to False.
            cache (Optional[dict], optional): The cache to store activations in, a new dict is created by default. Defaults to None.

        Returns:
            cache (dict): The cache where activations will be stored.
            fwd_hooks (list): The forward hooks.
            bwd_hooks (list): The backward hooks. Empty if incl_bwd is False.
        """
        # TODO: make sure device can only be None or CPU

        if cache is None:
            cache = {}

        if names_filter is None:
            names_filter = lambda name: True
        elif type(names_filter) == str:
            names_filter = lambda name: name == names_filter
        elif type(names_filter) == list:
            names_filter = lambda name: name in names_filter

        self.is_caching = True

        def save_hook(tensor, hook):
            if remove_batch_dim:
                cache[hook.name] = tensor.detach().to(device)[0]
            else:
                cache[hook.name] = tensor.detach().to(device)

        def save_hook_back(tensor, hook):
            if remove_batch_dim:
                cache[hook.name + "_grad"] = tensor.detach().to(device)[0]
            else:
                cache[hook.name + "_grad"] = tensor.detach().to(device)

        fwd_hooks = []
        bwd_hooks = []
        for name, hp in self.hook_dict.items():
            if names_filter(name):
                fwd_hooks.append((name, save_hook))
                if incl_bwd:
                    bwd_hooks.append((name, save_hook_back))

        return cache, fwd_hooks, bwd_hooks
