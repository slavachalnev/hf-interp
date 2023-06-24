from hf_interp.hooked_transformer import HookedTransformer


def my_hook(value, hook):
    print('in hook')
    return value


model = HookedTransformer.from_pretrained("gpt2-small")

loss = model.run_with_hooks("Hello there", fwd_hooks=[("blocks.0.attn.hook_v", my_hook)], return_type='loss')
print('loss', loss)
