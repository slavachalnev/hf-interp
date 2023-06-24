from hf_interp.hooked_transformer import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")
text = "Hi, my name is"

logits, loss = model.forward(text, return_type='both')
print('logits', logits)
max_logits = logits[0].argmax(dim=-1)
print(max_logits)

tokenizer = model.tokenizer
print(tokenizer.decode(max_logits.tolist()))
print('loss', loss)
