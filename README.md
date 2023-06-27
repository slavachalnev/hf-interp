# hf-interp

Aims to provide the same functionality as Neel Nanda's [Transformer Lens](https://github.com/neelnanda-io/TransformerLens) except it uses HuggingFace's Transformers library as the backend.

All the models, interpretability methods, and tests are taken from Neel's library. I just changed the base class from `torch.nn.Module` to `transformers.PreTrainedModel` and tweaked the loading, device handling, and configuration to work with HuggingFace's library.

## Installation

```
pip install -e .
```

## Testing

```
python -m pytest
```
