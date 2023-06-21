from transformers import PretrainedConfig


class HookedTransformerConfig(PretrainedConfig):
    model_type = "hooked_transformer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
