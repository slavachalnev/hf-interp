import torch.nn as nn


class HookPoint(nn.Module):
    def __init__(self, name: str = ""):
        super().__init__()
        self.name = name

    def forward(self, x):
        return x
