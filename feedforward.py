import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

class FeedForward(nn.Module):
    def __init__(self, d_model : int, d_ff : int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x) -> TensorType[float]:
        return self.net(x)
