import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

N = 6 # number of layers
h = 8 # number of heads
d_model = 512 # embedding dimensions
d_ff = 2048 # feedforward hidden layer