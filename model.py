import torch
import torch.nn as nn

class DeepCS(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.x = nn.Parameter(torch.ones(N, 1) * 0.5)

    def forward(self, A):
        b = A @ self.x
        return b