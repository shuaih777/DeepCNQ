import torch.nn as nn
from efficient_kan import KAN as EKan

class KANModel(nn.Module):
    def __init__(self, input_dim, dim_feedforward, layers=2, grid_size=5):
        super().__init__()
        layer_dims = [input_dim] + [dim_feedforward] * layers
        self.model = EKan(layer_dims, grid_size)

    def forward(self, x):
        return self.model(x)
