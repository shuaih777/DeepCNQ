import torch.nn as nn
import torch
from .transformer_ps import TransformerPS
from .transformer_kan import TransformerKAN, TransformerKAN2
from .kan_model import KANModel

class DeepQuantReg(nn.Module):
    def __init__(self, input_dim, output_dim, num_diseases, model_type, dim_feedforward, nhead, **kwargs):
        super().__init__()
        self.model_type = model_type
        self.output_dim = output_dim
        self.num_diseases = num_diseases
        self.output_dim_fc = output_dim * num_diseases
        self.relu = nn.ReLU()
        self.model_base = self.get_model_base(input_dim, dim_feedforward, nhead, **kwargs)
        self.output_layer = nn.Linear(dim_feedforward, output_dim)

    def get_model_base(self, input_dim, dim_feedforward, nhead, **kwargs):
        model_dict = {
            'KAN': lambda: KANModel(input_dim, dim_feedforward, layers=kwargs.get('layers', 2), grid_size=kwargs.get('grid_size', 5)),
            'TransformerPS': lambda: TransformerPS(input_dim, dim_feedforward, nhead, **kwargs),
            'Transformer_KAN_gaps': lambda: TransformerKAN(input_dim, dim_feedforward, nhead, **kwargs),
            'Transformer_KAN2_gaps': lambda: TransformerKAN2(input_dim, dim_feedforward, nhead, **kwargs),
        }
        return model_dict[self.model_type]()

    def forward(self, x):
        x = self.model_base(x)
        x = self.relu(x)
        raw_output = self.output_layer(x)
        raw_output = raw_output.view(-1, self.num_diseases, self.output_dim)
        if self.output_dim == 1:
            return raw_output
        h_0 = raw_output[:, :, 0].unsqueeze(-1)
        gaps = torch.log(1 + torch.exp(raw_output[:, :, 1:]))
        cumsum_gaps = torch.cumsum(gaps, dim=-1)
        return torch.cat((h_0, h_0 + cumsum_gaps), dim=-1)
