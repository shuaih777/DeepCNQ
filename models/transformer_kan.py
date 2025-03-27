import torch.nn as nn
from .positional_encoding import PositionalEncoding
from efficient_kan import KAN as ekan

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, batch_first=True, grid_size=5):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, batch_first=batch_first)
        self.kan = ekan([d_model, dim_feedforward, d_model], grid_size)

    def _ff_block(self, x):
        x = self.kan(x)
        return self.dropout2(x)

    def forward(self, *args, **kwargs):
        torch.backends.mha.set_fastpath_enabled(False)
        return super().forward(*args, **kwargs)

class TransformerKAN(nn.Module):
    def __init__(self, input_dim, dim_feedforward, nhead=2, layers=2, dropout=0.4, acfn='relu', grid_size=8):
        super().__init__()
        self.input_embed = nn.Linear(1, dim_feedforward)
        self.position_encodeT = PositionalEncoding(input_dim, dropout)
        self.norm = nn.LayerNorm(dim_feedforward)
        encoder_layer = CustomTransformerEncoderLayer(dim_feedforward, nhead, dim_feedforward, dropout, acfn, True, grid_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output_linear = ekan([input_dim * dim_feedforward, dim_feedforward], grid_size)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_embed(x)
        x = self.position_encodeT(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.norm(x)
        x = self.transformer_encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.output_linear(x)
        return x

class TransformerKAN2(TransformerKAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_linear = nn.Linear(kwargs[0] * kwargs[1], kwargs[1])
