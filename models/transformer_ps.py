import torch.nn as nn
from .positional_encoding import PositionalEncoding

class TransformerPS(nn.Module):
    def __init__(self, input_dim, dim_feedforward, nhead=2, layers=2, dropout=0.4, acfn='relu'):
        super(TransformerPS, self).__init__()
        self.input_embed = nn.Linear(1, dim_feedforward)
        self.position_encodeT = PositionalEncoding(input_dim, dropout)
        self.norm = nn.LayerNorm(dim_feedforward)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead,
                                                   dim_feedforward=2 * dim_feedforward, dropout=dropout,
                                                   activation=acfn, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output_linear = nn.Linear(input_dim * dim_feedforward, dim_feedforward)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_embed(x)
        x = self.position_encodeT(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.norm(x)
        x = self.transformer_encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.output_linear(x)
        return x
