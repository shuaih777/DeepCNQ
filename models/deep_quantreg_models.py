import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats
import matplotlib.pyplot as plt
from torch.autograd import Variable
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1).float() # [max_len, 1], [0., 1., 2., ..., torch.tensor(max_len-1).float()]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -torch.tensor(math.log(10000.0) / d_model)) 
        # torch.arange(0, d_model, 2).float(): [0., 2., 4., ..., d_model-2.]
        # use exp-log trick to calculate 10000^(2i/d_model)
        pe[:, 0::2] = torch.sin(position * div_term) # for every position, add sin to even columns
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term) # for every position, add cos to odd columns
        else:
            pe[:, 1::2] = torch.cos(position * div_term[-1])
        # pe[:, 1::2] = torch.cos(position * div_term) # for every position, add cos to odd columns
        pe = pe.unsqueeze(0) # torch.Size([1, max_len, d_model])
        self.register_buffer('pe', pe) # registers a tensor as a buffer (will not be updated by gradient descent)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False) 
        # x.size(1) is the sequence length. So, x.size(1) is the number of positions in the input.
        # pe[:, :x.size(1)]: [1, x.size(1), d_model]. Broadcasts the positional encoding to the batch size.
        # = [batch_size, x.size(1), d_model] + [1, x.size(1), d_model] 
        # = [batch_size, x.size(1), d_model] + [batch_size, x.size(1), d_model]
        return self.dropout(x)

class TransformerPS(nn.Module):
    def __init__(self, input_dim, dim_feedforward, nhead=2, layers=2, dropout=0.4, acfn='relu'):
        super(TransformerPS, self).__init__()
        print('TransformerLayer', dim_feedforward, nhead, layers, dropout, acfn)
        self.input_embed = nn.Linear(1,dim_feedforward)
        self.position_encodeT = PositionalEncoding(input_dim, dropout) # positional encoding on dim=2
        
        self.norm = nn.LayerNorm(dim_feedforward)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead, dim_feedforward=2*dim_feedforward, dropout=dropout, activation=acfn,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output_linear = nn.Linear(input_dim*dim_feedforward, dim_feedforward)

    def forward(self, x):
        x = x.unsqueeze(-1) # [bsize, input_dim, 1]
        x = self.input_embed(x) # [bsize, input_dim, embed_dim]
        x = self.position_encodeT(x.permute(0,2,1)).permute(0,2,1) # [bsize, input_dim, embed_dim]
        x = self.norm(x)
        x = self.transformer_encoder(x) # [bsize, input_dim, embed_dim]
        x = x.view(x.shape[0],-1)
        x = self.output_linear(x)
        return x

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, batch_first=True, grid_size=5):
        # Must specify batch_first=batch_first since it's not a positional argument.
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, batch_first=batch_first)
        # Replace only the linear layers with KAN
        # The input from the self-attention layer (with dimension embedding_dim, or d_model) is first passed through a linear layer.
        # This layer projects the d_model dimension to a larger size
        # # To replace the linear layer F.linear() (it's not 1-hidden layer MLP, just a linear Transformation)
        # self.linear1 = ekan([d_model, dim_feedforward], grid_size)
        self.linear1 = nn.Linear(1, 1)
        # # Output transformation: The expanded representation is passed through a second linear layer to reduce the dimension back down to d_model.
        # self.linear2 = ekan([dim_feedforward, d_model], grid_size)
        self.linear2 = nn.Linear(1, 1)
        self.kan = ekan([d_model, dim_feedforward, d_model], grid_size)
    def _ff_block(self, x):
        # Replace the feed-forward block with the KAN layer
        # print("Using modified _ff_block with KAN")
        x = self.kan(x)
        return self.dropout2(x)

    def forward(self, *args, **kwargs):
        torch.backends.mha.set_fastpath_enabled(False)
        return super().forward(*args, **kwargs)

class TransformerKAN(nn.Module):
    def __init__(self, input_dim, dim_feedforward, nhead=2, layers=2, dropout=0.4, acfn='relu', grid_size=8):
        super(TransformerKAN, self).__init__()
        print('TransformerLayer', dim_feedforward, nhead, layers, dropout, acfn)
        
        # self.input_embed = ekan([1, dim_feedforward], grid_size)
        self.input_embed = nn.Linear(1,dim_feedforward)
        self.position_encodeT = PositionalEncoding(input_dim, dropout)
        
        self.norm = nn.LayerNorm(dim_feedforward)
        
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=acfn,
            batch_first=True,
            grid_size=grid_size
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output_linear = ekan([input_dim*dim_feedforward, dim_feedforward], grid_size)
        # self.output_linear = nn.Linear(input_dim*dim_feedforward, dim_feedforward)

    def forward(self, x):
        x = x.unsqueeze(-1) # [bsize, input_dim, 1]
        x = self.input_embed(x) # in kan.py, assert x.size(-1) == self.in_features
        x = self.position_encodeT(x.permute(0,2,1)).permute(0,2,1)
        x = self.norm(x)
        x = self.transformer_encoder(x) # [bsize, input_dim, embed_dim]
        x = x.view(x.shape[0], -1)
        x = self.output_linear(x)
        return x

class TransformerKAN2(nn.Module):
    def __init__(self, input_dim, dim_feedforward, nhead=2, layers=2, dropout=0.4, acfn='relu', grid_size=8):
        super(TransformerKAN2, self).__init__()
        print('TransformerLayer', dim_feedforward, nhead, layers, dropout, acfn)
        
        # self.input_embed = ekan([1, dim_feedforward], grid_size)
        self.input_embed = nn.Linear(1,dim_feedforward)
        self.position_encodeT = PositionalEncoding(input_dim, dropout)
        
        self.norm = nn.LayerNorm(dim_feedforward)
        
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=acfn,
            batch_first=True,
            grid_size=grid_size
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        # self.output_linear = ekan([input_dim*dim_feedforward, dim_feedforward], grid_size)
        self.output_linear = nn.Linear(input_dim*dim_feedforward, dim_feedforward)

    def forward(self, x):
        x = x.unsqueeze(-1) # [bsize, input_dim, 1]
        x = self.input_embed(x) # in kan.py, assert x.size(-1) == self.in_features
        x = self.position_encodeT(x.permute(0,2,1)).permute(0,2,1)
        x = self.norm(x)
        x = self.transformer_encoder(x) # [bsize, input_dim, embed_dim]
        x = x.view(x.shape[0], -1)
        x = self.output_linear(x)
        return x

class DeepQuantReg(nn.Module):
    def __init__(self, input_dim, output_dim, num_diseases, model_type, dim_feedforward, nhead, **kwargs):
        super(DeepQuantReg, self).__init__()
        # define the layers in __init__() to ensure they are incorporated in gradient descent
        self.layer_norm = nn.LayerNorm(dim_feedforward)
        self.sigmoid = nn.Sigmoid()
        self.model_type = model_type
        self.output_dim = output_dim
        self.layern = kwargs.get('layers', 2)
        self.dropout = kwargs.get('dropout', 0.2)
        self.acfn = kwargs.get('acfn', 'relu')
        self.relu = nn.ReLU()
        # Multivariate
        self.num_diseases = num_diseases
        self.output_dim_fc = output_dim * num_diseases
        print('Transformer output dim:', output_dim)

        if model_type == 'KAN' or model_type == 'KAN_gaps':
            grid_size = kwargs.get('grid_size', 5)
            print('KAN grid size:', grid_size)
            kan_layers = [input_dim] + [dim_feedforward]*self.layern
            self.model_base = ekan(kan_layers, grid_size)
            self.output_layer = nn.Linear(dim_feedforward, output_dim)
        elif model_type == 'Transformer_KAN_gaps':
            grid_size = kwargs.get('grid_size', 5)
            # self.transformer = TransformerPS(input_dim, dim_feedforward, nhead, self.layern, self.dropout, self.acfn)
            # self.kan = ekan([dim_feedforward, dim_feedforward, dim_feedforward])
            self.model_base = TransformerKAN(input_dim, dim_feedforward, nhead, self.layern, self.dropout, self.acfn)
            self.output_layer = nn.Linear(dim_feedforward, output_dim)
        elif model_type == 'Transformer_KAN2_gaps':
            grid_size = kwargs.get('grid_size', 5)
            # self.transformer = TransformerPS(input_dim, dim_feedforward, nhead, self.layern, self.dropout, self.acfn)
            # self.kan = ekan([dim_feedforward, dim_feedforward, dim_feedforward])
            self.model_base = TransformerKAN2(input_dim, dim_feedforward, nhead, self.layern, self.dropout, self.acfn)
            self.output_layer = nn.Linear(dim_feedforward, output_dim)
        elif model_type == 'TransformerPS' or model_type == 'TransformerPS_gaps':
            self.layer_norm = nn.LayerNorm(dim_feedforward)
            self.model_base = TransformerPS(input_dim, dim_feedforward, nhead, **kwargs)
            self.output_layer = nn.Linear(dim_feedforward, self.output_dim_fc)

    def forward(self, x):
        # Apply the appropriate backbone based on model_type
        x = self.model_base(x)

        x = self.relu(x)

        # Multi-diseases
        raw_output = self.output_layer(x)  # shape: (batch, D * K)

        # Reshape to (batch, D, K)
        raw_output = raw_output.view(-1, self.num_diseases, self.output_dim)

        if self.output_dim == 1:
            feature_vec = raw_output

        else:
            # 取最小分位数 h_0
            h_0 = raw_output[:, :, 0].unsqueeze(-1)  # shape: (batch, D, 1)

            # 计算 gaps，并确保非负
            gaps = raw_output[:, :, 1:]  # shape: (batch, D, K-1)
            gaps = torch.log(1 + torch.exp(gaps))  # 取 exp 确保非负

            # 计算 cumulative sum 确保单调递增
            cumsum_gaps = torch.cumsum(gaps, dim=-1)  # shape: (batch, D, K-1)

            # 组合成最终 feature vector
            feature_vec = torch.cat((h_0, h_0 + cumsum_gaps), dim=-1)  # shape: (batch, D, K)
        
        return feature_vec