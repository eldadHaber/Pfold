import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.1, ntokenOut=-1):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Conv1d(ntoken, ninp, 7, padding=3) #nn.Linear(ntoken, ninp)
        self.ninp = ninp
        if ntokenOut < 0:
            ntokenOut = ntoken
        self.decoder = nn.Conv1d(ninp, ntokenOut, 7, padding=3) #nn.Linear(ninp, ntokenOut)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        #src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.encoder(src) * math.sqrt(self.ninp)
        #src = self.pos_encoder(src)
        output = self.transformer_encoder(src.squeeze(0).t().unsqueeze(0), self.src_mask)
        #output = self.decoder(output)
        output = self.decoder(output.squeeze(0).t().unsqueeze(0))

        D = tr2DistSmall(output)

        return (D,)


def tr2DistSmall(Y):

    k = Y.shape[1]
    Z = Y[0,:,:]
    Z = Z - torch.mean(Z, dim=1, keepdim=True)
    D = torch.sum(Z**2, dim=0).unsqueeze(0) + torch.sum(Z**2, dim=0).unsqueeze(1) - 2*Z.t() @ Z
    D = 3*D/k
    return torch.sqrt(torch.relu(D))
