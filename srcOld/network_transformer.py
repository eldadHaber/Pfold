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
        # self.encoder = nn.Embedding(ntoken, ninp)
        self.encoder = nn.Conv1d(ntoken, ninp, 7, padding=3) #nn.Linear(ntoken, ninp)
        # self.encoder = nn.Linear(ntoken, ninp)
        self.ninp = ninp
        if ntokenOut < 0:
            ntokenOut = ntoken
        # self.decoder = nn.Linear(ninp, ntoken)
        self.decoder = nn.Conv1d(ninp, ntokenOut, 7, padding=3) #nn.Linear(ninp, ntokenOut)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, mask):
        # We start by changing the shape of src from the conventional shape of (N,C,L) to (L,N,C), where N=Nbatch, C=#Channels, L= sequence length
        src = src.permute(2,0,1)

        #src = self.encoder(src) * math.sqrt(self.ninp)
        # src = self.encoder(src)
        src = self.encoder(src.permute(1,2,0))
        src = src.permute(2,0,1)
        src = self.pos_encoder(src)

        #src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=(mask==0))
        #output = self.decoder(output)
        # output = self.decoder(output) # * mask[:,None,:]
        output = self.decoder(output.permute(1,2,0)) # * mask[:,None,:]
        output = output.permute(2,0,1)

        # Now we change the shape back to normal again.
        output = output.permute(1,2,0)
        D = tr2DistSmall(output)

        return (D,)


def tr2DistSmall(Y):

    k = Y.shape[1]
    Z = Y[0,:,:]
    Z = Z - torch.mean(Z, dim=1, keepdim=True)
    D = torch.sum(Z**2, dim=0).unsqueeze(0) + torch.sum(Z**2, dim=0).unsqueeze(1) - 2*Z.t() @ Z
    D = 3*D/k
    return torch.sqrt(torch.relu(D))
