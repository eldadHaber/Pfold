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

    def __init__(self, chan_in, emsize, nhead, nhid, nlayers, dropout=0.1, chan_out=-1, stencil=7):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer


        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = CNN(chan_in, 2 * chan_in, 3 * chan_in, emsize, stencil)
        self.ninp = emsize
        if chan_out < 0:
            chan_out = chan_in
        self.decoder = CNN(emsize, 2 * emsize, 3 * emsize, chan_out, stencil)

    def forward(self, src, mask):
        # We start by changing the shape of src from the conventional shape of (N,C,L) to (L,N,C), where N=Nbatch, C=#Channels, L= sequence length
        src = src.permute(2,0,1)
        mask_e = mask.unsqueeze(1)

        src = self.encoder(src.permute(1,2,0), mask_e)
        src = src.permute(2,0,1)
        src = self.pos_encoder(src)

        #src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=(mask==0))
        output = self.decoder(output.permute(1,2,0), mask_e)
        output = output.permute(2,0,1)

        # Now we change the shape back to normal again.
        output = output.permute(1,2,0)
        dNN = tr2DistSmall(output[:,0:3,:])
        dCaCa = tr2DistSmall(output[:,3:6,:])
        dCbCb = tr2DistSmall(output[:,6:9,:])

        return (dNN,dCaCa,dCbCb), output



class CNN(nn.Module):

    def __init__(self, nIn, nhid1, nhid2, nOut, stencilsize):
        super(CNN, self).__init__()

        self.K1 = nn.Conv1d(nIn,   nhid1, stencilsize, padding=stencilsize // 2)
        self.K2 = nn.Conv1d(nhid1, nhid2, stencilsize, padding=stencilsize // 2)
        self.K3 = nn.Conv1d(nhid2, nhid1, stencilsize, padding=stencilsize // 2)
        self.K4 = nn.Conv1d(nhid1, nhid2, stencilsize, padding=stencilsize // 2)
        self.K5 = nn.Conv1d(nhid2, nhid1, stencilsize, padding=stencilsize // 2)
        self.K6 = nn.Conv1d(nhid1, nOut,  stencilsize, padding=stencilsize // 2)
        self.init_weights()

    def init_weights(self):
        initrange  = 0.1
        initrangeR = 0.001

        nn.init.uniform_(self.K1.weight, -initrange, initrange)
        nn.init.uniform_(self.K2.weight, -initrangeR, initrangeR)
        nn.init.uniform_(self.K3.weight, -initrangeR, initrangeR)
        nn.init.uniform_(self.K4.weight, -initrangeR, initrangeR)
        nn.init.uniform_(self.K5.weight, -initrangeR, initrangeR)
        nn.init.uniform_(self.K6.weight, -initrange, initrange)

    def forward(self, src, mask):
        z1 = torch.relu(self.K1(src) * mask)
        z2 = z1 + self.K3(torch.relu(self.K2(z1)) * mask) * mask
        z3 = z2 + self.K5(torch.relu(self.K4(z1)) * mask) * mask
        z3 = self.K6(z2) * mask
        return z3


def tr2DistSmall(Y):

    k = Y.shape[1]
    Z = Y - torch.mean(Y, dim=2, keepdim=True)
    D = torch.sum(Z**2, dim=1).unsqueeze(1) + torch.sum(Z**2, dim=1).unsqueeze(2) - 2*Z.transpose(1,2) @ Z
    D = 3*D/k
    return torch.sqrt(torch.relu(D))

