import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout, ntokenOut, stencilsize):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.encoder = nn.Linear(ntoken, ninp)
        self.encoder = nn.Conv1d(ntoken, ninp, stencilsize, padding=stencilsize // 2)
        self.ninp = ninp
        #self.decoder = nn.Linear(ninp, ntoken)
        self.decoder = nn.Conv1d(ninp, ntokenOut, stencilsize, padding=stencilsize // 2)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def reshapeF(self,src):
        return src.squeeze(1).t().unsqueeze(0)
    def reshapeB(self,src):
        return src.squeeze(0).t().unsqueeze(1)

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.reshapeF(src)
        src = self.encoder(src)
        src = self.reshapeB(src)

        output = self.transformer_encoder(src, self.src_mask)
        output = output - torch.mean(output, dim=0).unsqueeze(0)

        output = self.reshapeF(output)
        output = self.decoder(output)
        output = self.reshapeB(output)
        return output