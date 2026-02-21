import torch
from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super(TransformerBlock, self).__init__()
        self.m_att = nn.MultiheadAttention(dim, 1)
        self.ff = nn.Sequential(nn.Linear(dim, dim*4),
                                nn.ReLU(),
                                nn.Linear(dim*4, dim))
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        att_out, _ = self.m_att(x, x, x)
        x = self.ln1(x+ att_out)
        ff_out = self.ff(x)
        return self.ln2(x+ff_out)