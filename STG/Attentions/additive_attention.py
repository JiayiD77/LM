import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self,
                 dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                q, k, v,
                scale=1):
        
        self.scale = scale

        q_reshape = torch.unsqueeze(q, -2) # (batch, T, 1, dim)
        k_reshape = torch.unsqueeze(k, -3) # (batch, 1, T, dim)

        attn_weight = torch.sum(scale * torch.tanh(q_reshape+k_reshape), axis=-1) # (batch, T, T)
        
        attn_weight = F.softmax(attn_weight, dim=-1)
        attn_weight = self.dropout(attn_weight)

        out = attn_weight @ v

        return out


