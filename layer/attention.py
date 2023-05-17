import math
import torch
import numpy as np
from torch import nn, Tensor, softmax, dropout
from typing import Optional, Tuple
from torch.nn.init import constant_, uniform_, xavier_normal_


class KnowledgeAttention(nn.Module):
    def __init__(self, dim=0) -> None:
        super().__init__()
        self.attention_layer = self._scaled_dot_product_attention
        self.dim = dim
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        # self.w_o = nn.Linear(30 - dim, 20)
        self.dropout = nn.Dropout(p=0.2)
        self.apply(self._init_weights)

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        """

        q: Knowledge prompt (batch_size, 20, past_key_value_dim)
        k: Item sequence i feature (batch_size, num_seq_item, past_key_value_dim)
        v: == k
        """
        k = k.permute(0, 2, 1)
        k = self.w_k(k)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        v = self.w_v(v)
        v = v.permute(0, 2, 1)
        output, attn = self.attention_layer(q, k, v)
        return output

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1.0 / self.seq_len)
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.01)

    def _scaled_dot_product_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        B, Nt, E = q.shape
        q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(q, k.transpose(-2, -1))

        if attn_mask is not None:
            attn += attn_mask

        attn = softmax(attn, dim=-1)

        if dropout_p > 0.0:
            attn = self.dropout(attn)
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v)
        return output, attn


class ReasoningAttention(nn.Module):
    def __init__(self, dim=768, w_o=None) -> None:
        super().__init__()
        self.attention_layer = self._scaled_dot_product_attention
        self.dim = dim
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        if w_o is None:
            self.w_o = nn.Linear(dim, 18432)
        else:
            self.w_o = w_o
        self.apply(self._init_weights)

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        """
        k, v: Knowledge prompt (batch_size, 20, hidden_state_dim)  # 16, 20, 768
        q: Item sequence i feature (batch_size, num_seq_item, past_key_value_dim)  # 16, 10, 768
        """

        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        output, attn = self.attention_layer(q, k, v)
        output = self.w_o(output)
        return output

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1.0 / self.seq_len)
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.01)

    def _scaled_dot_product_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        B, Nt, E = q.shape
        q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(q, k.transpose(-2, -1))

        if attn_mask is not None:
            attn += attn_mask

        attn = softmax(attn, dim=-1)

        if dropout_p > 0.0:
            attn = self.dropout(attn)
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v)
        return output, attn
