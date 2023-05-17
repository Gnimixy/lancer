import torch
import numpy as np
from torch import nn
from torch.nn.init import constant_, uniform_, xavier_normal_


class PrefixPrompt(nn.Module):
    def __init__(
        self, n_layer=12, n_embd=768, n_head=12, seq_len=20, consist=2, device="cpu"
    ) -> None:
        super().__init__()
        self.n_layer = n_layer
        self.consist = consist
        self.n_head = n_head
        self.seq_len = seq_len
        self.n_embd = n_embd
        self.device = device

        self.task_token = torch.arange(self.seq_len, device=device).long()
        self.task_embedding_layer = nn.Embedding(self.seq_len, self.n_embd)
        self.mlp1 = nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd),
            nn.Tanh(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(self.n_embd, 2 * self.n_layer * self.n_embd),
            nn.Dropout(0.2),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1.0 / self.seq_len)
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def forward(self, attention_mask, beams=1, output="tuple"):
        batch_size = attention_mask.shape[0]
        task_embedding = self.task_embedding_layer(self.task_token)  # [seq_len, n_embd]
        
        if output == "tuple" or output == "tensor":
            task_embedding = self.mlp1(task_embedding)
            task_prefix = self.mlp2(task_embedding).repeat(
                batch_size * beams, 1
            )  # [batch_size * seq_len * beams, consist * n_layers * n_embd]

            task_prefix = task_prefix.view(
                batch_size * beams,
                self.seq_len,
                self.consist,
                self.n_layer,
                self.n_head,
                self.n_embd // self.n_head,
            )
            past_key_values = task_prefix.permute(
                3, 2, 0, 4, 1, 5
            )  # [n_layers, consist, batch_size, n_head, seq_len, n_embd // nhead]
            if output == "tuple":
                past_key_values = tuple(tuple(j for j in i) for i in past_key_values)
                
        elif output == "attn":
            task_embedding = self.mlp1(task_embedding)
            task_prefix = self.mlp2(task_embedding).repeat(batch_size, 1)
            past_key_values = task_prefix.view(batch_size, self.seq_len, -1).contiguous()
        
        elif output == "hidden_state":
            task_prefix = self.mlp1(task_embedding).repeat(batch_size, 1)
            past_key_values = task_prefix.view(batch_size, self.seq_len, -1).contiguous()
            return past_key_values, attention_mask
        
        prefix_attention_mask = torch.ones(batch_size, self.seq_len, device=self.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        
        return past_key_values, attention_mask
