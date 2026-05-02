import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Head(nn.Module):

    def __init__(self, x_emb, head_emb, masking_enabled = False, cross_attention = False):
        super().__init__()
        self.key = nn.Linear(x_emb, head_emb)
        self.query = nn.Linear(x_emb, head_emb)
        self.value = nn.Linear(x_emb, head_emb)
        self.masking_enabled = masking_enabled
        self.cross_attention = cross_attention

    def forward(self, x, key = None, query = None):
        B, T, C = x.shape
        k = self.key(x)                         # (batch_size, seq_len, head_emb)
        q = self.query(x)                       # (batch_size, seq_len, head_emb)
        v = self.value(x)                       # (batch_size, seq_len, head_emb)

        if self.cross_attention:
            k = key(x)
            q = query(x)

        logits = q @ k.transpose(-2, -1)        # (batch_size, seq_len, head_emb)
        logits = logits / (k.shape[1] ** 0.5)

        if self.masking_enabled:
            mask = torch.tril(torch.ones(T, T))
            logits = logits.masked_fill(mask == 0, float('-inf'))

        logits = F.softmax(logits)
        logits = logits @ v
        return logits
    
class MultiHeadAttention(nn.Module):

    def __init__(self, x_emb, head_emb, heads_num, cross_attention = False, masking_enabled = False):
        super().__init__()
        self.proj = nn.Linear(x_emb, x_emb)
        self.heads = nn.ModuleList([Head(x_emb//heads_num, head_emb, masking_enabled, cross_attention) for _ in range(heads_num)])

    def forward(self, x, key, query):
        head_size = x.shape[-1] // len(self.heads)
        logits = [
            head(x[:, :, i * head_size: (i + 1) * head_size], key, query)
            if head.cross_attention
            else head(x[:, :, i * head_size: (i + 1) * head_size])
            for i, head in enumerate(self.heads)
        ]
        logits = self.proj(logits)
        return logits
    
class MultiHeadBlock(nn.Module):

    def __init__(self, seq_len, x_emb, head_emb, heads_num, cross_attention = False, masking_enabled = False):
        super().__init__()
        self.heads = MultiHeadAttention(x_emb, head_emb, heads_num, cross_attention, masking_enabled)
        self.layer_norm = nn.LayerNorm((seq_len, x_emb))

    def forward(self, tokens, key=None, query=None):
        logits = self.heads(tokens, key, query)
        logits_norm = self.layer_norm(logits)
        return logits_norm + tokens
    
class FeedFwdBlock(nn.Module):

    def __init__(self, seq_len, x_emb):
        super().__init__()
        self.linear = nn.Linear(x_emb, x_emb)
        self.layer_norm = nn.LayerNorm((seq_len, x_emb))

    def forward(self, tokens):
        logits = self.linear(tokens)
        logits_norm = self.layer_norm(logits)
        return logits_norm + tokens
    
class DecoderArchitecture(nn.Module):

    def __init__(self, seq_len, x_emb, head_emb, heads_num):
        super().__init__()
        self.self_attention = MultiHeadBlock(seq_len, x_emb, head_emb, heads_num, cross_attention=False, masking_enabled=True)
        self.cross_attention = MultiHeadBlock(seq_len, x_emb, head_emb, heads_num, cross_attention=True, masking_enabled=True)
        self.feed_fwd = FeedFwdBlock(seq_len, x_emb)

    def forward(self, tokens, key, query):
        logits = self.self_attention(tokens)
        logits = self.self_attention(logits, key, query)
        logits = self.feed_fwd(logits)
        return logits
    
class Decoder(nn.Module):

    def __init__(self, vocab_size, seq_len, x_emb, head_emb, heads_num, decoder_num):
        super().__init__()
        self.architecture = nn.ModuleList([DecoderArchitecture(seq_len, x_emb, head_emb, heads_num)]*decoder_num)
        self.linear = nn.Linear(x_emb, x_emb)
        self.look_up_table = nn.Parameter(torch.randn((vocab_size+2, x_emb)))
        self.postional_enc = nn.Parameter(torch.randn((seq_len, x_emb)))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

    def forward(self, tokens, target = None, key = None, query = None):
        loss = None
        tokens = torch.tensor(tokens)
        x = self.look_up_table(tokens) + self.postional_enc(tokens)

        for decoder in self.architecture:
            x = decoder(tokens, key, query)

        if target:
            target = torch.tensor(target)
            B, T, C = x.shape
            loss = F.cross_entropy(x.view(B*T, C), target.view(B*T))

        return x, loss