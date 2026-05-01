import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import Tokenizer 

class Head(nn.Module):

    def __init__(self, x_emb, head_emb):
        super().__init__()
        self.k = nn.Linear(x_emb, head_emb)
        self.q = nn.Linear(x_emb, head_emb)
        self.v = nn.Linear(x_emb, head_emb)

    def forward(self, x):
        k = self.k(x)                                   # (seq_length, head_emb)
        q = self.q(x)                                   # (seq_length, head_emb)
        v = self.v(x)                                   # (seq_length, head_emb)

        x = q @ k.transpose(-2, -1)                     # (seq_length, seq_length) we have not used q@k.T since it will be invalid operation in case of batches
        x = x/ pow(k.shape[-1], 0.5)
        x = F.softmax(x, dim=-1)                        # (seq_length, seq_length)
        x = x @ v                                       # (seq_length, head_emb)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, x_emb, heads_num, head_emb):
        super().__init__()
        self.heads_num = heads_num
        self.heads = nn.ModuleList([Head(x_emb, head_emb) for _ in range(heads_num)])
        self.proj = nn.Linear(x_emb, x_emb)

    def forward(self, tokens):
        x = torch.cat([head.forward(tokens) for head in self.heads], dim=-1)
        x = self.proj(x)
        return x
        
class MultiheadBlock(nn.Module):

    def __init__ (self, heads_num, seq_length, x_emb):
        super().__init__()
        self.layer_norm = nn.LayerNorm((seq_length, x_emb))
        self.heads = MultiHeadAttention(x_emb, heads_num, x_emb//heads_num)

    def forward(self, tokens):
        x = self.heads(tokens)
        x = self.layer_norm(x)
        return tokens + x
    
class FeedFwdBlock(nn.Module):

    def __init__(self, seq_length, x_emb):
        super().__init__()
        self.layer = nn.Linear(x_emb, x_emb)
        self.layer_norm = nn.LayerNorm((seq_length, x_emb))

    def forward(self, tokens):
        x = self.layer(tokens)
        x = self.layer_norm(x)
        return tokens + x
    
class Encoder:

    def __init__(self, vocab_size: int, x_emb: int, seq_len: int, heads_num: int, special_tokens = {}):
        self.seq_len = seq_len
        self.special_tokens = special_tokens
        self.look_up_table = torch.randn((vocab_size+2, x_emb))
        self.postional_enc = torch.randn((x_emb, seq_len))
        self.architecture = nn.ModuleList([
            MultiheadBlock(heads_num, seq_len, x_emb),
            FeedFwdBlock(seq_len, x_emb)
        ])
        self.tokenizer = Tokenizer(special_tokens)

    def add_padding(self, tokens: list):
        if len(tokens) >= self.seq_len:
            return tokens[: self.seq_len]
        else:
            padding_size = self.seq_len - len(tokens)
            padding = [301]*padding_size
            return tokens + padding

    def forward(self, text: str, targets = None):
        # implement tokenizer here
        tokens = self.tokenizer.encode(text)
        tokens = self.add_padding(tokens)
        tokens = torch.tensor(tokens)

        loss = None

        # lookup_table + postional_enc
        x = self.look_up_table[tokens] + (tokens * self.postional_enc).T

        for block in self.architecture:
            x = block(x)
        
        if targets:
            B, T, C = x.shape
            x = x.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(x, targets)
        
        return x, loss
    
test_encoder = Encoder(300, 20, 100, 4, {"<|endoftext|>": 50256, "<|padding|>": 50257})
test_encoder.forward("Hi bharat")