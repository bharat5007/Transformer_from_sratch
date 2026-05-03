import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import sys, os
sys.path.insert(0, os.path.abspath(".."))
from python.tokenizer import Tokenizer

text = ""
with open('../dataset/1 - A Game of Thrones.txt', 'r', encoding='utf-8', errors='ignore') as f:
    text += f.read()

tokenizer = Tokenizer.load("../tokenizer.json")
tokens = tokenizer.encode(text)


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

    def __init__ (self, heads_num, x_emb):
        super().__init__()
        self.layer_norm = nn.LayerNorm(x_emb)
        self.heads = MultiHeadAttention(x_emb, heads_num, x_emb//heads_num)

    def forward(self, tokens):
        x = self.layer_norm(tokens)
        x = self.heads(x)
        return tokens + x

class FeedFwdBlock(nn.Module):

    def __init__(self, x_emb):
        super().__init__()
        self.layer = nn.Sequential(
        nn.Linear(x_emb, 4 * x_emb),
        nn.GELU(),
        nn.Linear(4 * x_emb, x_emb)
    )
        self.layer_norm = nn.LayerNorm(x_emb)

    def forward(self, tokens):
        x = self.layer_norm(tokens)
        x = self.layer(x)
        return tokens + x

class EncoderArchitecture(nn.Module):

    def __init__(self, x_emb, heads_num):
        super().__init__()
        self.multihead = MultiheadBlock(heads_num, x_emb)
        self.feed_fwd = FeedFwdBlock(x_emb)

    def forward(self, tokens):
        x = self.multihead(tokens)
        x = self.feed_fwd(x)
        return x

class Encoder(nn.Module):

    def __init__(self, vocab_size: int, x_emb: int, seq_len: int, heads_num: int, encoder_num: int):
        super().__init__()
        self.seq_len = seq_len
        self.MASK_TOKEN_ID = vocab_size+2
        self.look_up_table = nn.Parameter(torch.randn((vocab_size+3, x_emb)))
        self.postional_enc = nn.Parameter(torch.randn((seq_len, x_emb)))
        self.architecture = nn.ModuleList([EncoderArchitecture(x_emb, heads_num) for _ in range(encoder_num)])
        self.lm_head = nn.Linear(x_emb, vocab_size)

    def encode(self, tokens):
        T = tokens.shape[1]
        x = self.look_up_table[tokens] + self.postional_enc[torch.arange(T)]
        for block in self.architecture:
            x = block(x)
        return x  # (B, T, x_emb)

    def forward(self, tokens, targets=None):
        tokens = torch.tensor(tokens)
        x = self.encode(tokens)
        logits = self.lm_head(x)       # (B, T, vocab_size)

        if targets is not None:
            targets = torch.tensor(targets)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss

    def fit(self, tokens, epochs=100, batch_size=32):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        chunks = []
        for i in range(0, len(tokens) - self.seq_len, self.seq_len):
            chunks.append(tokens[i : i + self.seq_len])

        last = tokens[len(chunks) * (self.seq_len):]
        if len(last) > 0:
            last = last + [0] * (self.seq_len - len(last))
            chunks.append(last)

        x_chunks = []
        y_chunks = []
        for chunk in chunks:
            x = chunk.copy()
            y = [-100] * len(chunk)

            for i, token in enumerate(chunk):
                if random.random() < 0.15:
                    y[i] = token
                    x[i] = self.MASK_TOKEN_ID
            x_chunks.append(x)
            y_chunks.append(y)

        for epoch in range(epochs):
            combined = list(zip(x_chunks, y_chunks))
            random.shuffle(combined)
            x_shuffled, y_shuffled = zip(*combined)
            x, y = list(x_shuffled), list(y_shuffled)

            x_batch, y_batch = [], []
            for _ in range(batch_size):
                i = random.randint(0, len(x) - 1)
                x_batch.append(x[i])
                y_batch.append(y[i])

            output, loss = self.forward(x_batch, y_batch)
            print(f"Loss: {loss.item():.4f}, epoch: {epoch}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()