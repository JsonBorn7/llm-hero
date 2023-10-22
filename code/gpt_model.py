# coding: utf-8

import math
import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary


class GPTConfig:
    vocab_size: int = 16000
    seq_len: int = 128
    d_model: int = 128 # d_model
    n_layer: int = 4
    n_head: int = 4
    bias: bool = True
    dropout: float = 0.0
  

class SinusoidPE(nn.Module):
    """ sin/cos position encoding """
    
    def __init__(self, config):
        super().__init__()
        d_model, seq_len = config.d_model, config.seq_len
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('sinusoid_pe', pe)
        
    def forward(self, x):
        return self.sinusoid_pe[:, :x.shape[1], :]


class SelfAttention(nn.Module):
  """ multi-head attention """

  def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # output projection
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        # self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.dropout = config.dropout
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.seq_len, config.seq_len))
                                          .view(1, 1, config.seq_len, config.seq_len))

  def forward(self, x):
      B, C, E = x.size()

      q, k, v  = self.attn(x).split(self.d_model, dim=2)
      q = q.view(B, C, self.n_head, E // self.n_head).transpose(1, 2) # (B, nh, C, hs)
      k = k.view(B, C, self.n_head, E // self.n_head).transpose(1, 2) # (B, nh, C, hs)
      v = v.view(B, C, self.n_head, E // self.n_head).transpose(1, 2) # (B, nh, C, hs)

      # self-attention: (B, nh, C, hs) x (B, nh, hs, C) -> (B, nh, C, C)
      att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
      att = att.masked_fill(self.mask[:,:,:C,:C] == 0, float('-inf'))
      att = F.softmax(att, dim=-1)
      att = self.attn_dropout(att)
      y = att @ v # (B, nh, C, C) x (B, nh, C, hs) -> (B, nh, C, hs)
      y = y.transpose(1, 2).contiguous().view(B, C, E)

      return self.proj(y)


class FeedFoward(nn.Module):
    """ a two-layers mlp """

    def __init__(self, config):
        super().__init__()
        d_model = config.d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """ Decoder Block """ 

    def __init__(self, config):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.ffn = FeedFoward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTModel(nn.Module):
  
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embed_table = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed_table = SinusoidPE(config) 
        self.decoder_blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)]) 
        self.layer_norm = nn.LayerNorm(config.d_model, bias=config.bias)
        self.final_linear = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, features, targets=None):
        tok_emb = self.tok_embed_table(features) # (B,C,E)
        pos_emb = self.pos_embed_table(tok_emb)
        x = tok_emb + pos_emb # (B,C,E)
        x = self.decoder_blocks(x)
        
        x = self.layer_norm(x)
        if targets is not None:
          logits = self.final_linear(x) # (B,C,V)
          loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
          logits = self.final_linear(x[:, [-1], :])
          loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, seq, max_new_tokens):
        for _ in range(max_new_tokens):
            seq = seq[:, -self.config.seq_len:]
            logits, _ = self(seq)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, V)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, V)
            # sample from the distribution
            seq_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            seq = torch.cat((seq, seq_next), dim=1)
        return seq


def main():
    config = GPTConfig()
    model = GPTModel(config)
    summary(model, input_size=[(100, config.seq_len), (100, config.seq_len)],
            dtypes=[torch.long, torch.long])


if __name__ == '__main__':
    main()