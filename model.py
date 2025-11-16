import math
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
  def __init__(self, n_embd, n_head, block_size, dropout=0.1):
    super().__init__()
    assert n_embd % n_head == 0
    self.n_head = n_head
    self.head_dim = n_embd // n_head
    self.key = nn.Linear(n_embd, n_embd, bias=False)
    self.query = nn.Linear(n_embd, n_embd, bias=False)
    self.value = nn.Linear(n_embd, n_embd, bias=False)
    self.proj = nn.Linear(n_embd, n_embd)
    self.attn_drop = nn.Dropout(dropout)
    self.resid_drop = nn.Dropout(dropout)
    # Causal mask ensures token t can only attend to <= t
    self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

  def forward(self, x):
    B, T, C = x.size()
    k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
    q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
    v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # scaled dot-product
    att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))  # apply causal mask
    att = torch.softmax(att, dim=-1)
    att = self.attn_drop(att)
    y = att @ v  # attention output
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.resid_drop(self.proj(y))  # output projection
    return y


class Block(nn.Module):
  def __init__(self, n_embd, n_head, block_size, dropout=0.1):
    super().__init__()
    self.ln1 = nn.LayerNorm(n_embd)
    self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
    self.ln2 = nn.LayerNorm(n_embd)
    self.mlp = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.GELU(),
      nn.Linear(4 * n_embd, n_embd),
      nn.Dropout(dropout),
    )

  def forward(self, x):
    x = x + self.attn(self.ln1(x))  # pre-norm + residual
    x = x + self.mlp(self.ln2(x))  # MLP + residual
    return x


class TinyGPT(nn.Module):
  def __init__(self, vocab_size, n_layer=4, n_head=4, n_embd=256, block_size=256, dropout=0.1):
    super().__init__()
    self.block_size = block_size
    self.tok_emb = nn.Embedding(vocab_size, n_embd)  # token embeddings
    self.pos_emb = nn.Embedding(block_size, n_embd)  # learned positional embeddings
    self.drop = nn.Dropout(dropout)
    self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if isinstance(module, nn.Linear) and module.bias is not None:
      nn.init.zeros_(module.bias)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    assert T <= self.block_size, "Sequence length exceeds block_size"
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
    x = self.tok_emb(idx) + self.pos_emb(pos)  # token + position
    x = self.drop(x)
    for block in self.blocks:
      x = block(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)  # (B, T, vocab)

    loss = None
    if targets is not None:
      loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss

  @torch.no_grad()
  def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -self.block_size:]
      logits, _ = self.forward(idx_cond)
      logits = logits[:, -1, :] / max(temperature, 1e-5)  # last token
      if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('inf')  # top-k filter
      probs = torch.softmax(logits, dim=-1)
      next_id = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, next_id), dim=1)
    return idx
