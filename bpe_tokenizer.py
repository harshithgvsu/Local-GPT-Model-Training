# bpe_tokenizer.py
# Tiny, self-contained BPE tokenizer (character-based init + pair merges).
# Goal: better than pure char-level, still easy to read and hack.

import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<SYSTEM>", "<USER>", "<ASSISTANT>"]


def _split_to_chars(word: str) -> Tuple[str, ...]:
  # Represent a word as a tuple of individual chars; EOS marker inside word is not used.
  return tuple(word)


class BPETokenizer:
  def __init__(self, vocab: Dict[str, int], merges: List[Tuple[str, str]]):
    self.vocab = vocab  # token -> id
    self.id2tok = {i: t for t, i in vocab.items()}
    self.merges = merges  # list of merged pairs in order
    # Build the merge table for fast encoding
    self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

  @staticmethod
  def _get_vocab_from_corpus(texts: List[str], reserved: List[str]) -> Counter:
    # Start from individual characters seen in corpus plus reserved specials.
    chars = Counter()
    for t in texts:
      chars.update(t)
    for s in reserved:
      chars[s] += 1  # ensure presence
    return chars

  @staticmethod
  def train(texts: List[str], vocab_size: int = 2000, reserved_tokens: List[str] = None):
    """
        Train a tiny BPE over the given texts.
        - Initialize tokens as characters (plus reserved specials).
        - Iteratively merge most frequent adjacent pairs in 'words' (whitespace tokenized).
        """
    reserved = list(SPECIAL_TOKENS if reserved_tokens is None else reserved_tokens)
    # 1) Build initial vocab from characters
    init = BPETokenizer._get_vocab_from_corpus(texts, reserved)
    # Initial token set (string tokens)
    tokens = {ch for ch in init.keys()}
    # Guarantee reserved tokens are included
    for s in reserved:
      tokens.add(s)
    # 2) Corpus as list of words -> each word as tuple of chars
    corpus_words = []
    for t in texts:
      for w in t.split():
        corpus_words.append(_split_to_chars(w))

    def get_stats(words):
      pairs = Counter()
      for w in words:
        if len(w) < 2:
          continue
        for i in range(len(w) - 1):
          pairs[(w[i], w[i + 1])] += 1
      return pairs

    def merge_words(words, pair):
      a, b = pair
      new_words = []
      bigram = (a, b)
      for w in words:
        if len(w) < 2:
          new_words.append(w)
          continue
        merged = []
        i = 0
        while i < len(w):
          if i < len(w) - 1 and (w[i], w[i + 1]) == bigram:
            merged.append(a + b)
            i += 2
          else:
            merged.append(w[i])
            i += 1
        new_words.append(tuple(merged))
      return new_words

    merges = []
    # 3) Merge until vocab_size (or we run out of pairs)
    while len(tokens) < vocab_size:
      stats = get_stats(corpus_words)
      if not stats:
        break
      best = stats.most_common(1)[0][0]  # (a,b)
      new_token = best[0] + best[1]
      if new_token in tokens:
        # If already present, break to avoid infinite loop
        break
      tokens.add(new_token)
      merges.append(best)
      corpus_words = merge_words(corpus_words, best)

    # 4) Build final vocab mapping token -> id (stable order)
    vocab_list = sorted(tokens)
    vocab = {tok: i for i, tok in enumerate(vocab_list)}
    return BPETokenizer(vocab=vocab, merges=merges)

  def save(self, path: str):
    with open(path, "w", encoding="utf-8") as f:
      json.dump({
        "vocab": self.vocab,
        "merges": self.merges
      }, f, ensure_ascii=False)

  @staticmethod
  def load(path: str):
    with open(path, "r", encoding="utf-8") as f:
      obj = json.load(f)
    merges = [tuple(p) for p in obj["merges"]]
    return BPETokenizer(vocab=obj["vocab"], merges=merges)

  def _bpe_encode_word(self, word: str) -> List[int]:
    # Encode a single whitespace-delimited word using merge ranks
    if not word:
      return []
    symbols = list(word)
    # Greedy pair merges
    while True:
      # Find best ranked pair in 'symbols'
      pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
      if not pairs:
        break
      ranks = [self.merge_ranks.get(p, 1e12) for p in pairs]
      best_i = min(range(len(ranks)), key=lambda i: ranks[i])
      if ranks[best_i] == 1e12:
        break
      a, b = pairs[best_i]
      merged = a + b
      # Merge the best pair
      new_symbols = []
      i = 0
      while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
          new_symbols.append(merged)
          i += 2
        else:
          new_symbols.append(symbols[i])
          i += 1
      symbols = new_symbols
    # Map tokens to ids (unknown pieces fallback to char ids if present)
    ids = []
    for s in symbols:
      if s in self.vocab:
        ids.append(self.vocab[s])
      else:
        # Fallback: split to chars known in vocab
        for ch in s:
          if ch in self.vocab:
            ids.append(self.vocab[ch])
    return ids

  def encode(self, text: str, add_special: bool = False, role: str = None) -> List[int]:
    ids = []
    if add_special and "<BOS>" in self.vocab:
      ids.append(self.vocab["<BOS>"])
    if role and role in self.vocab:
      ids.append(self.vocab[role])
    # Basic whitespace tokenization; BPE merges handle subwords
    for w in text.split():
      ids.extend(self._bpe_encode_word(w))
      # Add a space token if it exists, else a literal space char
      if " " in self.vocab:
        ids.append(self.vocab[" "])
      else:
        # Ensure space is modeled (common in corpus)
        pass
    return ids

  def decode(self, ids: List[int]) -> str:
    tokens = [self.id2tok.get(i, "") for i in ids]
    # Remove special tokens from rendering
    specials = set(SPECIAL_TOKENS)
    tokens = [t for t in tokens if t not in specials and t not in ("<BOS>", "<EOS>")]
    text = "".join(tokens)
    return text
