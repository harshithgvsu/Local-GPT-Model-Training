# tokenizer.py
# Minimal character-level tokenizer (train/save/load/encode/decode).

import json
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class CharTokenizer:
    stoi: Dict[str, int]  # char -> id
    itos: Dict[int, str]  # id -> char

    @classmethod
    def train(cls, text: str):
        chars = sorted(list(set(text)))  # all unique characters
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        print("stoi-------", stoi)
        print("itos----------", itos)
        return cls(stoi=stoi, itos=itos)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[ch] for ch in s if ch in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.itos[i] for i in ids)

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'stoi': self.stoi}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        stoi = {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in data['stoi'].items()}
        itos = {v: k for k, v in stoi.items()}
        return cls(stoi=stoi, itos=itos)
