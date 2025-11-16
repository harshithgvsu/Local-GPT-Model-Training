# # train.py
# import argparse, torch, math
# from torch.utils.data import Dataset, DataLoader
# from tokenizer import CharTokenizer
# from model import TinyGPT
#
#
# class CharDataset(Dataset):
#   def __init__(self, data_ids, block_size):
#     self.data = data_ids
#     self.block_size = block_size
#
#   def __len__(self):
#     return max(0, len(self.data) - self.block_size - 1)
#
#   def __getitem__(self, idx):
#     x = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.long)
#     y = torch.tensor(self.data[idx + 1:idx + self.block_size + 1], dtype=torch.long)
#     return x, y
#
#
# def pick_device(requested: str):
#   if requested == "mps" and torch.backends.mps.is_available():
#     return torch.device("mps")
#   if requested == "cuda" and torch.cuda.is_available():
#     return torch.device("cuda")
#   return torch.device("cpu")
#
#
# def main():
#   ap = argparse.ArgumentParser()
#   ap.add_argument("--data", required=True, help="path to text file")
#   ap.add_argument("--out", default="ckpt.pt")
#   ap.add_argument("--epochs", type=int, default=2)
#   ap.add_argument("--batch_size", type=int, default=64)
#   ap.add_argument("--block_size", type=int, default=256)
#   ap.add_argument("--n_layer", type=int, default=4)
#   ap.add_argument("--n_head", type=int, default=4)
#   ap.add_argument("--n_embd", type=int, default=256)
#   ap.add_argument("--dropout", type=float, default=0.1)
#   ap.add_argument("--lr", type=float, default=3e-4)
#   ap.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
#   args = ap.parse_args()
#
#   text = open(args.data, "r", encoding="utf-8").read()
#
#   # 1) tokenizer
#   tokenizer = CharTokenizer.train(text)
#   vocab_size = len(tokenizer.stoi)
#   print(f"Vocab size: {vocab_size}")
#
#   # 2) encode all data, split train/val
#   ids = tokenizer.encode(text)
#   split = int(0.9 * len(ids)) if len(ids) > args.block_size * 2 else int(0.8 * len(ids))
#   train_ids = ids[:split]
#   val_ids = ids[split:] if split < len(ids) else ids[:]
#
#   train_dl = DataLoader(CharDataset(train_ids, args.block_size),
#                         batch_size=args.batch_size, shuffle=True, drop_last=True)
#   val_dl = DataLoader(CharDataset(val_ids, args.block_size),
#                       batch_size=args.batch_size, shuffle=False, drop_last=True)
#
#   device = pick_device(args.device)
#   print("Using device:", device)
#
#   model = TinyGPT(vocab_size, args.n_layer, args.n_head, args.n_embd,
#                   args.block_size, args.dropout).to(device)
#   optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
#   best_val = math.inf
#
#   def eval_loss(loader):
#     model.eval()
#     losses = []
#     with torch.no_grad():
#       for x, y in loader:
#         x, y = x.to(device), y.to(device)
#         _, loss = model(x, y)
#         losses.append(loss.item())
#     model.train()
#     return sum(losses) / len(losses) if losses else float('inf')
#
#   model.train()
#   for epoch in range(1, args.epochs + 1):
#     for i, (x, y) in enumerate(train_dl):
#       x, y = x.to(device), y.to(device)
#       _, loss = model(x, y)
#       optim.zero_grad()
#       loss.backward()
#       torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#       optim.step()
#
#       if (i + 1) % 50 == 0:
#         print(f"epoch {epoch} step {i + 1}/{len(train_dl)} loss {loss.item():.4f}")
#
#     val_loss = eval_loss(val_dl)
#     print(f"[epoch {epoch}] val_loss: {val_loss:.4f}")
#
#     if val_loss < best_val:
#       best_val = val_loss
#       torch.save({
#         "config": {
#           "vocab_size": vocab_size,
#           "n_layer": args.n_layer,
#           "n_head": args.n_head,
#           "n_embd": args.n_embd,
#           "block_size": args.block_size,
#           "dropout": args.dropout,
#         },
#         "model_state": model.state_dict(),
#         "tokenizer": {"stoi": tokenizer.stoi},
#       }, args.out)
#       print(f"Saved best checkpoint to {args.out} (val_loss={best_val:.4f})")
#
#   # Always save the final model even if no validation set
#   torch.save({
#     "config": {
#       "vocab_size": vocab_size,
#       "n_layer": args.n_layer,
#       "n_head": args.n_head,
#       "n_embd": args.n_embd,
#       "block_size": args.block_size,
#       "dropout": args.dropout,
#     },
#     "model_state": model.state_dict(),
#     "tokenizer": {"stoi": tokenizer.stoi},
#   }, args.out)
#   print(f"Saved final checkpoint to {args.out}")
#
#
# if __name__ == "__main__":
#   main()


# train.py
import argparse, torch, math, os
from torch.utils.data import Dataset, DataLoader
from model import TinyGPT
from tokenizer import CharTokenizer
from bpe_tokenizer import BPETokenizer, SPECIAL_TOKENS


def pick_device(requested: str):
  if requested == "mps" and torch.backends.mps.is_available():
    return torch.device("mps")
  if requested == "cuda" and torch.cuda.is_available():
    return torch.device("cuda")
  return torch.device("cpu")


def build_tokenizer(args, raw_text: str):
  if args.tokenizer == "char":
    tok = CharTokenizer.train(raw_text)
    meta = {"type": "char"}
    vocab_size = len(tok.stoi)
    return tok, vocab_size, meta
  else:  # "bpe"
    # You can concatenate multiple files later; start with one.
    tok = BPETokenizer.train([raw_text], vocab_size=args.vocab_size, reserved_tokens=SPECIAL_TOKENS)
    meta = {"type": "bpe"}
    vocab_size = len(tok.vocab)
    return tok, vocab_size, meta


def format_training_text_for_chat(raw_text: str) -> str:
  """
    Minimal 'instruction' formatting:
    Wrap your raw text as a single 'conversation' so the model sees markers.
    For best results, create a proper instruction dataset (USER/ASSISTANT pairs).
    """
  system = "You are a helpful, concise assistant for Harshith."
  return f"<SYSTEM> {system}\n<USER> {raw_text}\n<ASSISTANT>"


class LMWindows(Dataset):
  def __init__(self, all_ids, block_size):
    self.data = all_ids
    self.block_size = block_size

  def __len__(self):
    return max(0, len(self.data) - self.block_size - 1)

  def __getitem__(self, idx):
    x = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.long)
    y = torch.tensor(self.data[idx + 1:idx + self.block_size + 1], dtype=torch.long)
    return x, y


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--data", required=True, help="path to text file")
  ap.add_argument("--out", default="ckpt.pt")
  ap.add_argument("--epochs", type=int, default=2)
  ap.add_argument("--batch_size", type=int, default=64)
  ap.add_argument("--block_size", type=int, default=256)
  ap.add_argument("--n_layer", type=int, default=6)
  ap.add_argument("--n_head", type=int, default=6)
  ap.add_argument("--n_embd", type=int, default=384)
  ap.add_argument("--dropout", type=float, default=0.1)
  ap.add_argument("--lr", type=float, default=3e-4)
  ap.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
  ap.add_argument("--tokenizer", choices=["char", "bpe"], default="bpe")
  ap.add_argument("--vocab_size", type=int, default=2000, help="for BPE")
  args = ap.parse_args()

  raw = open(args.data, "r", encoding="utf-8").read()
  # Optional: use a simple chat wrapper so it learns the markers
  wrapped = raw  # format_training_text_for_chat(raw)

  tok, vocab_size, meta = build_tokenizer(args, wrapped)
  print(f"Tokenizer: {meta['type']} | vocab_size={vocab_size}")

  # Encode
  if meta["type"] == "char":
    ids = tok.encode(wrapped)
  else:
    # Add BOS to help decoding boundaries
    ids = tok.encode(wrapped, add_special=True)

  # Train/val split
  split = int(0.9 * len(ids)) if len(ids) > args.block_size * 2 else int(0.8 * len(ids))
  train_ids = ids[:split]
  val_ids = ids[split:] if split < len(ids) else ids[:]

  device = pick_device(args.device)
  print("Using device:", device)

  train_dl = DataLoader(LMWindows(train_ids, args.block_size),
                        batch_size=args.batch_size, shuffle=True, drop_last=True)
  val_dl = DataLoader(LMWindows(val_ids, args.block_size),
                      batch_size=args.batch_size, shuffle=False, drop_last=False)  # allow small val

  model = TinyGPT(vocab_size, args.n_layer, args.n_head, args.n_embd,
                  args.block_size, args.dropout).to(device)
  optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
  best_val = math.inf

  def eval_loss(loader):
    model.eval()
    losses = []
    with torch.no_grad():
      for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float('inf')

  model.train()
  for epoch in range(1, args.epochs + 1):
    for i, (x, y) in enumerate(train_dl):
      x, y = x.to(device), y.to(device)
      _, loss = model(x, y)
      optim.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optim.step()
      if (i + 1) % 50 == 0:
        print(f"epoch {epoch} step {i + 1}/{len(train_dl)} loss {loss.item():.4f}")

    val_loss = eval_loss(val_dl)
    print(f"[epoch {epoch}] val_loss: {val_loss:.4f}")
    if val_loss < best_val:
      best_val = val_loss
      checkpoint = {
        "config": {
          "vocab_size": vocab_size,
          "n_layer": args.n_layer,
          "n_head": args.n_head,
          "n_embd": args.n_embd,
          "block_size": args.block_size,
          "dropout": args.dropout,
        },
        "model_state": model.state_dict(),
        "tokenizer_meta": meta,
      }
      # Save tokenizer
      if meta["type"] == "char":
        checkpoint["tokenizer_char"] = {"stoi": tok.stoi}
      else:
        # persist BPE tokenizer to a sidecar file and path in ckpt
        tok_path = os.path.splitext(args.out)[0] + ".tokenizer.json"
        tok.save(tok_path)
        checkpoint["tokenizer_bpe_path"] = tok_path

      torch.save(checkpoint, args.out)
      print(f"Saved best checkpoint to {args.out} (val_loss={best_val:.4f})")

  # Always save final
  if best_val == math.inf:
    checkpoint = {
      "config": {
        "vocab_size": vocab_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "block_size": args.block_size,
        "dropout": args.dropout,
      },
      "model_state": model.state_dict(),
      "tokenizer_meta": meta,
    }
    if meta["type"] == "char":
      checkpoint["tokenizer_char"] = {"stoi": tok.stoi}
    else:
      tok_path = os.path.splitext(args.out)[0] + ".tokenizer.json"
      tok.save(tok_path)
      checkpoint["tokenizer_bpe_path"] = tok_path
    torch.save(checkpoint, args.out)
    print(f"Saved final checkpoint to {args.out}")


if __name__ == "__main__":
  main()
