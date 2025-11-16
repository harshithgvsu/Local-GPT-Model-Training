import argparse, torch, os
from model import TinyGPT
from tokenizer import CharTokenizer
from bpe_tokenizer import BPETokenizer


def pick_device(requested: str):
  if requested == "mps" and torch.backends.mps.is_available():
    return torch.device("mps")
  if requested == "cuda" and torch.cuda.is_available():
    return torch.device("cuda")
  return torch.device("cpu")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--prompt", default="Hello")
  ap.add_argument("--max_new_tokens", type=int, default=200)
  ap.add_argument("--temperature", type=float, default=0.8)
  ap.add_argument("--top_k", type=int, default=40)
  ap.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
  args = ap.parse_args()

  ckpt = torch.load(args.ckpt, map_location="cpu")
  meta = ckpt["tokenizer_meta"]
  cfg = ckpt["config"]

  # Load tokenizer
  if meta["type"] == "char":
    stoi = ckpt["tokenizer_char"]["stoi"]
    tok = CharTokenizer(stoi=stoi, itos={v: k for k, v in stoi.items()})
    enc = tok.encode(args.prompt)
    bos = []
  else:
    tok_path = ckpt["tokenizer_bpe_path"]
    tok = BPETokenizer.load(tok_path)
    # add BOS token to help generation start cleanly if present
    bos = [tok.vocab["<BOS>"]] if "<BOS>" in tok.vocab else []
    enc = bos + tok.encode(args.prompt)

  model = TinyGPT(cfg["vocab_size"], cfg["n_layer"], cfg["n_head"],
                  cfg["n_embd"], cfg["block_size"], cfg["dropout"])
  model.load_state_dict(ckpt["model_state"])
  device = pick_device(args.device)
  model.to(device).eval()

  x = torch.tensor([enc], dtype=torch.long).to(device)
  with torch.no_grad():
    y = model.generate(x, max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature, top_k=args.top_k)
  print(tok.decode(y[0].tolist()))


if __name__ == "__main__":
  main()
