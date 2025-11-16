# # chat.py
# import argparse, torch
# from model import TinyGPT
# from tokenizer import CharTokenizer
#
# SYSTEM = "You are a helpful local assistant."
#
# def pick_device(requested: str):
#     if requested == "mps" and torch.backends.mps.is_available():
#         return torch.device("mps")
#     if requested == "cuda" and torch.cuda.is_available():
#         return torch.device("cuda")
#     return torch.device("cpu")
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--ckpt", required=True)
#     ap.add_argument("--device", choices=["cpu","cuda","mps"], default="cpu")
#     args = ap.parse_args()
#
#     ckpt = torch.load(args.ckpt, map_location="cpu")
#     stoi = ckpt["tokenizer"]["stoi"]
#     tokenizer = CharTokenizer(stoi=stoi, itos={v:k for k,v in stoi.items()})
#     cfg = ckpt["config"]
#
#     model = TinyGPT(cfg["vocab_size"], cfg["n_layer"], cfg["n_head"],
#                     cfg["n_embd"], cfg["block_size"], cfg["dropout"])
#     model.load_state_dict(ckpt["model_state"])
#
#     device = pick_device(args.device)
#     model.to(device).eval()
#
#     history = []
#     print("Local Chat. Type ':quit' to exit.\n")
#     while True:
#         user = input("You: ").strip()
#         if user == ":quit":
#             print("Bye!"); break
#
#         # very simple prompt: system + all turns as plain text
#         full = SYSTEM + "\n"
#         for u,a in zip(history[::2], history[1::2]):
#             full += f"User: {u}\nAssistant: {a}\n"
#         full += f"User: {user}\nAssistant:"
#
#         x = torch.tensor([tokenizer.encode(full)], dtype=torch.long).to(device)
#         with torch.no_grad():
#             y = model.generate(x, max_new_tokens=256, temperature=0.8, top_k=40)
#         gen = tokenizer.decode(y[0].tolist())[len(full):]
#
#         # stop on next "User:" if the model echoes the pattern
#         cut = gen.find("\nUser:")
#         if cut != -1:
#             gen = gen[:cut].strip()
#
#         print(f"Assistant: {gen}\n")
#         history.extend([user, gen])
#
# if __name__ == "__main__":
#     main()




# chat.py
import argparse, torch
from model import TinyGPT
from tokenizer import CharTokenizer
from bpe_tokenizer import BPETokenizer

SYSTEM_TXT = "You are a helpful, concise assistant for Harshith."

def pick_device(requested: str):
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def build_prompt(history, user_msg, tok, is_bpe: bool):
    # Simple role-tagged transcript the model saw during training
    lines = [f"<SYSTEM> {SYSTEM_TXT}"]
    for u, a in history:
        lines.append(f"<USER> {u}")
        lines.append(f"<ASSISTANT> {a}")
    lines.append(f"<USER> {user_msg}")
    lines.append(f"<ASSISTANT>")
    text = "\n".join(lines)

    if is_bpe:
        ids = []
        if "<BOS>" in tok.vocab:
            ids.append(tok.vocab["<BOS>"])
        # Prefix role tokens (optional; we mainly use textual tags)
        ids += tok.encode(text)
        return ids
    else:
        return CharTokenizer(stoi=tok.stoi, itos=tok.itos).encode(text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", choices=["cpu","cuda","mps"], default="cpu")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    meta = ckpt["tokenizer_meta"]
    cfg = ckpt["config"]

    # Load tokenizer
    is_bpe = (meta["type"] == "bpe")
    if is_bpe:
        tok = BPETokenizer.load(ckpt["tokenizer_bpe_path"])
    else:
        stoi = ckpt["tokenizer_char"]["stoi"]
        tok = CharTokenizer(stoi=stoi, itos={v:k for k,v in stoi.items()})

    model = TinyGPT(cfg["vocab_size"], cfg["n_layer"], cfg["n_head"],
                    cfg["n_embd"], cfg["block_size"], cfg["dropout"])
    model.load_state_dict(ckpt["model_state"])
    device = pick_device(args.device)
    model.to(device).eval()

    history = []
    print("Local Chat (BPE aware). Type ':quit' to exit.\n")
    while True:
        user = input("You: ").strip()
        if user == ":quit":
            print("Bye!"); break

        ids = build_prompt(history, user, tok, is_bpe)
        x = torch.tensor([ids], dtype=torch.long).to(device)
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=256,
                               temperature=args.temperature, top_k=args.top_k)
        full = tok.decode(y[0].tolist())
        # Extract only the assistant continuation after the last "<ASSISTANT>"
        cut = full.rfind("<ASSISTANT>")
        reply = full[cut + len("<ASSISTANT>"):].strip() if cut != -1 else full.strip()
        # Stop if model started next turn
        next_turn = reply.find("<USER>")
        if next_turn != -1:
            reply = reply[:next_turn].strip()

        print(f"Assistant: {reply}\n")
        history.append((user, reply))
if __name__ == "__main__":
    main()
