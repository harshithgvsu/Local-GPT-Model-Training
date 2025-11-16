import argparse, torch, re
from model import TinyGPT
from tokenizer import CharTokenizer
from bpe_tokenizer import BPETokenizer
from web_retriever import get_web_context

SYSTEM_TXT = (
  "You are a helpful, concise assistant. The CONTEXT may contain text scraped from the web. "
  "You MUST pick ONLY the part that answers the user's question and restate it. "
  "If the answer is not in the context, say: 'I don't have that information in the provided context.' "
  "Do NOT invent details."
)

STOP_TOKENS = ["<USER>", "<SYSTEM>", "<CONTEXT>", "</CONTEXT>"]

FALLBACK_KB = {
  "who is albert einstein?": "Albert Einstein was a German-born theoretical physicist best known for developing the "
                             "theory of relativity and the equation E = mc^2.",
  "who is albert einstein": "Albert Einstein was a German-born theoretical physicist best known for developing the "
                            "theory of relativity and the equation E = mc^2.",
}


def pick_device(requested: str):
  if requested == "mps" and torch.backends.mps.is_available():
    return torch.device("mps")
  if requested == "cuda" and torch.cuda.is_available():
    return torch.device("cuda")
  return torch.device("cpu")


def clean_user_query(user: str) -> str:
  user = user.strip()
  user = re.sub(r"^(you|user)\s*:\s*", "", user, flags=re.IGNORECASE)
  return user.strip()


def build_prompt(system, context_blocks, history, user_msg):
  lines = [f"<SYSTEM> {system}"]
  if context_blocks:
    lines.append("<CONTEXT>")
    for i, blk in enumerate(context_blocks, 1):
      src = blk["source"]
      text = blk["text"][:400]
      lines.append(f"[{i}] ({src}) {text}")
    lines.append("</CONTEXT>")
  for u, a in history:
    lines.append(f"<USER> {u}")
    lines.append(f"<ASSISTANT> {a}")
  lines.append(f"<USER> {user_msg}")
  lines.append("<ASSISTANT>")
  return "\n".join(lines)


def load_tokenizer_from_ckpt(ckpt):
  meta = ckpt["tokenizer_meta"]
  if meta["type"] == "bpe":
    tok = BPETokenizer.load(ckpt["tokenizer_bpe_path"])
    return tok, "bpe"
  else:
    stoi = ckpt["tokenizer_char"]["stoi"]
    tok = CharTokenizer(stoi=stoi, itos={v: k for k, v in stoi.items()})
    return tok, "char"


def encode_text(tok, kind, text):
  if kind == "bpe":
    ids = []
    if "<BOS>" in tok.vocab:
      ids.append(tok.vocab["<BOS>"])
    ids += tok.encode(text)
    return ids
  else:
    return tok.encode(text)


def decode_ids(tok, kind, ids):
  return tok.decode(ids)


def cut_at_stop(reply: str) -> str:
  for st in STOP_TOKENS:
    idx = reply.find(st)
    if idx != -1:
      reply = reply[:idx].strip()
  return reply.strip()


def pick_best_sentence_from_context(context_blocks):
  """
    Rule-based: pick the first sentence from the first web block.
    This saves us when the model output is garbage.
    """
  if not context_blocks:
    return None
  txt = context_blocks[0]["text"]
  for sent in re.split(r'(?<=[.?!])\s+', txt):
    s = sent.strip()
    if len(s) > 30:
      return s
  return txt.strip() if txt else None


def looks_like_gibberish(s: str) -> bool:
  if not s:
    return True
  if len(s) < 20:
    return True
  if "context-base" in s.lower():
    return True
  if s.count("#") > 0:
    return True
  return False


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
  ap.add_argument("--temperature", type=float, default=0.5)
  ap.add_argument("--top_k", type=int, default=30)
  ap.add_argument("--web_k", type=int, default=2)
  args = ap.parse_args()

  # load model
  ckpt = torch.load(args.ckpt, map_location="cpu")
  cfg = ckpt["config"]
  tok, kind = load_tokenizer_from_ckpt(ckpt)

  model = TinyGPT(
    cfg["vocab_size"],
    cfg["n_layer"],
    cfg["n_head"],
    cfg["n_embd"],
    cfg["block_size"],
    cfg["dropout"],
  )
  device = pick_device(args.device)
  model.load_state_dict(ckpt["model_state"])
  model.to(device).eval()

  history = []
  print("Web-only RAG chat (with rule-based fallback). Type ':quit' to exit.\n")
  while True:
    raw_user = input("You: ").strip()
    if raw_user == ":quit":
      break

    user = clean_user_query(raw_user)
    # print(f"[DEBUG] Cleaned user query: {user!r}")

    kb_ans = FALLBACK_KB.get(user.lower())

    # 2) fetch web
    # print("[DEBUG] Fetching from web ...")
    web_ctx = get_web_context(user, k=args.web_k)

    # print("\n[DEBUG] Web context:")
    # for wc in web_ctx:
    #     print(f"source={wc['source']}")
    #     print(wc["text"][:160], "...\n")

    context_blocks = web_ctx

    # 3) build prompt
    prompt = build_prompt(SYSTEM_TXT, context_blocks, history, user)
    # print("\n[DEBUG] Final prompt:\n")
    # print(prompt)
    # print("\n[END PROMPT]\n")

    # 4) encode
    ids = encode_text(tok, kind, prompt)
    x = torch.tensor([ids], dtype=torch.long).to(device)

    # 5) generate
    with torch.no_grad():
      y = model.generate(
        x,
        max_new_tokens=160,
        temperature=args.temperature,
        top_k=args.top_k,
      )

    full = decode_ids(tok, kind, y[0].tolist())

    # 6) extract model answer
    cut = full.rfind("<ASSISTANT>")
    if cut != -1:
      reply = full[cut + len("<ASSISTANT>"):].strip()
    else:
      reply = full.strip()

    reply = cut_at_stop(reply)

    # 7) if model failed / garbled, fallback to context (rule-based)
    if looks_like_gibberish(reply):
      ctx_sent = pick_best_sentence_from_context(context_blocks)
      if ctx_sent:
        reply = ctx_sent
      elif kb_ans:
        reply = kb_ans

    # final fallback: kb
    if not reply.strip() and kb_ans:
      reply = kb_ans

    print(f"Assistant: {reply}\n")
    history.append((user, reply))


if __name__ == "__main__":
  main()
