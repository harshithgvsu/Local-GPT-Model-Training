import os, re, math, json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
STOP = set("""
a an the and or of to in on for with without at by from about into over after before as is are was were be being been it its that this these those you your yours me my we our they them their he she his her
""".split())


def tokenize(text: str) -> List[str]:
  return [w.lower() for w in WORD_RE.findall(text)]


def normalize_tokens(tokens: List[str]) -> List[str]:
  return [t for t in tokens if t not in STOP and len(t) > 1]


def read_text_file(path: str) -> str:
  try:
    return open(path, "r", encoding="utf-8").read()
  except Exception:
    # fallback if encoding odd
    return open(path, "r", encoding="latin-1", errors="ignore").read()


def chunk_text(text: str, max_words=300, overlap=60) -> List[str]:
  words = text.split()
  chunks = []
  i = 0
  while i < len(words):
    chunk = " ".join(words[i:i + max_words])
    if chunk.strip():
      chunks.append(chunk)
    i += max_words - overlap
  return chunks


def build_index(docs_dir: str, exts=(".txt", ".md"), max_words=300, overlap=60):
  """
    Scans docs_dir, chunks files, builds a TF-IDF index over chunks.
    Returns a dict with:
      - 'chunks': list of {'text','source','chunk_id'}
      - 'idf': idf dict
      - 'vocab': term -> column
      - 'vectors': list of sparse tf-idf dicts (term_index -> weight, L2-normalized)
    """
  chunks = []
  for root, _, files in os.walk(docs_dir):
    for fn in files:
      if fn.lower().endswith(exts):
        path = os.path.join(root, fn)
        raw = read_text_file(path)
        for j, ch in enumerate(chunk_text(raw, max_words=max_words, overlap=overlap)):
          chunks.append({"text": ch, "source": path, "chunk_id": len(chunks)})

  # build DF
  df = Counter()
  tokenized_chunks = []
  for ch in chunks:
    toks = normalize_tokens(tokenize(ch["text"]))
    tokenized_chunks.append(toks)
    df.update(set(toks))  # document frequency per chunk

  N = max(1, len(chunks))
  idf = {t: math.log((N + 1) / (df_t + 1)) + 1.0 for t, df_t in df.items()}  # smoothed IDF

  # vocab and vectors
  vocab = {t: i for i, t in enumerate(sorted(df.keys()))}
  vectors = []
  for toks in tokenized_chunks:
    tf = Counter(toks)
    vec = {}
    for t, f in tf.items():
      if t in vocab:
        j = vocab[t]
        vec[j] = (f / len(toks)) * idf.get(t, 0.0)
    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    vec = {j: v / norm for j, v in vec.items()}
    vectors.append(vec)

  return {"chunks": chunks, "idf": idf, "vocab": vocab, "vectors": vectors}


def _cosine_sparse(a: Dict[int, float], b: Dict[int, float]) -> float:
  # both are dicts term_index -> weight (already normalized)
  if len(a) > len(b):
    a, b = b, a
  score = 0.0
  for j, v in a.items():
    bv = b.get(j)
    if bv is not None:
      score += v * bv
  return score


def query(index, text: str, k=4) -> List[Tuple[float, Dict]]:
  toks = normalize_tokens(tokenize(text))
  if not toks:
    return []
  # query tf-idf
  tf = Counter(toks)
  q = {}
  for t, f in tf.items():
    if t in index["vocab"]:
      j = index["vocab"][t]
      q[j] = (f / len(toks)) * index["idf"].get(t, 0.0)
  norm = math.sqrt(sum(v * v for v in q.values())) or 1.0
  q = {j: v / norm for j, v in q.items()}

  scores = []
  for vec, ch in zip(index["vectors"], index["chunks"]):
    s = _cosine_sparse(q, vec)
    scores.append((s, ch))
  scores.sort(key=lambda x: x[0], reverse=True)
  return scores[:k]
