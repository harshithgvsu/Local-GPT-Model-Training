import requests
from bs4 import BeautifulSoup

HEADERS = {
  "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
}


def fetch_page(url: str) -> str:
  """Download raw HTML from a webpage."""
  resp = requests.get(url, headers=HEADERS, timeout=10)
  resp.raise_for_status()
  return resp.text


def extract_text(html: str, max_chars=1500) -> str:
  """Extract readable paragraph text from HTML."""
  soup = BeautifulSoup(html, "html.parser")
  parts = []
  for p in soup.find_all("p"):
    t = p.get_text(" ", strip=True)
    if t:
      parts.append(t)
    if sum(len(x) for x in parts) > max_chars:
      break
  return "\n".join(parts)


def duckduckgo_search_json(query: str, k: int = 3):
  """Use DuckDuckGo's public JSON API to get search results."""
  url = f"https://api.duckduckgo.com/?q={query}&format=json&no_redirect=1&no_html=1"
  resp = requests.get(url, headers=HEADERS, timeout=10)
  if resp.status_code != 200:
    return []
  data = resp.json()
  results = []

  # Direct Answer if exists
  if data.get("AbstractText"):
    results.append({"source": "DuckDuckGo_Abstract", "text": data["AbstractText"]})

  # RelatedTopics often contain useful links
  for topic in data.get("RelatedTopics", [])[:k]:
    if isinstance(topic, dict) and topic.get("Text"):
      results.append({
        "source": topic.get("FirstURL", "DuckDuckGo_Related"),
        "text": topic["Text"]
      })
  return results


def duckduckgo_search_html(query: str, k: int = 3):
  """Fallback HTML scraper if JSON API has no results."""
  url = "https://duckduckgo.com/html/?q=" + query.replace(" ", "+")
  html = fetch_page(url)
  soup = BeautifulSoup(html, "html.parser")
  results = []
  for a in soup.select(".result__a")[:k]:
    href = a.get("href")
    if href:
      results.append(href)
  return results


def get_web_context(query: str, k: int = 2):
  """Hybrid retriever: try JSON API first, then scrape pages."""
  contexts = []

  # Step 1: try JSON API
  json_results = duckduckgo_search_json(query, k=k)
  contexts.extend(json_results)

  # Step 2: fallback scrape
  if not contexts:
    urls = duckduckgo_search_html(query, k=k)
    for u in urls:
      try:
        html = fetch_page(u)
        text = extract_text(html, max_chars=1200)
        if text.strip():
          contexts.append({"source": u, "text": text})
      except Exception:
        continue

  # Step 3: ensure something always returns
  if not contexts:
    contexts.append({
      "source": "None",
      "text": f"No web results found for '{query}'."
    })

  return contexts
