"""Learning memory: rules and definitions the user wants the agent to remember.

Rules ("ignore test rows", "revenue = sum of Booking Value") are saved to a small
JSON file with an embedding each. Before every question the most relevant rules
are found by cosine similarity and given to the agent.

Embeddings come from Gemini (free tier). Without a Gemini key, or when the API
fails, a simple word-overlap score is used instead, so memory always works.
"""

import json
import re
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_PATH = Path("memory/rules.json")
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMS = 768
TOP_K = 3
MIN_SIMILARITY = 0.60   # cosine, gemini-embedding-001 (unrelated text scores ~0.45-0.55)
MIN_OVERLAP = 0.25      # word-overlap fallback: share of the rule's words found in the question

STOPWORDS = set("""a an the and or of to in on for by with is are was were be been it this that these
those as at from when what which who how do does did show me my i we you all any each per than then
there their its into about only always never use using""".split())


def gemini_embedder(api_key: str) -> Callable[[List[str], bool], List[List[float]]]:
    """Returns embed(texts, is_query) using Gemini embeddings."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    client = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=api_key)

    def embed(texts: List[str], is_query: bool = False) -> List[List[float]]:
        if is_query:
            return [client.embed_query(t, output_dimensionality=EMBED_DIMS) for t in texts]
        return client.embed_documents(texts, output_dimensionality=EMBED_DIMS)
    return embed


def _words(text: str) -> set:
    words = (w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in STOPWORDS and len(w) > 1)
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words}  # charts -> chart


def _overlap(question: str, rule: str) -> float:
    """Shared words / words in the shorter text, so a long rule can still match a short question."""
    rw, qw = _words(rule), _words(question)
    return len(rw & qw) / min(len(rw), len(qw)) if rw and qw else 0.0


class RuleMemory:
    """path=None keeps rules in memory only (used on the public app, so visitors don't share rules)."""

    def __init__(self, path: Optional[Path] = DEFAULT_PATH,
                 embed: Optional[Callable[[List[str], bool], List[List[float]]]] = None):
        self.path = Path(path) if path else None
        self.embed = embed
        self.rules: List[Dict] = []
        if self.path and self.path.exists():
            try:
                self.rules = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.rules = []

    # ------------------------------------------------------------ storage

    def _save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.rules, indent=1), encoding="utf-8")

    def _embed(self, texts: List[str], is_query: bool) -> Optional[List[List[float]]]:
        if not self.embed:
            return None
        try:
            return self.embed(texts, is_query)
        except Exception:  # quota / network: fall back to word overlap
            return None

    def add(self, text: str) -> Dict:
        text = text.strip()
        vec = self._embed([text], is_query=False)
        rule = {"id": uuid.uuid4().hex[:8], "text": text, "created": time.strftime("%Y-%m-%d %H:%M"),
                "embedding": vec[0] if vec else None}
        self.rules.append(rule)
        self._save()
        return rule

    def delete(self, rule_id: str):
        self.rules = [r for r in self.rules if r["id"] != rule_id]
        self._save()

    # ------------------------------------------------------------ retrieval

    def relevant(self, question: str, k: int = TOP_K) -> List[Tuple[Dict, float]]:
        """Up to k rules related to the question, best first, as (rule, score)."""
        if not self.rules:
            return []
        # Rules saved while embeddings were unavailable get one now, if possible.
        missing = [r for r in self.rules if not r.get("embedding")]
        if missing and (vecs := self._embed([r["text"] for r in missing], is_query=False)):
            for r, v in zip(missing, vecs):
                r["embedding"] = v
            self._save()

        qvec = self._embed([question], is_query=True)
        scored = []
        for r in self.rules:
            if qvec and r.get("embedding"):
                a, b = np.array(qvec[0]), np.array(r["embedding"])
                score = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
                if score >= MIN_SIMILARITY:
                    scored.append((r, score))
            else:
                score = _overlap(question, r["text"])
                if score >= MIN_OVERLAP:
                    scored.append((r, score))
        return sorted(scored, key=lambda x: -x[1])[:k]


REMEMBER_PATTERN = re.compile(r"^\s*(remember|note|rule)\s*[:\-]\s*(.+)$", re.I | re.S)


def parse_remember(message: str) -> Optional[str]:
    """'remember: revenue = sum of Booking Value' -> 'revenue = sum of Booking Value'."""
    m = REMEMBER_PATTERN.match(message)
    return m.group(2).strip() if m else None
