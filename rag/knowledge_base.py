"""
rag/knowledge_base.py
----------------------
A fully offline, lightweight retrieval layer used to ground the LLM's advice
in real agriculture reference material (extension guides, ICAR advisories,
pesticide dosage sheets, etc.) instead of letting it hallucinate.

Uses TF-IDF (scikit-learn) rather than a neural embedding model so it stays
small and fast on-device -- no extra model download required. Swap in a
sentence-embedding model + FAISS later if you want semantic search.

Put your source material (plain .txt or .pdf) into data/knowledge/ before use.
"""

import os
import glob
import pickle
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_DIR = os.environ.get("KS_KNOWLEDGE_DIR", "data/knowledge")
INDEX_CACHE = os.environ.get("KS_KB_CACHE", "data/kb_index.pkl")
CHUNK_SIZE = 500  # characters per chunk


@dataclass
class Chunk:
    text: str
    source: str


def _read_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _load_documents(knowledge_dir: str):
    docs = []
    for path in glob.glob(os.path.join(knowledge_dir, "**/*"), recursive=True):
        if path.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                docs.append((path, f.read()))
        elif path.lower().endswith(".pdf"):
            docs.append((path, _read_pdf(path)))
    return docs


def _chunk_text(text: str, source: str, size: int = CHUNK_SIZE):
    chunks = []
    for i in range(0, len(text), size):
        piece = text[i : i + size].strip()
        if len(piece) > 40:  # skip near-empty fragments
            chunks.append(Chunk(text=piece, source=source))
    return chunks


class KnowledgeBase:
    def __init__(self, knowledge_dir: str = KNOWLEDGE_DIR, cache_path: str = INDEX_CACHE):
        self.knowledge_dir = knowledge_dir
        self.cache_path = cache_path
        self.chunks: list[Chunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self._build_or_load()

    def _build_or_load(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
            self.chunks = data["chunks"]
            self.vectorizer = data["vectorizer"]
            self.matrix = data["matrix"]
            print(f"[kb] Loaded cached index: {len(self.chunks)} chunks")
            return

        documents = _load_documents(self.knowledge_dir)
        if not documents:
            print(
                f"[kb] No documents found in {self.knowledge_dir}. "
                f"Add .txt/.pdf agri-advisory files there to enable grounded RAG answers."
            )
            self.chunks = []
            return

        for path, text in documents:
            self.chunks.extend(_chunk_text(text, source=os.path.basename(path)))

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(
                {"chunks": self.chunks, "vectorizer": self.vectorizer, "matrix": self.matrix}, f
            )
        print(f"[kb] Built index: {len(self.chunks)} chunks from {len(documents)} docs")

    def retrieve(self, query: str, top_k: int = 3) -> list[Chunk]:
        if not self.chunks or self.vectorizer is None:
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in top_idx if sims[i] > 0]


if __name__ == "__main__":
    kb = KnowledgeBase()
    for chunk in kb.retrieve("tomato early blight treatment dosage"):
        print(f"--- from {chunk.source} ---")
        print(chunk.text[:200], "...\n")
