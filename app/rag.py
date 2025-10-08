import json
import pathlib
from dataclasses import dataclass
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedDocument:
    """Container that stores retrieved document metadata."""

    id: str
    title: str
    summary: str
    content: str
    source_name: str
    source_url: str
    last_updated: str
    score: float

    def to_source_payload(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "last_updated": self.last_updated,
            "score": round(float(self.score), 3),
        }


class RagPipeline:
    """Simple retrieval augmented generation pipeline using TF-IDF retrieval."""

    def __init__(self, corpus_path: str, top_k: int = 3) -> None:
        self.corpus_path = pathlib.Path(corpus_path)
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        with self.corpus_path.open("r", encoding="utf-8") as fp:
            raw_docs = json.load(fp)

        if not isinstance(raw_docs, list) or not raw_docs:
            raise ValueError("Corpus must be a non-empty list of documents")

        self.documents = raw_docs
        self.top_k = top_k

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform([doc["content"] for doc in self.documents])

    def retrieve(self, query: str) -> List[RetrievedDocument]:
        """Retrieve the top-k documents that are most similar to the query."""
        if not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_matrix).flatten()

        top_indices = similarities.argsort()[::-1][: self.top_k]

        retrieved = []
        for idx in top_indices:
            doc = self.documents[idx]
            retrieved.append(
                RetrievedDocument(
                    id=doc["id"],
                    title=doc["title"],
                    summary=doc.get("summary", ""),
                    content=doc["content"],
                    source_name=doc.get("source_name", ""),
                    source_url=doc.get("source_url", ""),
                    last_updated=doc.get("last_updated", ""),
                    score=float(similarities[idx]),
                )
            )
        return retrieved

    def synthesize_answer(self, query: str, documents: List[RetrievedDocument]) -> str:
        """Generate a concise answer by combining summaries from retrieved documents."""
        if not documents:
            return (
                "I could not find information related to your question in the knowledge base. "
                "Try rephrasing or ask about salaries, skills, hiring trends, or interview preparation."
            )

        intro = "Here is what I found:" if len(documents) > 1 else "Here's the most relevant insight:" 
        bullet_lines = []
        for doc in documents:
            bullet_lines.append(
                f"- {doc.summary} (Source: {doc.source_name}, updated {doc.last_updated})"
            )

        return "\n".join([intro, *bullet_lines])

    def answer(self, query: str) -> Dict[str, Any]:
        documents = self.retrieve(query)
        answer = self.synthesize_answer(query, documents)
        return {
            "answer": answer,
            "sources": [doc.to_source_payload() for doc in documents],
        }


__all__ = ["RagPipeline", "RetrievedDocument"]