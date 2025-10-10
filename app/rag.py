import json
import pathlib
import sqlite3
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

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
    """Simple retrieval augmented generation pipeline using TF-IDF retrieval.

    This pipeline can load documents from a JSON corpus file (the original
    behaviour) or from a SQLite database table. To use a DB-backed corpus,
    pass ``db_path`` and ``table`` to the constructor. The DB table is
    expected to expose at least a ``content`` column and a unique ``id``.
    """

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        db_path: Optional[str] = None,
        table: str = "rag_documents",
        top_k: int = 3,
    ) -> None:
        self.top_k = top_k

        if db_path:
            # Load documents from SQLite DB
            db_file = pathlib.Path(db_path)
            if not db_file.exists():
                raise FileNotFoundError(f"RAG DB not found: {db_file}")
            conn = sqlite3.connect(str(db_file))
            conn.row_factory = sqlite3.Row

            # If an FTS table exists, load content from it to improve retrieval
            try:
                fts_check = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rag_fts'").fetchone()
                if fts_check:
                    # Load full rows from rag_documents but use FTS to prefilter
                    # candidate ids (we pull a modest number to vectorise).
                    cur_ids = conn.execute("SELECT id FROM rag_fts WHERE rag_fts MATCH ? LIMIT 100", ("*",))
                    ids = [r[0] for r in cur_ids.fetchall()]
                    if ids:
                        placeholders = ",".join(["?" for _ in ids])
                        cur = conn.execute(f"SELECT * FROM {table} WHERE id IN ({placeholders})", ids)
                    else:
                        cur = conn.execute(f"SELECT * FROM {table}")
                else:
                    cur = conn.execute(f"SELECT * FROM {table}")
            except sqlite3.OperationalError:
                cur = conn.execute(f"SELECT * FROM {table}")

            raw_docs = [dict(r) for r in cur.fetchall()]
            conn.close()
        elif corpus_path:
            corpus_file = pathlib.Path(corpus_path)
            if not corpus_file.exists():
                raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
            with corpus_file.open("r", encoding="utf-8") as fp:
                raw_docs = json.load(fp)
        else:
            raise ValueError("Either corpus_path or db_path must be provided")

        if not isinstance(raw_docs, list) or not raw_docs:
            raise ValueError("Corpus must be a non-empty list of documents")

        # Normalise documents to expected keys
        def _doc_field(d: Dict[str, Any], key: str) -> Any:
            return d.get(key) or d.get(key.lower()) or ""

        self.documents = []
        for d in raw_docs:
            self.documents.append(
                {
                    "id": _doc_field(d, "id"),
                    "title": _doc_field(d, "title"),
                    "summary": _doc_field(d, "summary"),
                    "content": _doc_field(d, "content"),
                    "source_name": _doc_field(d, "source_name"),
                    "source_url": _doc_field(d, "source_url"),
                    "last_updated": _doc_field(d, "last_updated"),
                }
            )

        # Build id -> index lookup for quick candidate mapping
        self.id_to_index: Dict[str, int] = {str(doc.get("id")): i for i, doc in enumerate(self.documents)}

        # Remember DB path and whether an FTS table exists (checked lazily)
        self.db_path = str(db_file) if db_path else None
        self._fts_available = False
        if self.db_path:
            try:
                conn = sqlite3.connect(self.db_path)
                cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rag_fts'")
                self._fts_available = cur.fetchone() is not None
            except Exception:
                self._fts_available = False
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        self.vectorizer = TfidfVectorizer(stop_words="english")
        # Fit TF-IDF on all documents content (used for cosine similarity)
        self.doc_matrix = self.vectorizer.fit_transform([doc["content"] for doc in self.documents])

    def retrieve(self, query: str) -> List[RetrievedDocument]:
        """Retrieve the top-k documents that are most similar to the query."""
        if not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])

        # If we have an FTS table, use it to prefilter candidate ids and
        # compute an FTS-based hit score which we combine with TF-IDF score.
        if self.db_path and self._fts_available:
            try:
                conn = sqlite3.connect(self.db_path)
                cur = conn.execute("SELECT id, offsets(rag_fts) as offs FROM rag_fts WHERE rag_fts MATCH ? LIMIT 200", (query,))
                rows = cur.fetchall()
                conn.close()
            except Exception:
                rows = []

            ids = [str(r[0]) for r in rows]
            # crude hit-count derived from offsets() string (every match adds tokens)
            fts_hits = []
            for r in rows:
                offs = r[1] or ""
                if not offs:
                    fts_hits.append(0.0)
                else:
                    tokens = offs.split()
                    hits = max(0, len(tokens) // 4)
                    fts_hits.append(float(hits))

            # Map ids to indices and compute TF-IDF similarity only on candidates
            candidate_indices = [self.id_to_index.get(i) for i in ids]
            # filter invalid
            valid_pairs = [(idx, hit) for idx, hit in zip(candidate_indices, fts_hits) if idx is not None]
            if valid_pairs:
                indices, hits = zip(*valid_pairs)
                import numpy as _np

                subset = self.doc_matrix[list(indices)]
                tfidf_sim = cosine_similarity(query_vec, subset).flatten()

                # normalize both signals to [0,1]
                def _norm(arr):
                    a = _np.array(arr, dtype=float)
                    if a.size == 0:
                        return a
                    mn = a.min()
                    mx = a.max()
                    if mx <= mn:
                        return _np.zeros_like(a)
                    return (a - mn) / (mx - mn)

                tfidf_n = _norm(tfidf_sim)
                fts_n = _norm(hits)

                alpha = 0.6
                combined = alpha * fts_n + (1.0 - alpha) * tfidf_n

                # pick top_k by combined score
                order = _np.argsort(combined)[::-1][: self.top_k]
                retrieved = []
                for rank_idx in order:
                    doc_idx = indices[rank_idx]
                    doc = self.documents[doc_idx]
                    score = float(combined[rank_idx])
                    retrieved.append(
                        RetrievedDocument(
                            id=doc["id"],
                            title=doc["title"],
                            summary=doc.get("summary", ""),
                            content=doc["content"],
                            source_name=doc.get("source_name", ""),
                            source_url=doc.get("source_url", ""),
                            last_updated=doc.get("last_updated", ""),
                            score=score,
                        )
                    )
                return retrieved

        # Fallback: compute TF-IDF over all documents
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
            line = f"- {doc.summary} (Source: {doc.source_name}, updated {doc.last_updated})"
            bullet_lines.append(line)

            # If we have a DB and the source maps to a CSV table, include a
            # short sample of table rows to provide precise values on demand.
            if getattr(self, "db_path", None):
                table = None
                src = (doc.source_name or "").lower()
                if "glassdoor" in src:
                    table = "glassdoor_salary"
                elif "oews" in src or "oe" in src or "bls" in src:
                    # Prefer OEWS table for occupational rows
                    table = "oews_salary"
                elif "macro" in src or "bls_macro" in src:
                    table = "bls_macro_indicators"

                if table:
                    try:
                        conn = sqlite3.connect(self.db_path)
                        conn.row_factory = sqlite3.Row
                        cursor = conn.execute(f"SELECT * FROM '{table}' LIMIT 3")
                        rows = [dict(r) for r in cursor.fetchall()]
                        conn.close()
                        if rows:
                            # Render a tiny CSV-like snippet
                            headers = list(rows[0].keys())
                            snippet_lines = ["    | " + " | ".join(headers) + " |"]
                            for r in rows:
                                snippet_lines.append("    | " + " | ".join([str(r.get(h, "")) for h in headers]) + " |")
                            bullet_lines.extend(snippet_lines)
                    except Exception:
                        # If DB access fails, silently skip row inclusion
                        pass

        return "\n".join([intro, *bullet_lines])

    def answer(self, query: str) -> Dict[str, Any]:
        documents = self.retrieve(query)
        answer = self.synthesize_answer(query, documents)
        return {
            "answer": answer,
            "sources": [doc.to_source_payload() for doc in documents],
        }


__all__ = ["RagPipeline", "RetrievedDocument"]