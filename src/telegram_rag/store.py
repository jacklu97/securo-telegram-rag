"""SQLite persistence + in-memory hybrid (semantic + keyword) index.

The corpus is one Telegram group (thousands of messages, not millions), so
brute-force scoring over a numpy matrix is simpler and faster than running a
vector database. Embeddings are stored as float32 blobs and mirrored into
memory; the mirror refreshes incrementally as the ingester adds rows.

Ranking is hybrid: cosine similarity plus a keyword-overlap bonus. Dense
embeddings alone fail hard on acronyms and exact terms ("SV", "MSI",
bank names) — short chat messages cluster in embedding space and noise
outranks the one message that literally contains the term.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    msg_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    sender TEXT,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    UNIQUE (chat_id, msg_id)
);
"""


@dataclass
class Hit:
    date: str
    sender: str
    text: str
    score: float


class Store:
    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()
        self._matrix: np.ndarray | None = None
        self._rows: list[tuple[str, str, str]] = []  # (date, sender, text)
        self._load()

    def _load(self) -> None:
        cur = self._conn.execute("SELECT date, sender, text, embedding FROM messages ORDER BY id")
        rows, vecs = [], []
        for date, sender, text, blob in cur:
            rows.append((date, sender or "", text))
            vecs.append(np.frombuffer(blob, dtype=np.float32))
        self._rows = rows
        self._matrix = np.vstack(vecs) if vecs else None

    def known_ids(self, chat_id: int) -> set[int]:
        cur = self._conn.execute("SELECT msg_id FROM messages WHERE chat_id = ?", (chat_id,))
        return {r[0] for r in cur}

    def count(self) -> int:
        return len(self._rows)

    def latest_date(self) -> str | None:
        row = self._conn.execute("SELECT MAX(date) FROM messages").fetchone()
        return row[0] if row and row[0] else None

    def add(
        self,
        *,
        chat_id: int,
        msg_id: int,
        date: datetime,
        sender: str,
        text: str,
        embedding: np.ndarray,
    ) -> None:
        vec = np.asarray(embedding, dtype=np.float32)
        iso = date.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO messages (chat_id, msg_id, date, sender, text, embedding)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (chat_id, msg_id, iso, sender, text, vec.tobytes()),
                )
            except sqlite3.IntegrityError:
                return  # already ingested
            self._conn.commit()
            self._rows.append((iso, sender, text))
            self._matrix = (
                vec[None, :] if self._matrix is None else np.vstack([self._matrix, vec])
            )

    def search(
        self,
        query_vec: np.ndarray,
        *,
        limit: int,
        since: str | None = None,
        query_text: str = "",
    ) -> list[Hit]:
        with self._lock:
            if self._matrix is None:
                return []
            matrix, rows = self._matrix, list(self._rows)
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        scores = (matrix @ q) / norms

        terms = _query_terms(query_text)
        if terms:
            bonus = np.fromiter(
                (_keyword_overlap(text, terms) for _, _, text in rows),
                dtype=np.float32,
                count=len(rows),
            )
            # Overlap can add up to ~0.9 for a full exact match — enough for a
            # literal hit to beat the ~0.8 similarity noise floor of short chat
            # messages, without drowning semantic ranking on longer queries.
            scores = scores + bonus

        order = np.argsort(-scores)
        hits: list[Hit] = []
        for idx in order:
            date, sender, text = rows[int(idx)]
            if since and date < since:
                continue
            hits.append(Hit(date=date, sender=sender, text=text, score=float(scores[int(idx)])))
            if len(hits) >= limit:
                break
        return hits


_STOPWORDS = {
    "que", "como", "para", "por", "con", "los", "las", "del", "una", "uno",
    "the", "and", "for", "grupo", "telegram", "mensajes", "sobre",
}


def _query_terms(query_text: str) -> list[str]:
    terms = [t for t in re.findall(r"[\wáéíóúñü]+", query_text.lower()) if len(t) >= 2]
    return [t for t in terms if t not in _STOPWORDS]


def _keyword_overlap(text: str, terms: list[str]) -> float:
    lowered = text.lower()
    matched = 0
    for term in terms:
        if len(term) <= 3:
            # Word-boundary match for short tokens/acronyms so "sv" doesn't
            # fire inside unrelated words.
            if re.search(rf"\b{re.escape(term)}\b", lowered):
                matched += 1
        elif term in lowered:
            matched += 1
    if not matched:
        return 0.0
    return 0.9 * (matched / len(terms))
