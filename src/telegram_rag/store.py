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
        promo_intent: bool | None = None,
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

        if promo_intent is None:
            promo_intent = query_is_promo_intent(query_text)
        if promo_intent:
            scores = scores + np.fromiter(
                (_announcement_score(text) for _, _, text in rows),
                dtype=np.float32,
                count=len(rows),
            )

        terms = _query_terms(query_text)
        if terms:
            # Keyword weight scales with specificity: 1-2 terms means an
            # exact-term lookup (acronym, bank name) where a literal hit must
            # clear the ~0.8 cosine noise floor; generic multi-term queries
            # ("promociones tarjeta de crédito") match half the corpus, so a
            # flat 0.9 would just reward short keyword-dense chatter.
            weight = 0.9 if len(terms) <= 2 else 0.45
            bonus = np.fromiter(
                (_keyword_overlap(text, terms) for _, _, text in rows),
                dtype=np.float32,
                count=len(rows),
            )
            scores = scores + weight * bonus

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


# Signals that a message ANNOUNCES a promotion rather than asks about one.
# Chat corpora are dominated by questions ("¿aplica para TDC?"), which share
# vocabulary with promo queries and would otherwise outrank the actual offers.
_PROMO_QUERY = re.compile(r"promo|ofert|descuent|msi|cashback|bonificaci|benefic", re.I)
_PROMO_SIGNALS = re.compile(
    r"\d+\s*%|\d+\s*msi|meses sin intereses|bonificaci[oó]n|cashback|"
    r"v[aá]lido|vigencia|termina|hasta el \d|c[oó]digo|cup[oó]n",
    re.I,
)


def query_is_promo_intent(query_text: str) -> bool:
    return bool(_PROMO_QUERY.search(query_text))


def _announcement_score(text: str) -> float:
    score = 0.0
    signals = len(_PROMO_SIGNALS.findall(text))
    if signals:
        score += min(0.3, 0.15 * signals)
    if len(text) > 120:  # announcements run long; drive-by questions don't
        score += 0.1
    return score


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
    return matched / len(terms)
