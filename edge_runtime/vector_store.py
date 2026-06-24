from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CaseDocument:
    case_id: str
    site_id: str
    title: str
    problem: str
    resolution: str
    trade: str
    risk_level: str
    tags: list[str]
    source_uri: str
    vector: list[float]

    @property
    def searchable_text(self) -> str:
        return " ".join(
            [
                self.title,
                self.problem,
                self.resolution,
                self.trade,
                self.risk_level,
                " ".join(self.tags),
            ]
        )


@dataclass(frozen=True)
class SearchHit:
    case: CaseDocument
    score: float


class SQLiteVectorStore:
    """Portable SQLite vector store for the prototype.

    Production mobile builds should swap this brute-force JSON-vector
    implementation for SQLCipher plus sqlite-vec or USearch.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def upsert_cases(self, cases: Iterable[CaseDocument]) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.executemany(
                """
                INSERT INTO cases (
                    case_id, site_id, title, problem, resolution, trade,
                    risk_level, tags_json, source_uri, vector_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    site_id=excluded.site_id,
                    title=excluded.title,
                    problem=excluded.problem,
                    resolution=excluded.resolution,
                    trade=excluded.trade,
                    risk_level=excluded.risk_level,
                    tags_json=excluded.tags_json,
                    source_uri=excluded.source_uri,
                    vector_json=excluded.vector_json
                """,
                [
                    (
                        c.case_id,
                        c.site_id,
                        c.title,
                        c.problem,
                        c.resolution,
                        c.trade,
                        c.risk_level,
                        json.dumps(c.tags, ensure_ascii=False),
                        c.source_uri,
                        json.dumps(c.vector),
                    )
                    for c in cases
                ],
            )

    def search(self, query_vector: list[float], *, site_id: str, top_k: int = 4) -> list[SearchHit]:
        rows = self._fetch_site_cases(site_id)
        hits = [
            SearchHit(case=case, score=_cosine_similarity(query_vector, case.vector))
            for case in rows
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]

    def enqueue_outbox(self, event_type: str, payload: dict) -> int:
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute(
                "INSERT INTO outbox(event_type, payload_json) VALUES (?, ?)",
                (event_type, json.dumps(payload, ensure_ascii=False)),
            )
            return int(cur.lastrowid)

    def list_outbox(self, limit: int = 100) -> list[dict]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT id, event_type, payload_json, created_at FROM outbox ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": row[0],
                "event_type": row[1],
                "payload": json.loads(row[2]),
                "created_at": row[3],
            }
            for row in rows
        ]

    def _fetch_site_cases(self, site_id: str) -> list[CaseDocument]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                """
                SELECT case_id, site_id, title, problem, resolution, trade,
                       risk_level, tags_json, source_uri, vector_json
                FROM cases
                WHERE site_id = ?
                """,
                (site_id,),
            ).fetchall()
        return [
            CaseDocument(
                case_id=row[0],
                site_id=row[1],
                title=row[2],
                problem=row[3],
                resolution=row[4],
                trade=row[5],
                risk_level=row[6],
                tags=json.loads(row[7]),
                source_uri=row[8],
                vector=json.loads(row[9]),
            )
            for row in rows
        ]

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    site_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    problem TEXT NOT NULL,
                    resolution TEXT NOT NULL,
                    trade TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    vector_json TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS pack_manifest (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_cases_site_id ON cases(site_id)")


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Vector dimensions differ: {len(left)} != {len(right)}")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)

