from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SyncedHit:
    payload: dict
    score: float


class SyncedCaseStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def upsert(self, payload: dict, vector: list[float]) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                INSERT INTO synced_cases(id, payload_json, vector_json)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    vector_json=excluded.vector_json
                """,
                (payload["id"], json.dumps(payload, ensure_ascii=False), json.dumps(vector)),
            )

    def search(self, query_vector: list[float], top_k: int = 3) -> list[SyncedHit]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute("SELECT payload_json, vector_json FROM synced_cases").fetchall()
        hits = [
            SyncedHit(payload=json.loads(payload_json), score=_cosine(query_vector, json.loads(vector_json)))
            for payload_json, vector_json in rows
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as con:
            return int(con.execute("SELECT COUNT(*) FROM synced_cases").fetchone()[0])

    def _init(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS synced_cases (
                    id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    vector_json TEXT NOT NULL
                )
                """
            )


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Vector dimensions differ: {len(left)} != {len(right)}")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)

