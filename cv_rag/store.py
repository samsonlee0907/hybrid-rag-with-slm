from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from cv_rag.synthetic_data import Incident


@dataclass(frozen=True)
class CvHit:
    incident: Incident
    score: float
    image_path: str


class CvVectorStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def upsert(self, incident: Incident, image_path: str, vector: list[float]) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                INSERT INTO cv_incidents (
                    incident_id, payload_json, image_path, vector_json
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(incident_id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    image_path=excluded.image_path,
                    vector_json=excluded.vector_json
                """,
                (
                    incident.incident_id,
                    json.dumps(incident.__dict__, ensure_ascii=False),
                    image_path,
                    json.dumps(vector),
                ),
            )

    def search(self, query_vector: list[float], top_k: int = 3) -> list[CvHit]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute("SELECT payload_json, image_path, vector_json FROM cv_incidents").fetchall()
        hits = [
            CvHit(
                incident=Incident(**json.loads(payload)),
                image_path=image_path,
                score=_cosine(query_vector, json.loads(vector_json)),
            )
            for payload, image_path, vector_json in rows
        ]
        return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as con:
            return int(con.execute("SELECT COUNT(*) FROM cv_incidents").fetchone()[0])

    def _init(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS cv_incidents (
                    incident_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    image_path TEXT NOT NULL,
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

