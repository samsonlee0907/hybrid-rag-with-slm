from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
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

    def upsert(self, payload: dict, vector: list[float], sync_metadata: dict | None = None) -> None:
        sync_metadata = sync_metadata or {}
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                INSERT INTO synced_cases(
                    id, payload_json, vector_json, sync_sequence, content_hash, asset_tier, applied_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    vector_json=excluded.vector_json,
                    sync_sequence=excluded.sync_sequence,
                    content_hash=excluded.content_hash,
                    asset_tier=excluded.asset_tier,
                    applied_at=excluded.applied_at
                """,
                (
                    payload["id"],
                    json.dumps(payload, ensure_ascii=False),
                    json.dumps(vector),
                    int(sync_metadata.get("sync_sequence", payload.get("sync_sequence", 0))),
                    str(sync_metadata.get("content_hash", payload.get("content_hash", ""))),
                    str(sync_metadata.get("asset_tier", sync_metadata.get("selected_asset_tier", "metadata"))),
                    datetime.now(timezone.utc).isoformat(),
                ),
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

    def last_sync_sequence(self) -> int:
        with sqlite3.connect(self.db_path) as con:
            return int(con.execute("SELECT COALESCE(MAX(sync_sequence), 0) FROM synced_cases").fetchone()[0])

    def local_hashes(self) -> dict[str, str]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute("SELECT id, content_hash FROM synced_cases WHERE content_hash != ''").fetchall()
        return {case_id: content_hash for case_id, content_hash in rows}

    def _init(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS synced_cases (
                    id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    sync_sequence INTEGER NOT NULL DEFAULT 0,
                    content_hash TEXT NOT NULL DEFAULT '',
                    asset_tier TEXT NOT NULL DEFAULT 'metadata',
                    applied_at TEXT NOT NULL DEFAULT ''
                )
                """
            )
            _ensure_column(con, "synced_cases", "sync_sequence", "INTEGER NOT NULL DEFAULT 0")
            _ensure_column(con, "synced_cases", "content_hash", "TEXT NOT NULL DEFAULT ''")
            _ensure_column(con, "synced_cases", "asset_tier", "TEXT NOT NULL DEFAULT 'metadata'")
            _ensure_column(con, "synced_cases", "applied_at", "TEXT NOT NULL DEFAULT ''")


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Vector dimensions differ: {len(left)} != {len(right)}")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _ensure_column(con: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
