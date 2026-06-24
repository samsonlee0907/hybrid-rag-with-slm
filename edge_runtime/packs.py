from __future__ import annotations

import json
from pathlib import Path

from edge_runtime.embeddings import Embedder
from edge_runtime.vector_store import CaseDocument, SQLiteVectorStore


def load_cases_jsonl(path: str, embedder: Embedder) -> list[CaseDocument]:
    cases: list[CaseDocument] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            required = {"case_id", "site_id", "title", "problem", "resolution", "trade", "risk_level", "tags", "source_uri"}
            missing = sorted(required - set(raw))
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {', '.join(missing)}")
            text = " ".join(
                [
                    raw["title"],
                    raw["problem"],
                    raw["resolution"],
                    raw["trade"],
                    raw["risk_level"],
                    " ".join(raw["tags"]),
                ]
            )
            cases.append(
                CaseDocument(
                    case_id=raw["case_id"],
                    site_id=raw["site_id"],
                    title=raw["title"],
                    problem=raw["problem"],
                    resolution=raw["resolution"],
                    trade=raw["trade"],
                    risk_level=raw["risk_level"],
                    tags=raw["tags"],
                    source_uri=raw["source_uri"],
                    vector=embedder.embed_text(text),
                )
            )
    return cases


def build_pack(input_jsonl: str, db_path: str, embedder: Embedder) -> int:
    cases = load_cases_jsonl(input_jsonl, embedder)
    store = SQLiteVectorStore(db_path)
    store.upsert_cases(cases)
    return len(cases)

