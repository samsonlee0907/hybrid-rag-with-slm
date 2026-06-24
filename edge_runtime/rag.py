from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from edge_runtime.embeddings import Embedder
from edge_runtime.phi4_client import Generator
from edge_runtime.vector_store import SQLiteVectorStore, SearchHit


class CloudRetriever(Protocol):
    def retrieve(self, query: str, *, site_id: str, top_k: int) -> list[dict]:
        raise NotImplementedError


@dataclass(frozen=True)
class RagResponse:
    mode: str
    answer: str
    local_hits: list[SearchHit]
    cloud_hits: list[dict]


class HybridRagEngine:
    def __init__(
        self,
        *,
        site_id: str,
        embedder: Embedder,
        vector_store: SQLiteVectorStore,
        generator: Generator,
        cloud_retriever: CloudRetriever | None = None,
    ) -> None:
        self.site_id = site_id
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.cloud_retriever = cloud_retriever

    def answer(self, query: str, *, online: bool = False, top_k: int = 4, max_tokens: int = 512) -> RagResponse:
        query_vector = self.embedder.embed_text(query)
        local_hits = self.vector_store.search(query_vector, site_id=self.site_id, top_k=top_k)
        cloud_hits: list[dict] = []
        mode = "offline"

        if online and self.cloud_retriever is not None:
            cloud_hits = self.cloud_retriever.retrieve(query, site_id=self.site_id, top_k=top_k)
            mode = "hybrid"

        prompt = build_grounded_prompt(query=query, local_hits=local_hits, cloud_hits=cloud_hits)
        answer = self.generator.generate(prompt, max_tokens=max_tokens)
        self.vector_store.enqueue_outbox(
            "query_completed",
            {
                "site_id": self.site_id,
                "query": query,
                "mode": mode,
                "local_case_ids": [hit.case.case_id for hit in local_hits],
                "cloud_case_ids": [hit.get("case_id") for hit in cloud_hits],
            },
        )
        return RagResponse(mode=mode, answer=answer, local_hits=local_hits, cloud_hits=cloud_hits)


def build_grounded_prompt(query: str, local_hits: list[SearchHit], cloud_hits: list[dict]) -> str:
    evidence_lines: list[str] = []
    for index, hit in enumerate(local_hits, start=1):
        case = hit.case
        evidence_lines.append(
            f"[L{index}] {case.case_id} | score={hit.score:.3f} | risk={case.risk_level} | "
            f"title={case.title} | problem={case.problem} | resolution={case.resolution} | source={case.source_uri}"
        )
    for index, hit in enumerate(cloud_hits, start=1):
        evidence_lines.append(
            f"[C{index}] {hit.get('case_id', 'cloud-case')} | score={hit.get('score', 0):.3f} | "
            f"title={hit.get('title', '')} | snippet={hit.get('snippet', '')} | source={hit.get('source_uri', '')}"
        )

    evidence = "\n".join(evidence_lines) if evidence_lines else "No evidence retrieved."
    return f"""SYSTEM:
You are a construction field copilot. Give concise, evidence-grounded guidance.
Do not invent approvals, standards, or facts not present in evidence.
Escalate high-risk structural, safety, or compliance issues to the site engineer.

EVIDENCE:
{evidence}

USER QUESTION:
{query}

RESPONSE REQUIREMENTS:
1. Start with the most likely matching previous case.
2. Recommend practical next actions.
3. Cite evidence IDs like [L1] or [C1].
4. State offline/cloud limitations.
"""
