from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cloud_api.azure_search import AzureHybridRetriever
from cloud_api.settings import AzureSettings


app = FastAPI(title="Hybrid RAG Mobile BFF", version="0.1.0")


class QueryRequest(BaseModel):
    site_id: str = Field(default="demo-site")
    query: str
    online: bool = True
    top_k: int = Field(default=5, ge=1, le=20)


class OutboxEvent(BaseModel):
    device_id: str
    site_id: str
    event_type: str
    payload: dict[str, Any]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
async def query(request: QueryRequest) -> dict[str, Any]:
    if not request.online:
        return {
            "mode": "offline-required",
            "answer": "Device should answer from local case pack; cloud path was not requested.",
            "hits": [],
        }

    try:
        retriever = AzureHybridRetriever(AzureSettings.from_env())
        hits = await retriever.retrieve(request.query, site_id=request.site_id, top_k=request.top_k)
        answer = await retriever.grounded_chat(request.query, hits)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"mode": "cloud", "answer": answer, "hits": hits}


@app.post("/sync/outbox")
async def sync_outbox(events: list[OutboxEvent]) -> dict[str, Any]:
    # Production: validate Entra token/device attestation, write to Service Bus,
    # and persist raw evidence to Blob/Data Lake.
    return {"accepted": len(events), "next_pack_check_seconds": 300}


@app.get("/packs/{site_id}/manifest.json")
async def pack_manifest(site_id: str) -> dict[str, Any]:
    # Production: return signed TUF metadata and SAS-protected delta pack URLs.
    return {
        "site_id": site_id,
        "version": "dev-0",
        "signature": "not-signed-in-starter-kit",
        "packs": [],
    }
