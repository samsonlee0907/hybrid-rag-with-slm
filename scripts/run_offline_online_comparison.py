from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_rag.models import ClipEmbedder
from cv_rag.pipeline import build_cv_index, search_cv_index
from online_rag.azure_search import AzureSearchClient, SearchConfig
from online_rag.enriched_data import get_enriched_incidents
from online_rag.sync_store import SyncedCaseStore


QUERIES = [
    {
        "scenario": "Known offline concrete defect",
        "query": "A site photo shows concrete honeycombing and exposed aggregate after column formwork removal.",
        "offline_expected": "INC-002",
        "online_expected": "INC-002",
        "reason": "Both paths know the case, but online adds root cause and richer checklist.",
    },
    {
        "scenario": "Water ingress upgraded by electrical risk",
        "query": "Water seepage is close to a temporary electrical riser in the basement after heavy rain.",
        "offline_expected": "INC-001",
        "online_expected": "ONL-007",
        "reason": "Offline has generic water ingress; online has an enriched safety-critical variant.",
    },
    {
        "scenario": "Temporary works case missing offline",
        "query": "A temporary works prop is bowed and the base plate appears displaced after slab loading.",
        "offline_expected": None,
        "online_expected": "ONL-010",
        "reason": "Offline seed pack has no temporary works cases; online retrieval fills the gap.",
    },
    {
        "scenario": "Confined space safety case missing offline",
        "query": "A gas detector alarms during manhole drainage inspection before confined space entry.",
        "offline_expected": None,
        "online_expected": "ONL-011",
        "reason": "Online enriched corpus contains confined-space incident management guidance.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare offline CV-RAG, online enriched RAG, and synced offline delta pack.")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--index", default="construction-incidents-online")
    parser.add_argument("--workspace", default="data\\hybrid-comparison")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    offline_db = str(workspace / "offline_seed.sqlite")
    delta_db = str(workspace / "offline_delta.sqlite")
    Path(offline_db).unlink(missing_ok=True)
    Path(delta_db).unlink(missing_ok=True)

    build_cv_index(str(workspace / "offline"), offline_db, device=args.device)
    embedder = ClipEmbedder(device=args.device)
    search_client = AzureSearchClient(SearchConfig(args.endpoint, args.key, args.index))
    enriched_by_id = {incident.id: incident for incident in get_enriched_incidents()}
    enriched_vectors = {incident.id: embedder.embed_text(incident.content) for incident in enriched_by_id.values()}
    delta_store = SyncedCaseStore(delta_db)

    comparisons = []
    for item in QUERIES:
        query_vector = embedder.embed_text(item["query"])
        offline_hits = search_cv_index(item["query"], offline_db, device=args.device, top_k=3)
        online_hits = search_client.search(item["query"], query_vector, top=3)

        for hit in online_hits[:2]:
            incident = enriched_by_id[hit["id"]]
            delta_store.upsert(incident.to_search_doc(enriched_vectors[incident.id]), enriched_vectors[incident.id])

        synced_hits = delta_store.search(query_vector, top_k=3)
        comparisons.append(
            {
                **item,
                "offline_top": {
                    "id": offline_hits[0].incident.incident_id,
                    "title": offline_hits[0].incident.title,
                    "score": round(offline_hits[0].score, 4),
                },
                "online_top": _summarize_online(online_hits[0]),
                "online_top3": [_summarize_online(hit) for hit in online_hits],
                "synced_offline_top": _summarize_synced(synced_hits[0]) if synced_hits else None,
            }
        )

    report = {
        "mode": "offline-online-comparison",
        "online_component": "Azure AI Search vector + keyword hybrid index",
        "offline_seed_db": offline_db,
        "offline_delta_db": delta_db,
        "offline_delta_count": delta_store.count(),
        "queries": comparisons,
    }
    output_path = Path(args.output or workspace / "offline_online_comparison.json")
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _summarize_online(hit: dict) -> dict:
    return {
        "id": hit.get("id"),
        "title": hit.get("title"),
        "severity": hit.get("severity"),
        "score": round(hit.get("@search.score", 0.0), 4),
        "source_scope": hit.get("source_scope"),
        "action_checklist": hit.get("action_checklist", []),
        "escalation_rule": hit.get("escalation_rule"),
        "offline_cache_reason": hit.get("offline_cache_reason"),
    }


def _summarize_synced(hit) -> dict:
    payload = hit.payload
    return {
        "id": payload.get("id"),
        "title": payload.get("title"),
        "severity": payload.get("severity"),
        "score": round(hit.score, 4),
        "source_scope": payload.get("source_scope"),
    }


if __name__ == "__main__":
    main()

