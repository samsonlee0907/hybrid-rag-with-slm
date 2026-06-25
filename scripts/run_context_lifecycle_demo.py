from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_rag.enriched_dataset import convert_enriched_incident, convert_enriched_incidents
from cv_rag.models import ClipEmbedder, load_generator
from cv_rag.pipeline import build_prompt, fuse_vectors
from cv_rag.store import CvVectorStore
from cv_rag.synthetic_data import generate_dataset
from online_rag.azure_search import AzureSearchClient, SearchConfig
from online_rag.enriched_data import get_enriched_incidents


INITIAL_OFFLINE_IDS = {"INC-001", "INC-002", "INC-005"}

SEARCH_HISTORY = [
    {
        "scenario": "Known concrete defect already cached",
        "query": "A column face has honeycombing and exposed aggregate after formwork removal.",
        "online_expected": "INC-002",
    },
    {
        "scenario": "Water ingress becomes electrical safety issue",
        "query": "Water seepage is close to a temporary electrical riser in the basement after rain.",
        "online_expected": "ONL-007",
    },
    {
        "scenario": "Temporary works prop not in local pack",
        "query": "A temporary works prop is visibly bowed and the base plate has shifted under slab load.",
        "online_expected": "ONL-010",
    },
    {
        "scenario": "Confined space gas alarm not in local pack",
        "query": "A gas detector alarms before manhole drainage inspection and confined space entry.",
        "online_expected": "ONL-011",
    },
]

FULL_ONLINE_QUERIES = [
    "water seepage near temporary electrical riser",
    "falling object exclusion zone breach below overhead works",
    "concrete spalling at slab edge after formwork striking",
    "temporary works prop deformation under slab load",
    "confined space gas detector alarm before manhole entry",
    "mobile crane boom operating near overhead service line",
    "MEP duct clashes with ceiling service zone",
    "crack near lift core wall opening after concrete placement",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate limited offline context, online resume, delta sync, and later offline search.")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--index", default="construction-incidents-online")
    parser.add_argument("--incidents-json", default="notebooks/assets/online_comparison/gpt54mini_enriched_incidents.json")
    parser.add_argument("--workspace", default="notebooks/assets/context_lifecycle")
    parser.add_argument("--db", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--generator", choices=["template", "phi4"], default="template")
    parser.add_argument("--output", default="notebooks/assets/context_lifecycle/context_lifecycle_report.json")
    parser.add_argument("--summary-output", default="notebooks/assets/context_lifecycle/context_lifecycle_summary.json")
    parser.add_argument("--image-source-dir", default="notebooks/assets/cv_rag_enriched/images")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    db_path = args.db or str(workspace / "limited_plus_delta.sqlite")
    enriched = get_enriched_incidents(args.incidents_json)
    enriched_by_id = {incident.id: incident for incident in enriched}
    all_incidents = convert_enriched_incidents(enriched)
    generate_dataset(str(workspace), all_incidents, render_images=False)
    _copy_image_pack(Path(args.image_source_dir), workspace / "images", all_incidents)

    embedder = ClipEmbedder(device=args.device)
    store = CvVectorStore(db_path)
    store.clear()
    image_dir = workspace / "images"
    initial_incidents = [incident for incident in all_incidents if incident.incident_id in INITIAL_OFFLINE_IDS]
    _index_incidents(store, embedder, image_dir, initial_incidents)

    search_client = AzureSearchClient(SearchConfig(args.endpoint, args.key, args.index))
    generator = load_generator(args.generator, device=args.device)

    initial_results = []
    initial_hits_by_query = {}
    for item in SEARCH_HISTORY:
        offline_hits = store.search(embedder.embed_text(item["query"]), top_k=3)
        initial_hits_by_query[item["query"]] = offline_hits
        initial_results.append(
            {
                **item,
                "offline_top3": [_summarize_local_hit(hit, rank) for rank, hit in enumerate(offline_hits, start=1)],
            }
        )

    online_resume_results = []
    synced_ids: list[str] = []
    for item in SEARCH_HISTORY:
        online_hits = search_client.search(item["query"], embedder.embed_text(item["query"]), top=3)
        selected_ids = _select_delta_ids(online_hits)
        for incident_id in selected_ids:
            if incident_id in synced_ids:
                continue
            incident = convert_enriched_incident(enriched_by_id[incident_id])
            _index_incidents(store, embedder, image_dir, [incident])
            synced_ids.append(incident_id)
        online_resume_results.append(
            {
                **item,
                "online_top3": [_summarize_online_hit(hit, rank) for rank, hit in enumerate(online_hits, start=1)],
                "synced_ids": selected_ids,
            }
        )

    later_offline_results = []
    later_hits_by_query = {}
    for item in SEARCH_HISTORY:
        later_hits = store.search(embedder.embed_text(item["query"]), top_k=3)
        later_hits_by_query[item["query"]] = later_hits
        later_offline_results.append(
            {
                **item,
                "offline_after_sync_top3": [_summarize_local_hit(hit, rank) for rank, hit in enumerate(later_hits, start=1)],
            }
        )

    full_online_results = []
    for query in FULL_ONLINE_QUERIES:
        hits = search_client.search(query, embedder.embed_text(query), top=3)
        full_online_results.append({"query": query, "online_top3": [_summarize_online_hit(hit, rank) for rank, hit in enumerate(hits, start=1)]})

    sequence_steps = []
    for item, initial_result, online_result, later_result in zip(
        SEARCH_HISTORY, initial_results, online_resume_results, later_offline_results, strict=True
    ):
        initial_prompt = build_prompt(item["query"], initial_hits_by_query[item["query"]])
        later_prompt = build_prompt(item["query"], later_hits_by_query[item["query"]])
        sequence_steps.append(
            {
                "scenario": item["scenario"],
                "query": item["query"],
                "initial_offline": {
                    "top3": initial_result["offline_top3"],
                    "phi4_evidence_prompt": initial_prompt,
                    "answer": generator.generate(initial_prompt),
                },
                "enriched_offline_after_sync": {
                    "top3": later_result["offline_after_sync_top3"],
                    "phi4_evidence_prompt": later_prompt,
                    "answer": generator.generate(later_prompt),
                },
                "online_resume": {
                    "top3": online_result["online_top3"],
                    "synced_ids": online_result["synced_ids"],
                },
            }
        )

    phi4_examples = [
        {
            "scenario": item["scenario"],
            "query": item["query"],
            "evidence_prompt": step["enriched_offline_after_sync"]["phi4_evidence_prompt"],
            "answer": step["enriched_offline_after_sync"]["answer"],
        }
        for item, step in zip(SEARCH_HISTORY[1:3], sequence_steps[1:3], strict=True)
    ]

    report = {
        "mode": "context-lifecycle-hybrid-rag",
        "workspace": str(workspace),
        "db_path": db_path,
        "initial_offline_ids": sorted(INITIAL_OFFLINE_IDS),
        "synced_ids": synced_ids,
        "initial_offline_count": len(INITIAL_OFFLINE_IDS),
        "offline_after_sync_count": store.count(),
        "initial_offline_results": initial_results,
        "online_resume_results": online_resume_results,
        "later_offline_results": later_offline_results,
        "full_online_results": full_online_results,
        "sequence_steps": sequence_steps,
        "phi4_role": [
            "Phi-4-mini is not the vector database; it is the local reasoning layer after retrieval.",
            "Before sync it can only answer from the limited local evidence, so it should expose uncertainty and escalation.",
            "After connectivity resumes and relevant online cases are cached, Phi-4-mini can draft richer offline answers from the updated local evidence.",
        ],
        "phi4_examples": phi4_examples,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = _build_summary(report)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "synced_ids": synced_ids, "offline_after_sync_count": store.count()}, indent=2))


def _copy_image_pack(source_dir: Path, image_dir: Path, incidents) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for incident in incidents:
        source_path = source_dir / incident.image_file
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source image for {incident.incident_id}: {source_path}")
        shutil.copyfile(source_path, image_dir / incident.image_file)


def _index_incidents(store: CvVectorStore, embedder: ClipEmbedder, image_dir: Path, incidents) -> None:
    for incident in incidents:
        image_path = image_dir / incident.image_file
        image_vector = embedder.embed_image(str(image_path))
        text_vector = embedder.embed_text(incident.searchable_text)
        store.upsert(incident, str(image_path), fuse_vectors(image_vector, text_vector))


def _select_delta_ids(online_hits: list[dict]) -> list[str]:
    if not online_hits:
        return []
    top_hit = online_hits[0]
    if top_hit.get("source_scope") != "online_enriched_only":
        return []
    return [top_hit["id"]]


def _summarize_local_hit(hit, rank: int) -> dict:
    return {
        "rank": rank,
        "id": hit.incident.incident_id,
        "title": hit.incident.title,
        "severity": hit.incident.severity,
        "source_scope": hit.incident.source_scope,
        "score": round(hit.score, 4),
    }


def _summarize_online_hit(hit: dict, rank: int) -> dict:
    return {
        "rank": rank,
        "id": hit.get("id"),
        "title": hit.get("title"),
        "severity": hit.get("severity"),
        "source_scope": hit.get("source_scope"),
        "score": round(hit.get("@search.score", 0.0), 4),
        "action_checklist": hit.get("action_checklist", []),
        "escalation_rule": hit.get("escalation_rule"),
    }


def _build_summary(report: dict) -> dict:
    return {
        "initial_offline_ids": report["initial_offline_ids"],
        "synced_ids": report["synced_ids"],
        "offline_after_sync_count": report["offline_after_sync_count"],
        "rows": [
            {
                "scenario": initial["scenario"],
                "initial_top": initial["offline_top3"][0]["id"],
                "online_top": online["online_top3"][0]["id"],
                "synced_ids": online["synced_ids"],
                "later_top": later["offline_after_sync_top3"][0]["id"],
            }
            for initial, online, later in zip(
                report["initial_offline_results"],
                report["online_resume_results"],
                report["later_offline_results"],
                strict=True,
            )
        ],
        "full_online": [
            {
                "query": item["query"],
                "top": item["online_top3"][0]["id"],
            }
            for item in report["full_online_results"]
        ],
        "sequence": [
            {
                "scenario": item["scenario"],
                "initial_top": item["initial_offline"]["top3"][0]["id"],
                "enriched_offline_top": item["enriched_offline_after_sync"]["top3"][0]["id"],
                "online_top": item["online_resume"]["top3"][0]["id"],
                "synced_ids": item["online_resume"]["synced_ids"],
            }
            for item in report["sequence_steps"]
        ],
    }


if __name__ == "__main__":
    main()
