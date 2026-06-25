from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_rag.enriched_dataset import convert_enriched_incidents
from cv_rag.models import load_generator, set_offline_mode
from cv_rag.pipeline import build_cv_index, build_prompt, search_cv_index
from online_rag.enriched_data import get_enriched_incidents


OFFLINE_QUERIES = [
    {
        "scenario": "basement water ingress",
        "query": "A photo shows damp staining and active seepage along a basement retaining wall cold joint.",
        "expected": "INC-001",
    },
    {
        "scenario": "column honeycombing",
        "query": "A column face has honeycombing, exposed aggregate, and rough voids after formwork striking.",
        "expected": "INC-002",
    },
    {
        "scenario": "rebar congestion",
        "query": "Dense rebar at a beam-column joint blocks concrete flow and makes vibrator access difficult.",
        "expected": "INC-003",
    },
    {
        "scenario": "MEP ceiling clash",
        "query": "A duct is installed below the intended ceiling plane and clashes with sprinkler and lighting space.",
        "expected": "INC-004",
    },
    {
        "scenario": "open edge hazard",
        "query": "An unprotected slab edge is beside scaffold access with no guardrail or toe board.",
        "expected": "INC-005",
    },
    {
        "scenario": "lift core crack",
        "query": "There is a crack near the lift core wall opening shortly after concrete placement.",
        "expected": "INC-006",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline CV-RAG evaluation using GPT-enriched incident content.")
    parser.add_argument("--incidents-json", default="notebooks/assets/online_comparison/gpt54mini_enriched_incidents.json")
    parser.add_argument("--workspace", default="notebooks/assets/cv_rag_enriched")
    parser.add_argument("--db", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--generator", choices=["template", "phi4"], default="template")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", default="notebooks/assets/cv_rag_enriched/offline_eval_report.json")
    parser.add_argument("--summary-output", default="notebooks/assets/cv_rag_enriched/offline_eval_report_summary.json")
    parser.add_argument("--query-mode", choices=["text", "image-text"], default="image-text")
    parser.add_argument(
        "--preserve-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use already generated image assets instead of redrawing fallback diagrams.",
    )
    args = parser.parse_args()

    if args.offline:
        set_offline_mode()

    enriched = get_enriched_incidents(args.incidents_json)
    incidents = convert_enriched_incidents(enriched, source_scope="offline_seed_enriched")
    incidents_by_id = {incident.incident_id: incident for incident in incidents}
    db_path = args.db or str(Path(args.workspace) / "offline_cv_rag.sqlite")
    count = build_cv_index(
        args.workspace,
        db_path,
        device=args.device,
        incidents=incidents,
        clean=True,
        render_images=not args.preserve_images,
    )
    generator = load_generator(args.generator, device=args.device)

    query_results = []
    answer_examples = []
    for item in OFFLINE_QUERIES:
        query_image = None
        if args.query_mode == "image-text":
            query_image = str(Path(args.workspace) / "images" / incidents_by_id[item["expected"]].image_file)
        hits = search_cv_index(item["query"], db_path, device=args.device, top_k=3, query_image=query_image)
        prompt = build_prompt(item["query"], hits, query_image=query_image)
        if len(answer_examples) < 2:
            answer_examples.append(
                {
                    "scenario": item["scenario"],
                    "query": item["query"],
                    "query_image": query_image,
                    "phi4_evidence_prompt": prompt,
                    "answer": generator.generate(prompt),
                }
            )
        query_results.append(
            {
                **item,
                "query_image": Path(query_image).name if query_image else None,
                "query_vector_inputs": ["text", "image"] if query_image else ["text"],
                "top_hit": hits[0].incident.incident_id,
                "matched": hits[0].incident.incident_id == item["expected"],
                "hits": [_summarize_hit(hit, rank) for rank, hit in enumerate(hits, start=1)],
            }
        )

    report = {
        "mode": "enriched-offline-cv-rag",
        "workspace": args.workspace,
        "db_path": db_path,
        "indexed_count": count,
        "indexed_incidents": [_summarize_incident(incident) for incident in incidents],
        "queries": query_results,
        "query_mode": args.query_mode,
        "query_embedding_model": "openai/clip-vit-base-patch32",
        "top1_accuracy": sum(item["matched"] for item in query_results) / len(query_results),
        "answer_generator": args.generator,
        "phi4_role": [
            "CLIP image/text vectors retrieve the closest local cases from SQLite.",
            "Phi-4-mini receives only the compact evidence list, so it can draft guidance without storing the case library in context.",
            "The answer must cite incident IDs and preserve escalation rules from retrieved evidence.",
        ],
        "answer_examples": answer_examples,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "indexed_count": report["indexed_count"],
        "top1_accuracy": report["top1_accuracy"],
        "queries": [
            {
                "scenario": item["scenario"],
                "expected": item["expected"],
                "query_image": item["query_image"],
                "query_vector_inputs": item["query_vector_inputs"],
                "top_hit": item["top_hit"],
                "top3": [hit["incident_id"] for hit in item["hits"]],
            }
            for item in report["queries"]
        ],
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "indexed": count, "top1_accuracy": report["top1_accuracy"]}, indent=2))


def _summarize_incident(incident) -> dict:
    return {
        "incident_id": incident.incident_id,
        "title": incident.title,
        "category": incident.category,
        "severity": incident.severity,
        "image": f"images/{incident.image_file}",
        "visual_clues": incident.visual_clues,
        "root_cause_hypothesis": incident.root_cause_hypothesis,
        "recommended_action": incident.recommended_action,
        "escalation": incident.escalation,
    }


def _summarize_hit(hit, rank: int) -> dict:
    return {
        "rank": rank,
        "incident_id": hit.incident.incident_id,
        "title": hit.incident.title,
        "severity": hit.incident.severity,
        "score": round(hit.score, 4),
        "image": Path(hit.image_path).name,
    }


if __name__ == "__main__":
    main()
