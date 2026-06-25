from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_rag.enriched_dataset import convert_enriched_incidents
from cv_rag.models import BlipImageCaptioner, Phi4MiniOnnxGenerator, set_offline_mode
from cv_rag.pipeline import build_cv_index, build_prompt, search_cv_index
from online_rag.enriched_data import get_enriched_incidents


DEFAULT_QUERY = "The photo shows active water seepage at a basement wall joint. What previous case is most relevant and what should we do next?"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real local CV-RAG setup: BLIP caption + CLIP search + Phi-4-mini ONNX answer.")
    parser.add_argument("--incidents-json", default="notebooks/assets/online_comparison/gpt54mini_enriched_incidents.json")
    parser.add_argument("--workspace", default="notebooks/assets/cv_rag_enriched")
    parser.add_argument("--db", default=None)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument(
        "--query-image",
        default="notebooks/assets/cv_rag_enriched/images/inc_001_basement_wall_water_ingress_observed_at_cons.png",
    )
    parser.add_argument("--caption-model", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--phi4-onnx-model-dir", required=True)
    parser.add_argument("--phi4-execution-provider", default="cuda", choices=["cuda", "cpu", "follow_config"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=260)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", default="notebooks/assets/real_local_inference/real_local_inference_report.json")
    args = parser.parse_args()

    if args.offline:
        set_offline_mode()

    workspace = Path(args.workspace)
    query_image = Path(args.query_image)
    db_path = args.db or str(workspace / "real_local_cv_rag.sqlite")
    if not query_image.exists():
        raise FileNotFoundError(f"Query image not found: {query_image}")
    if not Path(args.phi4_onnx_model_dir).exists():
        raise FileNotFoundError(f"Phi-4-mini ONNX model directory not found: {args.phi4_onnx_model_dir}")

    enriched = get_enriched_incidents(args.incidents_json)
    incidents = convert_enriched_incidents(enriched, source_scope="offline_seed_enriched")
    indexed_count = build_cv_index(str(workspace), db_path, device=args.device, incidents=incidents, clean=True, render_images=False)

    caption_started = time.perf_counter()
    captioner = BlipImageCaptioner(args.caption_model, device=args.device)
    image_caption = captioner.caption(str(query_image))
    caption_duration = time.perf_counter() - caption_started
    del captioner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    enriched_query = f"{args.query}\nLocal image caption: {image_caption}"
    retrieval_started = time.perf_counter()
    hits = search_cv_index(enriched_query, db_path, device=args.device, top_k=args.top_k, query_image=str(query_image))
    retrieval_duration = time.perf_counter() - retrieval_started
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    prompt = build_prompt(enriched_query, hits, query_image=str(query_image))
    generation_started = time.perf_counter()
    generator = Phi4MiniOnnxGenerator(args.phi4_onnx_model_dir, execution_provider=args.phi4_execution_provider)
    answer = generator.generate(prompt, max_new_tokens=args.max_new_tokens)
    generation_duration = time.perf_counter() - generation_started

    report = {
        "mode": "real-local-cv-rag",
        "indexed_count": indexed_count,
        "query": args.query,
        "query_image": str(query_image),
        "query_image_caption_model": args.caption_model,
        "query_image_caption": image_caption,
        "query_embedding_model": "openai/clip-vit-base-patch32",
        "query_vector_inputs": ["text", "image", "caption"],
        "vector_store": "SQLite brute-force cosine similarity",
        "answer_model": "microsoft/Phi-4-mini-instruct-onnx",
        "answer_model_dir": args.phi4_onnx_model_dir,
        "answer_execution_provider": args.phi4_execution_provider,
        "hits": [_summarize_hit(hit, rank) for rank, hit in enumerate(hits, start=1)],
        "answer": answer,
        "timings_seconds": {
            "caption": round(caption_duration, 3),
            "retrieval": round(retrieval_duration, 3),
            "generation": round(generation_duration, 3),
            "model_decode_loop": round(getattr(generator, "last_duration_seconds", generation_duration), 3),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _summarize_hit(hit, rank: int) -> dict:
    return {
        "rank": rank,
        "incident_id": hit.incident.incident_id,
        "title": hit.incident.title,
        "severity": hit.incident.severity,
        "score": round(hit.score, 4),
        "image_path": hit.image_path,
        "recommended_action": hit.incident.recommended_action,
        "escalation": hit.incident.escalation,
    }


if __name__ == "__main__":
    main()
