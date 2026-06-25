from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_enriched_images import _request_image


DEFAULT_QUERIES = [
    {
        "id": "qry_001_water_ingress",
        "scenario": "Held-out basement water ingress field photo",
        "expected_incident_id": "INC-001",
        "query": "A new site photo shows active water seepage and damp staining along a basement retaining wall construction joint. What previous case is most relevant and what should we do next?",
        "image_file": "qry_001_water_ingress_heldout.png",
        "prompt": "Photorealistic smartphone photo taken by a site engineer in a basement plantroom corridor. Show active water seepage and damp staining along a concrete retaining wall construction joint, with puddling at the wall base, temporary lighting, dust, and construction materials nearby. The scene should look like a real inspection photo, not a diagram. No readable text, arrows, labels, logos, UI overlays, injuries, or dramatic flooding.",
    },
    {
        "id": "qry_002_column_honeycombing",
        "scenario": "Held-out concrete column honeycombing field photo",
        "expected_incident_id": "INC-002",
        "query": "A newly stripped concrete column face has honeycombing, exposed aggregate, and rough voids near the lower lift. Which previous case matches and what action is required?",
        "image_file": "qry_002_column_honeycombing_heldout.png",
        "prompt": "Photorealistic handheld construction inspection photo of a reinforced concrete column after formwork removal. Show honeycombing, rough voids, exposed aggregate, and uneven concrete texture near the lower lift, with formwork debris and a dusty slab in the background. Realistic lighting and slight smartphone perspective. No labels, annotations, readable text, logos, cartoons, or injuries.",
    },
    {
        "id": "qry_003_scaffold_edge_protection",
        "scenario": "Held-out scaffold edge protection field photo",
        "expected_incident_id": "INC-005",
        "query": "A field photo shows an open slab edge beside scaffold access with incomplete guardrails and a fall exposure. What similar case should we use and what immediate controls are needed?",
        "image_file": "qry_003_scaffold_edge_protection_heldout.png",
        "prompt": "Photorealistic construction-site smartphone photo from a site safety walk. Show an open slab edge beside scaffold access where guardrails or toe boards are incomplete, with temporary barriers nearby, scaffold stairs or access platform visible, and a clear fall hazard. Workers may appear far away in PPE with non-identifiable faces. No readable signs, text, arrows, labels, logos, gore, or active accident.",
    },
    {
        "id": "qry_004_rebar_congestion",
        "scenario": "Held-out rebar congestion field photo",
        "expected_incident_id": "INC-003",
        "query": "The photo shows congested reinforcement at a beam-column joint before concrete pour, with tight bar spacing and limited access for vibration. What previous case applies and what should be checked?",
        "image_file": "qry_004_rebar_congestion_heldout.png",
        "prompt": "Photorealistic close smartphone inspection photo of a beam-column joint before concrete pouring. Show dense congested reinforcement bars, tight spacing, couplers or links, formwork edges, and limited room for concrete placement and vibration. Make it look like a real construction-site quality inspection image with dust and natural site lighting. No diagrams, labels, readable text, arrows, logos, or UI overlays.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate held-out photorealistic query-only images for the real local CV-RAG demo.")
    parser.add_argument("--output-dir", default="notebooks/assets/real_local_inference/query_images")
    parser.add_argument("--manifest", default="notebooks/assets/real_local_inference/heldout_query_images.json")
    parser.add_argument("--endpoint", default=os.getenv("AZURE_IMAGE_ENDPOINT"))
    parser.add_argument("--deployment", default=os.getenv("AZURE_IMAGE_DEPLOYMENT"))
    parser.add_argument("--api-version", default=os.getenv("AZURE_IMAGE_API_VERSION", "2025-04-01-preview"))
    parser.add_argument("--api-key", default=os.getenv("AZURE_IMAGE_API_KEY"))
    parser.add_argument("--bearer-token", default=os.getenv("AZURE_IMAGE_BEARER_TOKEN"))
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--quality", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    if not args.endpoint or not args.deployment or not (args.api_key or args.bearer_token):
        raise ValueError("Provide --endpoint, --deployment, and either --api-key or --bearer-token.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_queries = []
    for query in DEFAULT_QUERIES:
        output_path = output_dir / query["image_file"]
        if not (args.skip_existing and output_path.exists()):
            image_bytes = _request_image(
                endpoint=args.endpoint,
                deployment=args.deployment,
                api_version=args.api_version,
                api_key=args.api_key,
                bearer_token=args.bearer_token,
                prompt=query["prompt"],
                size=args.size,
                quality=args.quality,
                max_retries=args.max_retries,
            )
            output_path.write_bytes(image_bytes)
        manifest_queries.append(
            {
                "id": query["id"],
                "scenario": query["scenario"],
                "expected_incident_id": query["expected_incident_id"],
                "query": query["query"],
                "query_image": output_path.as_posix(),
                "image_generation_prompt": query["prompt"],
            }
        )
        print(json.dumps({"id": query["id"], "image": output_path.as_posix()}, ensure_ascii=False))

    manifest = {
        "purpose": "held-out query-only images for real local CV-RAG retrieval; these images are not indexed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation": {
            "endpoint": args.endpoint,
            "deployment": args.deployment,
            "api_version": args.api_version,
            "size": args.size,
            "quality": args.quality,
            "prompt_style": "photorealistic smartphone construction-site field photo",
        },
        "queries": manifest_queries,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest": manifest_path.as_posix(), "queries": len(manifest_queries)}, indent=2))


if __name__ == "__main__":
    main()
