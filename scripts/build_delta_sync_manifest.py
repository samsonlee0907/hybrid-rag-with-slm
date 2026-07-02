from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from online_rag.delta_sync import (
    ASSET_TIER_REVIEW,
    DEFAULT_PROJECT_ID,
    DEFAULT_SITE_ID,
    DeltaCandidate,
    DeviceSyncState,
    build_search_sync_fields,
    plan_delta_manifest,
)
from online_rag.enriched_data import get_enriched_incidents


VECTOR_DIMENSIONS = 512


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a mobile delta-sync manifest from the enriched incident corpus.")
    parser.add_argument("--incidents-json", default="notebooks/assets/online_comparison/gpt54mini_enriched_incidents.json")
    parser.add_argument("--image-manifest", default="notebooks/assets/cv_rag_enriched/image_manifest.json")
    parser.add_argument("--last-sync-sequence", type=int, default=6)
    parser.add_argument("--max-bytes", type=int, default=4_000_000)
    parser.add_argument("--requested-asset-tier", default=ASSET_TIER_REVIEW)
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    parser.add_argument("--site-id", default=DEFAULT_SITE_ID)
    parser.add_argument("--output", default="notebooks/assets/context_lifecycle/delta_sync_manifest_demo.json")
    args = parser.parse_args()

    root = Path.cwd()
    incidents = get_enriched_incidents(args.incidents_json)
    images = _image_lookup(root / args.image_manifest)
    candidates = [
        DeltaCandidate.from_search_document(
            _candidate_document(
                incident=incident,
                sync_sequence=index,
                image_path=images.get(incident.id),
                project_id=args.project_id,
                site_id=args.site_id,
            )
        )
        for index, incident in enumerate(incidents, start=1)
    ]
    manifest = plan_delta_manifest(
        candidates,
        DeviceSyncState(
            last_sync_sequence=args.last_sync_sequence,
            max_bytes=args.max_bytes,
            requested_asset_tier=args.requested_asset_tier,
            project_id=args.project_id,
            site_id=args.site_id,
        ),
    )
    payload = manifest.to_dict()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _candidate_document(*, incident, sync_sequence: int, image_path: Path | None, project_id: str, site_id: str) -> dict:
    vector = _demo_vector(incident.content)
    document = incident.to_search_doc(vector)
    document.update(
        build_search_sync_fields(
            payload=document,
            content_vector=vector,
            sync_sequence=sync_sequence,
            image_path=image_path,
            project_id=project_id,
            site_id=site_id,
            thumb_bytes=_estimate_jpeg_bytes(image_path, max_dimension=512, quality=80),
            review_image_bytes=_estimate_jpeg_bytes(image_path, max_dimension=768, quality=82),
        )
    )
    return document


def _demo_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [round((digest[index % len(digest)] - 128) / 128, 6) for index in range(VECTOR_DIMENSIONS)]


def _image_lookup(manifest_path: Path) -> dict[str, Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    return {item["incident_id"]: base / item["image"] for item in manifest["images"]}


def _estimate_jpeg_bytes(image_path: Path | None, *, max_dimension: int, quality: int) -> int:
    if not image_path or not image_path.exists():
        return 0
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_dimension, max_dimension))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        return len(buffer.getvalue())


if __name__ == "__main__":
    main()
