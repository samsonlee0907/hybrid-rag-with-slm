from __future__ import annotations

import argparse
from io import BytesIO
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_rag.enriched_dataset import convert_enriched_incident
from cv_rag.models import ClipEmbedder
from cv_rag.pipeline import fuse_vectors
from online_rag.azure_search import AzureSearchClient, SearchConfig
from online_rag.delta_sync import build_search_sync_fields
from online_rag.enriched_data import get_enriched_incidents


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the online Azure AI Search enriched incident index.")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--index", default="construction-incidents-online")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--incidents-json", default=None, help="Optional generated enriched incident JSON file.")
    parser.add_argument("--image-source-dir", default="notebooks/assets/cv_rag_enriched/images")
    parser.add_argument("--project-id", default="demo-project")
    parser.add_argument("--site-id", default="demo-site")
    parser.add_argument("--asset-manifest-prefix", default="packs")
    args = parser.parse_args()

    embedder = ClipEmbedder(device=args.device)
    incidents = get_enriched_incidents(args.incidents_json)
    docs = [
        _to_search_doc(
            incident,
            embedder,
            Path(args.image_source_dir),
            sync_sequence=index,
            project_id=args.project_id,
            site_id=args.site_id,
            asset_manifest_prefix=args.asset_manifest_prefix,
        )
        for index, incident in enumerate(incidents, start=1)
    ]

    client = AzureSearchClient(SearchConfig(endpoint=args.endpoint, api_key=args.key, index_name=args.index))
    client.create_or_replace_index()
    client.upload_documents(docs)

    print(json.dumps({"index": args.index, "uploaded": len(docs), "endpoint": args.endpoint}, indent=2))


def _to_search_doc(
    incident,
    embedder: ClipEmbedder,
    image_source_dir: Path,
    *,
    sync_sequence: int,
    project_id: str,
    site_id: str,
    asset_manifest_prefix: str,
) -> dict:
    text_vector = embedder.embed_text(incident.content)
    local_incident = convert_enriched_incident(incident)
    image_path = image_source_dir / local_incident.image_file
    if not image_path.exists():
        vector = text_vector
    else:
        image_vector = embedder.embed_image(str(image_path))
        vector = fuse_vectors(image_vector, text_vector)
    doc = incident.to_search_doc(vector)
    doc.update(
        build_search_sync_fields(
            payload=doc,
            content_vector=vector,
            sync_sequence=sync_sequence,
            image_path=image_path if image_path.exists() else None,
            project_id=project_id,
            site_id=site_id,
            asset_manifest_prefix=asset_manifest_prefix,
            thumb_bytes=_estimate_jpeg_bytes(image_path, max_dimension=512, quality=80),
            review_image_bytes=_estimate_jpeg_bytes(image_path, max_dimension=768, quality=82),
        )
    )
    return doc


def _estimate_jpeg_bytes(image_path: Path, *, max_dimension: int, quality: int) -> int:
    if not image_path.exists():
        return 0
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_dimension, max_dimension))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        return len(buffer.getvalue())


if __name__ == "__main__":
    main()
