from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_rag.enriched_dataset import convert_enriched_incident
from cv_rag.models import ClipEmbedder
from cv_rag.pipeline import fuse_vectors
from online_rag.azure_search import AzureSearchClient, SearchConfig
from online_rag.enriched_data import get_enriched_incidents


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the online Azure AI Search enriched incident index.")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--index", default="construction-incidents-online")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--incidents-json", default=None, help="Optional generated enriched incident JSON file.")
    parser.add_argument("--image-source-dir", default="notebooks/assets/cv_rag_enriched/images")
    args = parser.parse_args()

    embedder = ClipEmbedder(device=args.device)
    incidents = get_enriched_incidents(args.incidents_json)
    docs = [_to_search_doc(incident, embedder, Path(args.image_source_dir)) for incident in incidents]

    client = AzureSearchClient(SearchConfig(endpoint=args.endpoint, api_key=args.key, index_name=args.index))
    client.create_or_replace_index()
    client.upload_documents(docs)

    print(json.dumps({"index": args.index, "uploaded": len(docs), "endpoint": args.endpoint}, indent=2))


def _to_search_doc(incident, embedder: ClipEmbedder, image_source_dir: Path) -> dict:
    text_vector = embedder.embed_text(incident.content)
    local_incident = convert_enriched_incident(incident)
    image_path = image_source_dir / local_incident.image_file
    if not image_path.exists():
        return incident.to_search_doc(text_vector)
    image_vector = embedder.embed_image(str(image_path))
    return incident.to_search_doc(fuse_vectors(image_vector, text_vector))


if __name__ == "__main__":
    main()
