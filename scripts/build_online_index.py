from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_rag.models import ClipEmbedder
from online_rag.azure_search import AzureSearchClient, SearchConfig
from online_rag.enriched_data import get_enriched_incidents


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the online Azure AI Search enriched incident index.")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--index", default="construction-incidents-online")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--incidents-json", default=None, help="Optional generated enriched incident JSON file.")
    args = parser.parse_args()

    embedder = ClipEmbedder(device=args.device)
    incidents = get_enriched_incidents(args.incidents_json)
    docs = [incident.to_search_doc(embedder.embed_text(incident.content)) for incident in incidents]

    client = AzureSearchClient(SearchConfig(endpoint=args.endpoint, api_key=args.key, index_name=args.index))
    client.create_or_replace_index()
    client.upload_documents(docs)

    print(json.dumps({"index": args.index, "uploaded": len(docs), "endpoint": args.endpoint}, indent=2))


if __name__ == "__main__":
    main()
