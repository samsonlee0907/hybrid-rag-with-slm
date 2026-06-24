from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_rag.models import load_generator, set_offline_mode
from cv_rag.pipeline import build_cv_index, build_prompt, search_cv_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline CV-RAG proof of concept.")
    parser.add_argument("--workspace", default="data\\cv-rag", help="Workspace for synthetic images and local index.")
    parser.add_argument("--db", default=None, help="SQLite vector index path.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Model device.")
    parser.add_argument("--query", default="A site photo shows honeycombing and exposed aggregate after formwork removal. What should we do?")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--generator", choices=["template", "phi4"], default="template")
    parser.add_argument("--offline", action="store_true", help="Use HF/Transformers offline mode after models are cached.")
    parser.add_argument("--skip-build", action="store_true", help="Reuse existing index.")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    db_path = args.db or str(workspace / "cv-rag.sqlite")
    if args.offline:
        set_offline_mode()

    if not args.skip_build:
        count = build_cv_index(str(workspace), db_path, device=args.device)
    else:
        count = -1

    hits = search_cv_index(args.query, db_path, device=args.device, top_k=args.top_k)
    prompt = build_prompt(args.query, hits)
    generator = load_generator(args.generator, device=args.device)
    answer = generator.generate(prompt)

    result = {
        "mode": "offline-cv-rag",
        "device": args.device,
        "workspace": str(workspace),
        "db_path": db_path,
        "index_count": count,
        "query": args.query,
        "hits": [
            {
                "incident_id": hit.incident.incident_id,
                "title": hit.incident.title,
                "severity": hit.incident.severity,
                "score": round(hit.score, 4),
                "image_path": hit.image_path,
            }
            for hit in hits
        ],
        "answer": answer,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

