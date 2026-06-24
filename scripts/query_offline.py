from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_runtime.config import RuntimeConfig
from edge_runtime.embeddings import DevelopmentHashingEmbedder
from edge_runtime.phi4_client import ExtractiveFallbackGenerator, Phi4OnnxGenerator
from edge_runtime.rag import HybridRagEngine
from edge_runtime.vector_store import SQLiteVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an offline hybrid RAG query.")
    parser.add_argument("--db", default=None, help="Local SQLite case-pack path.")
    parser.add_argument("--site-id", default=None, help="Site identifier.")
    parser.add_argument("--query", required=True, help="Field worker question.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of local hits.")
    args = parser.parse_args()

    config = RuntimeConfig.from_env()
    db_path = args.db or config.local_pack_db
    site_id = args.site_id or config.site_id
    top_k = args.top_k or config.offline_top_k

    generator = (
        Phi4OnnxGenerator(config.phi4_onnx_model_dir)
        if config.phi4_onnx_model_dir
        else ExtractiveFallbackGenerator()
    )
    engine = HybridRagEngine(
        site_id=site_id,
        embedder=DevelopmentHashingEmbedder(),
        vector_store=SQLiteVectorStore(db_path),
        generator=generator,
    )

    response = engine.answer(args.query, online=False, top_k=top_k, max_tokens=config.max_answer_tokens)
    print(f"Mode: {response.mode}")
    print(response.answer)
    print("\nCitations:")
    for hit in response.local_hits:
        print(f"- {hit.case.case_id} score={hit.score:.3f} title={hit.case.title} source={hit.case.source_uri}")


if __name__ == "__main__":
    main()
