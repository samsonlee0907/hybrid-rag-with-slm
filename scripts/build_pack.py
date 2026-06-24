from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_runtime.embeddings import DevelopmentHashingEmbedder
from edge_runtime.packs import build_pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local case pack SQLite database.")
    parser.add_argument("--input", required=True, help="Input JSONL case file.")
    parser.add_argument("--db", required=True, help="Output SQLite database path.")
    args = parser.parse_args()

    count = build_pack(args.input, args.db, DevelopmentHashingEmbedder())
    print(f"Built local case pack at {args.db} with {count} cases.")


if __name__ == "__main__":
    main()
