from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_rag.enriched_dataset import convert_enriched_incidents
from cv_rag.synthetic_data import generate_dataset
from online_rag.enriched_data import get_enriched_incidents


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate notebook image assets from enriched incident records.")
    parser.add_argument("--incidents-json", default="notebooks/assets/online_comparison/gpt54mini_enriched_incidents.json")
    parser.add_argument("--output-dir", default="notebooks/assets/cv_rag_enriched")
    parser.add_argument("--scope", choices=["all", "offline_seed_enriched", "online_enriched_only"], default="all")
    args = parser.parse_args()

    enriched = get_enriched_incidents(args.incidents_json)
    scope = None if args.scope == "all" else args.scope
    incidents = convert_enriched_incidents(enriched, source_scope=scope)
    incidents_path, records = generate_dataset(args.output_dir, incidents)

    manifest = {
        "source": args.incidents_json,
        "output_dir": args.output_dir,
        "incident_count": len(records),
        "incidents_jsonl": incidents_path.as_posix(),
        "images": [
            {
                "incident_id": incident.incident_id,
                "title": incident.title,
                "source_scope": incident.source_scope,
                "image": f"images/{incident.image_file}",
                "caption": incident.image_caption,
            }
            for incident in records
        ],
    }
    manifest_path = Path(args.output_dir) / "image_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "images": len(records)}, indent=2))


if __name__ == "__main__":
    main()
