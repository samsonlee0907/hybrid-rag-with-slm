from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_rag.enriched_dataset import convert_enriched_incidents
from cv_rag.synthetic_data import Incident
from cv_rag.synthetic_data import generate_dataset
from online_rag.enriched_data import get_enriched_incidents


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate notebook image assets from enriched incident records.")
    parser.add_argument("--incidents-json", default="notebooks/assets/online_comparison/gpt54mini_enriched_incidents.json")
    parser.add_argument("--output-dir", default="notebooks/assets/cv_rag_enriched")
    parser.add_argument("--scope", choices=["all", "offline_seed_enriched", "online_enriched_only"], default="all")
    parser.add_argument(
        "--mode",
        choices=["auto", "azure-openai", "diagram"],
        default="auto",
        help="Use Azure OpenAI image generation when configured, otherwise draw deterministic fallback diagrams.",
    )
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

    enriched = get_enriched_incidents(args.incidents_json)
    scope = None if args.scope == "all" else args.scope
    incidents = convert_enriched_incidents(enriched, source_scope=scope)
    use_azure = _should_use_azure(args)
    incidents_path, records = generate_dataset(args.output_dir, incidents, render_images=not use_azure)
    if use_azure:
        _generate_azure_images(args, records)

    manifest = {
        "source": args.incidents_json,
        "output_dir": args.output_dir,
        "incident_count": len(records),
        "incidents_jsonl": incidents_path.as_posix(),
        "generation": {
            "mode": "azure-openai" if use_azure else "diagram",
            "endpoint": args.endpoint if use_azure else None,
            "deployment": args.deployment if use_azure else None,
            "api_version": args.api_version if use_azure else None,
            "size": args.size if use_azure else "640x420",
            "quality": args.quality if use_azure else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "prompt_style": "photorealistic construction-site inspection photo" if use_azure else "deterministic PIL diagram",
        },
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
    print(json.dumps({"manifest": str(manifest_path), "images": len(records), "mode": manifest["generation"]["mode"]}, indent=2))


def _should_use_azure(args: argparse.Namespace) -> bool:
    has_auth = bool(args.api_key or args.bearer_token)
    configured = bool(args.endpoint and args.deployment and has_auth)
    if args.mode == "azure-openai" and not configured:
        raise ValueError(
            "Azure OpenAI image generation requires --endpoint, --deployment, and either "
            "--api-key or --bearer-token. Use --mode diagram for fallback diagrams."
        )
    return args.mode == "azure-openai" or (args.mode == "auto" and configured)


def _generate_azure_images(args: argparse.Namespace, incidents: Iterable[Incident]) -> None:
    image_dir = Path(args.output_dir) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for incident in incidents:
        output_path = image_dir / incident.image_file
        if args.skip_existing and output_path.exists():
            continue
        prompt = _build_image_prompt(incident)
        image_bytes = _request_image(
            endpoint=args.endpoint,
            deployment=args.deployment,
            api_version=args.api_version,
            api_key=args.api_key,
            bearer_token=args.bearer_token,
            prompt=prompt,
            size=args.size,
            quality=args.quality,
            max_retries=args.max_retries,
        )
        output_path.write_bytes(image_bytes)
        print(json.dumps({"incident_id": incident.incident_id, "image": str(output_path)}, ensure_ascii=False))


def _request_image(
    *,
    endpoint: str,
    deployment: str,
    api_version: str,
    api_key: str | None,
    bearer_token: str | None,
    prompt: str,
    size: str,
    quality: str,
    max_retries: int,
) -> bytes:
    url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/images/generations?api-version={api_version}"
    body = {
        "prompt": prompt,
        "n": 1,
        "size": size,
        "quality": quality,
        "output_format": "png",
    }
    headers = {"Content-Type": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    elif api_key:
        headers["api-key"] = api_key
    else:
        raise ValueError("Image generation requires an API key or bearer token.")

    payload = json.dumps(body).encode("utf-8")
    for attempt in range(1, max_retries + 1):
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                response_data = json.loads(response.read().decode("utf-8"))
                return base64.b64decode(response_data["data"][0]["b64_json"])
        except urllib.error.HTTPError as exc:
            error_payload = exc.read().decode("utf-8", errors="replace")
            if exc.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                time.sleep(min(45, 5 * attempt))
                continue
            raise RuntimeError(f"Image generation failed with HTTP {exc.code}: {error_payload}") from exc

    raise RuntimeError("Image generation failed after all retry attempts.")


def _build_image_prompt(incident: Incident) -> str:
    clues = "; ".join(incident.visual_clues)
    return f"""Photorealistic documentary construction-site inspection photo for a synthetic incident-management CV-RAG dataset.
Scene: {incident.image_caption}
Visible details to include: {clues}
Field observation: {incident.observation}
Make it look like a realistic smartphone photo taken by a site engineer: real concrete, steel, temporary works, MEP services, dust, natural site lighting, slight handheld perspective.
If workers appear, show them at a distance with standard PPE and non-identifiable faces. Do not show injuries, blood, gore, or active harm.
Do not include readable text, labels, annotations, arrows, icons, UI overlays, logos, watermarks, diagrams, cartoons, 3D render style, or before/after split panels."""


if __name__ == "__main__":
    main()
