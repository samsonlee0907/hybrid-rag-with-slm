from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from online_rag.enriched_data import get_enriched_incidents


SYSTEM_PROMPT = """You are a senior construction QA/QC and safety incident analyst.
Return JSON only. Do not include markdown. Use realistic field language, but do not name any real company or project."""


USER_PROMPT = """Generate 12 synthetic construction incident records for a hybrid offline/online CV-RAG demo.

Return exactly this JSON shape:
{
  "incidents": [
    {
      "id": "INC-001",
      "source_scope": "offline_seed_enriched",
      "title": "...",
      "category": "...",
      "severity": "medium|high|critical",
      "image_caption": "...",
      "visual_clues": ["..."],
      "observation": "...",
      "root_cause_hypothesis": "...",
      "action_checklist": ["...", "...", "...", "..."],
      "escalation_rule": "...",
      "offline_cache_reason": "..."
    }
  ]
}

Use these IDs and scenario anchors:
- INC-001: basement wall water ingress.
- INC-002: concrete column honeycombing.
- INC-003: rebar congestion at beam-column joint.
- INC-004: MEP duct and ceiling coordination clash.
- INC-005: unsafe open edge near scaffold access.
- INC-006: crack near lift core wall.
- ONL-007: water ingress close to temporary electrical riser.
- ONL-008: falling object exclusion-zone breach.
- ONL-009: concrete spalling at slab edge after formwork striking.
- ONL-010: temporary works prop deformation.
- ONL-011: confined-space gas detector alarm during drainage inspection.
- ONL-012: mobile crane lift near overhead service.

The first six records must use source_scope "offline_seed_enriched".
The last six records must use source_scope "online_enriched_only".
Make each checklist practical and safety-aware. Keep the content synthetic and vendor-neutral."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate enriched incident JSON using a Foundry / Azure OpenAI chat deployment.")
    parser.add_argument("--output", default="notebooks/assets/online_comparison/gpt54mini_enriched_incidents.json")
    parser.add_argument("--endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT"))
    parser.add_argument("--deployment", default=os.getenv("AZURE_OPENAI_DEPLOYMENT"))
    parser.add_argument("--api-version", default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"))
    parser.add_argument("--offline", action="store_true", help="Write the deterministic built-in enriched dataset without calling a model.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.offline or not args.endpoint or not args.deployment:
        payload = {
            "generator": "deterministic-fallback",
            "reason": "Model endpoint/deployment was not supplied or --offline was selected.",
            "incidents": [asdict(incident) for incident in get_enriched_incidents()],
        }
    else:
        payload = _generate_with_chat(args.endpoint, args.deployment, args.api_version)
        payload["generator"] = {
            "provider": "azure-ai-foundry",
            "deployment": args.deployment,
            "api_version": args.api_version,
        }

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "incidents": len(payload["incidents"]), "generator": payload["generator"]}, indent=2))


def _generate_with_chat(endpoint: str, deployment: str, api_version: str) -> dict:
    url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "max_completion_tokens": 6000,
    }
    response = httpx.post(url, headers=_auth_headers(), json=body, timeout=120)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    payload = json.loads(_extract_json(content))
    incidents = payload["incidents"]
    if len(incidents) != 12:
        raise ValueError(f"Expected 12 generated incidents, got {len(incidents)}")
    return {"incidents": incidents}


def _auth_headers() -> dict[str, str]:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if api_key:
        return {"api-key": api_key, "Content-Type": "application/json"}

    bearer = os.getenv("AZURE_OPENAI_BEARER_TOKEN") or _get_azure_cli_token()
    return {"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"}


def _get_azure_cli_token() -> str:
    az_cli = shutil.which("az") or shutil.which("az.cmd")
    if not az_cli:
        raise RuntimeError("Azure CLI was not found. Set AZURE_OPENAI_API_KEY or AZURE_OPENAI_BEARER_TOKEN instead.")
    result = subprocess.run(
        [
            az_cli,
            "account",
            "get-access-token",
            "--resource",
            "https://cognitiveservices.azure.com/",
            "--query",
            "accessToken",
            "-o",
            "tsv",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _extract_json(content: str) -> str:
    starts = [index for index in [content.find("{"), content.find("[")] if index >= 0]
    if not starts:
        raise ValueError("Model response did not contain JSON.")
    start = min(starts)
    end = max(content.rfind("}"), content.rfind("]"))
    if end < start:
        raise ValueError("Model response did not contain JSON.")
    return content[start : end + 1]


if __name__ == "__main__":
    main()
