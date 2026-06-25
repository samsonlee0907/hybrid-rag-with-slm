from __future__ import annotations

import argparse
import base64
from html import escape
from io import BytesIO
import json
from pathlib import Path
import shutil
import textwrap
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
ASSETS_DIR = NOTEBOOKS_DIR / "assets"
REPORTS_DIR = NOTEBOOKS_DIR / "reports"
LOCAL_REPORT_DIR = REPORTS_DIR / "local_offline_rag"
HYBRID_REPORT_DIR = REPORTS_DIR / "hybrid_rag"

SOURCE_LOCAL_DIR = ASSETS_DIR / "real_local_inference"
SOURCE_CONTEXT_DIR = ASSETS_DIR / "context_lifecycle"
SOURCE_CASE_DIR = ASSETS_DIR / "cv_rag_enriched"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final Local-Offline-RAG and Hybrid-RAG notebooks.")
    parser.add_argument("--local-output", default=str(NOTEBOOKS_DIR / "Local-Offline-RAG.ipynb"))
    parser.add_argument("--hybrid-output", default=str(NOTEBOOKS_DIR / "Hybrid-RAG.ipynb"))
    args = parser.parse_args()

    _prepare_report_folders()
    build_local_offline_notebook(Path(args.local_output))
    build_hybrid_notebook(Path(args.hybrid_output))


def _prepare_report_folders() -> None:
    for path in (LOCAL_REPORT_DIR, HYBRID_REPORT_DIR):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    _copy_report(SOURCE_LOCAL_DIR / "real_local_inference_report.json", LOCAL_REPORT_DIR / "blip_phi4_report.json")
    _copy_report(SOURCE_LOCAL_DIR / "moondream_real_local_inference_report.json", LOCAL_REPORT_DIR / "moondream_phi4_report.json")
    _copy_report(SOURCE_LOCAL_DIR / "heldout_query_images.json", LOCAL_REPORT_DIR / "heldout_query_images.json")
    if (SOURCE_LOCAL_DIR / "moondream_runtime_status.txt").exists():
        _copy_report(SOURCE_LOCAL_DIR / "moondream_runtime_status.txt", LOCAL_REPORT_DIR / "moondream_runtime_status.txt")

    _copy_report(SOURCE_CONTEXT_DIR / "context_lifecycle_report.json", HYBRID_REPORT_DIR / "hybrid_lifecycle_report.json")
    _copy_report(SOURCE_CONTEXT_DIR / "context_lifecycle_summary.json", HYBRID_REPORT_DIR / "hybrid_lifecycle_summary.json")
    if (SOURCE_CONTEXT_DIR / "context_lifecycle_moondream_smoke_report.json").exists():
        _copy_report(SOURCE_CONTEXT_DIR / "context_lifecycle_moondream_smoke_report.json", HYBRID_REPORT_DIR / "moondream_hybrid_smoke_report.json")
    if (SOURCE_CONTEXT_DIR / "context_lifecycle_moondream_smoke_summary.json").exists():
        _copy_report(SOURCE_CONTEXT_DIR / "context_lifecycle_moondream_smoke_summary.json", HYBRID_REPORT_DIR / "moondream_hybrid_smoke_summary.json")


def _copy_report(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Required source report not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_local_offline_notebook(output_path: Path) -> None:
    blip_report = _read_json(LOCAL_REPORT_DIR / "blip_phi4_report.json")
    moondream_report = _read_json(LOCAL_REPORT_DIR / "moondream_phi4_report.json")
    query_manifest = _read_json(LOCAL_REPORT_DIR / "heldout_query_images.json")
    indexed_cases = _read_jsonl(SOURCE_CASE_DIR / "incidents.jsonl")
    offline_cases = [item for item in indexed_cases if item.get("source_scope") == "offline_seed_enriched"]

    blip_queries = blip_report["queries"]
    moondream_queries = moondream_report["queries"]
    decision = _captioner_decision(blip_queries, moondream_queries)

    cells = [
        _markdown(
            "# Local-Offline-RAG\n\n"
            "This notebook only exercises the disconnected path. It uses held-out query photos that are not indexed, "
            "local image captioning, local CLIP image/text/caption embeddings, a local SQLite vector store, and "
            "Phi-4-mini ONNX CPU/mobile for grounded answer drafting.\n\n"
            f"**Downstream selection for Hybrid-RAG:** {decision}"
        ),
        _markdown(
            "## Offline execution path\n\n"
            "```mermaid\n"
            "flowchart LR\n"
            "  Q[Held-out worker photo + question] --> C[BLIP or Moondream local caption]\n"
            "  Q --> E[CLIP image embedding]\n"
            "  C --> T[CLIP text/caption embedding]\n"
            "  E --> F[Fused query vector]\n"
            "  T --> F\n"
            "  F --> S[SQLite local vector store]\n"
            "  S --> R[Top cited historical cases]\n"
            "  R --> P[Evidence prompt]\n"
            "  P --> Phi[Phi-4-mini ONNX CPU/mobile]\n"
            "  Phi --> A[Grounded field response]\n"
            "```\n"
        ),
        _markdown(
            "## Clean report locations\n\n"
            + _markdown_table(
                ["Artifact", "Path"],
                [
                    ["BLIP + Phi-4-mini report", "`notebooks/reports/local_offline_rag/blip_phi4_report.json`"],
                    ["Moondream + Phi-4-mini report", "`notebooks/reports/local_offline_rag/moondream_phi4_report.json`"],
                    ["Held-out query manifest", "`notebooks/reports/local_offline_rag/heldout_query_images.json`"],
                    ["Query image folder", "`notebooks/assets/real_local_inference/query_images/`"],
                    ["Indexed case image folder", "`notebooks/assets/cv_rag_enriched/images/`"],
                ],
            )
        ),
        _markdown(
            "## What is available offline\n\n"
            "The local vector store indexes six enriched seed incidents. The held-out query photos are separate and are not indexed.\n\n"
            + _markdown_table(
                ["ID", "Severity", "Category", "Why it belongs in the offline pack"],
                [
                    [
                        item["incident_id"],
                        item["severity"],
                        item["category"],
                        item.get("offline_cache_reason", ""),
                    ]
                    for item in offline_cases
                ],
            )
        ),
        _html_output_cell(
            "from IPython.display import HTML, display\n"
            "display(HTML(indexed_offline_cases_html))\n",
            _case_gallery_html(offline_cases, title="Indexed offline case pack"),
            execution_count=1,
        ),
        _markdown(
            "## BLIP vs Moondream: semantic and retrieval comparison\n\n"
            "Both captioners were evaluated against the same four held-out field photos. BLIP is the fast operational baseline "
            "on the A10 VM. Moondream2 provides richer semantics, but it did not fit on the 4 GB A10-4Q CUDA profile and used CPU.\n\n"
            + _captioner_comparison_table(blip_queries, moondream_queries)
        ),
        _html_output_cell(
            "from IPython.display import HTML, display\n"
            "display(HTML(offline_query_comparison_html))\n",
            _offline_query_comparison_html(blip_queries, moondream_queries),
            execution_count=2,
        ),
        _markdown(
            "## Phi-4-mini grounded answers\n\n"
            "Phi-4-mini-instruct is text-only: it receives the retrieved case evidence and the local caption as text. "
            "The responses below are generated by Phi-4-mini ONNX CPU/mobile from local evidence only.\n\n"
            + "\n\n".join(_local_answer_block(blip, moon) for blip, moon in zip(blip_queries, moondream_queries, strict=True))
        ),
        _markdown(
            "## Offline conclusion\n\n"
            "- Both BLIP and Moondream achieved top-1 matches for all four held-out query images.\n"
            "- Moondream captions are more semantic and field-readable, especially for water ingress and rebar scenes.\n"
            "- On this A10 VM profile, BLIP is the practical caption choice because Moondream ran on CPU at roughly 220 seconds per image.\n"
            "- For iPhone/mobile, Moondream remains worth evaluating through Core ML / MLX; the A10 CPU fallback is not representative of an Apple-optimized package.\n"
        ),
    ]

    _write_notebook(output_path, cells)
    _write_summary(
        LOCAL_REPORT_DIR / "local_offline_summary.json",
        {
            "notebook": output_path.as_posix(),
            "query_images_indexed": False,
            "indexed_offline_count": len(offline_cases),
            "captioner_decision_for_hybrid": decision,
            "comparison": [_comparison_row(blip, moon) for blip, moon in zip(blip_queries, moondream_queries, strict=True)],
        },
    )


def build_hybrid_notebook(output_path: Path) -> None:
    report = _read_json(HYBRID_REPORT_DIR / "hybrid_lifecycle_report.json")
    summary = _read_json(HYBRID_REPORT_DIR / "hybrid_lifecycle_summary.json")
    moondream_smoke = _read_json_optional(HYBRID_REPORT_DIR / "moondream_hybrid_smoke_report.json")
    selected_captioner = "Moondream"

    cells = [
        _markdown(
            "# Hybrid-RAG\n\n"
            "This notebook demonstrates the connected/disconnected lifecycle. It starts with a deliberately limited offline vector store, "
            "uses search history and Azure AI Search during connectivity resume to identify useful online cases, stages those cases back "
            "into the offline vector store, and compares retrieval across initial offline, enriched offline, and full online search.\n\n"
            f"**Captioner selected for the target iOS architecture:** {selected_captioner}. "
            "This is an explicit iOS/Core ML / MLX assumption, not an A10 VM throughput result. "
            "The A10 run showed BLIP is the fast VM baseline, while Moondream2 produced richer semantic captions but fell back to CPU on the 4 GB A10-4Q profile. "
            "For an iPhone-targeted design, Moondream is selected because the field copilot benefits more from richer visual understanding and VQA-style prompting, pending validation on an optimized iOS runtime."
        ),
        _markdown(
            "## Hybrid execution path\n\n"
            "```mermaid\n"
            "flowchart LR\n"
            "  Q[Worker photo + text query] --> L1[Initial offline SQLite pack]\n"
            "  L1 --> A1[Phi-4-mini answer from limited evidence]\n"
            "  A1 --> H[Search history: misses, low-specificity matches, escalations]\n"
            "  H --> Planner[Phi-4-mini staging planner]\n"
            "  Planner --> Online[Azure AI Search full index]\n"
            "  Online --> Delta[Relevant online-only cases selected]\n"
            "  Delta --> L2[Offline vector store enriched]\n"
            "  L2 --> A2[Later offline answer from richer evidence]\n"
            "  Online --> Full[Full online search comparison]\n"
            "```\n\n"
            "In the current report, the staging decision is reproducible: stage the top Azure AI Search result when it is an online-only case that is not already in the local pack. "
            "This is the deterministic policy view of the same role Phi-4-mini should play in production: read search history, summarize the missing intent, and decide which online evidence is worth caching."
        ),
        _markdown(
            "## iOS target assumptions for Moondream selection\n\n"
            + _markdown_table(
                ["Assumption", "Implication"],
                [
                    [
                        "Target runtime is a recent iPhone with an optimized Core ML / MLX Moondream package.",
                        "Moondream is selected for richer image understanding and VQA-style prompting, not because the A10 VM proved fast Moondream inference.",
                    ],
                    [
                        "The A10 VM has only a 4 GB A10-4Q framebuffer.",
                        "Moondream2 did not fit on CUDA there and ran on CPU, so the VM latency should not be treated as an iPhone latency estimate.",
                    ],
                    [
                        "CLIP/MobileCLIP-style embeddings still perform local vector retrieval.",
                        "Moondream supplies semantic visual context to improve query understanding and answer grounding; it does not replace the vector store.",
                    ],
                    [
                        "Phi-4-mini ONNX remains text-only.",
                        "Phi-4-mini receives retrieved evidence plus Moondream visual context as text and drafts the grounded response.",
                    ],
                ],
            )
        ),
        _markdown(
            "## Clean report locations\n\n"
            + _markdown_table(
                ["Artifact", "Path"],
                [
                    ["Hybrid lifecycle report", "`notebooks/reports/hybrid_rag/hybrid_lifecycle_report.json`"],
                    ["Hybrid summary", "`notebooks/reports/hybrid_rag/hybrid_lifecycle_summary.json`"],
                    ["Moondream smoke report", "`notebooks/reports/hybrid_rag/moondream_hybrid_smoke_report.json`" if moondream_smoke else "Not generated in this build"],
                    ["Notebook", "`notebooks/Hybrid-RAG.ipynb`"],
                    ["Images", "`notebooks/assets/cv_rag_enriched/images/`"],
                ],
            )
        ),
        *_moondream_smoke_cells(moondream_smoke),
        _markdown(
            "## Context states\n\n"
            + _markdown_table(
                ["State", "Available case IDs", "Purpose"],
                [
                    [
                        "Initial offline",
                        ", ".join(report["initial_offline_ids"]),
                        "Small local pack that works for common/high-value cases but misses specialist incidents.",
                    ],
                    [
                        "Enriched offline after sync",
                        ", ".join(report["initial_offline_ids"] + report["synced_ids"]),
                        "Local store after connectivity resumes and selected online-only cases are staged.",
                    ],
                    [
                        "Full online",
                        "Azure AI Search full enriched index",
                        "Broader retrieval surface for all cases, including online-only specialist incidents.",
                    ],
                ],
            )
        ),
        _markdown(
            "## Run 1: initial offline search with limited context\n\n"
            + _markdown_table(
                ["Scenario", "Query", "Initial offline top-3"],
                [
                    [item["scenario"], item["query"], _format_context_hits(item["initial_offline"]["top3"])]
                    for item in report["sequence_steps"]
                ],
            )
        ),
        _markdown(
            "## Connectivity resumes: search-history-driven staging\n\n"
            + _markdown_table(
                ["Scenario", "Missing intent interpreted from search history", "Azure AI Search top-3", "Staged into offline store", "Planner rationale"],
                [
                    [
                        step["scenario"],
                        _missing_intent(step),
                        _format_context_hits(step["online_resume"]["top3"]),
                        ", ".join(step["online_resume"]["synced_ids"]) or "none",
                        _staging_rationale(step),
                    ]
                    for step in report["sequence_steps"]
                ],
            )
        ),
        _markdown(
            "## Run 2: later offline search after delta sync\n\n"
            + _markdown_table(
                ["Scenario", "Initial offline top", "Enriched offline top", "Full online top", "What changed"],
                [
                    [
                        step["scenario"],
                        _top_id(step["initial_offline"]["top3"]),
                        _top_id(step["enriched_offline_after_sync"]["top3"]),
                        _top_id(step["online_resume"]["top3"]),
                        _hybrid_change_note(step),
                    ]
                    for step in report["sequence_steps"]
                ],
            )
        ),
        _html_output_cell(
            "from IPython.display import HTML, display\n"
            "display(HTML(hybrid_flow_html))\n",
            _hybrid_flow_html(report["sequence_steps"]),
            execution_count=1,
        ),
        _markdown(
            "## Run 3: full Azure AI Search capability\n\n"
            + _markdown_table(
                ["Online query", "Top-3 full online results"],
                [[item["query"], _format_context_hits(item["online_top3"])] for item in report["full_online_results"]],
            )
        ),
        _markdown(
            "## Side-by-side comparison across the three runs\n\n"
            + _markdown_table(
                ["Scenario", "Initial offline", "Enriched offline", "Full online during resume", "Synced"],
                [
                    [
                        row["scenario"],
                        row["initial_top"],
                        row["enriched_offline_top"],
                        row["online_top"],
                        ", ".join(row["synced_ids"]) or "none",
                    ]
                    for row in summary["sequence"]
                ],
            )
        ),
        _markdown(
            "## Example responses before and after enrichment\n\n"
            + "\n\n".join(_hybrid_answer_block(step) for step in report["sequence_steps"][1:])
        ),
        _markdown(
            "## Hybrid conclusion\n\n"
            "- Initial offline search is acceptable for already-cached cases, but specialist cases can map to only approximate local evidence.\n"
            "- Connectivity resume uses the full AI Search index to identify online-only evidence that matches search-history intent.\n"
            "- Staging `ONL-007`, `ONL-010`, and `ONL-011` changes later offline top hits from approximate cases to the relevant specialist cases.\n"
            "- Full online search remains broader than the staged local pack; the staged pack should cache only the most useful cases for the site/user history.\n"
        ),
    ]

    _write_notebook(output_path, cells)
    _write_summary(
        HYBRID_REPORT_DIR / "hybrid_demo_summary.json",
        {
            "notebook": output_path.as_posix(),
            "selected_captioner": selected_captioner,
            "selection_assumption": (
                "Moondream is selected for the iOS target architecture under an assumed Core ML / MLX optimized runtime. "
                "The A10 VM result is not an iPhone benchmark; it only shows Moondream CPU fallback semantics and BLIP VM-speed baseline."
            ),
            "initial_offline_ids": report["initial_offline_ids"],
            "synced_ids": report["synced_ids"],
            "comparison": summary["sequence"],
            "moondream_smoke": _moondream_smoke_summary(moondream_smoke),
        },
    )


def _captioner_decision(blip_queries: list[dict[str, Any]], moondream_queries: list[dict[str, Any]]) -> str:
    blip_all_matched = all(item.get("matched_expected") for item in blip_queries)
    moondream_all_matched = all(item.get("matched_expected") for item in moondream_queries)
    blip_avg = sum(item["timings_seconds"]["caption"] for item in blip_queries) / len(blip_queries)
    moondream_avg = sum(item["timings_seconds"]["caption"] for item in moondream_queries) / len(moondream_queries)
    if blip_all_matched and moondream_all_matched and blip_avg < moondream_avg:
        return "Use BLIP for the A10 VM hybrid flow; keep Moondream as the richer semantic comparator/future mobile-optimized candidate."
    if moondream_all_matched:
        return "Use Moondream for richer semantic captions."
    return "Use BLIP as the more stable baseline for this A10 VM flow."


def _comparison_row(blip: dict[str, Any], moon: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": blip["id"],
        "scenario": blip["scenario"],
        "blip_top": blip["hits"][0]["incident_id"],
        "blip_score": blip["hits"][0]["score"],
        "blip_caption_seconds": blip["timings_seconds"]["caption"],
        "moondream_top": moon["hits"][0]["incident_id"],
        "moondream_score": moon["hits"][0]["score"],
        "moondream_caption_seconds": moon["timings_seconds"]["caption"],
        "moondream_caption": moon["query_image_caption"],
    }


def _moondream_smoke_cells(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not report:
        return []
    caption_seconds = [
        step.get("query_image_caption_seconds", 0.0)
        for step in report["sequence_steps"]
        if step.get("query_image_caption_seconds") is not None
    ]
    avg_caption_seconds = sum(caption_seconds) / len(caption_seconds) if caption_seconds else 0.0
    return [
        _markdown(
            "## Moondream 2-scenario hybrid smoke run\n\n"
            "This smaller run validates the Hybrid-RAG flow with Moondream visual context before paying the full CPU-captioning cost. "
            "It uses two online-enrichment scenarios, captions each query image once, appends that local visual caption to the worker's text query, "
            "then compares initial offline, Azure AI Search resume, and later enriched-offline retrieval.\n\n"
            f"**Captioner:** {report.get('caption_model')} ({report.get('captioner')}, revision {report.get('moondream_revision')})  \n"
            f"**Caption device on the A10 VM run:** {report.get('caption_device')}  \n"
            f"**Average caption time:** {avg_caption_seconds:.1f}s/image  \n"
            f"**Answer generator in this smoke run:** {report.get('answer_generator')}\n\n"
            + _markdown_table(
                ["Scenario", "Moondream caption", "Initial offline top", "Online top", "Enriched offline top", "Staged", "Caption time"],
                [
                    [
                        step["scenario"],
                        _compact(step.get("query_image_caption", ""), 220),
                        _top_id(step["initial_offline"]["top3"]),
                        _top_id(step["online_resume"]["top3"]),
                        _top_id(step["enriched_offline_after_sync"]["top3"]),
                        ", ".join(step["online_resume"]["synced_ids"]) or "none",
                        f"{step.get('query_image_caption_seconds', 0.0):.1f}s",
                    ]
                    for step in report["sequence_steps"]
                ],
            )
        ),
        _html_output_cell(
            "from IPython.display import HTML, display\n"
            "display(HTML(moondream_hybrid_smoke_html))\n",
            _hybrid_flow_html(report["sequence_steps"]),
            execution_count=2,
        ),
    ]


def _moondream_smoke_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    return {
        "captioner": report.get("captioner"),
        "caption_model": report.get("caption_model"),
        "caption_device": report.get("caption_device"),
        "scenarios": [
            {
                "scenario": step["scenario"],
                "initial_top": _top_id(step["initial_offline"]["top3"]),
                "online_top": _top_id(step["online_resume"]["top3"]),
                "enriched_offline_top": _top_id(step["enriched_offline_after_sync"]["top3"]),
                "synced_ids": step["online_resume"]["synced_ids"],
                "caption_seconds": step.get("query_image_caption_seconds"),
            }
            for step in report["sequence_steps"]
        ],
    }


def _captioner_comparison_table(blip_queries: list[dict[str, Any]], moondream_queries: list[dict[str, Any]]) -> str:
    return _markdown_table(
        [
            "Scenario",
            "BLIP caption",
            "Moondream caption",
            "BLIP top hit",
            "Moondream top hit",
            "Caption latency",
        ],
        [
            [
                blip["scenario"],
                _compact(blip["query_image_caption"], 160),
                _compact(moon["query_image_caption"], 220),
                f"{blip['hits'][0]['incident_id']} ({blip['hits'][0]['score']:.4f})",
                f"{moon['hits'][0]['incident_id']} ({moon['hits'][0]['score']:.4f})",
                f"BLIP {blip['timings_seconds']['caption']:.3f}s / Moondream {moon['timings_seconds']['caption']:.1f}s",
            ]
            for blip, moon in zip(blip_queries, moondream_queries, strict=True)
        ],
    )


def _offline_query_comparison_html(blip_queries: list[dict[str, Any]], moondream_queries: list[dict[str, Any]]) -> str:
    blocks = ["<div style='font-family:Segoe UI,Arial,sans-serif'>"]
    for blip, moon in zip(blip_queries, moondream_queries, strict=True):
        query_image = REPO_ROOT / blip["query_image"]
        blip_hit = blip["hits"][0]
        moon_hit = moon["hits"][0]
        blocks.extend(
            [
                f"<h2>{escape(blip['scenario'])}</h2>",
                f"<p><b>User question:</b> {escape(blip['query'])}</p>",
                "<div style='display:flex; flex-wrap:wrap; gap:14px; margin-bottom:18px'>",
                _image_card(query_image, "Held-out query photo", "Not indexed"),
                _image_card(REPO_ROOT / blip_hit["image_path"], f"BLIP top: {blip_hit['incident_id']}", f"{blip_hit['title']} | score={blip_hit['score']:.4f}"),
                _image_card(REPO_ROOT / moon_hit["image_path"], f"Moondream top: {moon_hit['incident_id']}", f"{moon_hit['title']} | score={moon_hit['score']:.4f}"),
                "</div>",
                "<table style='border-collapse:collapse; width:100%; margin-bottom:18px'>"
                "<tr><th style='text-align:left;border:1px solid #ddd;padding:6px'>BLIP caption</th>"
                "<th style='text-align:left;border:1px solid #ddd;padding:6px'>Moondream caption</th></tr>"
                f"<tr><td style='border:1px solid #ddd;padding:6px'>{escape(blip['query_image_caption'])}</td>"
                f"<td style='border:1px solid #ddd;padding:6px'>{escape(moon['query_image_caption'])}</td></tr></table>",
            ]
        )
    blocks.append("</div>")
    return "\n".join(blocks)


def _case_gallery_html(cases: list[dict[str, Any]], *, title: str) -> str:
    blocks = [
        "<div style='font-family:Segoe UI,Arial,sans-serif'>",
        f"<h2>{escape(title)}</h2>",
        "<div style='display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:14px'>",
    ]
    for item in cases:
        blocks.append(
            _image_card(
                SOURCE_CASE_DIR / "images" / item["image_file"],
                f"{item['incident_id']} - {item['title']}",
                f"Severity: {item['severity']} | {item.get('category', '')}",
            )
        )
    blocks.append("</div></div>")
    return "\n".join(blocks)


def _hybrid_flow_html(steps: list[dict[str, Any]]) -> str:
    image_dir = SOURCE_CASE_DIR / "images"
    blocks = ["<div style='font-family:Segoe UI,Arial,sans-serif'>"]
    for step in steps:
        initial_top = step["initial_offline"]["top3"][0]
        enriched_top = step["enriched_offline_after_sync"]["top3"][0]
        online_top = step["online_resume"]["top3"][0]
        blocks.extend(
            [
                f"<h2>{escape(step['scenario'])}</h2>",
                f"<p><b>Query:</b> {escape(step['query'])}</p>",
                "<div style='display:flex; flex-wrap:wrap; gap:14px; margin-bottom:18px'>",
                _image_card(image_dir / step["query_image"], "Query image", "Used for image-text retrieval"),
                _hybrid_hit_card(initial_top, "Run 1 initial offline"),
                _hybrid_hit_card(enriched_top, "Run 2 enriched offline"),
                _hybrid_hit_card(online_top, "Run 3 online/resume"),
                "</div>",
            ]
        )
    blocks.append("</div>")
    return "\n".join(blocks)


def _hybrid_hit_card(hit: dict[str, Any], label: str) -> str:
    return (
        "<div style='width:245px; border:1px solid #ddd; border-radius:8px; padding:10px'>"
        f"<div style='font-weight:700'>{escape(label)}</div>"
        f"<div>{escape(hit.get('id', ''))}</div>"
        f"<div>{escape(hit.get('title', ''))}</div>"
        f"<div>severity={escape(hit.get('severity', ''))}</div>"
        f"<div>score={hit.get('score', 0):.4f}</div>"
        f"<div>scope={escape(hit.get('source_scope', ''))}</div>"
        "</div>"
    )


def _image_card(path: Path, title: str, subtitle: str) -> str:
    return (
        "<div style='width:245px; border:1px solid #ddd; border-radius:8px; padding:10px'>"
        f"<img src='{_thumbnail_data_uri(path)}' style='width:100%; height:155px; object-fit:cover; border-radius:4px'>"
        f"<div style='font-weight:700; margin-top:6px'>{escape(title)}</div>"
        f"<div style='font-size:12px'>{escape(subtitle)}</div>"
        "</div>"
    )


def _local_answer_block(blip: dict[str, Any], moon: dict[str, Any]) -> str:
    return (
        f"### {blip['scenario']}\n\n"
        f"**User question:** {blip['query']}\n\n"
        f"**BLIP + Phi-4-mini answer:**\n\n{_quote(_compact(blip['answer'], 1200))}\n\n"
        f"**Moondream + Phi-4-mini answer:**\n\n{_quote(_compact(moon['answer'], 1200))}"
    )


def _hybrid_answer_block(step: dict[str, Any]) -> str:
    return (
        f"### {step['scenario']}\n\n"
        f"**Initial offline answer from limited evidence:**\n\n"
        f"{_quote(_compact(step['initial_offline']['answer'], 900))}\n\n"
        f"**Later offline answer after sync:**\n\n"
        f"{_quote(_compact(step['enriched_offline_after_sync']['answer'], 900))}"
    )


def _missing_intent(step: dict[str, Any]) -> str:
    online_top = step["online_resume"]["top3"][0]
    initial_top = step["initial_offline"]["top3"][0]
    if online_top["id"] == initial_top["id"]:
        return "Local pack already contains the best matching case."
    return f"Search history points to `{online_top['id']}` ({online_top['title']}) rather than the approximate local `{initial_top['id']}` match."


def _staging_rationale(step: dict[str, Any]) -> str:
    synced = step["online_resume"]["synced_ids"]
    if not synced:
        return "Do not stage: top online evidence is already present in the local pack."
    staged = step["online_resume"]["top3"][0]
    return f"Stage `{staged['id']}` because it is online-only, top-ranked, and more specific to the search-history intent."


def _hybrid_change_note(step: dict[str, Any]) -> str:
    initial_top = _top_id(step["initial_offline"]["top3"])
    enriched_top = _top_id(step["enriched_offline_after_sync"]["top3"])
    if initial_top == enriched_top:
        return "No specialist delta required; local pack already had the right case."
    return f"Delta sync changed the offline top hit from `{initial_top}` to `{enriched_top}`."


def _format_context_hits(hits: list[dict[str, Any]]) -> str:
    return "<br>".join(
        f"{hit['rank']}. {hit['id']} - {hit['title']} (`{hit['score']:.4f}`)"
        for hit in hits
    )


def _top_id(hits: list[dict[str, Any]]) -> str:
    return hits[0]["id"] if hits else ""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_optional(path: Path) -> Any:
    if not path.exists():
        return None
    return _read_json(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_summary(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _thumbnail_data_uri(path: Path, max_size: tuple[int, int] = (420, 280)) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Image not found for notebook embedding: {path}")
    with Image.open(path) as image:
        image.thumbnail(max_size)
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=78, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _compact(text: str, limit: int = 900) -> str:
    normalized = "\n".join(line.strip() for line in text.strip().splitlines() if line.strip())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" for line in text.splitlines() if line.strip())


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _markdown_cell(value: Any) -> str:
    return str(value).replace("\n", "<br>").replace("|", "\\|")


def _markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _source_lines(source)}


def _html_output_cell(source: str, html: str, *, execution_count: int) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": [
            {
                "output_type": "display_data",
                "data": {"text/html": html},
                "metadata": {},
            }
        ],
        "source": _source_lines(source),
    }


def _source_lines(source: str) -> list[str]:
    normalized = textwrap.dedent(source).strip("\n")
    return [line + "\n" for line in normalized.splitlines()]


def _write_notebook(path: Path, cells: list[dict[str, Any]]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
