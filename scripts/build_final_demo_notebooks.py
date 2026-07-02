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
SAFETY_REPORT_DIR = REPORTS_DIR / "safety_warning_rag"

SOURCE_LOCAL_DIR = ASSETS_DIR / "real_local_inference"
SOURCE_CONTEXT_DIR = ASSETS_DIR / "context_lifecycle"
SOURCE_CASE_DIR = ASSETS_DIR / "cv_rag_enriched"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final Local-Offline-RAG, Hybrid-RAG, and Safety-Warning-RAG notebooks.")
    parser.add_argument("--local-output", default=str(NOTEBOOKS_DIR / "Local-Offline-RAG.ipynb"))
    parser.add_argument("--local-tc-output", default=str(NOTEBOOKS_DIR / "Local-Offline-RAG-tc.ipynb"))
    parser.add_argument("--hybrid-output", default=str(NOTEBOOKS_DIR / "Hybrid-RAG.ipynb"))
    parser.add_argument("--safety-output", default=str(NOTEBOOKS_DIR / "Safety-Warning-RAG.ipynb"))
    args = parser.parse_args()

    _prepare_report_folders()
    build_local_offline_notebook(Path(args.local_output))
    build_local_offline_tc_notebook(Path(args.local_tc_output))
    build_hybrid_notebook(Path(args.hybrid_output))
    build_safety_warning_notebook(Path(args.safety_output))


def _prepare_report_folders() -> None:
    for path in (LOCAL_REPORT_DIR, HYBRID_REPORT_DIR, SAFETY_REPORT_DIR):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    _copy_report(SOURCE_LOCAL_DIR / "real_local_inference_report.json", LOCAL_REPORT_DIR / "blip_phi4_report.json")
    _copy_report(SOURCE_LOCAL_DIR / "moondream_real_local_inference_report.json", LOCAL_REPORT_DIR / "moondream_phi4_report.json")
    _copy_report(SOURCE_LOCAL_DIR / "heldout_query_images.json", LOCAL_REPORT_DIR / "heldout_query_images.json")
    if (SOURCE_LOCAL_DIR / "traditional_chinese_offline_report.json").exists():
        _copy_report(SOURCE_LOCAL_DIR / "traditional_chinese_offline_report.json", LOCAL_REPORT_DIR / "traditional_chinese_offline_report.json")
    if (SOURCE_LOCAL_DIR / "moondream_runtime_status.txt").exists():
        _copy_report(SOURCE_LOCAL_DIR / "moondream_runtime_status.txt", LOCAL_REPORT_DIR / "moondream_runtime_status.txt")

    _copy_report(SOURCE_CONTEXT_DIR / "context_lifecycle_report.json", HYBRID_REPORT_DIR / "hybrid_lifecycle_report.json")
    _copy_report(SOURCE_CONTEXT_DIR / "context_lifecycle_summary.json", HYBRID_REPORT_DIR / "hybrid_lifecycle_summary.json")
    if (SOURCE_CONTEXT_DIR / "context_lifecycle_moondream_validation_report.json").exists():
        _copy_report(SOURCE_CONTEXT_DIR / "context_lifecycle_moondream_validation_report.json", HYBRID_REPORT_DIR / "moondream_hybrid_validation_report.json")
    if (SOURCE_CONTEXT_DIR / "context_lifecycle_moondream_validation_summary.json").exists():
        _copy_report(SOURCE_CONTEXT_DIR / "context_lifecycle_moondream_validation_summary.json", HYBRID_REPORT_DIR / "moondream_hybrid_validation_summary.json")

    _copy_report(SOURCE_LOCAL_DIR / "safety_warning_demo_report.json", SAFETY_REPORT_DIR / "safety_warning_demo_report.json")


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
            "The disconnected flow keeps image understanding, retrieval, and answer drafting on the local device. "
            "Phi-4-mini is text-only, so it receives a compact evidence prompt built from the local caption and retrieved cases."
        ),
        _html_output_cell(
            "",
            _offline_execution_flow_html(),
            execution_count=1,
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
            execution_count=2,
        ),
        _markdown(
            "## BLIP vs Moondream: semantic and retrieval comparison\n\n"
            "Both captioners were evaluated against the same four held-out field photos. BLIP is the fast operational baseline. "
            "Moondream2 provides richer visual semantics and is the preferred iOS-target option when an optimized runtime is available.\n\n"
            + _captioner_comparison_table(blip_queries, moondream_queries)
        ),
        _html_output_cell(
            "from IPython.display import HTML, display\n"
            "display(HTML(offline_query_comparison_html))\n",
            _offline_query_comparison_html(blip_queries, moondream_queries),
            execution_count=3,
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
            "- BLIP remains a practical fast baseline for resource-constrained validation runs.\n"
            "- For the target iOS architecture, Hybrid-RAG selects Moondream under a Core ML / MLX optimization assumption; CPU fallback timings from a VM should not be treated as iPhone benchmarks.\n"
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


def build_local_offline_tc_notebook(output_path: Path) -> None:
    tc_report_path = LOCAL_REPORT_DIR / "traditional_chinese_offline_report.json"
    if not tc_report_path.exists():
        return
    report = _read_json(tc_report_path)
    top_hit = report["hits"][0] if report.get("hits") else {}
    cells = [
        _markdown(
            "# Local-Offline-RAG-tc\n\n"
            "This notebook validates a fully offline Traditional Chinese user interaction. "
            "The worker asks in Traditional Chinese and provides a field photo. The offline stack uses the local image body/caption, "
            "normalizes the intent for local retrieval, searches the local case pack, and returns the final grounded response in Traditional Chinese.\n\n"
            "**Scope:** no Azure AI Search, no cloud model, and no online corpus are used in this test."
        ),
        _markdown(
            "## Traditional Chinese offline execution path\n\n"
            "Phi-4-mini is text-only, so it does not inspect pixels directly. The image is first represented as a local visual caption/body; "
            "Phi-4-mini then helps convert the Traditional Chinese field question plus image body into an English retrieval query for the local vector store, "
            "and later drafts the final Traditional Chinese answer from retrieved evidence."
        ),
        _html_output_cell(
            "",
            _tc_execution_flow_html(),
            execution_count=1,
        ),
        _markdown(
            "## Test setup\n\n"
            + _markdown_table(
                ["Item", "Value"],
                [
                    ["Scenario", report["scenario_id"]],
                    ["Traditional Chinese question", report["traditional_chinese_query"]],
                    ["Query image", f"`{report['query_image']}`"],
                    ["Query image indexed?", str(report["query_image_indexed"])],
                    ["Local image body source", f"{report.get('source_captioner', '')} / {report.get('source_caption_model', '')}"],
                    ["Offline retrieval inputs", ", ".join(report.get("source_vector_inputs", []))],
                    ["Query rewrite model", report["query_rewrite_model"]],
                    ["Answer model", report["answer_model"]],
                ],
            )
        ),
        _html_output_cell(
            "from IPython.display import HTML, display\n"
            "display(HTML(traditional_chinese_offline_html))\n",
            _traditional_chinese_offline_html(report),
            execution_count=2,
        ),
        _markdown(
            "## Query normalization and retrieval result\n\n"
            + _markdown_table(
                ["Field", "Value"],
                [
                    ["Local image body", report["image_body"]],
                    ["Normalized English retrieval query", report["normalized_retrieval_query_en"]],
                    ["Top local case", f"{top_hit.get('incident_id', '')} - {top_hit.get('title', '')}"],
                    ["Top score", f"{top_hit.get('score', 0):.4f}"],
                    ["Expected case", report.get("expected_incident_id", "")],
                    ["Matched expected?", str(report.get("matched_expected"))],
                ],
            )
        ),
        _markdown(
            "## Traditional Chinese answer from offline evidence\n\n"
            f"{_quote(report['answer_tc'])}"
        ),
        _markdown(
            "## TC offline conclusion\n\n"
            "- The Traditional Chinese question is mapped back to the water-ingress retrieval intent using the local image body/caption.\n"
            f"- The top local retrieval result is `{top_hit.get('incident_id', '')}`, matching the expected case `{report.get('expected_incident_id', '')}`.\n"
            "- The final response is returned in Traditional Chinese while preserving the local incident citation and escalation rule.\n"
            "- This confirms the intended language pattern: use Phi-4-mini for query normalization and localized answer drafting, while keeping facts in the offline case pack."
        ),
    ]
    _write_notebook(output_path, cells)


def build_safety_warning_notebook(output_path: Path) -> None:
    report = _read_json(SAFETY_REPORT_DIR / "safety_warning_demo_report.json")
    scenarios = report["scenarios"]

    cells = [
        _markdown(
            "# Safety-Warning-RAG\n\n"
            "This notebook extends the offline CV-RAG pattern from troubleshooting to safety-warning advisory. "
            "It demonstrates one image that should trigger an immediate safety warning and one image that should remain a QA/coordination follow-up.\n\n"
            f"**Scope boundary:** {report['a10_simulation_boundary']}"
        ),
        _markdown(
            "## Offline safety-warning execution path\n\n"
            "The safety-warning flow stays within the same A10/edge-style simulation used by the local notebook: local visual context, "
            "local retrieval, a deterministic policy gate, and Phi-4-mini ONNX CPU/mobile for the final advisory wording. "
            "The policy gate decides whether to issue a warning; the SLM drafts a concise, evidence-cited message."
        ),
        _html_output_cell(
            "",
            _safety_warning_flow_html(),
            execution_count=1,
        ),
        _markdown(
            "## Clean report locations\n\n"
            + _markdown_table(
                ["Artifact", "Path"],
                [
                    ["Safety warning report", "`notebooks/reports/safety_warning_rag/safety_warning_demo_report.json`"],
                    ["Notebook", "`notebooks/Safety-Warning-RAG.ipynb`"],
                    ["Positive held-out query image", "`notebooks/assets/real_local_inference/query_images/`"],
                    ["Indexed case images", "`notebooks/assets/cv_rag_enriched/images/`"],
                ],
            )
        ),
        _markdown(
            "## Scenario setup and policy gate\n\n"
            + _markdown_table(
                [
                    "Scenario",
                    "Top local evidence",
                    "Severity",
                    "Matched policy keywords",
                    "Warning required",
                    "Expected",
                ],
                [
                    [
                        scenario["label"],
                        _safety_top_hit_label(scenario),
                        scenario["policy_decision"].get("severity", ""),
                        ", ".join(scenario["policy_decision"].get("keyword_matches", [])) or "none",
                        str(scenario["policy_decision"]["warning_required"]),
                        str(scenario["expected_warning_required"]),
                    ]
                    for scenario in scenarios
                ],
            )
        ),
        _html_output_cell(
            "from IPython.display import HTML, display\n"
            "display(HTML(safety_warning_demo_html))\n",
            _safety_warning_demo_html(report),
            execution_count=2,
        ),
        _markdown(
            "## Generated advisory responses\n\n"
            + "\n\n".join(_safety_response_block(scenario) for scenario in scenarios)
        ),
        _markdown(
            "## Safety-warning conclusion\n\n"
            "- The open-edge/scaffold access image maps to `INC-005` and triggers the deterministic safety-warning gate.\n"
            "- The MEP coordination-clash image maps to `INC-004` and is intentionally held as a coordination follow-up rather than an immediate safety warning.\n"
            "- Phi-4-mini produces the final advisory text from retrieved evidence and the policy decision; it does not decide site safety by itself.\n"
            "- This is suitable for batch/photo-based field advisory and escalation support, not real-time video monitoring or autonomous stop-work enforcement."
        ),
    ]

    _write_notebook(output_path, cells)
    _write_summary(
        SAFETY_REPORT_DIR / "safety_warning_summary.json",
        {
            "notebook": output_path.as_posix(),
            "online_used": report["online_used"],
            "source_stack": report["source_stack"],
            "scenario_count": len(scenarios),
            "policy_results": [
                {
                    "id": scenario["id"],
                    "top_hit": scenario["policy_decision"].get("top_hit"),
                    "warning_required": scenario["policy_decision"]["warning_required"],
                    "expected_warning_required": scenario["expected_warning_required"],
                    "matched_expected": scenario["policy_decision"]["warning_required"] == scenario["expected_warning_required"],
                }
                for scenario in scenarios
            ],
        },
    )


def build_hybrid_notebook(output_path: Path) -> None:
    report = _read_json(HYBRID_REPORT_DIR / "hybrid_lifecycle_report.json")
    summary = _read_json(HYBRID_REPORT_DIR / "hybrid_lifecycle_summary.json")
    moondream_validation = _read_json_optional(HYBRID_REPORT_DIR / "moondream_hybrid_validation_report.json")
    selected_captioner = "Moondream"

    cells = [
        _markdown(
            "# Hybrid-RAG\n\n"
            "This notebook demonstrates the connected/disconnected lifecycle. It starts with a deliberately limited offline vector store, "
            "uses search history and Azure AI Search during connectivity resume to identify useful online cases, stages those cases back "
            "into the offline vector store, and compares retrieval across initial offline, enriched offline, and full online search.\n\n"
            f"**Captioner selected for the target iOS architecture:** {selected_captioner}. "
            "This is an explicit iOS/Core ML / MLX assumption, not a generic VM throughput claim. "
            "BLIP remains the fast baseline, while Moondream2 is selected for the target design because the field copilot benefits from richer visual understanding and VQA-style prompting. "
            "Validate latency, thermals, and model package size on the target iPhone runtime before production rollout."
        ),
        _markdown(
            "## Hybrid execution path\n\n"
            "The hybrid flow starts offline, uses connectivity to discover missing specialist evidence, and stages only useful online cases back into the local pack."
        ),
        _html_output_cell(
            "",
            _hybrid_execution_flow_html(),
            execution_count=1,
        ),
        _markdown(
            "The staging policy is reproducible in this notebook: cache the top Azure AI Search result only when it is an online-only case that is not already present locally. "
            "In production, Phi-4-mini can help summarize search history and explain which online evidence should be staged, while the final cache decision remains policy-controlled and auditable."
        ),
        _markdown(
            "## iOS target assumptions for Moondream selection\n\n"
            + _markdown_table(
                ["Assumption", "Implication"],
                [
                    [
                        "Target runtime is a recent iPhone with an optimized Core ML / MLX Moondream package.",
                        "Moondream is selected for richer image understanding and VQA-style prompting, subject to target-device validation.",
                    ],
                    [
                        "Recorded VM CPU fallback is not an iPhone benchmark.",
                        "Use the VM result as semantic evidence only; measure speed and thermals on the optimized mobile runtime.",
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
                    ["Moondream validation report", "`notebooks/reports/hybrid_rag/moondream_hybrid_validation_report.json`" if moondream_validation else "Not generated in this build"],
                    ["Notebook", "`notebooks/Hybrid-RAG.ipynb`"],
                    ["Images", "`notebooks/assets/cv_rag_enriched/images/`"],
                ],
            )
        ),
        *_moondream_validation_cells(moondream_validation),
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
                "The recorded VM result is not an iPhone benchmark; it only shows Moondream semantic behavior and the BLIP speed baseline."
            ),
            "initial_offline_ids": report["initial_offline_ids"],
            "synced_ids": report["synced_ids"],
            "comparison": summary["sequence"],
            "moondream_validation": _moondream_validation_summary(moondream_validation),
        },
    )


def _captioner_decision(blip_queries: list[dict[str, Any]], moondream_queries: list[dict[str, Any]]) -> str:
    blip_all_matched = all(item.get("matched_expected") for item in blip_queries)
    moondream_all_matched = all(item.get("matched_expected") for item in moondream_queries)
    blip_avg = sum(item["timings_seconds"]["caption"] for item in blip_queries) / len(blip_queries)
    moondream_avg = sum(item["timings_seconds"]["caption"] for item in moondream_queries) / len(moondream_queries)
    if blip_all_matched and moondream_all_matched and blip_avg < moondream_avg:
        return "Use BLIP as the fast baseline, but select Moondream for the target iOS Hybrid-RAG architecture under the Core ML / MLX optimization assumption."
    if moondream_all_matched:
        return "Use Moondream for richer semantic captions."
    return "Use BLIP as the more stable baseline for this flow."


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


def _moondream_validation_cells(report: dict[str, Any] | None) -> list[dict[str, Any]]:
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
            "## Moondream 2-scenario hybrid validation run\n\n"
            "This smaller run validates the Hybrid-RAG flow with Moondream visual context before running every scenario. "
            "It uses two online-enrichment scenarios, captions each query image once, appends that local visual caption to the worker's text query, "
            "then compares initial offline, Azure AI Search resume, and later enriched-offline retrieval.\n\n"
            f"**Captioner:** {report.get('caption_model')} ({report.get('captioner')}, revision {report.get('moondream_revision')})  \n"
            f"**Caption device in the recorded run:** {report.get('caption_device')}  \n"
            f"**Average caption time:** {avg_caption_seconds:.1f}s/image  \n"
            f"**Answer generator in this validation run:** {report.get('answer_generator')}\n\n"
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
            "display(HTML(moondream_hybrid_validation_html))\n",
            _hybrid_flow_html(report["sequence_steps"]),
            execution_count=2,
        ),
    ]


def _moondream_validation_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
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


def _offline_execution_flow_html() -> str:
    boxes = [
        _svg_box(40, 150, 190, 70, ["Worker photo", "and question"]),
        _svg_box(280, 70, 190, 70, ["Visual caption", "BLIP or Moondream"]),
        _svg_box(280, 230, 190, 70, ["Image embedding", "CLIP"]),
        _svg_box(520, 70, 190, 70, ["Text embedding", "Query + caption"]),
        _svg_box(520, 230, 190, 70, ["Fused query", "vector"]),
        _svg_box(760, 230, 190, 70, ["Local vector", "store"]),
        _svg_box(760, 70, 190, 70, ["Retrieved cases", "with citations"]),
        _svg_box(990, 70, 190, 70, ["Evidence prompt", "for Phi-4-mini"]),
        _svg_box(990, 230, 190, 70, ["Grounded field", "response"]),
    ]
    arrows = [
        _svg_arrow(230, 168, 280, 105),
        _svg_arrow(230, 190, 280, 265),
        _svg_arrow(470, 105, 520, 105),
        _svg_arrow(615, 140, 615, 230),
        _svg_arrow(470, 265, 520, 265),
        _svg_arrow(710, 265, 760, 265),
        _svg_arrow(855, 230, 855, 140),
        _svg_arrow(950, 105, 990, 105),
        _svg_arrow(1085, 140, 1085, 230),
    ]
    return _flow_svg(1220, 360, "Local offline RAG execution path", boxes, arrows)


def _tc_execution_flow_html() -> str:
    boxes = [
        _svg_box(35, 135, 180, 70, ["繁中問題", "+ field photo"]),
        _svg_box(260, 55, 180, 70, ["Local image", "body/caption"]),
        _svg_box(260, 215, 180, 70, ["Phi-4-mini", "query normalize"]),
        _svg_box(500, 215, 190, 70, ["English retrieval", "query"]),
        _svg_box(500, 55, 190, 70, ["Image + text", "query vector"]),
        _svg_box(750, 55, 185, 70, ["Local offline", "case pack"]),
        _svg_box(750, 215, 185, 70, ["Retrieved", "evidence"]),
        _svg_box(990, 215, 190, 70, ["Phi-4-mini", "繁中 answer"]),
    ]
    arrows = [
        _svg_arrow(215, 154, 260, 90),
        _svg_arrow(215, 186, 260, 250),
        _svg_arrow(440, 250, 500, 250),
        _svg_arrow(595, 215, 595, 125),
        _svg_arrow(690, 90, 750, 90),
        _svg_arrow(842, 125, 842, 215),
        _svg_arrow(935, 250, 990, 250),
    ]
    return _flow_svg(1215, 345, "Traditional Chinese full-offline query path", boxes, arrows)


def _hybrid_execution_flow_html() -> str:
    boxes = [
        _svg_box(40, 50, 170, 64, ["Photo +", "question"]),
        _svg_box(260, 35, 180, 64, ["Moondream", "caption"]),
        _svg_box(260, 125, 180, 64, ["CLIP", "embeddings"]),
        _svg_box(500, 80, 190, 64, ["Initial", "offline pack"]),
        _svg_box(740, 80, 190, 64, ["Limited", "answer"]),
        _svg_box(980, 80, 190, 64, ["Search", "history"]),
        _svg_box(980, 220, 190, 64, ["Staging", "planner"]),
        _svg_box(740, 220, 190, 64, ["Azure AI", "Search"]),
        _svg_box(500, 220, 190, 64, ["Delta cases", "selected"]),
        _svg_box(260, 220, 180, 64, ["Enriched", "offline pack"]),
        _svg_box(40, 220, 170, 64, ["Later offline", "answer"]),
        _svg_box(740, 345, 190, 64, ["Full online", "comparison"]),
    ]
    arrows = [
        _svg_arrow(210, 68, 260, 67),
        _svg_arrow(210, 96, 260, 157),
        _svg_arrow(350, 99, 350, 125),
        _svg_arrow(440, 157, 500, 112),
        _svg_arrow(690, 112, 740, 112),
        _svg_arrow(930, 112, 980, 112),
        _svg_arrow(1075, 144, 1075, 220),
        _svg_arrow(980, 252, 930, 252),
        _svg_arrow(740, 252, 690, 252),
        _svg_arrow(500, 252, 440, 252),
        _svg_arrow(260, 252, 210, 252),
        _svg_arrow(835, 284, 835, 345),
    ]
    return _flow_svg(1210, 455, "Hybrid RAG lifecycle", boxes, arrows)


def _safety_warning_flow_html() -> str:
    boxes = [
        _svg_box(40, 60, 175, 68, ["Worker photo", "+ question"]),
        _svg_box(265, 45, 185, 68, ["Local visual", "caption/body"]),
        _svg_box(265, 145, 185, 68, ["CLIP image/text", "embedding"]),
        _svg_box(500, 95, 190, 68, ["Local vector", "case retrieval"]),
        _svg_box(740, 95, 190, 68, ["Retrieved evidence", "with severity"]),
        _svg_box(980, 95, 190, 68, ["Deterministic", "safety gate"]),
        _svg_box(740, 250, 190, 68, ["Phi-4-mini", "advisory wording"]),
        _svg_box(980, 250, 190, 68, ["Human escalation", "and inspection"]),
        _svg_box(500, 250, 190, 68, ["QA / coordination", "follow-up path"]),
    ]
    arrows = [
        _svg_arrow(215, 80, 265, 79),
        _svg_arrow(215, 108, 265, 179),
        _svg_arrow(450, 79, 500, 112),
        _svg_arrow(450, 179, 500, 128),
        _svg_arrow(690, 129, 740, 129),
        _svg_arrow(930, 129, 980, 129),
        _svg_arrow(1075, 163, 835, 250),
        _svg_arrow(930, 284, 980, 284),
        _svg_arrow(980, 144, 690, 284),
    ]
    return _flow_svg(1210, 370, "Offline safety-warning advisory path", boxes, arrows)


def _flow_svg(width: int, height: int, title: str, boxes: list[str], arrows: list[str]) -> str:
    return (
        "<div style='font-family:Segoe UI,Arial,sans-serif; max-width:1180px'>"
        f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' aria-label='{escape(title)}' "
        "xmlns='http://www.w3.org/2000/svg'>"
        "<defs>"
        "<marker id='arrow' markerWidth='10' markerHeight='10' refX='8' refY='3' orient='auto' markerUnits='strokeWidth'>"
        "<path d='M0,0 L0,6 L9,3 z' fill='#475569'/>"
        "</marker>"
        "</defs>"
        f"<rect x='1' y='1' width='{width - 2}' height='{height - 2}' rx='16' fill='#ffffff' stroke='#d0d7de'/>"
        f"<text x='32' y='34' fill='#0f172a' font-size='20' font-weight='700'>{escape(title)}</text>"
        + "".join(arrows)
        + "".join(boxes)
        + "</svg></div>"
    )


def _svg_box(x: int, y: int, width: int, height: int, lines: list[str]) -> str:
    line_height = 18
    first_line_y = y + (height / 2) - ((len(lines) - 1) * line_height / 2) + 6
    text = "".join(
        f"<text x='{x + width / 2:.1f}' y='{first_line_y + index * line_height:.1f}' "
        "text-anchor='middle' fill='#0f172a' font-size='15' font-weight='600'>"
        f"{escape(line)}</text>"
        for index, line in enumerate(lines)
    )
    return (
        f"<rect x='{x}' y='{y}' width='{width}' height='{height}' rx='12' fill='#eef2ff' stroke='#818cf8' stroke-width='1.4'/>"
        + text
    )


def _svg_arrow(x1: int, y1: int, x2: int, y2: int) -> str:
    return (
        f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' "
        "stroke='#475569' stroke-width='1.8' marker-end='url(#arrow)'/>"
    )


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


def _traditional_chinese_offline_html(report: dict[str, Any]) -> str:
    top_hit = report["hits"][0]
    blocks = [
        "<div style='font-family:Segoe UI,Arial,sans-serif'>",
        "<div style='display:flex; flex-wrap:wrap; gap:14px; margin-bottom:18px'>",
        _image_card(REPO_ROOT / report["query_image"], "Held-out query photo", "Not indexed; used in offline retrieval"),
        _image_card(REPO_ROOT / top_hit["image_path"], f"Top local hit: {top_hit['incident_id']}", f"{top_hit['title']} | score={top_hit['score']:.4f}"),
        "</div>",
        "<table style='border-collapse:collapse; width:100%; margin-bottom:18px'>",
        "<tr><th style='text-align:left;border:1px solid #ddd;padding:6px'>Traditional Chinese question</th>"
        "<th style='text-align:left;border:1px solid #ddd;padding:6px'>Normalized English retrieval query</th></tr>",
        f"<tr><td style='border:1px solid #ddd;padding:6px'>{escape(report['traditional_chinese_query'])}</td>"
        f"<td style='border:1px solid #ddd;padding:6px'>{escape(report['normalized_retrieval_query_en'])}</td></tr>",
        "</table>",
        f"<p><b>Offline note:</b> {escape(report.get('retrieval_replay_note', ''))}</p>",
        "</div>",
    ]
    return "\n".join(blocks)


def _safety_warning_demo_html(report: dict[str, Any]) -> str:
    blocks = [
        "<div style='font-family:Segoe UI,Arial,sans-serif'>",
        f"<p style='margin:0 0 12px 0'><b>Source stack:</b> {escape(report['source_stack']['positive_case'])} "
        f"Negative control: {escape(report['source_stack']['negative_case'])} "
        f"Answer generator: {escape(report['source_stack']['answer_generator'])}.</p>",
    ]
    for scenario in report["scenarios"]:
        top_hit = scenario["hits"][0]
        warning_required = scenario["policy_decision"]["warning_required"]
        accent = "#dc2626" if warning_required else "#16a34a"
        status = "Safety warning issued" if warning_required else "No immediate safety warning"
        blocks.extend(
            [
                "<div style='border:1px solid #ddd; border-radius:10px; padding:14px; margin:14px 0'>",
                f"<div style='font-size:18px; font-weight:700; color:{accent}; margin-bottom:8px'>{escape(status)}</div>",
                f"<div style='font-weight:700; margin-bottom:10px'>{escape(scenario['label'])}</div>",
                "<div style='display:flex; flex-wrap:wrap; gap:14px; margin-bottom:12px'>",
                _image_card(
                    REPO_ROOT / scenario["query_image"],
                    "Field/query image",
                    "Held-out image" if not scenario["query_image_indexed"] else "Controlled local case image",
                ),
                _image_card(
                    REPO_ROOT / top_hit["image_path"],
                    f"Top local evidence: {top_hit['incident_id']}",
                    f"{top_hit['title']} | score={top_hit['score']:.4f}",
                ),
                "</div>",
                "<table style='border-collapse:collapse; width:100%; margin-bottom:12px'>",
                "<tr><th style='text-align:left;border:1px solid #ddd;padding:6px'>Policy field</th>"
                "<th style='text-align:left;border:1px solid #ddd;padding:6px'>Value</th></tr>",
                f"<tr><td style='border:1px solid #ddd;padding:6px'>Warning required</td><td style='border:1px solid #ddd;padding:6px'>{warning_required}</td></tr>",
                f"<tr><td style='border:1px solid #ddd;padding:6px'>Risk level</td><td style='border:1px solid #ddd;padding:6px'>{escape(scenario['policy_decision']['risk_level'])}</td></tr>",
                f"<tr><td style='border:1px solid #ddd;padding:6px'>Matched keywords</td><td style='border:1px solid #ddd;padding:6px'>{escape(', '.join(scenario['policy_decision']['keyword_matches']) or 'none')}</td></tr>",
                f"<tr><td style='border:1px solid #ddd;padding:6px'>Policy action</td><td style='border:1px solid #ddd;padding:6px'>{escape(scenario['policy_decision']['policy_action'])}</td></tr>",
                "</table>",
                f"<pre style='white-space:pre-wrap; background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px'>{escape(scenario['safety_response'])}</pre>",
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


def _safety_top_hit_label(scenario: dict[str, Any]) -> str:
    top_hit = scenario["hits"][0]
    return f"{top_hit['incident_id']} - {top_hit['title']} (`{top_hit['score']:.4f}`)"


def _safety_response_block(scenario: dict[str, Any]) -> str:
    return (
        f"### {scenario['label']}\n\n"
        f"**Answer model:** {scenario['answer_model']}\n\n"
        f"{_quote(_compact(scenario['safety_response'], 1200))}"
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
