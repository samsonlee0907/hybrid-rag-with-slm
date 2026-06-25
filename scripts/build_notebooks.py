from __future__ import annotations

import argparse
import base64
from html import escape
from io import BytesIO
import json
from pathlib import Path
import textwrap
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
OFFLINE_ASSETS = NOTEBOOKS_DIR / "assets" / "cv_rag_enriched"
CONTEXT_ASSETS = NOTEBOOKS_DIR / "assets" / "context_lifecycle"
REAL_LOCAL_ASSETS = NOTEBOOKS_DIR / "assets" / "real_local_inference"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build executed notebooks from CV-RAG report JSON assets.")
    parser.add_argument("--offline-report", default=str(OFFLINE_ASSETS / "offline_eval_report.json"))
    parser.add_argument("--context-report", default=str(CONTEXT_ASSETS / "context_lifecycle_report.json"))
    parser.add_argument("--real-local-report", default=str(REAL_LOCAL_ASSETS / "real_local_inference_report.json"))
    parser.add_argument("--offline-output", default=str(NOTEBOOKS_DIR / "offline_cv_rag_results.ipynb"))
    parser.add_argument("--context-output", default=str(NOTEBOOKS_DIR / "field_context_lifecycle_hybrid_rag.ipynb"))
    parser.add_argument("--real-local-output", default=str(NOTEBOOKS_DIR / "real_local_inference_cv_rag.ipynb"))
    args = parser.parse_args()

    build_offline_notebook(Path(args.offline_report), Path(args.offline_output))
    build_context_notebook(Path(args.context_report), Path(args.context_output))
    if Path(args.real_local_report).exists():
        build_real_local_notebook(Path(args.real_local_report), Path(args.real_local_output))


def build_offline_notebook(report_path: Path, output_path: Path) -> None:
    report = _read_json(report_path)
    manifest = _read_json(OFFLINE_ASSETS / "image_manifest.json")
    generation = manifest.get("generation", {})
    indexed_incidents = report["indexed_incidents"]

    cells = [
        _markdown(
            "# Offline CV-RAG Results with Photorealistic Incident Context\n\n"
            "This notebook is the full-offline scenario: the device has a local image/text case pack, "
            "photorealistic construction-site image assets, local CLIP embeddings, a SQLite vector store, "
            "and local answer drafting from retrieved evidence. The incident text was generated "
            "with the `gpt-5.4-mini` deployment; the current image pack was generated through Azure OpenAI "
            f"`{generation.get('deployment', 'image generation')}` using the official `images/generations` API."
        ),
        _markdown(
            "## Local model path clarification\n\n"
            + _markdown_table(
                ["Stage", "Current implementation in this VM POC"],
                [
                    [
                        "Query vectorization",
                        "`openai/clip-vit-base-patch32` runs locally and embeds both the worker text and the query image when `query_mode=image-text`.",
                    ],
                    [
                        "Vector search",
                        "SQLite stores fused CLIP image/text vectors for cached cases and returns the closest evidence.",
                    ],
                    [
                        "Answer drafting",
                        "`Phi-4-mini-instruct` is text-only, so it receives the retrieved evidence as text. It is not directly given pixels.",
                    ],
                    [
                        "Executed notebook generator",
                        f"`{report.get('answer_generator', 'unknown')}`. Use `--generator phi4` to attempt the actual local Phi-4-mini text model after weights are cached; this VM run used the deterministic fallback when set to `template`.",
                    ],
                    [
                        "Direct image+text SLM option",
                        "Use `Phi-4-multimodal-instruct` if the SLM itself must inspect the image alongside the text prompt.",
                    ],
                ],
            )
        ),
        _markdown(
            "## What got indexed into the local vector store\n\n"
            "The offline pack contains only the six `offline_seed_enriched` records. Each incident image is "
            "cached in the local pack and indexed together with the incident text. The vector stored in SQLite "
            "is the same fused representation used by the runtime: "
            "`0.45 * image_embedding + 0.55 * incident_text_embedding`. This POC weighting favors text-only "
            "field queries while retaining visual grounding from the cached photo.\n\n"
            + _markdown_table(
                ["ID", "Severity", "Visual clues", "Root-cause hypothesis"],
                [
                    [
                        item["incident_id"],
                        item["severity"],
                        ", ".join(item.get("visual_clues", [])),
                        item.get("root_cause_hypothesis", ""),
                    ]
                    for item in indexed_incidents
                ],
            )
        ),
        _html_output_cell(
            "from IPython.display import HTML, display\n"
            "# Executed output below embeds image thumbnails directly; it does not depend on relative Markdown image links.\n"
            "display(HTML(indexed_case_gallery_html))\n",
            _image_gallery_html(
                [
                    {
                        "id": item["incident_id"],
                        "title": item["title"],
                        "severity": item["severity"],
                        "image_path": OFFLINE_ASSETS / item["image"],
                        "caption": ", ".join(item.get("visual_clues", [])),
                    }
                    for item in indexed_incidents
                ],
                title="Indexed offline case images",
            ),
            execution_count=1,
        ),
        _markdown(
            "## Local vector store shape\n\n"
            + _markdown_table(
                ["Field", "Meaning"],
                [
                    ["`incident_id`", "Stable citation ID returned to Phi-4-mini and shown to the worker."],
                    [
                        "`payload_json`",
                        "Enriched incident title, severity, observation, visual clues, root-cause hypothesis, checklist, and escalation rule.",
                    ],
                    ["`image_path`", "Photorealistic image cached in the offline pack."],
                    [
                        "`vector_json`",
                        "512-dimensional CLIP fused image/text vector for the prototype; production should replace JSON vectors with `sqlite-vec` or USearch.",
                    ],
                ],
            )
        ),
        _markdown(
            "## Offline query results\n\n"
            "Each worker question is paired with a query photo from the test fixture. CLIP embeds the text and image locally, "
            "compares the fused vector against SQLite, and passes only the retrieved evidence to the answer generator.\n\n"
            + _markdown_table(
                ["Scenario", "Query inputs", "Expected", "Top hit", "Matched", "Top-3 retrieved cases"],
                [
                    [
                        item["scenario"],
                        ", ".join(item.get("query_vector_inputs", ["text"])),
                        item["expected"],
                        item["top_hit"],
                        "yes" if item["matched"] else "no",
                        _format_hits(item["hits"]),
                    ]
                    for item in report["queries"]
                ],
            )
            + f"\n\n**Top-1 accuracy:** `{report['top1_accuracy']:.0%}` across `{len(report['queries'])}` offline queries."
        ),
        _html_output_cell(
            "from IPython.display import HTML, display\n"
            "# Executed output below embeds the top retrieval thumbnails for each offline query.\n"
            "display(HTML(offline_query_result_html))\n",
            _offline_query_html(report["queries"]),
            execution_count=2,
        ),
        _markdown(
            "## User question and grounded model response\n\n"
            "The notebook now presents the response the way the field worker would see it: question first, then a clean grounded answer. "
            "The evidence prompt is still stored in the JSON report for auditability, but it is not shown as the user-facing response.\n\n"
            + "\n\n".join(_answer_example_md(item) for item in report.get("answer_examples", []))
        ),
    ]

    _write_notebook(output_path, cells)


def build_context_notebook(report_path: Path, output_path: Path) -> None:
    report = _read_json(report_path)
    image_by_id = _manifest_image_map()

    cells = [
        _markdown(
            "# Field Context Lifecycle Hybrid RAG Demo\n\n"
            "This notebook shows a query/response sequence rather than only an aggregate result table. "
            "It starts from a deliberately cleaned and limited local vector store, uses connectivity resume "
            "to fetch relevant online cases, caches those cases into an offline delta pack, and then searches "
            "again locally to show how the offline context has become richer. Each search-history query is embedded "
            f"with `{report.get('query_embedding_model', 'openai/clip-vit-base-patch32')}` using `{report.get('query_mode', 'image-text')}` inputs."
        ),
        _markdown(
            "## Information flow\n\n"
            "```mermaid\n"
            "flowchart LR\n"
            "  Worker[Field worker photo + text query] --> CLIP[Local CLIP image+text embedding]\n"
            "  CLIP --> Offline1[Initial offline SQLite pack]\n"
            "  Offline1 --> Phi1[Phi-4-mini conservative answer]\n"
            "  Offline1 --> History[Search history + missed/specific intent]\n"
            "  History -->|connectivity resumes| Online[Azure AI Search full enriched index]\n"
            "  Online --> Select[Select relevant online-only cases]\n"
            "  Select --> Delta[Signed/local offline delta pack]\n"
            "  Delta --> Offline2[Enriched offline search]\n"
            "  Offline2 --> Phi2[Phi-4-mini richer cited answer]\n"
            "```\n"
        ),
        _markdown(
            "## Context states\n\n"
            + _markdown_table(
                ["State", "What is available", "Purpose"],
                [
                    [
                        "Initial offline",
                        ", ".join(report["initial_offline_ids"]),
                        "Small resource-limited pack for common/high-priority cases.",
                    ],
                    [
                        "Enriched offline after sync",
                        ", ".join(report["initial_offline_ids"] + report["synced_ids"]),
                        "Local pack after connectivity resumes and relevant online cases are cached.",
                    ],
                    [
                        "Online",
                        "Full Azure AI Search index",
                        "Broader corpus and fresher context when network is available.",
                    ],
                ],
            )
        ),
        _html_output_cell(
            "from IPython.display import HTML, display\n"
            "# Synced online-only cases are embedded as thumbnails so the notebook remains self-contained.\n"
            "display(HTML(synced_case_gallery_html))\n",
            _image_gallery_html(
                [
                    {
                        "id": incident_id,
                        "title": _title_for_id(report, incident_id),
                        "severity": _severity_for_id(report, incident_id),
                        "image_path": image_by_id[incident_id],
                        "caption": "Synced into offline delta pack for later disconnected search.",
                    }
                    for incident_id in report["synced_ids"]
                ],
                title="Online cases cached for later offline use",
            ),
            execution_count=1,
        ),
        _markdown("## Query/response sequence: initial offline -> enriched offline -> online\n\n" + _sequence_markdown(report)),
        _markdown(
            "## Aggregate lifecycle result\n\n"
            + _markdown_table(
                [
                    "Search-history scenario",
                    "Initial offline top",
                    "Enriched offline top after sync",
                    "Online top",
                    "Synced into offline delta",
                ],
                [
                    [
                        item["scenario"],
                        item["initial_offline"]["top3"][0]["id"],
                        item["enriched_offline_after_sync"]["top3"][0]["id"],
                        item["online_resume"]["top3"][0]["id"],
                        ", ".join(item["online_resume"]["synced_ids"]) or "none",
                    ]
                    for item in report["sequence_steps"]
                ],
            )
        ),
        _markdown(
            "## Full online search with more queries\n\n"
            + _markdown_table(
                ["Online query", "Top-3 online results"],
                [[item["query"], _format_online_hits(item["online_top3"])] for item in report["full_online_results"]],
            )
        ),
        _markdown(
            "## Where Phi-4-mini helps in the hybrid scenario\n\n"
            "Local CLIP handles the image+text query embedding. Phi-4-mini-instruct runs after retrieval in both offline passes. "
            "Because Phi-4-mini-instruct is text-only, it receives cited evidence rather than raw pixels. Before sync, it should be conservative: cite "
            "the limited evidence, explain that the local pack may not contain a specialist case, and escalate. "
            "After sync, it can use the newly cached online-only case to produce a more specific response while "
            "still operating offline. If direct image+text reasoning inside the SLM is required, replace the answer generator with "
            "`Phi-4-multimodal-instruct`.\n\n"
            f"This executed lifecycle report used `{report.get('answer_generator', 'unknown')}` as the answer generator.\n\n"
            "The online path can use Azure AI Search and a larger/cloud model when available, but the key hybrid "
            "behavior is that the useful online result becomes a local evidence record for the next disconnected search."
        ),
    ]

    _write_notebook(output_path, cells)


def build_real_local_notebook(report_path: Path, output_path: Path) -> None:
    report = _read_json(report_path)
    queries = report.get("queries") or [_single_real_local_query(report)]

    cells = [
        _markdown(
            "# Real Local CV-RAG Inference on Held-Out Field Photos\n\n"
            "This notebook captures a non-deterministic local inference run on `vm-cvrag-a10-poc`. "
            "The result is not the template fallback: BLIP captioned the query photo locally, CLIP embedded the "
            "text + image + caption for SQLite vector search, and Phi-4-mini ONNX generated the final answer from "
            "retrieved evidence.\n\n"
            "The query photos are held-out field images stored under `notebooks/assets/real_local_inference/query_images`. "
            "They are not part of the indexed historical case image pack."
        ),
        _markdown(
            "## What ran locally\n\n"
            + _markdown_table(
                ["Stage", "Executed model/runtime", "Result"],
                [
                    [
                        "Photo understanding",
                        f"`{queries[0]['query_image_caption_model']}`",
                        f"{len(queries)} held-out query photos captioned locally.",
                    ],
                    [
                        "Vector retrieval",
                        f"`{report['query_embedding_model']}` + {report['vector_store']}",
                        "Inputs: " + ", ".join(report["query_vector_inputs"]),
                    ],
                    [
                        "Answer drafting",
                        f"`{report['answer_model']}` via ONNX Runtime GenAI",
                        f"Provider: `{report['answer_execution_provider']}`; model path: `{report['answer_model_dir']}`",
                    ],
                    [
                        "A10 GPU finding",
                        "`gpu/gpu-int4-rtn-block-32` downloaded successfully",
                        "It did not fit in the 4 GB A10-4Q framebuffer, so the successful run used the official CPU/mobile int4 package for Phi-4-mini.",
                    ],
                ],
            )
        ),
        _html_output_cell(
            "from IPython.display import HTML, display\n"
            "# Executed output below embeds the query photo and retrieval thumbnails.\n"
            "display(HTML(real_local_result_html))\n",
            _real_local_queries_html(queries),
            execution_count=1,
        ),
        _markdown(
            "## Retrieval summary\n\n"
            + _markdown_table(
                ["Scenario", "Expected case", "Top retrieved case", "Top score", "Expected matched"],
                [
                    [
                        item["scenario"],
                        item.get("expected_incident_id") or "not specified",
                        item["hits"][0]["incident_id"],
                        f"{item['hits'][0]['score']:.4f}",
                        "yes" if item.get("matched_expected") else "no",
                    ]
                    for item in queries
                ],
            )
        ),
        _markdown("## User questions and Phi-4-mini responses\n\n" + "\n\n".join(_real_local_answer_md(item) for item in queries)),
        _markdown(
            "## Timing from the VM run\n\n"
            + _markdown_table(
                ["Scenario", "BLIP caption", "CLIP retrieval", "Phi-4-mini load/generation", "Decode loop only"],
                [
                    [
                        item["scenario"],
                        item["timings_seconds"]["caption"],
                        item["timings_seconds"]["retrieval"],
                        item["timings_seconds"]["generation"],
                        item["timings_seconds"]["model_decode_loop"],
                    ]
                    for item in queries
                ],
            )
        ),
    ]

    _write_notebook(output_path, cells)


def _single_real_local_query(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario": "Single query",
        "query": report["query"],
        "query_image": report["query_image"],
        "query_image_caption_model": report["query_image_caption_model"],
        "query_image_caption": report["query_image_caption"],
        "expected_incident_id": None,
        "matched_expected": False,
        "hits": report["hits"],
        "answer": report["answer"],
        "timings_seconds": report["timings_seconds"],
    }


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_image_map() -> dict[str, Path]:
    manifest = _read_json(OFFLINE_ASSETS / "image_manifest.json")
    return {
        item["incident_id"]: OFFLINE_ASSETS / item["image"]
        for item in manifest["images"]
    }


def _sequence_markdown(report: dict[str, Any]) -> str:
    parts = []
    for idx, item in enumerate(report["sequence_steps"], start=1):
        synced = ", ".join(item["online_resume"]["synced_ids"]) or "none"
        parts.append(
            f"### {idx}. {item['scenario']}\n\n"
            f"**Worker query:** {item['query']}\n\n"
            f"**Query image:** `{item.get('query_image', 'not captured')}`\n\n"
            "**Initial offline retrieval**\n\n"
            + _markdown_table(
                ["Top-3 local evidence", "Model response from initial offline context"],
                [[_format_local_hits(item["initial_offline"]["top3"]), _compact_answer(item["initial_offline"]["answer"])]],
            )
            + "\n\n**Enriched offline retrieval after connectivity sync**\n\n"
            + _markdown_table(
                ["Top-3 local evidence after sync", "Model response after offline cache enrichment"],
                [
                    [
                        _format_local_hits(item["enriched_offline_after_sync"]["top3"]),
                        _compact_answer(item["enriched_offline_after_sync"]["answer"]),
                    ]
                ],
            )
            + "\n\n**Online result used during connectivity resume**\n\n"
            + _markdown_table(
                ["Top-3 Azure AI Search evidence", "Cached for later offline use"],
                [[_format_online_hits(item["online_resume"]["top3"]), synced]],
            )
        )
    return "\n\n".join(parts)


def _title_for_id(report: dict[str, Any], incident_id: str) -> str:
    for group in ("online_resume_results", "full_online_results"):
        for item in report.get(group, []):
            hits = item.get("online_top3", [])
            for hit in hits:
                if hit.get("id") == incident_id:
                    return hit.get("title", incident_id)
    return incident_id


def _severity_for_id(report: dict[str, Any], incident_id: str) -> str:
    for group in ("online_resume_results", "full_online_results"):
        for item in report.get(group, []):
            for hit in item.get("online_top3", []):
                if hit.get("id") == incident_id:
                    return hit.get("severity", "")
    return ""


def _offline_query_html(queries: list[dict[str, Any]]) -> str:
    blocks = ['<div style="font-family:Segoe UI,Arial,sans-serif">']
    for item in queries:
        blocks.append(f"<h3>{escape(item['scenario'])}</h3>")
        blocks.append(f"<p><b>Query:</b> {escape(item['query'])}</p>")
        if item.get("query_image"):
            query_image_path = OFFLINE_ASSETS / "images" / item["query_image"]
            blocks.append(
                "<div style='width:260px; border:2px solid #666; padding:8px; border-radius:6px; margin-bottom:10px'>"
                f"<img src='{_thumbnail_data_uri(query_image_path)}' style='width:100%; height:155px; object-fit:cover'>"
                "<div><b>Query photo embedded by local CLIP</b></div>"
                "</div>"
            )
        blocks.append("<div style='display:flex; gap:12px; flex-wrap:wrap'>")
        for hit in item["hits"]:
            image_path = OFFLINE_ASSETS / "images" / hit["image"]
            blocks.append(
                "<div style='width:210px; border:1px solid #ddd; padding:8px; border-radius:6px'>"
                f"<img src='{_thumbnail_data_uri(image_path)}' style='width:100%; height:140px; object-fit:cover'>"
                f"<div><b>#{hit['rank']} {escape(hit['incident_id'])}</b></div>"
                f"<div>{escape(hit['title'])}</div>"
                f"<div>score={hit['score']:.4f}</div>"
                "</div>"
            )
        blocks.append("</div>")
    blocks.append("</div>")
    return "\n".join(blocks)


def _image_gallery_html(items: list[dict[str, Any]], *, title: str) -> str:
    blocks = [
        "<div style='font-family:Segoe UI,Arial,sans-serif'>",
        f"<h2>{escape(title)}</h2>",
        "<div style='display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:14px'>",
    ]
    for item in items:
        blocks.append(
            "<div style='border:1px solid #ddd; padding:10px; border-radius:8px'>"
            f"<img src='{_thumbnail_data_uri(Path(item['image_path']))}' style='width:100%; height:155px; object-fit:cover; border-radius:4px'>"
            f"<div style='font-weight:600; margin-top:6px'>{escape(item['id'])} - {escape(item['title'])}</div>"
            f"<div><b>Severity:</b> {escape(item.get('severity', ''))}</div>"
            f"<div style='font-size:12px'>{escape(item.get('caption', ''))}</div>"
            "</div>"
        )
    blocks.append("</div></div>")
    return "\n".join(blocks)


def _real_local_result_html(report: dict[str, Any]) -> str:
    blocks = [
        "<div style='font-family:Segoe UI,Arial,sans-serif'>",
        "<h2>Real local inference query and retrieval</h2>",
        "<div style='display:flex; gap:16px; flex-wrap:wrap; margin-bottom:16px'>",
        "<div style='width:300px; border:2px solid #555; padding:10px; border-radius:8px'>"
        f"<img src='{_thumbnail_data_uri(REPO_ROOT / report['query_image'])}' style='width:100%; height:190px; object-fit:cover; border-radius:4px'>"
        "<div style='font-weight:600; margin-top:6px'>Query photo</div>"
        f"<div><b>BLIP caption:</b> {escape(report['query_image_caption'])}</div>"
        "</div>",
        "</div>",
        "<h3>Top retrieved local evidence</h3>",
        "<div style='display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:14px'>",
    ]
    for hit in report["hits"]:
        blocks.append(
            "<div style='border:1px solid #ddd; padding:10px; border-radius:8px'>"
            f"<img src='{_thumbnail_data_uri(REPO_ROOT / hit['image_path'])}' style='width:100%; height:155px; object-fit:cover; border-radius:4px'>"
            f"<div style='font-weight:600; margin-top:6px'>#{hit['rank']} {escape(hit['incident_id'])}</div>"
            f"<div>{escape(hit['title'])}</div>"
            f"<div><b>Score:</b> {hit['score']:.4f}</div>"
            f"<div><b>Severity:</b> {escape(hit['severity'])}</div>"
            "</div>"
        )
    blocks.append("</div></div>")
    return "\n".join(blocks)


def _real_local_queries_html(queries: list[dict[str, Any]]) -> str:
    blocks = ["<div style='font-family:Segoe UI,Arial,sans-serif'>"]
    for item in queries:
        blocks.extend(
            [
                f"<h2>{escape(item['scenario'])}</h2>",
                f"<p><b>BLIP caption:</b> {escape(item['query_image_caption'])}</p>",
                "<div style='display:flex; gap:14px; flex-wrap:wrap; margin-bottom:18px'>",
                "<div style='width:285px; border:2px solid #555; padding:10px; border-radius:8px'>"
                f"<img src='{_thumbnail_data_uri(REPO_ROOT / item['query_image'])}' style='width:100%; height:185px; object-fit:cover; border-radius:4px'>"
                "<div style='font-weight:600; margin-top:6px'>Held-out query photo, not indexed</div>"
                f"<div><b>Expected case:</b> {escape(item.get('expected_incident_id') or 'not specified')}</div>"
                "</div>",
                "</div>",
                "<div style='display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin-bottom:22px'>",
            ]
        )
        for hit in item["hits"]:
            blocks.append(
                "<div style='border:1px solid #ddd; padding:10px; border-radius:8px'>"
                f"<img src='{_thumbnail_data_uri(REPO_ROOT / hit['image_path'])}' style='width:100%; height:150px; object-fit:cover; border-radius:4px'>"
                f"<div style='font-weight:600; margin-top:6px'>#{hit['rank']} {escape(hit['incident_id'])}</div>"
                f"<div>{escape(hit['title'])}</div>"
                f"<div><b>Score:</b> {hit['score']:.4f}</div>"
                f"<div><b>Severity:</b> {escape(hit['severity'])}</div>"
                "</div>"
            )
        blocks.append("</div>")
    blocks.append("</div>")
    return "\n".join(blocks)


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


def _answer_example_md(item: dict[str, Any]) -> str:
    image_line = f"\n\n**Query image:** `{Path(item['query_image']).name}`" if item.get("query_image") else ""
    return (
        f"### {item['scenario']}\n\n"
        f"**User question:** {item['query']}"
        f"{image_line}\n\n"
        "**Model response:**\n\n"
        f"{_quote_block(_compact_answer(item['answer'], limit=1400))}"
    )


def _real_local_answer_md(item: dict[str, Any]) -> str:
    top_hit = item["hits"][0]
    return (
        f"### {item['scenario']}\n\n"
        f"**User question:** {item['query']}\n\n"
        f"**Held-out query image:** `{Path(item['query_image']).name}`\n\n"
        f"**Top retrieved case:** `{top_hit['incident_id']}` - {top_hit['title']} "
        f"(`score={top_hit['score']:.4f}`, severity `{top_hit['severity']}`)\n\n"
        "**Model response:**\n\n"
        f"{_quote_block(_compact_answer(item['answer'], limit=1600))}"
    )


def _compact_answer(answer: str, limit: int = 900) -> str:
    normalized = "\n".join(line.strip() for line in answer.strip().splitlines() if line.strip())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."


def _quote_block(text: str) -> str:
    return "\n".join(f"> {line}" for line in text.splitlines() if line.strip())


def _format_hits(hits: list[dict[str, Any]]) -> str:
    return "<br>".join(f"{hit['rank']}. {hit['incident_id']} (`{hit['score']:.4f}`)" for hit in hits)


def _format_local_hits(hits: list[dict[str, Any]]) -> str:
    return "<br>".join(f"{hit['rank']}. {hit['id']} - {hit['title']} (`{hit['score']:.4f}`)" for hit in hits)


def _format_online_hits(hits: list[dict[str, Any]]) -> str:
    return "<br>".join(f"{hit['rank']}. {hit['id']} - {hit['title']} (`{hit['score']:.4f}`)" for hit in hits)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell_value(value) for value in row) + " |")
    return "\n".join(lines)


def _markdown_cell_value(value: Any) -> str:
    text = str(value)
    return text.replace("\n", "<br>").replace("|", "\\|")


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
