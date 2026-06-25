from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_REPORT = REPO_ROOT / "notebooks" / "reports" / "local_offline_rag" / "moondream_phi4_report.json"
DEFAULT_OUTPUT = REPO_ROOT / "notebooks" / "assets" / "real_local_inference" / "traditional_chinese_offline_report.json"
DEFAULT_SCENARIO_ID = "qry_001_water_ingress"
DEFAULT_TC_QUERY = (
    "現場相片顯示地下室擋土牆的施工縫有明顯滲水、牆身有水漬，地面亦有積水。"
    "請幫我用過往案例找出最相似的情況，並用繁體中文說明下一步應該怎樣處理，以及甚麼情況需要升級處理。"
)
DEFAULT_RETRIEVAL_QUERY = (
    "Basement retaining wall construction joint water ingress with damp staining, puddling at the wall base, "
    "possible waterproofing membrane or joint seal failure, requiring similar previous case and next actions."
)
DEFAULT_TC_GLOSSARY = [
    {
        "source": "offline / disconnected",
        "target": "離線 / 無網絡",
        "note": "Do not use 無線.",
        "forbidden": ["無線", "无线"],
    },
    {
        "source": "active electrical equipment",
        "target": "帶電設備",
        "note": "Use for live electrical equipment affected by water.",
        "forbidden": ["電器設備", "电器设备"],
    },
    {
        "source": "elevation / chainage references",
        "target": "標高／樁號或位置參考",
        "note": "Use Hong Kong construction-site wording for location references.",
        "forbidden": [],
    },
    {
        "source": "tie-hole repairs",
        "target": "拉桿孔修補",
        "note": "Use when describing waterproofing penetrations or repairs.",
        "forbidden": [],
    },
    {
        "source": "sump",
        "target": "集水井",
        "note": "Use for drainage/sump operation checks.",
        "forbidden": [],
    },
    {
        "source": "soil washout",
        "target": "泥土流失或泥土沖刷",
        "note": "Use for leakage-related erosion or washout escalation.",
        "forbidden": ["土壤洗濁"],
    },
    {
        "source": "inspect / check",
        "target": "檢查",
        "note": "Use Traditional Chinese script consistently.",
        "forbidden": ["检查"],
    },
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Run an offline Traditional Chinese query-normalization and answer-generation validation."
    )
    parser.add_argument("--source-report", default=str(DEFAULT_SOURCE_REPORT))
    parser.add_argument("--scenario-id", default=DEFAULT_SCENARIO_ID)
    parser.add_argument("--traditional-chinese-query", default=DEFAULT_TC_QUERY)
    parser.add_argument("--phi4-onnx-model-dir", default=os.getenv("PHI4_ONNX_MODEL_DIR", ""))
    parser.add_argument("--phi4-execution-provider", default="follow_config")
    parser.add_argument("--require-phi4", action="store_true")
    parser.add_argument("--max-rewrite-tokens", type=int, default=160)
    parser.add_argument("--max-answer-tokens", type=int, default=520)
    parser.add_argument(
        "--glossary",
        default=None,
        help="Optional JSON glossary file loaded at runtime. Accepts a list of entries or an object with a `terms` list.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    source_report_path = Path(args.source_report)
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    source_query = _find_query(source_report, args.scenario_id)
    model_dir = Path(args.phi4_onnx_model_dir) if args.phi4_onnx_model_dir else None
    generator = _load_phi4(model_dir, args.phi4_execution_provider, require_phi4=args.require_phi4)
    glossary = _load_glossary(Path(args.glossary) if args.glossary else None)
    glossary_prompt = _format_glossary_for_prompt(glossary)

    image_body = source_query["query_image_caption"]
    rewrite_prompt = _build_rewrite_prompt(args.traditional_chinese_query, image_body)
    answer_prompt = None

    timings: dict[str, float] = {}
    if generator:
        rewrite_started = time.perf_counter()
        normalized_query = _clean_retrieval_query(
            generator.generate(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are an offline query planner for construction incident retrieval. "
                            "Convert Traditional Chinese field questions plus local image captions into concise English retrieval queries. "
                            "Do not add facts that are not present in the question or caption."
                        ),
                    },
                    {"role": "user", "content": rewrite_prompt},
                ],
                max_new_tokens=args.max_rewrite_tokens,
            )
        )
        timings["query_rewrite"] = round(time.perf_counter() - rewrite_started, 3)
    else:
        normalized_query = DEFAULT_RETRIEVAL_QUERY

    answer_prompt = _build_answer_prompt(
        traditional_chinese_query=args.traditional_chinese_query,
        image_body=image_body,
        normalized_query=normalized_query,
        hits=source_query["hits"],
        query_image=source_query["query_image"],
        glossary_prompt=glossary_prompt,
    )
    if generator:
        answer_started = time.perf_counter()
        answer_tc_raw = generator.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an offline construction field copilot. Answer in Traditional Chinese only. "
                        "Use only the retrieved local evidence. Cite incident IDs exactly, preserve escalation rules, "
                        "and do not invent approvals, standards, or facts. Apply the supplied glossary exactly. "
                        "Before finalizing, self-check that the answer uses Traditional Chinese script and contains none of the forbidden terms."
                    ),
                },
                {"role": "user", "content": answer_prompt},
            ],
            max_new_tokens=args.max_answer_tokens,
        ).strip()
        answer_tc = answer_tc_raw
        timings["answer_generation"] = round(time.perf_counter() - answer_started, 3)
    else:
        answer_tc_raw = _fallback_traditional_chinese_answer(source_query["hits"])
        answer_tc = answer_tc_raw

    top_hit = source_query["hits"][0] if source_query["hits"] else {}
    answer_validation_warnings = _validate_answer_against_glossary(answer_tc, glossary)
    report = {
        "mode": "full-offline-traditional-chinese-validation",
        "online_used": False,
        "scenario_id": args.scenario_id,
        "source_report": _relative(source_report_path),
        "source_captioner": source_report.get("query_image_captioner"),
        "source_caption_model": source_report.get("query_image_caption_model"),
        "source_vector_inputs": source_report.get("query_vector_inputs", []),
        "retrieval_replay_note": (
            "The local vector hits are replayed from the recorded offline Moondream + CLIP image/text/caption search "
            "for the same held-out image and local case pack. No Azure AI Search, cloud model, or online corpus was used."
        ),
        "traditional_chinese_query": args.traditional_chinese_query,
        "query_image": source_query["query_image"],
        "query_image_indexed": False,
        "image_body": image_body,
        "query_rewrite_model": "microsoft/Phi-4-mini-instruct-onnx" if generator else "deterministic-reference-fallback",
        "answer_model": "microsoft/Phi-4-mini-instruct-onnx" if generator else "deterministic-reference-fallback",
        "phi4_model_dir": str(model_dir) if model_dir else "",
        "answer_execution_provider": args.phi4_execution_provider if generator else "",
        "glossary_source": str(Path(args.glossary)) if args.glossary else "default-construction-tc-glossary",
        "glossary_terms": glossary,
        "normalized_retrieval_query_en": normalized_query,
        "hits": source_query["hits"],
        "expected_incident_id": source_query.get("expected_incident_id"),
        "top_hit": top_hit.get("incident_id"),
        "matched_expected": bool(source_query.get("expected_incident_id") and top_hit.get("incident_id") == source_query.get("expected_incident_id")),
        "answer_tc": answer_tc,
        "answer_tc_raw": answer_tc_raw,
        "answer_postprocessed": False,
        "answer_validation_warnings": answer_validation_warnings,
        "timings_seconds": timings,
        "rewrite_prompt": rewrite_prompt,
        "answer_prompt": answer_prompt,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


class Phi4MiniOnnxTextGenerator:
    def __init__(self, model_dir: Path, execution_provider: str) -> None:
        try:
            import onnxruntime_genai as og
        except ImportError as exc:
            raise RuntimeError("onnxruntime-genai is not installed. Install `pip install --pre onnxruntime-genai`.") from exc

        self._og = og
        self.model_dir = model_dir
        self.execution_provider = execution_provider
        config = og.Config(str(model_dir))
        if execution_provider not in {"follow_config", "cpu"}:
            config.clear_providers()
            config.append_provider(execution_provider)
        self.model = og.Model(config)
        self.tokenizer = og.Tokenizer(self.model)
        self.stream = self.tokenizer.create_stream()

    def generate(self, messages: list[dict[str, str]], *, max_new_tokens: int) -> str:
        prompt_text = self._apply_chat_template(messages)
        input_tokens = self.tokenizer.encode(prompt_text)
        params = self._og.GeneratorParams(self.model)
        params.set_search_options(
            max_length=len(input_tokens) + max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )
        generator = self._og.Generator(self.model, params)
        generator.append_tokens(input_tokens)

        tokens: list[str] = []
        while not generator.is_done():
            generator.generate_next_token()
            tokens.append(self.stream.decode(generator.get_next_tokens()[0]))
        del generator
        return "".join(tokens).strip()

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        try:
            template_path = self.model_dir / "chat_template.jinja"
            template = template_path.read_text(encoding="utf-8") if template_path.exists() else ""
            return self.tokenizer.apply_chat_template(
                messages=json.dumps(messages, ensure_ascii=False),
                add_generation_prompt=True,
                template_str=template,
            )
        except Exception:
            lines = []
            for message in messages:
                lines.append(f"{message['role'].title()}: {message['content']}")
            lines.append("Assistant:")
            return "\n".join(lines)


def _load_phi4(model_dir: Path | None, execution_provider: str, *, require_phi4: bool) -> Phi4MiniOnnxTextGenerator | None:
    if not model_dir:
        if require_phi4:
            raise ValueError("--phi4-onnx-model-dir or PHI4_ONNX_MODEL_DIR is required when --require-phi4 is set.")
        return None
    if not model_dir.exists():
        if require_phi4:
            raise FileNotFoundError(model_dir)
        return None
    return Phi4MiniOnnxTextGenerator(model_dir, execution_provider)


def _find_query(report: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    for item in report.get("queries", []):
        if item.get("id") == scenario_id:
            return item
    raise ValueError(f"Scenario {scenario_id!r} not found in {report.get('query_set') or 'source report'}")


def _build_rewrite_prompt(traditional_chinese_query: str, image_body: str) -> str:
    return (
        "Worker question in Traditional Chinese:\n"
        f"{traditional_chinese_query}\n\n"
        "Local image body/caption generated offline from the query photo:\n"
        f"{image_body}\n\n"
        "Return one concise English retrieval query for searching a local construction incident case pack. "
        "Include the defect/hazard, location, likely cause terms, and needed response. Return only the query text."
    )


def _build_answer_prompt(
    *,
    traditional_chinese_query: str,
    image_body: str,
    normalized_query: str,
    hits: list[dict[str, Any]],
    query_image: str,
    glossary_prompt: str,
) -> str:
    evidence = "\n".join(_hit_evidence_line(hit) for hit in hits)
    return (
        "Retrieved local evidence:\n"
        f"{evidence}\n\n"
        "Worker question in Traditional Chinese:\n"
        f"{traditional_chinese_query}\n\n"
        "Local image body/caption:\n"
        f"{image_body}\n\n"
        "Normalized English retrieval query used for local search:\n"
        f"{normalized_query}\n\n"
        f"Query image used for retrieval: {query_image}\n\n"
        "Answer requirements:\n"
        "- Reply in Traditional Chinese only.\n"
        "- Start with the most relevant incident ID and why it matches.\n"
        "- Provide practical next actions grounded in the retrieved evidence.\n"
        "- Preserve the escalation condition from the evidence.\n"
        "- State that this is an offline answer from the local case pack.\n"
        "- Use the glossary below exactly; do not use any forbidden terms.\n"
        "- Before finalizing, check your answer against the glossary and rewrite internally if needed.\n\n"
        f"{glossary_prompt}"
    )


def _hit_evidence_line(hit: dict[str, Any]) -> str:
    return (
        f"[{hit['incident_id']}] score={hit['score']:.3f}; title={hit['title']}; severity={hit['severity']}; "
        f"image={hit.get('image_path', '')}; action={hit.get('recommended_action', '')}; escalation={hit.get('escalation', '')}"
    )


def _clean_retrieval_query(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json|text)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    if cleaned.startswith("{"):
        try:
            payload = json.loads(cleaned)
            for key in ("retrieval_query_en", "retrieval_query", "query"):
                if payload.get(key):
                    return str(payload[key]).strip()
        except json.JSONDecodeError:
            pass
    return cleaned.strip('"')


def _fallback_traditional_chinese_answer(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "離線案例包未找到相似案例。請先保持現場安全、補拍相片，並升級至負責工程師覆核。"
    top = hits[0]
    actions = [item.strip() for item in top.get("recommended_action", "").split(";") if item.strip()]
    lines = [
        f"最相似的本地案例是 {top['incident_id']}（{top['title']}），相似度分數為 {top['score']:.3f}。",
        "此離線判斷來自已安裝在裝置上的本地案例包和查詢相片的本地影像描述。",
        "",
        "建議下一步：",
    ]
    lines.extend(f"{index}. {_tc_action(action)}" for index, action in enumerate(actions[:5], start=1))
    lines.extend(
        [
            "",
            f"升級條件：{_tc_action(top.get('escalation', '如情況涉及結構、安全或合規風險，應升級處理。'))}",
            f"引用依據：{top['incident_id']}。",
        ]
    )
    return "\n".join(lines)


def _tc_action(action: str) -> str:
    replacements = {
        "Mark and photograph the full extent of seepage and note elevation/chainage references.": "標記並拍攝滲水範圍，記錄標高及位置參考。",
        "Inspect the joint detailing, membrane continuity, and any penetrations or tie-hole repairs.": "檢查施工縫做法、防水膜連續性，以及任何穿牆位置或拉桿孔修補。",
        "Check adjacent drainage, sump operation, and external water management conditions.": "檢查相鄰排水、集水井運作及外圍水管理情況。",
        "Implement temporary water control measures and isolate any affected electrical or finish-sensitive areas.": "先採取臨時控水措施，並隔離受影響的電氣或對水敏感的裝修區域。",
        "Raise a corrective action for waterproofing specialist review and retesting after repair.": "提出整改行動，安排防水專家覆核，並在修補後重新測試。",
        "Escalate immediately if leakage increases, reaches active electrical equipment, or shows signs of wall movement, cracking, or soil washout.": "如滲水增加、接近帶電設備，或出現牆體移動、裂縫、泥土流失跡象，應立即升級處理。",
    }
    return replacements.get(action, action)


def _load_glossary(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_TC_GLOSSARY
    payload = json.loads(path.read_text(encoding="utf-8"))
    terms = payload.get("terms", payload) if isinstance(payload, dict) else payload
    if not isinstance(terms, list):
        raise ValueError("Glossary JSON must be a list or an object containing a `terms` list.")
    normalized_terms = []
    for index, term in enumerate(terms, start=1):
        if not isinstance(term, dict):
            raise ValueError(f"Glossary term {index} must be an object.")
        if not term.get("source") or not term.get("target"):
            raise ValueError(f"Glossary term {index} must include `source` and `target`.")
        forbidden = term.get("forbidden", [])
        if isinstance(forbidden, str):
            forbidden = [forbidden]
        normalized_terms.append(
            {
                "source": str(term["source"]),
                "target": str(term["target"]),
                "note": str(term.get("note", "")),
                "forbidden": [str(item) for item in forbidden],
            }
        )
    return normalized_terms


def _format_glossary_for_prompt(glossary: list[dict[str, Any]]) -> str:
    lines = [
        "Traditional Chinese glossary for this answer:",
        "| Source concept / English term | Required Traditional Chinese wording | Forbidden terms | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for term in glossary:
        forbidden = ", ".join(term.get("forbidden", [])) or "-"
        lines.append(f"| {term['source']} | {term['target']} | {forbidden} | {term.get('note', '')} |")
    return "\n".join(lines)


def _validate_answer_against_glossary(answer: str, glossary: list[dict[str, Any]]) -> list[str]:
    warnings = []
    for term in glossary:
        for forbidden in term.get("forbidden", []):
            if forbidden and forbidden in answer:
                warnings.append(
                    f"Forbidden term `{forbidden}` found; prefer `{term['target']}` for `{term['source']}`."
                )
    if re.search(r"[\u4e00-\u9fff]", answer) and re.search(r"检查|无线|电器|土壤洗濁", answer):
        warnings.append("Potential simplified Chinese or non-preferred construction wording detected.")
    return warnings


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
