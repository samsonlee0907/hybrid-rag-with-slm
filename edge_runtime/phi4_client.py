from __future__ import annotations

from abc import ABC, abstractmethod


class Generator(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        raise NotImplementedError


class Phi4OnnxGenerator(Generator):
    """Thin adapter for Phi-4-mini ONNX Runtime GenAI.

    The ONNX Runtime GenAI Python API is still evolving. Keep this adapter
    isolated so API changes do not leak into RAG orchestration code.
    """

    def __init__(self, model_dir: str) -> None:
        try:
            import onnxruntime_genai as og
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime-genai is not installed. Install it with "
                "`pip install --pre onnxruntime-genai` or use ExtractiveFallbackGenerator."
            ) from exc

        self._og = og
        self._model = og.Model(model_dir)
        self._tokenizer = og.Tokenizer(self._model)

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        og = self._og
        params = og.GeneratorParams(self._model)
        params.set_search_options(max_length=max_tokens, temperature=0.2, top_p=0.9)
        params.input_ids = self._tokenizer.encode(prompt)
        output_tokens = self._model.generate(params)
        first_sequence = output_tokens if output_tokens and isinstance(output_tokens[0], int) else output_tokens[0]
        return self._tokenizer.decode(first_sequence)


class ExtractiveFallbackGenerator(Generator):
    """Grounded non-LLM fallback for development and offline validation runs."""

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        evidence = _section(prompt, "EVIDENCE", "USER QUESTION")
        question = _section(prompt, "USER QUESTION", "RESPONSE REQUIREMENTS")
        lines = [line.strip() for line in evidence.splitlines() if line.strip()]
        cited = [line for line in lines if line.startswith("[")]
        answer_lines = [
            "Offline draft based on the installed case pack.",
            f"Question: {question.strip()}",
            "",
            "Most relevant previous cases:",
        ]
        answer_lines.extend(cited[:4] or ["No matching case found in the local pack."])
        answer_lines.extend(
            [
                "",
                "Suggested action:",
                "1. Compare the field condition with the cited cases and photos.",
                "2. Follow the approved resolution only if site conditions match.",
                "3. Escalate high-risk structural, safety, or compliance cases to the site engineer.",
                "",
                "Limitations: generated in offline mode; cloud retrieval/VLM checks may improve recall when available.",
            ]
        )
        return "\n".join(answer_lines)


def _section(text: str, start: str, end: str) -> str:
    marker = f"{start}:"
    if marker not in text:
        return ""
    remainder = text.split(marker, 1)[1]
    end_marker = f"{end}:"
    if end_marker in remainder:
        return remainder.split(end_marker, 1)[0]
    return remainder
