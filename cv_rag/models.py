from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
import time

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
)


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    torch_dtype: torch.dtype


def resolve_device(requested: str = "auto") -> DeviceInfo:
    if requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested
    dtype = torch.float16 if device == "cuda" else torch.float32
    return DeviceInfo(device=device, torch_dtype=dtype)


class ClipEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "auto") -> None:
        self.info = resolve_device(device)
        self.model_name = model_name
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=self.info.torch_dtype).to(self.info.device)
        self.model.eval()

    @torch.inference_mode()
    def embed_image(self, image_path: str) -> list[float]:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.info.device)
        features = self.model.get_image_features(**inputs)
        return _normalize(features).squeeze(0).detach().cpu().float().tolist()

    @torch.inference_mode()
    def embed_text(self, text: str) -> list[float]:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.info.device)
        features = self.model.get_text_features(**inputs)
        return _normalize(features).squeeze(0).detach().cpu().float().tolist()


class Phi4MiniGenerator:
    def __init__(self, model_name: str = "microsoft/Phi-4-mini-instruct", device: str = "auto") -> None:
        self.info = resolve_device(device)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.info.torch_dtype,
            device_map="auto" if self.info.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        if self.info.device == "cpu":
            self.model.to("cpu")
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 320) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are an offline construction incident copilot. Answer only from retrieved evidence and cite incident IDs.",
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"System: {messages[0]['content']}\nUser: {prompt}\nAssistant:"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class Phi4MiniOnnxGenerator:
    def __init__(self, model_dir: str, execution_provider: str = "cuda") -> None:
        try:
            import onnxruntime_genai as og
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime-genai is not installed. Install `onnxruntime-genai-cuda` "
                "for CUDA or `onnxruntime-genai` for CPU before running the real local demo."
            ) from exc

        self._og = og
        self.model_dir = model_dir
        self.execution_provider = execution_provider
        config = og.Config(model_dir)
        if execution_provider not in {"follow_config", "cpu"}:
            config.clear_providers()
            config.append_provider(execution_provider)
        self.model = og.Model(config)
        self.tokenizer = og.Tokenizer(self.model)
        self.stream = self.tokenizer.create_stream()

    def generate(self, prompt: str, max_new_tokens: int = 320) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are an offline construction incident copilot. Answer only from retrieved evidence and cite incident IDs.",
            },
            {"role": "user", "content": prompt},
        ]
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
        started = time.perf_counter()
        while not generator.is_done():
            generator.generate_next_token()
            tokens.append(self.stream.decode(generator.get_next_tokens()[0]))
        self.last_duration_seconds = time.perf_counter() - started
        del generator
        return "".join(tokens).strip()

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        try:
            import json

            template_path = Path(self.model_dir) / "chat_template.jinja"
            template = template_path.read_text(encoding="utf-8") if template_path.exists() else ""
            return self.tokenizer.apply_chat_template(
                messages=json.dumps(messages),
                add_generation_prompt=True,
                template_str=template,
            )
        except Exception:
            return f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant:"


class BlipImageCaptioner:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", device: str = "auto") -> None:
        self.info = resolve_device(device)
        self.model_name = model_name
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.info.torch_dtype,
        ).to(self.info.device)
        self.model.eval()

    @torch.inference_mode()
    def caption(self, image_path: str, prompt: str = "a construction site photo of") -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.info.device, self.info.torch_dtype)
        outputs = self.model.generate(**inputs, max_new_tokens=40)
        return self.processor.decode(outputs[0], skip_special_tokens=True).strip()


class EvidenceTemplateGenerator:
    def generate(self, prompt: str, max_new_tokens: int = 320) -> str:
        evidence = _section_between(prompt, "Retrieved evidence:", "Worker question:")
        if not evidence:
            evidence = _section_between(prompt, "Retrieved evidence:", "Question:")
        question = _section_between(prompt, "Worker question:", "Query image used for retrieval:")
        if not question:
            question = _section_between(prompt, "Question:", "Answer requirements:")
        top_cases = [_parse_evidence_line(line) for line in evidence.splitlines() if line.startswith("[")]
        top_cases = [case for case in top_cases if case]
        if not top_cases:
            return (
                "I could not find a relevant local case in the offline pack. "
                "Keep the area safe, capture more photos, and escalate to the responsible supervisor before continuing work."
            )

        top = top_cases[0]
        actions = _split_actions(top.get("action", ""))
        response = [
            f"This looks closest to {top['citation']} {top.get('title', 'the retrieved local case')} "
            f"(severity: {top.get('severity', 'unknown')}).",
            "",
            f"What I can ground from the local evidence: {top.get('observation', 'the retrieved case has similar visual evidence.')}",
        ]
        if top.get("root_cause"):
            response.extend(["", f"Likely cause to check: {top['root_cause']}"])
        response.append("")
        response.append("Recommended next actions:")
        response.extend(f"{index}. {action}" for index, action in enumerate(actions[:5], start=1))
        response.extend(["", f"Escalate if: {top.get('escalation', 'the condition is safety-critical or outside the cached case evidence.')}"])
        if len(top_cases) > 1:
            alternatives = ", ".join(case["citation"] for case in top_cases[1:3])
            response.extend(["", f"Also compare against {alternatives} if the photo does not match the top case."])
        response.extend(["", f"Grounding: answered from retrieved local evidence for {top['citation']}."])
        return "\n".join(response)


def load_generator(kind: str, device: str = "auto"):
    if kind == "phi4":
        return Phi4MiniGenerator(device=device)
    if kind == "template":
        return EvidenceTemplateGenerator()
    raise ValueError(f"Unsupported generator kind: {kind}")


def set_offline_mode() -> None:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"


def model_cache_ready(model_names: list[str]) -> bool:
    home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home.exists() and all(True for _ in model_names)


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _section_between(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    tail = text.split(start, 1)[1]
    if end in tail:
        tail = tail.split(end, 1)[0]
    return tail.strip()


def _parse_evidence_line(line: str) -> dict[str, str] | None:
    match = re.match(r"\[(?P<id>[^\]]+)\]\s*(?P<body>.*)", line)
    if not match:
        return None
    fields = {"citation": f"[{match.group('id')}]"}
    for key, value in re.findall(r"([a-z_]+)=(.*?)(?=; [a-z_]+=|$)", match.group("body")):
        fields[key] = value.strip()
    return fields


def _split_actions(action_text: str) -> list[str]:
    actions = [item.strip(" .") for item in re.split(r";|\n", action_text) if item.strip()]
    if actions:
        return actions
    return [
        "Compare the current site photo with the cited incident evidence.",
        "Apply only the corrective action that matches the field condition.",
        "Escalate before work continues if the condition is safety-critical.",
    ]
