from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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


class EvidenceTemplateGenerator:
    def generate(self, prompt: str, max_new_tokens: int = 320) -> str:
        evidence = prompt.split("Retrieved evidence:", 1)[-1].split("Question:", 1)[0].strip()
        question = prompt.split("Question:", 1)[-1].strip()
        lines = [line for line in evidence.splitlines() if line.startswith("[")]
        response = [
            "Offline CV-RAG answer from local image vectors and incident records.",
            f"Question: {question}",
            "",
            "Most relevant visual incidents:",
            *(lines[:3] or ["No incidents retrieved."]),
            "",
            "Recommended response:",
            "1. Compare the current site photo with the cited incident image and observation.",
            "2. Apply only the corrective action that matches the field condition.",
            "3. Escalate critical safety or structural issues before work continues.",
            "",
            "Offline limitation: no cloud search, no remote standards lookup, and no supervisor approval automation.",
        ]
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

