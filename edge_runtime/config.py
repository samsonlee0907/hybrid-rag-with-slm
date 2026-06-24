from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    site_id: str = "demo-site"
    local_pack_db: str = "data\\site-pack.sqlite"
    phi4_onnx_model_dir: str | None = None
    offline_top_k: int = 4
    max_answer_tokens: int = 512

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        return cls(
            site_id=os.getenv("FIELD_COPILOT_SITE_ID", "demo-site"),
            local_pack_db=os.getenv("LOCAL_PACK_DB", "data\\site-pack.sqlite"),
            phi4_onnx_model_dir=os.getenv("PHI4_ONNX_MODEL_DIR") or None,
            offline_top_k=int(os.getenv("OFFLINE_TOP_K", "4")),
            max_answer_tokens=int(os.getenv("MAX_ANSWER_TOKENS", "512")),
        )
