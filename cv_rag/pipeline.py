from __future__ import annotations

from pathlib import Path
import math
from typing import Iterable

from cv_rag.models import ClipEmbedder
from cv_rag.store import CvHit, CvVectorStore
from cv_rag.synthetic_data import Incident, generate_dataset, load_incidents


def build_cv_index(
    workspace: str,
    db_path: str,
    device: str = "auto",
    incidents: Iterable[Incident] | None = None,
    clean: bool = True,
) -> int:
    incidents_path, incidents = generate_dataset(workspace, incidents)
    embedder = ClipEmbedder(device=device)
    store = CvVectorStore(db_path)
    if clean:
        store.clear()
    image_dir = Path(workspace) / "images"
    for incident in incidents:
        image_path = image_dir / incident.image_file
        image_vector = embedder.embed_image(str(image_path))
        text_vector = embedder.embed_text(incident.searchable_text)
        vector = fuse_vectors(image_vector, text_vector)
        store.upsert(incident, str(image_path), vector)
    return len(load_incidents(str(incidents_path)))


def search_cv_index(query: str, db_path: str, device: str = "auto", top_k: int = 3) -> list[CvHit]:
    embedder = ClipEmbedder(device=device)
    store = CvVectorStore(db_path)
    return store.search(embedder.embed_text(query), top_k=top_k)


def build_prompt(query: str, hits: list[CvHit]) -> str:
    lines = []
    for i, hit in enumerate(hits, start=1):
        inc = hit.incident
        lines.append(
            f"[{inc.incident_id}] score={hit.score:.3f}; title={inc.title}; "
            f"severity={inc.severity}; image={hit.image_path}; observation={inc.observation}; "
            f"action={inc.recommended_action}; escalation={inc.escalation}"
        )
    evidence = "\n".join(lines)
    return f"""Retrieved evidence:
{evidence}

Question:
{query}

Write a concise construction incident response. Include top visual match, action steps, escalation condition, and citations."""


def fuse_vectors(image_vector: list[float], text_vector: list[float]) -> list[float]:
    if len(image_vector) != len(text_vector):
        raise ValueError(f"Vector dimensions differ: {len(image_vector)} != {len(text_vector)}")
    # Keep the image vector primary, but add incident text so text queries can
    # reliably retrieve the matching visual case in small offline packs.
    fused = [(0.65 * image) + (0.35 * text) for image, text in zip(image_vector, text_vector)]
    norm = math.sqrt(sum(value * value for value in fused))
    if norm == 0:
        return fused
    return [value / norm for value in fused]
