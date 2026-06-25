from __future__ import annotations

from pathlib import Path
import math
from typing import Iterable

from cv_rag.models import ClipEmbedder
from cv_rag.store import CvHit, CvVectorStore
from cv_rag.synthetic_data import Incident, generate_dataset, load_incidents

INDEX_IMAGE_WEIGHT = 0.45
INDEX_TEXT_WEIGHT = 0.55
QUERY_IMAGE_WEIGHT = 0.60
QUERY_TEXT_WEIGHT = 0.40


def build_cv_index(
    workspace: str,
    db_path: str,
    device: str = "auto",
    incidents: Iterable[Incident] | None = None,
    clean: bool = True,
    render_images: bool = True,
) -> int:
    incidents_path, incidents = generate_dataset(workspace, incidents, render_images=render_images)
    embedder = ClipEmbedder(device=device)
    store = CvVectorStore(db_path)
    if clean:
        store.clear()
    image_dir = Path(workspace) / "images"
    for incident in incidents:
        image_path = image_dir / incident.image_file
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image asset not found for {incident.incident_id}: {image_path}. "
                "Generate images first or call build_cv_index with render_images=True."
            )
        image_vector = embedder.embed_image(str(image_path))
        text_vector = embedder.embed_text(incident.searchable_text)
        vector = fuse_vectors(image_vector, text_vector, image_weight=INDEX_IMAGE_WEIGHT, text_weight=INDEX_TEXT_WEIGHT)
        store.upsert(incident, str(image_path), vector)
    return len(load_incidents(str(incidents_path)))


def search_cv_index(
    query: str,
    db_path: str,
    device: str = "auto",
    top_k: int = 3,
    query_image: str | None = None,
) -> list[CvHit]:
    embedder = ClipEmbedder(device=device)
    store = CvVectorStore(db_path)
    return store.search(build_query_vector(embedder, query, query_image=query_image), top_k=top_k)


def build_query_vector(embedder: ClipEmbedder, query: str, *, query_image: str | None = None) -> list[float]:
    text_vector = embedder.embed_text(query)
    if not query_image:
        return text_vector
    image_path = Path(query_image)
    if not image_path.exists():
        raise FileNotFoundError(f"Query image not found: {image_path}")
    image_vector = embedder.embed_image(str(image_path))
    return fuse_vectors(image_vector, text_vector, image_weight=QUERY_IMAGE_WEIGHT, text_weight=QUERY_TEXT_WEIGHT)


def build_prompt(query: str, hits: list[CvHit], *, query_image: str | None = None) -> str:
    lines = []
    for i, hit in enumerate(hits, start=1):
        inc = hit.incident
        lines.append(
            f"[{inc.incident_id}] score={hit.score:.3f}; title={inc.title}; "
            f"severity={inc.severity}; image={hit.image_path}; visual_clues={', '.join(inc.visual_clues)}; "
            f"observation={inc.observation}; root_cause={inc.root_cause_hypothesis}; "
            f"action={inc.recommended_action}; escalation={inc.escalation}"
        )
    evidence = "\n".join(lines)
    query_image_line = f"\nQuery image used for retrieval: {query_image}" if query_image else "\nQuery image used for retrieval: none"
    return f"""Retrieved evidence:
{evidence}

Worker question:
{query}
{query_image_line}

Answer requirements:
- Answer the worker directly in natural language.
- Ground every recommendation in the retrieved evidence.
- Cite the most relevant incident ID.
- Preserve escalation rules and do not invent policy."""


def fuse_vectors(
    image_vector: list[float],
    text_vector: list[float],
    *,
    image_weight: float = INDEX_IMAGE_WEIGHT,
    text_weight: float = INDEX_TEXT_WEIGHT,
) -> list[float]:
    if len(image_vector) != len(text_vector):
        raise ValueError(f"Vector dimensions differ: {len(image_vector)} != {len(text_vector)}")
    if image_weight < 0 or text_weight < 0 or image_weight + text_weight == 0:
        raise ValueError("Vector fusion weights must be non-negative and not both zero.")
    total = image_weight + text_weight
    normalized_image_weight = image_weight / total
    normalized_text_weight = text_weight / total
    fused = [
        (normalized_image_weight * image) + (normalized_text_weight * text)
        for image, text in zip(image_vector, text_vector)
    ]
    norm = math.sqrt(sum(value * value for value in fused))
    if norm == 0:
        return fused
    return [value / norm for value in fused]
