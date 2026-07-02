from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


ASSET_TIER_METADATA = "metadata"
ASSET_TIER_THUMB = "thumb512"
ASSET_TIER_REVIEW = "review768"
ASSET_TIER_FULL = "full"
ASSET_TIER_ORDER = [ASSET_TIER_METADATA, ASSET_TIER_THUMB, ASSET_TIER_REVIEW, ASSET_TIER_FULL]
DEFAULT_PROJECT_ID = "demo-project"
DEFAULT_SITE_ID = "demo-site"


@dataclass(frozen=True)
class DeviceSyncState:
    last_sync_sequence: int = 0
    max_bytes: int = 20_000_000
    requested_asset_tier: str = ASSET_TIER_REVIEW
    local_hashes: dict[str, str] = field(default_factory=dict)
    project_id: str = DEFAULT_PROJECT_ID
    site_id: str = DEFAULT_SITE_ID
    include_deleted: bool = False


@dataclass(frozen=True)
class DeltaCandidate:
    id: str
    title: str
    severity: str
    source_scope: str
    sync_sequence: int
    updated_at: str
    content_hash: str
    vector_hash: str
    metadata_bytes: int
    vector_bytes: int
    asset_tiers: list[str]
    asset_manifest_path: str
    safety_critical: bool = False
    is_deleted: bool = False
    cache_priority: int = 0
    thumb_hash: str = ""
    review_image_hash: str = ""
    full_asset_hash: str = ""
    thumb_bytes: int = 0
    review_image_bytes: int = 0
    full_asset_bytes: int = 0
    project_id: str = DEFAULT_PROJECT_ID
    site_id: str = DEFAULT_SITE_ID

    @classmethod
    def from_search_document(cls, document: dict[str, Any]) -> "DeltaCandidate":
        return cls(
            id=str(document["id"]),
            title=str(document.get("title", "")),
            severity=str(document.get("severity", "")),
            source_scope=str(document.get("source_scope", "")),
            sync_sequence=int(document.get("sync_sequence", 0)),
            updated_at=str(document.get("updated_at", "")),
            content_hash=str(document.get("content_hash", "")),
            vector_hash=str(document.get("vector_hash", "")),
            metadata_bytes=int(document.get("metadata_bytes", 0)),
            vector_bytes=int(document.get("vector_bytes", 0)),
            asset_tiers=list(document.get("asset_tiers", [])),
            asset_manifest_path=str(document.get("asset_manifest_path", "")),
            safety_critical=bool(document.get("safety_critical", False)),
            is_deleted=bool(document.get("is_deleted", False)),
            cache_priority=int(document.get("cache_priority", 0)),
            thumb_hash=str(document.get("thumb_hash", "")),
            review_image_hash=str(document.get("review_image_hash", "")),
            full_asset_hash=str(document.get("full_asset_hash", "")),
            thumb_bytes=int(document.get("thumb_bytes", 0)),
            review_image_bytes=int(document.get("review_image_bytes", 0)),
            full_asset_bytes=int(document.get("full_asset_bytes", 0)),
            project_id=str(document.get("project_id", DEFAULT_PROJECT_ID)),
            site_id=str(document.get("site_id", DEFAULT_SITE_ID)),
        )


@dataclass(frozen=True)
class DeltaManifestItem:
    id: str
    title: str
    sync_sequence: int
    action: str
    reason: str
    selected_asset_tier: str
    estimated_bytes: int
    content_hash: str
    vector_hash: str
    asset_hash: str
    asset_manifest_path: str
    safety_critical: bool
    severity: str
    source_scope: str


@dataclass(frozen=True)
class DeltaManifest:
    project_id: str
    site_id: str
    generated_at: str
    requested_asset_tier: str
    max_bytes: int
    total_estimated_bytes: int
    next_sync_sequence: int
    has_more: bool
    items: list[DeltaManifestItem]
    skipped: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["items"] = [asdict(item) for item in self.items]
        return payload


def build_search_sync_fields(
    *,
    payload: dict[str, Any],
    content_vector: list[float],
    sync_sequence: int,
    image_path: Path | None,
    project_id: str = DEFAULT_PROJECT_ID,
    site_id: str = DEFAULT_SITE_ID,
    asset_manifest_prefix: str = "packs",
    updated_at: datetime | None = None,
    thumb_bytes: int = 0,
    review_image_bytes: int = 0,
) -> dict[str, Any]:
    incident_id = str(payload["id"])
    image_bytes = image_path.stat().st_size if image_path and image_path.exists() else 0
    available_tiers = [ASSET_TIER_METADATA]
    if thumb_bytes:
        available_tiers.append(ASSET_TIER_THUMB)
    if review_image_bytes:
        available_tiers.append(ASSET_TIER_REVIEW)
    if image_bytes:
        available_tiers.append(ASSET_TIER_FULL)
    return {
        "project_id": project_id,
        "site_id": site_id,
        "sync_sequence": sync_sequence,
        "updated_at": (updated_at or datetime.now(timezone.utc)).isoformat(),
        "content_hash": stable_json_hash(_content_hash_payload(payload)),
        "vector_hash": vector_hash(content_vector),
        "thumb_hash": file_hash(image_path) if image_path and image_path.exists() and thumb_bytes else "",
        "review_image_hash": file_hash(image_path) if image_path and image_path.exists() and review_image_bytes else "",
        "full_asset_hash": file_hash(image_path) if image_path and image_path.exists() else "",
        "metadata_bytes": len(json.dumps(_content_hash_payload(payload), ensure_ascii=False).encode("utf-8")),
        "vector_bytes": len(content_vector) * 4,
        "thumb_bytes": thumb_bytes,
        "review_image_bytes": review_image_bytes,
        "full_asset_bytes": image_bytes,
        "asset_tiers": available_tiers,
        "asset_manifest_path": f"{asset_manifest_prefix.rstrip('/')}/{site_id}/{incident_id}.json",
        "safety_critical": is_safety_critical(payload),
        "cache_priority": cache_priority(payload),
        "is_deleted": False,
    }


def plan_delta_manifest(candidates: list[DeltaCandidate], state: DeviceSyncState) -> DeltaManifest:
    if state.requested_asset_tier not in ASSET_TIER_ORDER:
        raise ValueError(f"Unsupported asset tier: {state.requested_asset_tier}")

    selected: list[DeltaManifestItem] = []
    skipped: list[dict[str, Any]] = []
    total_bytes = 0
    next_sequence = state.last_sync_sequence

    filtered = [
        candidate
        for candidate in candidates
        if candidate.project_id == state.project_id
        and candidate.site_id == state.site_id
        and (state.include_deleted or not candidate.is_deleted)
        and _needs_sync(candidate, state)
    ]
    filtered.sort(key=lambda item: (-int(item.safety_critical), -item.cache_priority, item.sync_sequence, item.id))

    for candidate in filtered:
        tier = _select_asset_tier(candidate, state.requested_asset_tier)
        estimated_bytes = candidate.metadata_bytes + candidate.vector_bytes + _tier_bytes(candidate, tier)
        if total_bytes + estimated_bytes > state.max_bytes:
            skipped.append(
                {
                    "id": candidate.id,
                    "sync_sequence": candidate.sync_sequence,
                    "reason": "byte_budget_exceeded",
                    "estimated_bytes": estimated_bytes,
                }
            )
            continue
        selected.append(
            DeltaManifestItem(
                id=candidate.id,
                title=candidate.title,
                sync_sequence=candidate.sync_sequence,
                action="delete" if candidate.is_deleted else "upsert",
                reason=_sync_reason(candidate, state),
                selected_asset_tier=tier,
                estimated_bytes=estimated_bytes,
                content_hash=candidate.content_hash,
                vector_hash=candidate.vector_hash,
                asset_hash=_tier_hash(candidate, tier),
                asset_manifest_path=candidate.asset_manifest_path,
                safety_critical=candidate.safety_critical,
                severity=candidate.severity,
                source_scope=candidate.source_scope,
            )
        )
        total_bytes += estimated_bytes
        next_sequence = max(next_sequence, candidate.sync_sequence)

    if skipped:
        next_sequence = min(item["sync_sequence"] for item in skipped) - 1

    return DeltaManifest(
        project_id=state.project_id,
        site_id=state.site_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        requested_asset_tier=state.requested_asset_tier,
        max_bytes=state.max_bytes,
        total_estimated_bytes=total_bytes,
        next_sync_sequence=next_sequence,
        has_more=bool(skipped),
        items=selected,
        skipped=skipped,
    )


def stable_json_hash(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def vector_hash(vector: list[float]) -> str:
    body = json.dumps([round(value, 7) for value in vector], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def file_hash(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_safety_critical(payload: dict[str, Any]) -> bool:
    severity = str(payload.get("severity", "")).lower()
    text = " ".join(str(payload.get(key, "")) for key in ("title", "category", "observation", "escalation_rule")).lower()
    return severity == "critical" or any(keyword in text for keyword in ("safety", "fall", "electrical", "confined", "crane"))


def cache_priority(payload: dict[str, Any]) -> int:
    severity = str(payload.get("severity", "")).lower()
    if severity == "critical":
        return 100
    if severity == "high":
        return 80
    if str(payload.get("source_scope", "")).endswith("only"):
        return 60
    return 40


def _content_hash_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"content_vector", "@search.score", "@search.rerankerScore"}
    }


def _needs_sync(candidate: DeltaCandidate, state: DeviceSyncState) -> bool:
    local_hash = state.local_hashes.get(candidate.id)
    if local_hash == candidate.content_hash:
        return False
    if candidate.sync_sequence > state.last_sync_sequence:
        return True
    return local_hash is not None


def _sync_reason(candidate: DeltaCandidate, state: DeviceSyncState) -> str:
    if candidate.sync_sequence > state.last_sync_sequence:
        return "newer_sync_sequence"
    return "content_hash_changed"


def _select_asset_tier(candidate: DeltaCandidate, requested_tier: str) -> str:
    requested_index = ASSET_TIER_ORDER.index(requested_tier)
    available = [tier for tier in candidate.asset_tiers if tier in ASSET_TIER_ORDER]
    if not available:
        return ASSET_TIER_METADATA
    for tier in reversed(ASSET_TIER_ORDER[: requested_index + 1]):
        if tier in available:
            return tier
    return ASSET_TIER_METADATA


def _tier_bytes(candidate: DeltaCandidate, tier: str) -> int:
    if tier == ASSET_TIER_FULL:
        return candidate.full_asset_bytes
    if tier == ASSET_TIER_REVIEW:
        return candidate.review_image_bytes
    if tier == ASSET_TIER_THUMB:
        return candidate.thumb_bytes
    return 0


def _tier_hash(candidate: DeltaCandidate, tier: str) -> str:
    if tier == ASSET_TIER_FULL:
        return candidate.full_asset_hash
    if tier == ASSET_TIER_REVIEW:
        return candidate.review_image_hash
    if tier == ASSET_TIER_THUMB:
        return candidate.thumb_hash
    return candidate.content_hash
