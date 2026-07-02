from __future__ import annotations

from dataclasses import dataclass

import httpx

from online_rag.delta_sync import DEFAULT_PROJECT_ID, DEFAULT_SITE_ID, DeviceSyncState


API_VERSION = "2024-07-01"
VECTOR_DIMENSIONS = 512
DELTA_SELECT_FIELDS = [
    "id",
    "title",
    "severity",
    "source_scope",
    "project_id",
    "site_id",
    "sync_sequence",
    "updated_at",
    "content_hash",
    "vector_hash",
    "thumb_hash",
    "review_image_hash",
    "full_asset_hash",
    "metadata_bytes",
    "vector_bytes",
    "thumb_bytes",
    "review_image_bytes",
    "full_asset_bytes",
    "asset_tiers",
    "asset_manifest_path",
    "safety_critical",
    "cache_priority",
    "is_deleted",
]


@dataclass(frozen=True)
class SearchConfig:
    endpoint: str
    api_key: str
    index_name: str

    @property
    def base_url(self) -> str:
        return self.endpoint.rstrip("/")


class AzureSearchClient:
    def __init__(self, config: SearchConfig) -> None:
        self.config = config

    def create_or_replace_index(self) -> None:
        url = f"{self.config.base_url}/indexes/{self.config.index_name}?api-version={API_VERSION}"
        response = httpx.put(url, headers=self._headers(), json=_index_schema(self.config.index_name), timeout=60)
        response.raise_for_status()

    def upload_documents(self, docs: list[dict]) -> None:
        url = f"{self.config.base_url}/indexes/{self.config.index_name}/docs/index?api-version={API_VERSION}"
        payload = {"value": [{"@search.action": "upload", **doc} for doc in docs]}
        response = httpx.post(url, headers=self._headers(), json=payload, timeout=120)
        response.raise_for_status()

    def search(self, query: str, vector: list[float], top: int = 5) -> list[dict]:
        url = f"{self.config.base_url}/indexes/{self.config.index_name}/docs/search?api-version={API_VERSION}"
        payload = {
            "search": query,
            "top": top,
            "vectorQueries": [
                {
                    "kind": "vector",
                    "vector": vector,
                    "fields": "content_vector",
                    "k": top,
                }
            ],
            "select": ",".join(
                [
                    "id",
                    "source_scope",
                    "title",
                    "category",
                    "severity",
                    "image_caption",
                    "visual_clues",
                    "observation",
                    "root_cause_hypothesis",
                    "action_checklist",
                    "escalation_rule",
                    "offline_cache_reason",
                ]
            ),
        }
        response = httpx.post(url, headers=self._headers(), json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("value", [])

    def delta_candidates(self, state: DeviceSyncState, top: int = 200) -> list[dict]:
        url = f"{self.config.base_url}/indexes/{self.config.index_name}/docs/search?api-version={API_VERSION}"
        payload = {
            "search": "*",
            "top": top,
            "filter": _delta_filter(state),
            "orderby": "sync_sequence asc",
            "select": ",".join(DELTA_SELECT_FIELDS),
        }
        response = httpx.post(url, headers=self._headers(), json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("value", [])

    def _headers(self) -> dict[str, str]:
        return {"api-key": self.config.api_key, "Content-Type": "application/json"}


def _index_schema(index_name: str) -> dict:
    return {
        "name": index_name,
        "fields": [
            {"name": "id", "type": "Edm.String", "key": True, "filterable": True, "sortable": True},
            {"name": "project_id", "type": "Edm.String", "searchable": False, "filterable": True, "facetable": True},
            {"name": "site_id", "type": "Edm.String", "searchable": False, "filterable": True, "facetable": True},
            {"name": "source_scope", "type": "Edm.String", "filterable": True, "facetable": True},
            {"name": "title", "type": "Edm.String", "searchable": True},
            {"name": "category", "type": "Edm.String", "searchable": True, "filterable": True, "facetable": True},
            {"name": "severity", "type": "Edm.String", "searchable": True, "filterable": True, "facetable": True},
            {"name": "image_caption", "type": "Edm.String", "searchable": True},
            {"name": "visual_clues", "type": "Collection(Edm.String)", "searchable": True, "filterable": True},
            {"name": "observation", "type": "Edm.String", "searchable": True},
            {"name": "root_cause_hypothesis", "type": "Edm.String", "searchable": True},
            {"name": "action_checklist", "type": "Collection(Edm.String)", "searchable": True},
            {"name": "escalation_rule", "type": "Edm.String", "searchable": True},
            {"name": "offline_cache_reason", "type": "Edm.String", "searchable": True},
            {"name": "content", "type": "Edm.String", "searchable": True},
            {"name": "sync_sequence", "type": "Edm.Int64", "filterable": True, "sortable": True},
            {"name": "updated_at", "type": "Edm.DateTimeOffset", "filterable": True, "sortable": True},
            {"name": "content_hash", "type": "Edm.String", "searchable": False, "filterable": True},
            {"name": "vector_hash", "type": "Edm.String", "searchable": False, "filterable": True},
            {"name": "thumb_hash", "type": "Edm.String", "searchable": False, "filterable": True},
            {"name": "review_image_hash", "type": "Edm.String", "searchable": False, "filterable": True},
            {"name": "full_asset_hash", "type": "Edm.String", "searchable": False, "filterable": True},
            {"name": "metadata_bytes", "type": "Edm.Int64", "filterable": True, "sortable": True},
            {"name": "vector_bytes", "type": "Edm.Int64", "filterable": True, "sortable": True},
            {"name": "thumb_bytes", "type": "Edm.Int64", "filterable": True, "sortable": True},
            {"name": "review_image_bytes", "type": "Edm.Int64", "filterable": True, "sortable": True},
            {"name": "full_asset_bytes", "type": "Edm.Int64", "filterable": True, "sortable": True},
            {"name": "asset_tiers", "type": "Collection(Edm.String)", "filterable": True, "facetable": True},
            {"name": "asset_manifest_path", "type": "Edm.String", "searchable": False},
            {"name": "safety_critical", "type": "Edm.Boolean", "filterable": True, "facetable": True},
            {"name": "cache_priority", "type": "Edm.Int32", "filterable": True, "sortable": True},
            {"name": "is_deleted", "type": "Edm.Boolean", "filterable": True, "facetable": True},
            {
                "name": "content_vector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "dimensions": VECTOR_DIMENSIONS,
                "vectorSearchProfile": "clip-vector-profile",
            },
        ],
        "vectorSearch": {
            "algorithms": [
                {
                    "name": "clip-hnsw",
                    "kind": "hnsw",
                    "hnswParameters": {"metric": "cosine"},
                }
            ],
            "profiles": [
                {
                    "name": "clip-vector-profile",
                    "algorithm": "clip-hnsw",
                }
            ],
        },
    }


def _delta_filter(state: DeviceSyncState) -> str:
    filters = [
        f"project_id eq '{_escape_odata_string(state.project_id or DEFAULT_PROJECT_ID)}'",
        f"site_id eq '{_escape_odata_string(state.site_id or DEFAULT_SITE_ID)}'",
        f"sync_sequence gt {int(state.last_sync_sequence)}",
    ]
    if not state.include_deleted:
        filters.append("is_deleted eq false")
    return " and ".join(filters)


def _escape_odata_string(value: str) -> str:
    return value.replace("'", "''")
