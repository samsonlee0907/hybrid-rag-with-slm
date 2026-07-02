# Online-to-offline delta sync flow

This document describes how the construction field copilot should use Azure AI Search to discover useful online cases and sync only the minimum offline delta to mobile devices.

## Design goal

The mobile app must remain useful on construction sites with poor connectivity. When connectivity returns, the app should improve the offline pack without downloading the full online corpus or large media files.

The recommended pattern is:

```text
Azure AI Search
  -> searchable full corpus, vector/hybrid retrieval, sync candidate metadata

Mobile BFF
  -> Entra authorization, security trimming, byte-budget planning, signed manifest

Blob Storage / CDN
  -> compressed metadata/vector packs, thumbnails, review images, optional full assets

Mobile device
  -> hash verification, SQLite/vector upsert, tiered media cache, LRU eviction
```

Azure AI Search is the discovery and ranking index. Blob Storage is the artifact store. The mobile app should not pull large image or PDF payloads directly from Azure AI Search query responses.

## Azure AI Search index fields

The online index should contain both retrieval fields and sync-control fields.

| Field | Type | Purpose |
| --- | --- | --- |
| `id` | `Edm.String` key | Stable incident/case ID. |
| `project_id`, `site_id` | filterable strings | Security trimming and site-specific pack selection. |
| `title`, `category`, `severity`, `image_caption`, `observation`, `action_checklist`, `escalation_rule` | searchable fields | Hybrid keyword and semantic retrieval. |
| `content_vector` | `Collection(Edm.Single)` | 512-d CLIP-compatible vector in this POC. |
| `sync_sequence` | filterable/sortable integer | Monotonic delta cursor. Prefer this over timestamp-only paging. |
| `updated_at` | sortable timestamp | Human-readable audit/debug field. |
| `content_hash`, `vector_hash`, `thumb_hash`, `review_image_hash`, `full_asset_hash` | hash strings | Device checks whether local copies are still current. |
| `metadata_bytes`, `vector_bytes`, `thumb_bytes`, `review_image_bytes`, `full_asset_bytes` | sortable integers | Backend enforces network and storage budgets before returning a manifest. |
| `asset_tiers` | filterable string collection | Available tiers: `metadata`, `thumb512`, `review768`, `full`. |
| `asset_manifest_path` | string | Points to the Blob Storage manifest or pack item path. |
| `safety_critical`, `cache_priority`, `is_deleted` | filterable/sortable fields | Priority staging, safety packs, and tombstones. |

The starter-kit implementation extends `online_rag/azure_search.py` with these fields.

## Query pattern

The mobile app should call a backend endpoint, not Azure AI Search directly. The backend uses managed identity or a service key to query Azure AI Search.

Example delta-candidate query:

```json
{
  "search": "*",
  "filter": "project_id eq 'demo-project' and site_id eq 'demo-site' and sync_sequence gt 6 and is_deleted eq false",
  "orderby": "sync_sequence asc",
  "top": 200,
  "select": "id,title,severity,source_scope,sync_sequence,content_hash,vector_hash,metadata_bytes,vector_bytes,thumb_bytes,review_image_bytes,asset_tiers,asset_manifest_path,safety_critical,cache_priority,is_deleted"
}
```

For search-history-driven staging, the backend can run hybrid/vector search first, then pass only the selected online-only hits into the delta planner:

```json
{
  "search": "water ingress near temporary electrical riser",
  "vectorQueries": [
    {
      "kind": "vector",
      "vector": [],
      "fields": "content_vector",
      "k": 20
    }
  ],
  "filter": "project_id eq 'demo-project' and site_id eq 'demo-site' and source_scope eq 'online_enriched_only'",
  "top": 5,
  "select": "id,title,severity,safety_critical,sync_sequence,content_hash,asset_manifest_path"
}
```

## Device sync algorithm

1. Device sends:
   - `last_sync_sequence`
   - local `id -> content_hash`
   - requested asset tier, such as `metadata`, `thumb512`, or `review768`
   - max byte budget for the current network condition
   - project/site scope
2. Backend queries Azure AI Search for candidates newer than the cursor or candidates selected by a hybrid search.
3. Backend builds a manifest:
   - sort safety-critical and high-priority cases first
   - include metadata and vectors before images
   - select the highest requested asset tier available within the byte budget
   - skip oversized cases and set `has_more = true`
4. Device downloads the manifest items from Blob Storage:
   - metadata/vector pack first
   - thumbnails second
   - review images third
   - original media only on Wi-Fi, charging, or user open
5. Device verifies hashes, upserts local SQLite/vector rows, and advances `last_sync_sequence`.
6. Device keeps tombstones for deleted cases until all clients have advanced past the delete sequence.

The starter BFF exposes this as `POST /sync/delta`:

```json
{
  "device_id": "iphone-demo-01",
  "project_id": "demo-project",
  "site_id": "demo-site",
  "last_sync_sequence": 6,
  "max_bytes": 4000000,
  "requested_asset_tier": "review768",
  "local_hashes": {}
}
```

The response is a manifest containing `items[]`, `total_estimated_bytes`, `next_sync_sequence`, and `has_more`.

## Asset tiers and sizing methodology

Measured in the current demo corpus:

| Tier | Current / recommended planning size |
| --- | ---: |
| Current demo PNG + JSON | about 2.04 MB per case |
| 768px review JPEG + JSON/vector/index overhead | about 150-200 KB per case |
| 512px thumbnail JPEG + JSON/vector/index overhead | about 80-110 KB per case |
| Metadata + vectors only | about 10-25 KB per case |

For mobile planning, use **200 KB/case** for a practical `review768` tier. That gives roughly **10,000 cases in a 2 GB case-pack envelope**. The current PNG files are useful for demo visuals but should not be used as the production sync payload.

## Bandwidth controls

- **Manifest-first:** return IDs, hashes, byte sizes, and Blob paths before assets.
- **Budget-aware:** cap each sync request by network type, for example 2 MB cellular background, 20 MB manual cellular, 200 MB Wi-Fi.
- **Tiered assets:** sync metadata/vector first, then thumbnail/review images, then full media.
- **Resumable downloads:** use content hashes, ETags, and chunk/range support for larger packs.
- **Priority queues:** safety-critical, current site, current trade, and search-history misses stage first.
- **Eviction:** keep pinned safety packs; evict low-priority review images before metadata/vectors.

## Starter-kit implementation map

| File | Role |
| --- | --- |
| `online_rag/delta_sync.py` | Device state, delta candidate model, manifest planner, byte-budget selection, hash helpers. |
| `online_rag/azure_search.py` | Azure AI Search schema extensions and delta-candidate query. |
| `online_rag/sync_store.py` | Local SQLite synced-case store with sync sequence and content hash tracking. |
| `cloud_api/main.py` | Mobile BFF route `POST /sync/delta`; keeps Azure AI Search credentials off the device. |
| `scripts/build_online_index.py` | Uploads enriched incidents with sync metadata into Azure AI Search. |
| `scripts/build_delta_sync_manifest.py` | Offline review/demo script that builds a manifest from local corpus assets without Azure credentials. |

## Official references

- Azure AI Search vector search overview: <https://learn.microsoft.com/azure/search/vector-search-overview>
- Create an Azure AI Search index: <https://learn.microsoft.com/azure/search/search-how-to-create-search-index>
- Shape search results with `select`, `top`, and ordering: <https://learn.microsoft.com/azure/search/search-pagination-page-layout>
- OData filters in Azure AI Search: <https://learn.microsoft.com/azure/search/search-query-odata-filter>
