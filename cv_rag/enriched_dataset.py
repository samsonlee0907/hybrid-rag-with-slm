from __future__ import annotations

import re

from cv_rag.synthetic_data import Incident
from online_rag.enriched_data import EnrichedIncident


def convert_enriched_incident(incident: EnrichedIncident) -> Incident:
    return Incident(
        incident_id=incident.id,
        title=incident.title,
        category=incident.category,
        severity=incident.severity,
        image_file=_image_file_name(incident.id, incident.title),
        observation=incident.observation,
        recommended_action="; ".join(incident.action_checklist),
        escalation=incident.escalation_rule,
        image_caption=incident.image_caption,
        visual_clues=list(incident.visual_clues),
        root_cause_hypothesis=incident.root_cause_hypothesis,
        offline_cache_reason=incident.offline_cache_reason,
        source_scope=incident.source_scope,
    )


def convert_enriched_incidents(
    incidents: list[EnrichedIncident],
    *,
    source_scope: str | None = None,
    include_ids: set[str] | None = None,
) -> list[Incident]:
    converted = []
    for incident in incidents:
        if source_scope and incident.source_scope != source_scope:
            continue
        if include_ids and incident.id not in include_ids:
            continue
        converted.append(convert_enriched_incident(incident))
    return converted


def _image_file_name(incident_id: str, title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:44]
    return f"{incident_id.lower().replace('-', '_')}_{slug}.png"
