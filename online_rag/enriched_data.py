from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EnrichedIncident:
    id: str
    source_scope: str
    title: str
    category: str
    severity: str
    image_caption: str
    visual_clues: list[str]
    observation: str
    root_cause_hypothesis: str
    action_checklist: list[str]
    escalation_rule: str
    offline_cache_reason: str

    @property
    def content(self) -> str:
        return " ".join(
            [
                self.title,
                self.category,
                self.severity,
                self.image_caption,
                " ".join(self.visual_clues),
                self.observation,
                self.root_cause_hypothesis,
                " ".join(self.action_checklist),
                self.escalation_rule,
                self.offline_cache_reason,
            ]
        )

    def to_search_doc(self, vector: list[float]) -> dict:
        doc = asdict(self)
        doc["content"] = self.content
        doc["content_vector"] = vector
        return doc


ENRICHED_INCIDENTS: list[EnrichedIncident] = [
    EnrichedIncident(
        id="INC-001",
        source_scope="offline_seed_enriched",
        title="Basement wall water ingress",
        category="waterproofing",
        severity="medium",
        image_caption="Water staining and seepage lines visible on a basement retaining wall construction joint.",
        visual_clues=["blue seepage line", "retaining wall joint", "water stain", "basement"],
        observation="Water staining and seepage at basement retaining wall construction joint after heavy rain.",
        root_cause_hypothesis="Likely discontinuity at construction joint or local waterstop defect exposed by high groundwater pressure.",
        action_checklist=[
            "Mark seepage path and photograph before repair",
            "Inspect joint, kicker, and waterstop alignment",
            "Inject approved PU grout into active seepage path",
            "Monitor moisture for 72 hours after repair",
        ],
        escalation_rule="Escalate if seepage is active near electrical services or if structural cracks are visible.",
        offline_cache_reason="Common rainy-season defect; keep in offline pack for basement works.",
    ),
    EnrichedIncident(
        id="INC-002",
        source_scope="offline_seed_enriched",
        title="Concrete column honeycombing",
        category="concrete",
        severity="high",
        image_caption="Column surface shows dark voids, rough exposed aggregate, and honeycombing after formwork removal.",
        visual_clues=["voids", "exposed aggregate", "column", "formwork removal", "repair mortar"],
        observation="Honeycombing and exposed aggregate observed after column formwork removal.",
        root_cause_hypothesis="Potential insufficient vibration, congested reinforcement, or concrete segregation at lower column zone.",
        action_checklist=[
            "Stop covering works and isolate the affected location",
            "Notify QA/QC and structural engineer if depth or rebar exposure is uncertain",
            "Remove loose concrete and measure repair depth",
            "Repair with approved non-shrink mortar after acceptance",
        ],
        escalation_rule="Escalate to structural engineer before concealment or load transfer work.",
        offline_cache_reason="High-frequency concrete quality case that benefits from offline photo matching.",
    ),
    EnrichedIncident(
        id="INC-003",
        source_scope="offline_seed_enriched",
        title="Rebar congestion at beam column joint",
        category="reinforcement",
        severity="high",
        image_caption="Dense reinforcement cage at a beam-column joint leaves limited opening for concrete flow and vibrator access.",
        visual_clues=["dense rebar", "beam column joint", "vibrator access", "concrete flow"],
        observation="Dense reinforcement blocks concrete flow and limits vibrator access at beam column joint.",
        root_cause_hypothesis="Bar arrangement and lap congestion may prevent compaction unless pour method is adjusted.",
        action_checklist=[
            "Pause pour sequence before compaction failure occurs",
            "Review workability, pour direction, and poker vibrator access",
            "Capture photos for engineer review",
            "Use approved sequence or local detailing change only after acceptance",
        ],
        escalation_rule="Escalate before continuing the pour if concrete cannot be compacted properly.",
        offline_cache_reason="Critical pour-time condition where field teams may lack connectivity.",
    ),
    EnrichedIncident(
        id="INC-004",
        source_scope="offline_seed_enriched",
        title="MEP duct and ceiling clash",
        category="coordination",
        severity="medium",
        image_caption="Duct and cable tray occupy the planned false ceiling zone in a corridor.",
        visual_clues=["duct", "cable tray", "false ceiling", "coordination clash", "corridor"],
        observation="Duct and cable tray occupy planned false ceiling zone in corridor.",
        root_cause_hypothesis="Latest coordinated MEP elevation may not match ceiling installation level or shop drawing revision.",
        action_checklist=[
            "Raise BIM coordination issue with photo and gridline",
            "Verify approved ceiling level and MEP elevation",
            "Request coordinated shop drawing revision",
            "Avoid ad-hoc cutting without approval",
        ],
        escalation_rule="Escalate if fire-rated services, access panels, or statutory clearances are affected.",
        offline_cache_reason="Useful offline because ceiling/MEP inspections often occur in enclosed areas with poor signal.",
    ),
    EnrichedIncident(
        id="INC-005",
        source_scope="offline_seed_enriched",
        title="Unsafe open edge near scaffold",
        category="safety",
        severity="critical",
        image_caption="Temporary barrier is missing at an open edge beside scaffold access route.",
        visual_clues=["open edge", "missing guardrail", "scaffold access", "warning signage"],
        observation="Temporary barrier missing at open edge beside scaffold access route.",
        root_cause_hypothesis="Temporary edge protection was removed or not reinstated after access change.",
        action_checklist=[
            "Stop access immediately",
            "Install compliant guardrail or barricade",
            "Place warning signage and update access route",
            "Record corrective action before reopening",
        ],
        escalation_rule="Escalate immediately to safety officer and site supervisor.",
        offline_cache_reason="Critical safety case that must be searchable without network.",
    ),
    EnrichedIncident(
        id="INC-006",
        source_scope="offline_seed_enriched",
        title="Crack near lift core wall",
        category="structure",
        severity="medium",
        image_caption="Hairline crack appears near a lift core wall shortly after concrete pour.",
        visual_clues=["hairline crack", "lift core", "concrete pour", "monitoring gauge"],
        observation="Hairline crack near lift core wall after concrete pour.",
        root_cause_hypothesis="Likely shrinkage or thermal crack; movement and water ingress must be ruled out.",
        action_checklist=[
            "Measure crack width and mark endpoints",
            "Check curing and temperature records",
            "Inspect for water ingress",
            "Start crack monitoring log",
        ],
        escalation_rule="Escalate to structural engineer if crack widens, leaks, or aligns with a critical load path.",
        offline_cache_reason="Recurring structural observation suitable for local evidence pack.",
    ),
    EnrichedIncident(
        id="ONL-007",
        source_scope="online_enriched_only",
        title="Water ingress near temporary electrical riser",
        category="waterproofing safety",
        severity="critical",
        image_caption="Water seepage tracks down a basement wall within splash distance of a temporary electrical riser cabinet.",
        visual_clues=["water seepage", "temporary electrical riser", "basement wall", "wet floor", "critical isolation"],
        observation="Active seepage is close to temporary electrical distribution equipment after heavy rain.",
        root_cause_hypothesis="Joint seepage became a safety-critical issue because temporary electrical equipment was installed inside the wet zone.",
        action_checklist=[
            "Isolate affected electrical circuit through authorized personnel",
            "Barricade wet zone and prevent access",
            "Relocate or protect temporary electrical equipment",
            "Repair seepage path only after electrical risk is controlled",
        ],
        escalation_rule="Escalate immediately to electrical supervisor, safety officer, and site manager.",
        offline_cache_reason="Cache after first online retrieval because it upgrades a generic water-ingress case to a critical safety scenario.",
    ),
    EnrichedIncident(
        id="ONL-008",
        source_scope="online_enriched_only",
        title="Falling object exclusion zone breach",
        category="safety lifting",
        severity="critical",
        image_caption="Materials are stacked above a pedestrian route with no toe board or exclusion barrier below.",
        visual_clues=["falling object", "overhead materials", "pedestrian route", "missing exclusion zone"],
        observation="Workers pass below an area where loose materials are stored near an edge.",
        root_cause_hypothesis="Housekeeping and exclusion-zone controls were not updated after material staging changed.",
        action_checklist=[
            "Stop passage below the drop zone",
            "Remove or secure loose materials",
            "Install toe boards, netting, or exclusion barrier",
            "Brief subcontractors before reopening route",
        ],
        escalation_rule="Escalate immediately if materials are above live access routes.",
        offline_cache_reason="High-severity safety case to cache for repeated toolbox and inspection usage.",
    ),
    EnrichedIncident(
        id="ONL-009",
        source_scope="online_enriched_only",
        title="Concrete spalling at slab edge after striking formwork",
        category="concrete edge repair",
        severity="medium",
        image_caption="Spalled slab edge with chipped concrete and exposed small aggregate after formwork strike.",
        visual_clues=["spalling", "slab edge", "chipped concrete", "formwork strike"],
        observation="Local spalling is visible at slab edge after formwork removal.",
        root_cause_hypothesis="Likely mechanical damage during striking rather than deep structural honeycombing.",
        action_checklist=[
            "Confirm damage depth and whether reinforcement is exposed",
            "Clean loose material",
            "Apply approved edge repair method",
            "Record before/after photos for QA closeout",
        ],
        escalation_rule="Escalate if reinforcement is exposed or slab edge supports facade/anchor loads.",
        offline_cache_reason="Useful distinction from honeycombing; cache when concrete quality queries occur.",
    ),
    EnrichedIncident(
        id="ONL-010",
        source_scope="online_enriched_only",
        title="Temporary works prop deformation",
        category="temporary works",
        severity="critical",
        image_caption="A temporary prop appears bowed with a displaced base plate near a recently loaded slab zone.",
        visual_clues=["temporary prop", "bowing", "base plate displacement", "loaded slab", "temporary works"],
        observation="Temporary support member appears deformed after load was introduced.",
        root_cause_hypothesis="Possible overload, base settlement, or incorrect prop installation sequence.",
        action_checklist=[
            "Stop work and cordon off supported area",
            "Do not adjust the prop without temporary works coordinator approval",
            "Check latest temporary works design and inspection tag",
            "Arrange engineer inspection before loading continues",
        ],
        escalation_rule="Escalate immediately to temporary works coordinator and structural engineer.",
        offline_cache_reason="Cache after online hit because offline base pack has no temporary works cases.",
    ),
    EnrichedIncident(
        id="ONL-011",
        source_scope="online_enriched_only",
        title="Confined space gas alarm during drainage inspection",
        category="confined space safety",
        severity="critical",
        image_caption="Gas detector alarm appears during manhole inspection with worker preparing to enter.",
        visual_clues=["gas detector", "manhole", "confined space", "entry permit", "alarm"],
        observation="Portable gas detector alarmed before confined space entry.",
        root_cause_hypothesis="Atmosphere may be oxygen deficient or contaminated; permit controls must be revalidated.",
        action_checklist=[
            "Stop entry immediately",
            "Ventilate and retest atmosphere",
            "Check permit, rescue plan, and standby person",
            "Resume only after competent person approval",
        ],
        escalation_rule="Escalate to safety officer and confined-space competent person.",
        offline_cache_reason="Critical low-connectivity case for basement and drainage works.",
    ),
    EnrichedIncident(
        id="ONL-012",
        source_scope="online_enriched_only",
        title="Mobile crane lift near overhead service",
        category="lifting electrical safety",
        severity="critical",
        image_caption="Mobile crane boom operates close to an overhead cable or service corridor.",
        visual_clues=["mobile crane", "overhead service", "exclusion zone", "lifting plan"],
        observation="Lifting activity appears too close to overhead service corridor.",
        root_cause_hypothesis="Lift plan exclusion distance or banksman control may not reflect actual site constraint.",
        action_checklist=[
            "Stop lift before boom enters exclusion zone",
            "Verify overhead service status and safe clearance",
            "Review lifting plan and banksman position",
            "Resume only after appointed person approval",
        ],
        escalation_rule="Escalate immediately to lifting supervisor and electrical/service owner.",
        offline_cache_reason="Cache for repeated lifting operations where signal may be unreliable.",
    ),
]


def get_enriched_incidents() -> list[EnrichedIncident]:
    return list(ENRICHED_INCIDENTS)

