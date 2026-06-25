from __future__ import annotations

import json
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Incident:
    incident_id: str
    title: str
    category: str
    severity: str
    image_file: str
    observation: str
    recommended_action: str
    escalation: str
    image_caption: str = ""
    visual_clues: list[str] = field(default_factory=list)
    root_cause_hypothesis: str = ""
    offline_cache_reason: str = ""
    source_scope: str = "offline_seed"

    @property
    def searchable_text(self) -> str:
        return " ".join(
            [
                self.title,
                self.category,
                self.severity,
                self.observation,
                self.image_caption,
                " ".join(self.visual_clues),
                self.root_cause_hypothesis,
                self.recommended_action,
                self.escalation,
                self.offline_cache_reason,
            ]
        )


SCENARIOS: list[Incident] = [
    Incident(
        incident_id="INC-001",
        title="Basement wall water ingress",
        category="waterproofing",
        severity="medium",
        image_file="inc_001_water_ingress.png",
        observation="Water staining and seepage at basement retaining wall construction joint after heavy rain.",
        recommended_action="Inspect joint and waterstop, mark seepage path, inject approved PU grout, and monitor moisture for 72 hours.",
        escalation="Escalate if seepage is active near electrical services or if structural cracks are visible.",
    ),
    Incident(
        incident_id="INC-002",
        title="Concrete column honeycombing",
        category="concrete",
        severity="high",
        image_file="inc_002_honeycombing.png",
        observation="Honeycombing and exposed aggregate observed after column formwork removal.",
        recommended_action="Stop covering works, notify QA/QC, remove loose concrete, assess depth and rebar exposure, and repair with approved mortar.",
        escalation="Escalate to structural engineer before any concealment or load transfer work.",
    ),
    Incident(
        incident_id="INC-003",
        title="Rebar congestion at beam column joint",
        category="reinforcement",
        severity="high",
        image_file="inc_003_rebar_congestion.png",
        observation="Dense reinforcement blocks concrete flow and limits vibrator access at beam column joint.",
        recommended_action="Pause pour sequence, review workability and vibrator access, capture photos, and obtain engineer direction.",
        escalation="Escalate before continuing the pour if concrete cannot be compacted properly.",
    ),
    Incident(
        incident_id="INC-004",
        title="MEP duct and ceiling clash",
        category="coordination",
        severity="medium",
        image_file="inc_004_mep_clash.png",
        observation="Duct and cable tray occupy planned false ceiling zone in corridor.",
        recommended_action="Raise BIM coordination issue, verify approved ceiling level, and request coordinated shop drawing revision.",
        escalation="Escalate if fire-rated services, access panels, or statutory clearances are affected.",
    ),
    Incident(
        incident_id="INC-005",
        title="Unsafe open edge near scaffold",
        category="safety",
        severity="critical",
        image_file="inc_005_open_edge.png",
        observation="Temporary barrier missing at open edge beside scaffold access route.",
        recommended_action="Stop access, install compliant guardrail or barricade, place warning signage, and record corrective action.",
        escalation="Escalate immediately to safety officer and site supervisor.",
    ),
    Incident(
        incident_id="INC-006",
        title="Crack near lift core wall",
        category="structure",
        severity="medium",
        image_file="inc_006_lift_core_crack.png",
        observation="Hairline crack observed near lift core wall shortly after concrete pour.",
        recommended_action="Measure crack width, check curing and temperature records, inspect water ingress, and start monitoring log.",
        escalation="Escalate to structural engineer if crack widens, leaks, or aligns with critical load path.",
    ),
]


def generate_dataset(
    output_dir: str,
    incidents: Iterable[Incident] | None = None,
    *,
    render_images: bool = True,
) -> tuple[Path, list[Incident]]:
    root = Path(output_dir)
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    incidents_path = root / "incidents.jsonl"
    records = list(incidents or SCENARIOS)

    if render_images:
        for incident in records:
            _draw_incident_image(image_dir / incident.image_file, incident)

    with incidents_path.open("w", encoding="utf-8") as handle:
        for incident in records:
            handle.write(json.dumps(asdict(incident), ensure_ascii=False) + "\n")

    return incidents_path, records


def load_incidents(path: str) -> list[Incident]:
    incidents: list[Incident] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                incidents.append(Incident(**json.loads(line)))
    return incidents


def _draw_incident_image(path: Path, incident: Incident) -> None:
    theme = _theme(incident)
    image = Image.new("RGB", (640, 420), _palette(theme)[0])
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default()
    label_font = ImageFont.load_default()
    bg, accent = _palette(theme)

    draw.rectangle([0, 0, 640, 70], fill=accent)
    draw.text((24, 22), incident.title.upper(), fill=(255, 255, 255), font=title_font)
    draw.text((24, 48), f"{incident.category} | severity: {incident.severity}", fill=(255, 255, 255), font=label_font)

    if theme == "waterproofing":
        draw.rectangle([120, 120, 520, 340], outline=(80, 80, 80), width=6)
        draw.line([320, 120, 320, 340], fill=(30, 90, 180), width=7)
        draw.line([320, 210, 250, 250, 220, 320], fill=(30, 90, 180), width=5)
        draw.text((350, 210), "water seepage", fill=(0, 0, 80), font=label_font)
    elif theme == "electrical-water":
        draw.rectangle([95, 115, 365, 340], outline=(80, 80, 80), width=5)
        draw.line([225, 115, 225, 340], fill=(30, 90, 180), width=7)
        draw.rectangle([405, 145, 540, 305], fill=(235, 235, 210), outline=(70, 70, 70), width=4)
        draw.line([425, 185, 520, 185], fill=(230, 180, 0), width=5)
        draw.polygon([(465, 205), (505, 265), (425, 265)], fill=(250, 220, 0), outline=(120, 80, 0))
        draw.text((434, 238), "LIVE", fill=(0, 0, 0), font=label_font)
    elif theme == "concrete":
        draw.rectangle([210, 105, 430, 350], fill=(165, 165, 165), outline=(80, 80, 80), width=5)
        for x, y in [(250, 185), (310, 220), (360, 180), (285, 285), (375, 300)]:
            draw.ellipse([x, y, x + 38, y + 24], fill=(65, 65, 65))
        draw.text((230, 360), "honeycombing / voids", fill=(0, 0, 0), font=label_font)
    elif theme == "spalling":
        draw.rectangle([120, 125, 520, 285], fill=(175, 175, 175), outline=(80, 80, 80), width=5)
        draw.polygon([(125, 285), (260, 285), (245, 330), (135, 345)], fill=(120, 120, 120), outline=(70, 70, 70))
        draw.line([160, 295, 245, 318], fill=(90, 90, 90), width=3)
        draw.text((275, 315), "slab edge spalling", fill=(0, 0, 0), font=label_font)
    elif theme == "reinforcement":
        for x in range(120, 520, 34):
            draw.line([x, 110, x + 80, 345], fill=(120, 60, 30), width=6)
        for y in range(145, 330, 35):
            draw.line([110, y, 540, y], fill=(120, 60, 30), width=6)
        draw.rectangle([230, 180, 410, 270], outline=(200, 0, 0), width=5)
        draw.text((225, 285), "congested joint", fill=(120, 0, 0), font=label_font)
    elif theme == "coordination":
        draw.rectangle([90, 120, 560, 320], fill=(230, 230, 230), outline=(100, 100, 100), width=4)
        draw.rectangle([105, 160, 430, 215], fill=(120, 150, 165), outline=(70, 90, 100), width=3)
        draw.rectangle([290, 205, 535, 245], fill=(220, 170, 70), outline=(120, 80, 20), width=3)
        draw.line([90, 260, 560, 260], fill=(200, 0, 0), width=5)
        draw.text((170, 280), "ceiling clash zone", fill=(130, 0, 0), font=label_font)
    elif theme == "temporary-works":
        draw.rectangle([95, 300, 545, 335], fill=(155, 155, 155), outline=(80, 80, 80), width=3)
        draw.line([230, 300, 250, 125], fill=(85, 85, 85), width=12)
        draw.line([390, 300, 365, 125], fill=(85, 85, 85), width=12)
        draw.rectangle([205, 112, 420, 132], fill=(130, 130, 130))
        draw.ellipse([218, 290, 268, 335], fill=(180, 60, 40), outline=(90, 30, 20), width=3)
        draw.text((285, 245), "deformed prop", fill=(120, 0, 0), font=label_font)
    elif theme == "confined-space":
        draw.ellipse([120, 105, 520, 340], fill=(90, 90, 90), outline=(40, 40, 40), width=8)
        draw.rectangle([260, 155, 380, 285], fill=(235, 235, 200), outline=(70, 70, 70), width=4)
        draw.ellipse([295, 190, 345, 240], fill=(230, 40, 40))
        draw.text((282, 250), "GAS ALARM", fill=(0, 0, 0), font=label_font)
    elif theme == "lifting":
        draw.rectangle([90, 300, 530, 335], fill=(110, 110, 110))
        draw.line([180, 300, 360, 130], fill=(230, 160, 40), width=12)
        draw.line([360, 130, 500, 150], fill=(230, 160, 40), width=8)
        draw.line([500, 150, 500, 235], fill=(50, 50, 50), width=3)
        draw.line([505, 100, 505, 330], fill=(80, 80, 80), width=4)
        draw.text((395, 250), "overhead service", fill=(90, 0, 0), font=label_font)
    elif theme == "falling-object":
        draw.rectangle([70, 285, 570, 335], fill=(120, 120, 120))
        draw.rectangle([210, 120, 470, 165], fill=(140, 100, 60), outline=(80, 50, 20), width=3)
        draw.line([100, 245, 540, 245], fill=(230, 50, 40), width=5)
        draw.text((170, 260), "missing exclusion zone", fill=(120, 0, 0), font=label_font)
    elif theme == "safety":
        draw.rectangle([70, 260, 570, 340], fill=(120, 120, 120))
        draw.rectangle([390, 95, 455, 260], fill=(180, 180, 180), outline=(80, 80, 80), width=4)
        draw.line([70, 250, 570, 250], fill=(230, 180, 0), width=4)
        draw.polygon([(250, 135), (315, 240), (185, 240)], fill=(250, 220, 0), outline=(120, 80, 0))
        draw.text((222, 190), "OPEN EDGE", fill=(0, 0, 0), font=label_font)
    else:
        draw.rectangle([175, 110, 470, 335], fill=(170, 170, 170), outline=(80, 80, 80), width=5)
        draw.line([300, 135, 335, 205, 315, 280], fill=(120, 0, 0), width=5)
        draw.text((350, 210), "crack", fill=(120, 0, 0), font=label_font)

    draw.rectangle([24, 365, 616, 405], fill=bg, outline=accent, width=2)
    footer = incident.image_caption or incident.observation
    wrapped = textwrap.wrap(footer, width=86)[:2]
    for idx, line in enumerate(wrapped):
        draw.text((34, 374 + idx * 15), line, fill=(20, 20, 20), font=label_font)
    image.save(path)


def _theme(incident: Incident) -> str:
    text = " ".join(
        [
            incident.incident_id,
            incident.title,
            incident.category,
            incident.observation,
            incident.image_caption,
            " ".join(incident.visual_clues),
        ]
    ).lower()
    if "electrical" in text and "water" in text:
        return "electrical-water"
    if "temporary works" in text or "prop" in text:
        return "temporary-works"
    if "confined" in text or "gas detector" in text or "manhole" in text:
        return "confined-space"
    if "crane" in text or "lifting" in text or "overhead service" in text:
        return "lifting"
    if "falling object" in text or "drop zone" in text:
        return "falling-object"
    if "spalling" in text or "slab edge" in text:
        return "spalling"
    if "water" in text or "seepage" in text or "waterproof" in text:
        return "waterproofing"
    if "honeycomb" in text or "concrete" in text:
        return "concrete"
    if "rebar" in text or "reinforcement" in text:
        return "reinforcement"
    if "mep" in text or "duct" in text or "coordination" in text:
        return "coordination"
    if "open edge" in text or "guardrail" in text or "safety" in text:
        return "safety"
    if "crack" in text or "structure" in text:
        return "structure"
    return "generic"


def _palette(theme: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    palettes = {
        "waterproofing": ((225, 240, 255), (0, 90, 180)),
        "electrical-water": ((230, 240, 255), (25, 75, 170)),
        "concrete": ((238, 238, 238), (90, 90, 90)),
        "spalling": ((242, 242, 235), (105, 105, 85)),
        "reinforcement": ((247, 235, 220), (150, 75, 30)),
        "coordination": ((245, 245, 230), (180, 120, 25)),
        "temporary-works": ((240, 238, 230), (120, 95, 60)),
        "confined-space": ((230, 235, 235), (45, 70, 80)),
        "lifting": ((235, 240, 245), (210, 145, 35)),
        "falling-object": ((255, 240, 220), (205, 80, 40)),
        "safety": ((255, 244, 210), (210, 70, 30)),
        "structure": ((242, 238, 248), (110, 70, 150)),
    }
    return palettes.get(theme, ((245, 245, 245), (50, 100, 150)))
