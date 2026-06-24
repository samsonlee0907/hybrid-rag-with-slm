from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

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

    @property
    def searchable_text(self) -> str:
        return " ".join(
            [
                self.title,
                self.category,
                self.severity,
                self.observation,
                self.recommended_action,
                self.escalation,
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


def generate_dataset(output_dir: str) -> tuple[Path, list[Incident]]:
    root = Path(output_dir)
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    incidents_path = root / "incidents.jsonl"

    for incident in SCENARIOS:
        _draw_incident_image(image_dir / incident.image_file, incident)

    with incidents_path.open("w", encoding="utf-8") as handle:
        for incident in SCENARIOS:
            handle.write(json.dumps(asdict(incident), ensure_ascii=False) + "\n")

    return incidents_path, SCENARIOS


def load_incidents(path: str) -> list[Incident]:
    incidents: list[Incident] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                incidents.append(Incident(**json.loads(line)))
    return incidents


def _draw_incident_image(path: Path, incident: Incident) -> None:
    image = Image.new("RGB", (640, 420), _palette(incident.category)[0])
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default()
    label_font = ImageFont.load_default()
    bg, accent = _palette(incident.category)

    draw.rectangle([0, 0, 640, 70], fill=accent)
    draw.text((24, 22), incident.title.upper(), fill=(255, 255, 255), font=title_font)
    draw.text((24, 48), f"{incident.category} | severity: {incident.severity}", fill=(255, 255, 255), font=label_font)

    if incident.category == "waterproofing":
        draw.rectangle([120, 120, 520, 340], outline=(80, 80, 80), width=6)
        draw.line([320, 120, 320, 340], fill=(30, 90, 180), width=7)
        draw.line([320, 210, 250, 250, 220, 320], fill=(30, 90, 180), width=5)
        draw.text((350, 210), "water seepage", fill=(0, 0, 80), font=label_font)
    elif incident.category == "concrete":
        draw.rectangle([210, 105, 430, 350], fill=(165, 165, 165), outline=(80, 80, 80), width=5)
        for x, y in [(250, 185), (310, 220), (360, 180), (285, 285), (375, 300)]:
            draw.ellipse([x, y, x + 38, y + 24], fill=(65, 65, 65))
        draw.text((230, 360), "honeycombing / voids", fill=(0, 0, 0), font=label_font)
    elif incident.category == "reinforcement":
        for x in range(120, 520, 34):
            draw.line([x, 110, x + 80, 345], fill=(120, 60, 30), width=6)
        for y in range(145, 330, 35):
            draw.line([110, y, 540, y], fill=(120, 60, 30), width=6)
        draw.rectangle([230, 180, 410, 270], outline=(200, 0, 0), width=5)
        draw.text((225, 285), "congested joint", fill=(120, 0, 0), font=label_font)
    elif incident.category == "coordination":
        draw.rectangle([90, 120, 560, 320], fill=(230, 230, 230), outline=(100, 100, 100), width=4)
        draw.rectangle([105, 160, 430, 215], fill=(120, 150, 165), outline=(70, 90, 100), width=3)
        draw.rectangle([290, 205, 535, 245], fill=(220, 170, 70), outline=(120, 80, 20), width=3)
        draw.line([90, 260, 560, 260], fill=(200, 0, 0), width=5)
        draw.text((170, 280), "ceiling clash zone", fill=(130, 0, 0), font=label_font)
    elif incident.category == "safety":
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
    draw.text((34, 378), incident.observation[:95], fill=(20, 20, 20), font=label_font)
    image.save(path)


def _palette(category: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    palettes = {
        "waterproofing": ((225, 240, 255), (0, 90, 180)),
        "concrete": ((238, 238, 238), (90, 90, 90)),
        "reinforcement": ((247, 235, 220), (150, 75, 30)),
        "coordination": ((245, 245, 230), (180, 120, 25)),
        "safety": ((255, 244, 210), (210, 70, 30)),
        "structure": ((242, 238, 248), (110, 70, 150)),
    }
    return palettes.get(category, ((245, 245, 245), (50, 100, 150)))

