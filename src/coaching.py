import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from .models import CoachResponse
from .prompts import SYSTEM_PROMPT, build_user_prompt

FILLER_PATTERNS = [
    r"\bi think\b",
    r"\bkind of\b",
    r"\bsort of\b",
    r"\bmaybe\b",
    r"\bjust\b",
    r"\blike\b",
    r"\byou know\b",
    r"\ba little bit\b",
    r"\bthere are a few things\b",
]


def detect_fillers(text: str) -> dict[str, int]:
    lower_text = text.lower()
    counts: dict[str, int] = {}
    for pattern in FILLER_PATTERNS:
        matches = re.findall(pattern, lower_text)
        if matches:
            counts[pattern.replace(r"\b", "")] = len(matches)
    return counts


def _enforce_closed_objects(schema: dict) -> dict:
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
        for value in schema.values():
            if isinstance(value, dict):
                _enforce_closed_objects(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _enforce_closed_objects(item)
    return schema


def _build_quality_instructions(audience: str, mode: str) -> str:
    audience_map = {
        "CEO": "Focus on speed, decision quality, business risk, priorities, and what leadership should do next",
        "CFO": "Focus on financial tradeoffs, efficiency, operating risk, resource implications, and cost of action versus inaction",
        "CMO": "Focus on growth, performance, measurement, execution risk, and business impact",
        "Board": "Focus on strategic implications, risk, governance, and confidence in the recommendation",
        "Peer Leader": "Focus on alignment, execution, accountability, and cross-functional implications",
        "Recruiter": "Focus on clarity, seniority, business relevance, and concise executive framing",
    }

    mode_map = {
        "Reframe": "Prioritize a sharper executive rewrite with the cleanest possible framing",
        "Practice": "Prioritize a natural spoken answer that sounds polished and ready to say aloud",
        "Challenge": "Make the leadership challenge tougher and more realistic for the audience",
        "Polish": "Tighten wording, remove generic phrases, and make the answer more confident and direct",
    }

    return (
        f"Audience guidance: {audience_map.get(audience, 'Keep the answer leadership-ready and business-focused')}\n"
        f"Mode guidance: {mode_map.get(mode, 'Provide a strong executive response')}\n"
        "Quality bar:\n"
        "- Headline should be specific, not generic\n"
        "- Recommendation should be action-oriented and tied to the actual issue\n"
        "- Supporting points should avoid clichés and stay close to the transcript\n"
        "- Stronger closing line should sound decisive, not vague\n"
        "- Leadership challenge should be realistic for the stated audience\n"
        "- Scores should be meaningful, not inflated by default\n"
        "- Penalize generic, repetitive, or overly polished wording that loses the user's original meaning"
    )


def build_coach_response(
    client: OpenAI,
    transcript: str,
    audience: str,
    tone: str,
    response_length: str,
    mode: str,
) -> CoachResponse:
    schema = CoachResponse.model_json_schema()
    schema = _enforce_closed_objects(schema)

    user_prompt = build_user_prompt(
        transcript=transcript,
        audience=audience,
        tone=tone,
        response_length=response_length,
        mode=mode,
    )

    quality_instructions = _build_quality_instructions(audience, mode)

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{user_prompt}\n\nAdditional quality instructions:\n{quality_instructions}",
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "executive_coach_response",
                "schema": schema,
                "strict": True,
            }
        },
    )

    content = response.output_text
    return CoachResponse.model_validate_json(content)


def save_session(session_payload: dict[str, Any], data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(data_dir.glob("session_*.json"))
    next_number = len(existing) + 1
    session_path = data_dir / f"session_{next_number:03d}.json"
    session_path.write_text(json.dumps(session_payload, indent=2), encoding="utf-8")
    return session_path