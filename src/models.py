from pydantic import BaseModel, Field, ConfigDict


class ScoreCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clarity: int = Field(ge=1, le=10)
    concision: int = Field(ge=1, le=10)
    executive_presence: int = Field(ge=1, le=10)
    business_focus: int = Field(ge=1, le=10)
    actionability: int = Field(ge=1, le=10)


class CoachResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    executive_headline: str
    why_it_matters: str
    recommendation: str
    support_points: list[str]
    polished_spoken_version: str
    stronger_closing_line: str
    tough_question: str
    coaching_feedback: list[str]
    scores: ScoreCard