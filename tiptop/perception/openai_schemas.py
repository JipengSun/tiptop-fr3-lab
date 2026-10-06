"""Pydantic schemas for OpenAI Structured Outputs (vision detection)."""

from pydantic import BaseModel, Field


class DetectedBBox(BaseModel):
    """One object on a resolution-independent 1000x1000 grid (ymin, xmin, ymax, xmax)."""

    label: str = Field(description="Unique snake_case object name relevant to the task")
    ymin: int = Field(ge=0, le=1000, description="Top edge; 0=top of image, 1000=bottom")
    xmin: int = Field(ge=0, le=1000, description="Left edge; 0=left of image, 1000=right")
    ymax: int = Field(ge=0, le=1000, description="Bottom edge; must be greater than ymin")
    xmax: int = Field(ge=0, le=1000, description="Right edge; must be greater than xmin")


class GoalPredicate(BaseModel):
    name: str
    args: list[str]


class DetectAndTranslateResponse(BaseModel):
    bboxes: list[DetectedBBox]
    predicates: list[GoalPredicate]


class BboxListResponse(BaseModel):
    """Gripper mask and other single-purpose bbox prompts."""

    bboxes: list[DetectedBBox]
