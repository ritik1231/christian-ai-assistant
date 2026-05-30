from __future__ import annotations
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message:      str = Field(..., min_length=1, max_length=4000)
    session_id:   str | None = None
    denomination: str | None = None


class VerseSource(BaseModel):
    ref:   str
    text:  str
    score: float


class ChatResponse(BaseModel):
    reply:          str
    sources:        list[VerseSource]
    unverified:     list[str]
    session_id:     str
    denomination:   str
    provider:       str
    blocked:        bool
    block_category: str
    output_flagged: bool
    is_image:       bool
    image_url:      str | None


class ImageRequest(BaseModel):
    prompt:     str = Field(..., min_length=1, max_length=1000)
    session_id: str | None = None


class ImageResponse(BaseModel):
    success:          bool
    image_url:        str | None
    sanitised_prompt: str
    block_reason:     str
    message:          str
    session_id:       str


class HealthResponse(BaseModel):
    status:  str
    model:   str
    version: str


class SessionClearResponse(BaseModel):
    cleared:    bool
    session_id: str
