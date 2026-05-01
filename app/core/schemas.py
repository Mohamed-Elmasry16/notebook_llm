from pydantic import BaseModel
from enum import Enum


# ─────────────────────────────────────────
# Filter schemas
# ─────────────────────────────────────────
class FilterStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class RejectionReason(str, Enum):
    INVALID_TYPE = "invalid_file_type"
    TOO_LARGE = "file_too_large"
    TOO_SHORT = "content_too_short"
    NOT_EDUCATIONAL = "not_educational_content"
    UNSAFE_CONTENT = "unsafe_content"


class FilterResult(BaseModel):
    status: FilterStatus
    reason: RejectionReason | None = None
    message: str
    word_count: int | None = None
    detected_topic: str | None = None
    confidence: float | None = None


# ─────────────────────────────────────────
# Summary schemas
# ─────────────────────────────────────────
class SummaryResult(BaseModel):
    summary: str
    key_points: list[str]


# ─────────────────────────────────────────
# Q&A schemas
# ─────────────────────────────────────────
class QAPair(BaseModel):
    question: str
    answer: str


class QAResult(BaseModel):
    qa_pairs: list[QAPair]
    total: int


# ─────────────────────────────────────────
# YouTube schemas (new)
# ─────────────────────────────────────────
class YouTubeVideo(BaseModel):
    title: str
    url: str
    thumbnail: str
    channel: str
    description: str
    search_query: str


class YouTubeResult(BaseModel):
    videos: list[YouTubeVideo]
    total: int
    search_query: str


# ─────────────────────────────────────────
# Final process response — all outputs
# ─────────────────────────────────────────
class ProcessResponse(BaseModel):
    # Filter info
    word_count: int | None = None
    detected_topic: str | None = None
    confidence: float | None = None
    # Outputs
    summary: SummaryResult | None = None
    qa: QAResult | None = None
    youtube: YouTubeResult | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str
