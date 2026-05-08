import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.core.schemas import (
    FilterResult, FilterStatus,
    ProcessResponse, SummaryResult,
    QAResult, QAPair,
    YouTubeResult, YouTubeVideo,
)
from app.services.filter_service import run_filter_pipeline
from app.services.extractor_service import extract_text
from app.services.summarizer_service import summarize
from app.services.qa_service import generate_qa
from app.services.youtube_service import search_youtube

router = APIRouter()


# ─────────────────────────────────────────
# /filter
# ─────────────────────────────────────────
@router.post(
    "/filter",
    response_model=FilterResult,
    summary="Filter uploaded document",
    description="Validates file type, size, educational content, and safety.",
)
async def filter_document(file: UploadFile = File(...)):
    content = await file.read()
    try:
        extracted_text, word_count = extract_text(content, file.filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not extract text: {str(e)}")

    result = await run_filter_pipeline(
        file=file, content=content,
        extracted_text=extracted_text, word_count=word_count,
    )
    if result.status == FilterStatus.REJECTED:
        raise HTTPException(status_code=400, detail=result.model_dump())
    return result


# ─────────────────────────────────────────
# /process
# ─────────────────────────────────────────
@router.post(
    "/process",
    response_model=ProcessResponse,
    summary="Process document",
    description="""
Full pipeline: filter → summarize + Q&A + YouTube.

**Outputs:**
- `summary`: Academic-style summary of the document
- `key_points`: 5 most important points
- `explanation`: Simple beginner-friendly explanation (no jargon)
- `qa`: Question and answer pairs
- `youtube`: Related educational YouTube videos

**LLM Fallback Chain:**
OpenRouter (Qwen3 480B) → Gemini 2.5 Flash Lite → Gemini 2.5 Flash → Groq Llama
""",
)
async def process_document(
    file: UploadFile = File(...),
    include_summary: bool = Query(default=True, description="Include summary + explanation"),
    include_qa: bool = Query(default=True, description="Include Q&A pairs"),
    include_youtube: bool = Query(default=True, description="Include YouTube videos"),
    num_questions: int = Query(default=5, ge=1, le=10, description="Number of Q&A pairs (1-10)"),
    num_videos: int = Query(default=5, ge=1, le=10, description="Number of YouTube videos (1-10)"),
):
    content = await file.read()

    # Step 1: Extract text
    try:
        extracted_text, word_count = extract_text(content, file.filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not extract text: {str(e)}")

    # Step 2: Filter (Groq)
    filter_result = await run_filter_pipeline(
        file=file, content=content,
        extracted_text=extracted_text, word_count=word_count,
    )
    if filter_result.status == FilterStatus.REJECTED:
        raise HTTPException(status_code=400, detail=filter_result.model_dump())

    topic = filter_result.detected_topic
    response = ProcessResponse(
        word_count=filter_result.word_count,
        detected_topic=topic,
        confidence=filter_result.confidence,
    )

    # Step 3: YouTube runs in background (no LLM, no rate limit)
    youtube_task = None
    if include_youtube:
        youtube_task = asyncio.create_task(
            search_youtube(topic=topic, max_results=num_videos)
        )

    # Step 4: Summary + Explanation (uses LLM fallback chain)
    if include_summary:
        try:
            summary_data = await summarize(text=extracted_text, topic=topic)
            response.summary = SummaryResult(
                summary=summary_data["summary"],
                key_points=summary_data["key_points"],
                explanation=summary_data["explanation"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

    # Step 5: Q&A (uses LLM fallback chain)
    if include_qa:
        try:
            qa_data = await generate_qa(
                text=extracted_text, topic=topic, num_questions=num_questions
            )
            response.qa = QAResult(
                qa_pairs=[QAPair(**pair) for pair in qa_data],
                total=len(qa_data),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Q&A generation failed: {str(e)}")

    # Step 6: Collect YouTube results
    if youtube_task:
        try:
            yt_result = await youtube_task
            query = yt_result[0]["search_query"] if yt_result else topic or ""
            response.youtube = YouTubeResult(
                videos=[YouTubeVideo(**v) for v in yt_result],
                total=len(yt_result),
                search_query=query,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"YouTube search failed: {str(e)}")

    return response
