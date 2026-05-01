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
@router.post("/filter", response_model=FilterResult)
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
@router.post("/process", response_model=ProcessResponse)
async def process_document(
    file: UploadFile = File(...),
    include_summary: bool = Query(default=True),
    include_qa: bool = Query(default=True),
    include_youtube: bool = Query(default=True),
    num_questions: int = Query(default=5, ge=1, le=10),
    num_videos: int = Query(default=5, ge=1, le=10),
):
    content = await file.read()

    # Extract
    try:
        extracted_text, word_count = extract_text(content, file.filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not extract text: {str(e)}")

    # Filter
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

    # YouTube مش بيستخدم Gemini — يشتغل بالـ parallel مع الباقي
    # Summary و Q&A بيشتغلوا sequential عشان الـ Gemini rate limit
    if include_youtube:
        youtube_task = asyncio.create_task(
            search_youtube(topic=topic, max_results=num_videos)
        )

    # Summary أول
    if include_summary:
        try:
            summary_data = await summarize(text=extracted_text, topic=topic)
            response.summary = SummaryResult(**summary_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

    # Q&A تاني (بعد ما summary خلص — Gemini rate limit)
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

    # YouTube نجيب نتيجته (كان بيشتغل في الـ background)
    if include_youtube:
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
