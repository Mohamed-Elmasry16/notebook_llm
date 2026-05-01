import httpx
from app.core.config import settings


# ─────────────────────────────────────────
# Keyword Extractor
# ─────────────────────────────────────────
def _build_search_query(topic: str | None, summary: str | None) -> str:
    """
    Builds the best YouTube search query.
    Priority: topic first, fallback to first words of summary.
    """
    if topic:
        return topic

    if summary:
        words = summary.split()[:10]
        return " ".join(words)

    return "educational content"


# ─────────────────────────────────────────
# YouTube Search
# ─────────────────────────────────────────
async def search_youtube(
    topic: str | None = None,
    summary: str | None = None,
    max_results: int = 5,
) -> list[dict]:
    """
    Searches YouTube Data API v3 for videos related to the document topic.
    Returns list of videos with title, url, thumbnail, channel, description.
    """
    query = _build_search_query(topic, summary)

    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "relevanceLanguage": "en",
        "safeSearch": "strict",
        "key": settings.YOUTUBE_API_KEY,
    }

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            "https://www.googleapis.com/youtube/v3/search",
            params=params,
        )
        response.raise_for_status()
        data = response.json()

    return _parse_results(data, query)


def _parse_results(data: dict, query: str) -> list[dict]:
    """Parses YouTube API response into clean list of video dicts."""
    videos = []

    for item in data.get("items", []):
        snippet = item.get("snippet", {})
        video_id = item.get("id", {}).get("videoId", "")

        if not video_id:
            continue

        videos.append({
            "title": snippet.get("title", ""),
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
            "channel": snippet.get("channelTitle", ""),
            "description": snippet.get("description", "")[:200],
            "search_query": query,
        })

    return videos
