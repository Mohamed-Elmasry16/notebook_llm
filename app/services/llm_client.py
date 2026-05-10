"""
LLM Client for Processing (Summarizer + Q&A)
---------------------------------------------
Fallback chain — Groq intentionally EXCLUDED:
Groq is reserved for the Filter Layer only.
Using it here too causes TPM rate limit errors.

Chain:
1. OpenRouter (Qwen3 Coder 480B) — Primary, free
2. Gemini 2.5 Flash Lite          — Backup 1
3. Gemini 2.5 Flash               — Backup 2

On 429 (rate limit): immediately moves to next provider.
On other errors: also moves to next provider.
"""
import asyncio
import httpx
import google.generativeai as genai
from app.core.config import settings

genai.configure(api_key=settings.GEMINI_API_KEY)


async def _call_openrouter(prompt: str) -> str:
    """Primary: OpenRouter — Qwen3 Coder 480B (free)."""
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://notebook-llm.app",
        "X-Title": "Notebook LLM",
    }
    payload = {
        "model": settings.OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{settings.OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        # OpenRouter sometimes returns empty content
        content = data["choices"][0]["message"]["content"]
        if not content or not content.strip():
            raise ValueError("OpenRouter returned empty response")

        return content.strip()


async def _call_gemini_model(prompt: str, model_name: str) -> str:
    """Gemini backup call."""
    model = genai.GenerativeModel(model_name)
    response = await asyncio.to_thread(model.generate_content, prompt)
    text = response.text.strip()
    if not text:
        raise ValueError(f"Gemini {model_name} returned empty response")
    return text


def _is_rate_limit_error(e: Exception) -> bool:
    """Detects 429 rate limit errors from any provider."""
    msg = str(e).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg or "quota" in msg


async def call_llm(prompt: str) -> str:
    """
    Tries each provider in order.
    On 429 or any error → immediately moves to next provider.
    Groq is NOT in this chain (reserved for filter layer).
    """
    providers = [
        ("OpenRouter (Qwen3 480B)", lambda: _call_openrouter(prompt)),
        ("Gemini 2.5 Flash Lite",   lambda: _call_gemini_model(prompt, settings.GEMINI_BACKUP_1)),
        ("Gemini 2.5 Flash",        lambda: _call_gemini_model(prompt, settings.GEMINI_BACKUP_2)),
    ]

    last_error = None
    for name, call in providers:
        try:
            result = await call()
            print(f"[LLM] Success: {name}")
            return result
        except Exception as e:
            last_error = e
            reason = "rate limit" if _is_rate_limit_error(e) else "error"
            print(f"[LLM Fallback] {name} failed ({reason}): {e} — trying next...")
            continue

    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
