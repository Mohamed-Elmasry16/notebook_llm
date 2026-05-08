"""
LLM Client with automatic fallback chain:
1. OpenRouter (Qwen3 Coder 480B) — Primary, free
2. Gemini 2.5 Flash Lite          — Backup 1
3. Gemini 2.5 Flash               — Backup 2  
4. Groq Llama 3.3 70B             — Backup 3 (last resort)
"""
import asyncio
import httpx
import google.generativeai as genai
from groq import Groq
from app.core.config import settings

# Initialize clients
genai.configure(api_key=settings.GEMINI_API_KEY)
groq_client = Groq(api_key=settings.GROQ_API_KEY)


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
        return data["choices"][0]["message"]["content"].strip()


async def _call_gemini_model(prompt: str, model_name: str) -> str:
    """Gemini backup call with given model name."""
    model = genai.GenerativeModel(model_name)
    response = await asyncio.to_thread(model.generate_content, prompt)
    return response.text.strip()


def _call_groq_backup(prompt: str) -> str:
    """Last resort: Groq Llama 3.3 70B."""
    response = groq_client.chat.completions.create(
        model=settings.CLASSIFIER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


async def call_llm(prompt: str) -> str:
    """
    Tries each provider in order, falls back automatically on failure.
    Order: OpenRouter → Gemini Backup 1 → Gemini Backup 2 → Groq
    """
    providers = [
        ("OpenRouter (Qwen3 480B)", lambda: _call_openrouter(prompt)),
        ("Gemini 2.5 Flash Lite",   lambda: _call_gemini_model(prompt, settings.GEMINI_BACKUP_1)),
        ("Gemini 2.5 Flash",        lambda: _call_gemini_model(prompt, settings.GEMINI_BACKUP_2)),
        ("Groq Llama 3.3",          lambda: asyncio.to_thread(_call_groq_backup, prompt)),
    ]

    last_error = None
    for name, call in providers:
        try:
            result = await call()
            return result
        except Exception as e:
            last_error = e
            print(f"[LLM Fallback] {name} failed: {e} — trying next...")
            continue

    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
