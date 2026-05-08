from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ─────────────────────────────────────────
    # Groq — Filter Layer + Processing Backup 3
    # ─────────────────────────────────────────
    GROQ_API_KEY: str = ""
    CLASSIFIER_MODEL: str = "llama-3.3-70b-versatile"

    # ─────────────────────────────────────────
    # OpenRouter — Primary Processing
    # Get key from: openrouter.ai
    # ─────────────────────────────────────────
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "qwen/qwen3-coder-480b-a35b:free"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # ─────────────────────────────────────────
    # Gemini — Backup 1 & 2
    # Get key from: aistudio.google.com
    # ─────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_BACKUP_1: str = "gemini-2.5-flash-lite-preview-06-17"  # Backup 1
    GEMINI_BACKUP_2: str = "gemini-2.5-flash-preview-05-20"        # Backup 2

    # ─────────────────────────────────────────
    # YouTube Data API v3
    # Get key from: console.cloud.google.com
    # ─────────────────────────────────────────
    YOUTUBE_API_KEY: str = ""

    # ─────────────────────────────────────────
    # File limits
    # ─────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 50
    MIN_WORD_COUNT: int = 50
    ALLOWED_EXTENSIONS: set = {"pdf", "docx", "txt"}
    ALLOWED_MIME_TYPES: dict = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "txt": "text/plain",
    }

    class Config:
        env_file = ".env"


settings = Settings()
