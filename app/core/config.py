from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ─────────────────────────────────────────
    # Groq — Filter Layer (primary)
    # Get key from: console.groq.com
    # ─────────────────────────────────────────
    GROQ_API_KEY: str = ""
    CLASSIFIER_MODEL: str = "llama-3.3-70b-versatile"

    # ─────────────────────────────────────────
    # OpenRouter — Processing Primary + Filter Backups
    # Get key from: openrouter.ai
    # Models used:
    #   Filter Backup 1 : qwen/qwen3-coder-480b-a35b:free
    #   Filter Backup 2 : z-ai/glm-4.5-air:free
    #   Processing      : qwen/qwen3-coder-480b-a35b:free
    # ─────────────────────────────────────────
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "qwen/qwen3-coder-480b-a35b:free"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # ─────────────────────────────────────────
    # Gemini — Processing Backup 1 & 2
    # Get key from: aistudio.google.com
    # ─────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_BACKUP_1 = "gemini-2.5-flash-lite"
    GEMINI_BACKUP_2 = "gemini-2.5-flash"

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
