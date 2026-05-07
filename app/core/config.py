from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Groq — Filter Layer (free)
    GROQ_API_KEY: str = "put your groq api key here "
    CLASSIFIER_MODEL: str = "llama-3.3-70b-versatile"

    # Gemini — Processing Layer (free tier)
    GEMINI_API_KEY: str = "put your gimni api key here"
    PROCESSING_MODEL: str = "gemma-3-27b"

    # YouTube Data API v3 (free)
    YOUTUBE_API_KEY: str = "put your youtube api key here"

    # File limits
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
