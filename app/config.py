from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # === LLM ===
    GROQ_API_KEY: str = "placeholder"

    # === Search ===
    TAVILY_API_KEY: str = "placeholder"

    # === Email recipient ===
    RECIPIENT_EMAIL: str = "user@example.com"
    SENDER_EMAIL: str = "sender@example.com"

    # === Google OAuth (for Gmail + Calendar) ===
    GOOGLE_CREDENTIALS_JSON: Optional[str] = None
    GOOGLE_CREDENTIALS_PATH: Optional[str] = "credentials/google_credentials.json"
    GOOGLE_TOKEN_PATH: str = "credentials/token.json"

    # === Timezone ===
    USER_TIMEZONE: str = "Asia/Kolkata"

    # === Agent behaviour ===
    LOOKBACK_HOURS: int = 24
    MAX_STORIES: int = 8
    MIN_STORIES_BEFORE_SEND: int = 3
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # === Phase 1: PostgreSQL (checkpointing + memory) ===
    # Format: postgresql://user:password@host:5432/dbname
    # On Render: use your Internal Database URL
    DATABASE_URL: str 

    # === Phase 1: Alert agent ===
    # Urgency score (1-10) at or above which a breaking alert email is sent immediately
    ALERT_URGENCY_THRESHOLD: int = 8
    # How often (in hours) the alert agent polls for breaking news
    ALERT_POLL_HOURS: int = 2

    # === Phase 1: Story memory / deduplication ===
    # Cosine similarity threshold - stories above this score are considered duplicates
    SIMILARITY_THRESHOLD: float = 0.85
    # How many days back to check for near-duplicate stories
    MEMORY_LOOKBACK_DAYS: int = 7

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()