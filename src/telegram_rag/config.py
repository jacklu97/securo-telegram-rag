"""Environment-driven configuration. Every knob a deployment needs, nothing more."""
import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Telegram (user-account MTProto session; see scripts/telegram_login.py)
    api_id: int = _int("TELEGRAM_API_ID", 0)
    api_hash: str = os.environ.get("TELEGRAM_API_HASH", "")
    session_string: str = os.environ.get("TELEGRAM_SESSION_STRING", "")
    # Group to index: @username, invite title, or numeric id.
    group: str = os.environ.get("TELEGRAM_GROUP", "")
    backfill_limit: int = _int("TELEGRAM_BACKFILL_LIMIT", 5000)

    # MCP auth — must equal securo's AGENTS_MCP_JWT_SECRET so tokens minted by
    # the securo backend verify here.
    jwt_secret: str = os.environ.get("MCP_JWT_SECRET", "")
    jwt_audience: str = "securo-mcp"

    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = _int("PORT", 8900)

    db_path: str = os.environ.get("DB_PATH", "/data/messages.db")
    # Same multilingual model securo's native embedder defaults to.
    embed_model: str = os.environ.get(
        "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    min_message_chars: int = _int("MIN_MESSAGE_CHARS", 12)

    missing: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        problems = []
        if not self.api_id or not self.api_hash:
            problems.append("TELEGRAM_API_ID / TELEGRAM_API_HASH")
        if not self.session_string:
            problems.append("TELEGRAM_SESSION_STRING")
        if not self.group:
            problems.append("TELEGRAM_GROUP")
        if not self.jwt_secret:
            problems.append("MCP_JWT_SECRET")
        return problems


settings = Settings()
