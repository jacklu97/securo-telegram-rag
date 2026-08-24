from types import SimpleNamespace

from telegram_rag.config import Settings
from telegram_rag.ingest import _sender_name


def test_validate_reports_all_missing():
    problems = Settings(api_id=0, api_hash="", session_string="", group="", jwt_secret="").validate()
    joined = " ".join(problems)
    assert "TELEGRAM_API_ID" in joined
    assert "TELEGRAM_SESSION_STRING" in joined
    assert "TELEGRAM_GROUP" in joined
    assert "MCP_JWT_SECRET" in joined


def test_validate_ok_when_complete():
    s = Settings(api_id=1, api_hash="h", session_string="s", group="@g", jwt_secret="k")
    assert s.validate() == []


def test_sender_name_prefers_full_name():
    sender = SimpleNamespace(first_name="Ana", last_name="Pérez", username="anap")
    assert _sender_name(sender) == "Ana Pérez"


def test_sender_name_falls_back_to_username():
    sender = SimpleNamespace(first_name=None, last_name=None, username="anap")
    assert _sender_name(sender) == "anap"


def test_sender_name_handles_none():
    assert _sender_name(None) == ""
