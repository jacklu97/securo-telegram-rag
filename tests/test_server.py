"""JSON-RPC contract tests, exercised exactly the way securo's MCPClient
calls the endpoint (Bearer HS256 JWT, aud securo-mcp)."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from jose import jwt

from telegram_rag.server import create_app

SECRET = "test-secret"


def _token(*, aud: str = "securo-mcp", secret: str = SECRET, ttl: int = 300) -> str:
    now = int(time.time())
    return jwt.encode(
        {"sub": "u1", "iss": "securo-backend", "aud": aud, "iat": now, "exp": now + ttl},
        secret,
        algorithm="HS256",
    )


@pytest.fixture
def client(store, embedder):
    seeds = [
        ("BBVA tiene 20% de bonificación en restaurantes", None),
        ("Banorte lanza 12 MSI en Amazon", None),
        ("Promo BBVA vieja del año pasado", datetime.now(timezone.utc) - timedelta(days=400)),
    ]
    for i, (text, date) in enumerate(seeds):
        [vec] = embedder.embed([text])
        store.add(
            chat_id=1, msg_id=i + 1, date=date or datetime.now(timezone.utc),
            sender="tester", text=text, embedding=vec,
        )
    app = create_app(store, embedder, {"status": "test"})
    return TestClient(app)


def _rpc(client, method, params=None, *, token=None, rpc_id=7):
    headers = {"Authorization": f"Bearer {token or _token()}"}
    body = {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params or {}}
    return client.post("/mcp", json=body, headers=headers).json()


def test_rejects_missing_token(client):
    res = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    assert res.json()["error"]["code"] == -32001


def test_rejects_wrong_secret(client):
    res = _rpc(client, "tools/list", token=_token(secret="other-secret"))
    assert res["error"]["code"] == -32001


def test_rejects_wrong_audience(client):
    res = _rpc(client, "tools/list", token=_token(aud="not-securo"))
    assert res["error"]["code"] == -32001


def test_rejects_expired_token(client):
    res = _rpc(client, "tools/list", token=_token(ttl=-10))
    assert res["error"]["code"] == -32001


def test_initialize(client):
    res = _rpc(client, "initialize")
    assert res["result"]["serverInfo"]["name"] == "securo-telegram-rag"
    assert "tools" in res["result"]["capabilities"]


def test_tools_list(client):
    res = _rpc(client, "tools/list")
    names = {t["name"] for t in res["result"]["tools"]}
    assert names == {"search_group_messages", "group_stats"}
    for tool in res["result"]["tools"]:
        assert tool["inputSchema"]["type"] == "object"


def test_unknown_method(client):
    res = _rpc(client, "resources/list")
    assert res["error"]["code"] == -32601


def test_unknown_tool(client):
    res = _rpc(client, "tools/call", {"name": "nope", "arguments": {}})
    assert res["error"]["code"] == -32602


def test_search_returns_structured_and_text(client):
    res = _rpc(client, "tools/call", {
        "name": "search_group_messages",
        "arguments": {"query": "promociones BBVA", "limit": 2},
    })
    result = res["result"]
    assert result["isError"] is False
    hits = result["structuredContent"]["results"]
    assert len(hits) == 2
    assert "BBVA" in hits[0]["text"]
    assert {"date", "sender", "text", "score"} <= set(hits[0])
    # text content mirrors the hits for providers without structured support
    assert "BBVA" in result["content"][0]["text"]


def test_search_since_days_excludes_old(client):
    res = _rpc(client, "tools/call", {
        "name": "search_group_messages",
        "arguments": {"query": "BBVA", "since_days": 60, "limit": 10},
    })
    texts = [h["text"] for h in res["result"]["structuredContent"]["results"]]
    assert any("bonificación" in t for t in texts)
    assert not any("vieja" in t for t in texts)


def test_search_limit_clamped(client):
    res = _rpc(client, "tools/call", {
        "name": "search_group_messages",
        "arguments": {"query": "BBVA", "limit": 999},
    })
    assert len(res["result"]["structuredContent"]["results"]) <= 20


def test_search_empty_query_is_error(client):
    res = _rpc(client, "tools/call", {
        "name": "search_group_messages",
        "arguments": {"query": "  "},
    })
    assert res["result"]["isError"] is True


def test_group_stats(client):
    res = _rpc(client, "tools/call", {"name": "group_stats", "arguments": {}})
    data = res["result"]["structuredContent"]
    assert data["indexed_messages"] == 3
    assert data["latest_message"]


def test_health_open_without_auth(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["indexed"] == 3
