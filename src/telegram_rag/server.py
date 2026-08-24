"""JSON-RPC 2.0 MCP endpoint in the dialect securo's agent runtime speaks.

securo's MCPClient POSTs `tools/list` and `tools/call` with a Bearer JWT
(HS256, aud "securo-mcp") minted by the securo backend using the shared
AGENTS_MCP_JWT_SECRET. We verify with the same secret: only the paired
securo instance can query the corpus.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from jose import JWTError, jwt

from .config import settings
from .embedder import Embedder
from .store import Store

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_group_messages",
        "description": (
            "Search the indexed Telegram finance group. Works for ANY topic: slang or "
            "abbreviations used in the group, product opinions, promotions ('promociones', "
            "'MSI', cashback, descuentos), or community advice. Search directly with the "
            "user's own term — no bank name is required. Add a bank name only when the "
            "question is about that bank's cards. Query in Spanish. Results are already "
            "the best matches in the whole corpus: do NOT repeat similar queries; one "
            "call per distinct topic, then answer with what you have."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in Spanish."},
                "limit": {"type": "integer", "description": "Max results (default 8, max 20)."},
                "since_days": {
                    "type": "integer",
                    "description": "Only messages newer than N days. Promotions expire — use 60-90 for anything time-sensitive.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "group_stats",
        "description": "Corpus status: how many messages are indexed and how fresh they are.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def create_app(store: Store, embedder: Embedder, ingest_status: dict[str, str]) -> FastAPI:
    app = FastAPI()

    def _authorized(request: Request) -> bool:
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return False
        try:
            jwt.decode(
                auth[7:],
                settings.jwt_secret,
                algorithms=["HS256"],
                audience=settings.jwt_audience,
            )
            return True
        except JWTError:
            return False

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"indexed": store.count(), "latest": store.latest_date(), "ingest": ingest_status.get("status", "")}

    @app.post("/mcp")
    async def mcp(request: Request) -> JSONResponse:
        body = await request.json()
        rpc_id = body.get("id")

        def result(payload: Any) -> JSONResponse:
            return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": payload})

        def error(code: int, message: str) -> JSONResponse:
            return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}})

        if not _authorized(request):
            return error(-32001, "unauthorized")

        method = body.get("method", "")
        params = body.get("params") or {}

        if method == "initialize":
            return result({
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "securo-telegram-rag", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            })
        if method.startswith("notifications/"):
            return result({})
        if method == "tools/list":
            return result({"tools": TOOLS})
        if method == "tools/call":
            return await _call_tool(params, store, embedder, result, error)
        return error(-32601, f"method not found: {method}")

    return app


async def _call_tool(params: dict[str, Any], store: Store, embedder: Embedder, result, error):  # noqa: ANN001
    name = params.get("name", "")
    args = params.get("arguments") or {}

    if name == "group_stats":
        data = {"indexed_messages": store.count(), "latest_message": store.latest_date()}
        return result(_tool_result(data, f"{data['indexed_messages']} messages indexed, latest {data['latest_message']}"))

    if name == "search_group_messages":
        query = str(args.get("query") or "").strip()
        if not query:
            return result(_tool_result({"results": []}, "empty query", is_error=True))
        limit = max(1, min(int(args.get("limit") or 8), 20))
        # Small models loop when results are long or repetitive: truncate each
        # message and drop near-duplicates so a round of results stays terse.
        MAX_CHARS = 280
        since = None
        if args.get("since_days"):
            cutoff = datetime.now(timezone.utc) - timedelta(days=int(args["since_days"]))
            since = cutoff.strftime("%Y-%m-%d %H:%M")
        [qvec] = await embedder.embed_async([query])
        hits = store.search(qvec, limit=limit * 2, since=since, query_text=query)
        seen: set[str] = set()
        unique = []
        for h in hits:
            key = h.text[:120].lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(h)
            if len(unique) >= limit:
                break
        data = {
            "results": [
                {
                    "date": h.date,
                    "sender": h.sender,
                    "text": h.text[:MAX_CHARS] + ("…" if len(h.text) > MAX_CHARS else ""),
                    "score": round(h.score, 3),
                }
                for h in unique
            ]
        }
        lines = "\n\n".join(
            f"[{h.date}] {h.sender}: {h.text[:MAX_CHARS]}" for h in unique
        )
        text = (
            lines + "\n\n(These are the best matches in the corpus — answer now, do not search again for the same topic.)"
            if unique
            else "no matches — do not retry the same query"
        )
        return result(_tool_result(data, text))

    return error(-32602, f"unknown tool: {name}")


def _tool_result(data: Any, text: str, *, is_error: bool = False) -> dict[str, Any]:
    # Mirrors securo's built-in server shape: structuredContent + text content.
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": data,
        "isError": is_error,
    }
