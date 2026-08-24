"""Entry point: HTTP MCP server + Telegram ingester on one event loop."""
from __future__ import annotations

import asyncio
import logging

import uvicorn

from .config import settings
from .embedder import Embedder
from .ingest import Ingester
from .server import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("telegram_rag")


async def _run() -> None:
    missing = settings.validate()
    ingest_status: dict[str, str] = {"status": "starting"}

    from .store import Store

    store = Store(settings.db_path)
    embedder = Embedder(settings.embed_model)
    app = create_app(store, embedder, ingest_status)

    server = uvicorn.Server(uvicorn.Config(app, host=settings.host, port=settings.port, log_level="info"))

    async def ingest_task() -> None:
        if missing:
            # Serve searches over whatever is already in the DB, but say
            # loudly why nothing new is arriving.
            ingest_status["status"] = f"disabled — missing env: {', '.join(missing)}"
            logger.error(ingest_status["status"])
            return
        ingester = Ingester(store, embedder, ingest_status)
        while True:
            try:
                ingester.status = "connecting"
                await ingester.run()
            except Exception:  # noqa: BLE001
                logger.exception("ingester crashed; retrying in 60s")
                ingest_status["status"] = "crashed — retrying"
            await asyncio.sleep(60)

    await asyncio.gather(server.serve(), ingest_task())


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
