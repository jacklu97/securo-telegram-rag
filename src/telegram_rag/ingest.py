"""Telegram ingestion: one-time history backfill + live subscription.

Runs inside the same asyncio loop as the HTTP server. Uses a user-account
MTProto session (StringSession) because bots cannot read group history —
and the promo knowledge lives in the past messages.
"""
from __future__ import annotations

import logging

from telethon import TelegramClient, events
from telethon.sessions import StringSession

from .config import settings
from .embedder import Embedder
from .store import Store

logger = logging.getLogger(__name__)

_BATCH = 64


def _sender_name(sender) -> str:
    if sender is None:
        return ""
    name = " ".join(p for p in [getattr(sender, "first_name", None), getattr(sender, "last_name", None)] if p)
    return name or getattr(sender, "username", "") or ""


class Ingester:
    def __init__(self, store: Store, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self.client = TelegramClient(
            StringSession(settings.session_string), settings.api_id, settings.api_hash
        )
        self.status = "starting"

    async def run(self) -> None:
        await self.client.connect()
        if not await self.client.is_user_authorized():
            self.status = "session invalid — regenerate with scripts/telegram_login.py"
            logger.error(self.status)
            return

        group = settings.group
        entity = await self.client.get_entity(int(group) if group.lstrip("-").isdigit() else group)
        chat_id = entity.id
        logger.info("indexing group %s (%s)", getattr(entity, "title", group), chat_id)

        await self._backfill(entity, chat_id)
        self.status = f"live ({self.store.count()} messages indexed)"

        @self.client.on(events.NewMessage(chats=entity))
        async def _on_message(event) -> None:  # noqa: ANN001
            text = (event.message.message or "").strip()
            if len(text) < settings.min_message_chars:
                return
            sender = _sender_name(await event.get_sender())
            [vec] = await self.embedder.embed_async([text])
            self.store.add(
                chat_id=chat_id,
                msg_id=event.message.id,
                date=event.message.date,
                sender=sender,
                text=text,
                embedding=vec,
            )
            logger.info("ingested live message %s", event.message.id)

        await self.client.run_until_disconnected()

    async def _backfill(self, entity, chat_id: int) -> None:  # noqa: ANN001
        known = self.store.known_ids(chat_id)
        pending: list[tuple[int, object, str, str]] = []
        scanned = 0
        self.status = "backfilling"

        async def flush() -> None:
            if not pending:
                return
            vecs = await self.embedder.embed_async([p[3] for p in pending])
            for (msg_id, date, sender, text), vec in zip(pending, vecs):
                self.store.add(
                    chat_id=chat_id, msg_id=msg_id, date=date, sender=sender,
                    text=text, embedding=vec,
                )
            pending.clear()

        async for message in self.client.iter_messages(entity, limit=settings.backfill_limit):
            scanned += 1
            if message.id in known:
                continue
            text = (message.message or "").strip()
            if len(text) < settings.min_message_chars:
                continue
            sender = _sender_name(message.sender)
            pending.append((message.id, message.date, sender, text))
            if len(pending) >= _BATCH:
                await flush()
        await flush()
        logger.info("backfill done: scanned %s, corpus now %s", scanned, self.store.count())
