"""One-time interactive login: prints the StringSession to put in
TELEGRAM_SESSION_STRING on the server. Run locally:

    pip install telethon
    python scripts/telegram_login.py

Get api_id/api_hash at https://my.telegram.org -> API development tools.
The session grants access AS YOUR ACCOUNT — treat it like a password.
"""
import getpass

from telethon.sync import TelegramClient
from telethon.sessions import StringSession

api_id = int(input("api_id: ").strip())
api_hash = getpass.getpass("api_hash: ").strip()

with TelegramClient(StringSession(), api_id, api_hash) as client:
    print("\nTELEGRAM_SESSION_STRING (copy the whole line):\n")
    print(client.session.save())
