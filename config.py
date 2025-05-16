from __future__ import annotations

import os
from pathlib import Path
from requests import Session
from openai import OpenAI

# ── Proxy settings ────────────────────────────────────────────────
PROXIES: dict[str, str] = {
    "http":  os.getenv("HTTP_PROXY",  "http://127.0.0.1:7890"),
    "https": os.getenv("HTTPS_PROXY", "http://127.0.0.1:7890"),
}

# ── Keys & tokens (read from env) ─────────────────────────────────
OPENAI_API_KEY: str | None     = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN: str | None = os.getenv("TELEGRAM_BOT_TOKEN")
NEWSAPI_KEY: str | None        = os.getenv("NEWSAPI_KEY")
GNEWS_API_KEY: str | None      = os.getenv("GNEWS_API_KEY")

ADMIN_USER_ID: int             = int(os.getenv("ADMIN_USER_ID", "0"))
DAILY_DIGEST_ANALYST: str      = os.getenv("DAILY_DIGEST_ANALYST_CHAT_ID", "")

# ── Apply proxy to yfinance ───────────────────────────────────────
import yfinance.shared as _yf_shared  # type: ignore
_proxy_session = Session()
_proxy_session.proxies = PROXIES
_yf_shared._requests = _proxy_session  # pyright: ignore[reportPrivateUsage]

# ── OpenAI client ─────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)