import os
from requests import Session
from openai import OpenAI
from openai import AsyncOpenAI

PROXIES = {}

OPENAI_API_KEY = 'YOUR_API_KEY'
TELEGRAM_BOT_TOKEN = 'YOUR_API_KEY'
NEWSAPI_KEY = 'YOUR_API_KEY'
GNEWS_API_KEY = 'YOUR_API_KEY'
ADMIN_USER_ID = 'YOUR_API_KEY'
DAILY_DIGEST_ANALYST = "channel_id"

import yfinance.shared as yf_shared
_proxy_session = Session(); _proxy_session.proxies = PROXIES
yf_shared._requests = _proxy_session

# ── OpenAI client ────────────────────────────────────
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
