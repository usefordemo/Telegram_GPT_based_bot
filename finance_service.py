# finance_service.py — market snapshot + strict-source news → analysis
from __future__ import annotations
from typing import Any, Dict, List
from datetime import date, timedelta

import requests
import yfinance as yf
from API import PROXIES, openai_client
from data import (
    stock_sectors,
    crypto_keywords,
    company_to_ticker,
    CHARACTER_DESCRIPTION,
    QUANT_ANALYST_PROMPT,
    macro_topics,
)
from news_service import fetch_general_news, fetch_newsapi_top_news, _clamp_recent_window

# ----------------------- Market data helpers -----------------------
def get_stock_data(tickers: List[str]) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for tkr in tickers:
        try:
            hist = yf.Ticker(tkr).history(period="1d")
            if not hist.empty:
                data[tkr] = {k: (float(v) if hasattr(v, "item") else v) for k, v in hist.iloc[-1].to_dict().items()}
        except Exception as exc:
            print(f"[Stock Fetch Error] {tkr}: {exc}")
    return data

def get_crypto_prices(ids: List[str]) -> Dict[str, Any]:
    if not ids: return {}
    ids_param = ",".join(x.strip().lower().replace(" ", "-") for x in ids)
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids_param}&vs_currencies=usd"
    try:
        r = requests.get(url, proxies=PROXIES, timeout=10); r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"[Crypto Fetch Error] {exc}")
        return {}

# ----------------------- News helpers (strict sources) -----------------------
def _build_boolean_query(max_terms: int = 12) -> str:
    sector_terms = []
    for v in (stock_sectors or {}).values():
        sector_terms.extend(v or [])
    raw_terms = list(dict.fromkeys(sector_terms + list(macro_topics or []) + list(crypto_keywords or [])))
    terms = [t for t in raw_terms if isinstance(t, str) and t.strip()][:max_terms]
    return " OR ".join(f'"{t}"' for t in terms) if terms else "markets OR stocks OR crypto OR rates"

def _fetch_news_for_analysis(max_articles: int = 20) -> List[Dict]:
    q = _build_boolean_query()
    today = date.today()
    start_guess = (today - timedelta(days=14)).isoformat()
    end_guess = today.isoformat()
    frm, to = _clamp_recent_window(start_guess, end_guess)
    print(f"[finance_service] news window: {frm} → {to} | q={q}")
    arts = fetch_general_news(q=q, from_date=frm, to_date=to, max_articles=max_articles, lang="en")
    if not arts:
        print("[finance_service] /everything empty or blocked → fallback to /top-headlines")
        arts = fetch_newsapi_top_news(q=q, lang="en", limit=max_articles)
    return arts or []

# ----------------------- Main analysis entrypoint -----------------------
async def generate_investment_analysis(lang: str = "en", role: str | None = None) -> str:
    all_tickers = list({t for t in (company_to_ticker or {}).values() if isinstance(t, str) and t})
    stocks = get_stock_data(all_tickers)
    crypto_ids = [c for c in (crypto_keywords or []) if isinstance(c, str) and c]
    crypto = get_crypto_prices([c.lower() for c in crypto_ids])
    news_articles = _fetch_news_for_analysis(max_articles=20)
    news_titles = [a.get("title", "") for a in news_articles if a.get("title")]

    system_prompt = role or QUANT_ANALYST_PROMPT or CHARACTER_DESCRIPTION
    user_prompt = (
        "You are a pragmatic sell-side macro/quant analyst.\n"
        "Summarize the following market data (≤800 words) and propose 3–5 actionable trade ideas.\n\n"
        f"Stocks (latest daily OHLCV): {stocks}\n\n"
        f"Crypto (USD): {crypto}\n\n"
        f"News headlines (trusted outlets): {news_titles}\n\n"
        "For each idea, include rationale, key risks, a hedge, and an indicative holding horizon."
    )

    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[Analysis Generation Error] {exc}")
        return "Could not generate the investment analysis at this time."

