from typing import List, Dict, Any
import yfinance as yf, requests, asyncio
from config import PROXIES, openai_client,GNEWS_API_KEY,NEWSAPI_KEY, PROXIES
from data import (stock_sectors, crypto_keywords, company_to_ticker,
                  CHARACTER_DESCRIPTION, QUANT_ANALYST_PROMPT)
from data import macro_topics, crypto_keywords, company_to_ticker, stock_sectors
from news_service import fetch_combined_news


def get_stock_data(tickers: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    try:
        for tkr in tickers:
            hist = yf.Ticker(tkr).history(period="1d")
            if not hist.empty:
                data[tkr] = hist.iloc[-1].to_dict()
    except Exception as exc:
        print(f"[Stock Fetch Error] {exc}")
    return data

def get_crypto_prices(ids: list[str]) -> dict[str, Any]:
    ids_param = ",".join(ids)
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids_param}&vs_currencies=usd"
    try:
        r = requests.get(url, proxies=PROXIES, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"[Crypto Fetch Error] {exc}")
        return {}


# 整合投资分析函数
async def generate_investment_analysis(lang: str = "en", role: str | None = None) -> str:
    # 收集数据
    keywords = sum(stock_sectors.values(), []) + crypto_keywords + macro_topics
    news = fetch_combined_news(keywords, max_articles=20, lang=lang)
    crypto = get_crypto_prices([c.lower() for c in crypto_keywords])
    all_tickers = list(company_to_ticker.values())
    stocks = get_stock_data(all_tickers)

    # 构建 prompt
    if lang == "zh":
        user_prompt = (
            "请用**简体中文**总结以下市场数据（≤2000字），并给出 3–5 条具体交易建议。\n\n"
            f"主要指数: {stocks}\n\n"
            f"加密价格: {crypto}\n\n"
            f"新闻标题: {[a['title'] for a in news]}"
        )
    else:
        user_prompt = (
            "You are a financial analyst. Summarize the market data below (≤2000 words), and give 3–5 trade ideas.\n\n"
            f"Indexes: {stocks}\n\n"
            f"Crypto: {crypto}\n\n"
            f"News: {[a['title'] for a in news]}"
        )

    system_prompt = role or CHARACTER_DESCRIPTION

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

async def daily_digest(_):
    try:
        brief = await make_headline_brief("zh")
    except Exception as e:
        print("[DailyDigest Error]", e)
        return

    for cid in DAILY_DIGEST_TARGETS:
        try:
            await app.bot.send_message(chat_id=cid, text=brief[:4096])
        except Exception as e:
            print(f"[SendError → {cid}]", e)