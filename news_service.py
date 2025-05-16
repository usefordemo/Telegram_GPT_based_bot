from datetime import date, timedelta
import re, urllib.parse, requests
from typing import List, Dict
from config import PROXIES, NEWSAPI_KEY, openai_client,GNEWS_API_KEY
from data import (AUTHORITATIVE_SOURCES, TOPIC_MAP,
                  CHARACTER_DESCRIPTION,ASIA_SUB_KEYS)
from utils import detect_lang


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

def fetch_newsapi_top_news(q: str = "technology OR politics", lang: str = "en", limit: int = 5) -> list[dict]:
    url = (
        f"https://newsapi.org/v2/top-headlines?"
        f"q={urllib.parse.quote_plus(q)}&"
        f"language={lang}&pageSize={limit}&apiKey={NEWSAPI_KEY}"
    )
    try:
        r = requests.get(url, proxies=PROXIES, timeout=10)
        r.raise_for_status()
        return r.json().get("articles", [])
    except Exception as e:
        print("[NewsAPI Fetch Error]", e)
        return []

def fetch_combined_news(
        keywords: List[str],
        *,
        max_articles: int = 30,
        lang: str = "en"
    ) -> List[Dict]:
    """一次 OR 查询拉全部关键词新闻；返回 ≤ max_articles 篇文章"""
    if not keywords:
        return []

    query = " OR ".join(f'"{k}"' for k in keywords)
    url   = (
        "https://gnews.io/api/v4/search?"
        f"q={urllib.parse.quote_plus(query)}&"
        f"token={GNEWS_API_KEY}&lang={lang}&max={max_articles}"
    )

    try:
        resp = requests.get(url, proxies=PROXIES, timeout=10)
        resp.raise_for_status()
        return resp.json().get("articles", [])
    except Exception as exc:
        print(f"[News Fetch Error] {exc}")
        return []


def fetch_authoritative_headlines(max_articles: int = 5) -> list[dict]:
    """
    只从白名单英文源拉 <max_articles> 条最新头条
    """
    url = (
        "https://newsapi.org/v2/top-headlines?"
        f"sources={','.join(AUTHORITATIVE_SOURCES)}"
        f"&pageSize={max_articles}"
        f"&apiKey={NEWSAPI_KEY}"
    )
    try:
        r = requests.get(url, proxies=PROXIES, timeout=10); r.raise_for_status()
        return r.json().get("articles", [])[:max_articles]
    except Exception as e:
        print("[NewsAPI Fetch Error]", e)
        return []


def fetch_general_news(*,
                       q: str,
                       from_date: str,
                       to_date: str,
                       max_articles: int,
                       lang: str = "en") -> list[dict]:
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={urllib.parse.quote_plus(q)}"
        f"&from={from_date}&to={to_date}"
        f"&language={lang}"
        f"&sources={','.join(AUTHORITATIVE_SOURCES)}"
        f"&sortBy=publishedAt&pageSize={max_articles}"
        f"&apiKey={NEWSAPI_KEY}"
    )
    try:
        r = requests.get(url, proxies=PROXIES, timeout=10)
        r.raise_for_status()
        return r.json().get("articles", [])
    except Exception as exc:
        print("[NewsAPI Fetch Error]", exc)
        return []

async def make_headline_brief(target_lang: str) -> str:
    """
    ① 拉 5 条英文头条
    ② GPT 翻译/播报（含编号）
    ③ 在每条标题后**单独起一行**附上原文链接
    ④ GPT 评论（若含受限主题则跳过该条）
    """
    arts = fetch_authoritative_headlines(max_articles=5)
    if not arts:
        return {
            "zh": "暂时抓不到新闻 🥲",
            "ja": "最新ニュースが取得できません 🥲",
            "en": "No fresh news found 🥲",
        }[target_lang]

    # --- 1. 纯英文标题串 -------------------------------------------------
    english_block = "\n".join(
        f"{i+1}. {a['title']} ({a['source']['name']})"
        for i, a in enumerate(arts)
    )

    # --- 2. 翻译 / 播报 --------------------------------------------------
    if target_lang == "zh":
        reporter_prompt = (
            "请把下列英文新闻标题翻译成**简体中文**并保持序号：\n\n"
            + english_block
        )
        reporter_header = "【📢 今日头条】\n"
        comment_prompt = (
            "请对上述新闻逐条幽默点评（≤150字）。"
            "若某条涉及受限主题，**跳过该条**，"
            "不要输出任何内容或序号。"
        )
    elif target_lang == "ja":
        reporter_prompt = (
            "以下の英語ニュース見出しを日本語に翻訳し、番号を保ったまま列挙してください：\n\n"
            + english_block
        )
        reporter_header = "【📢 今週のヘッドライン】\n"
        comment_prompt = (
            "上記ニュースについて、ユーモアを交えた短いコメント（150字以内）を逐条でお願いします。"
            "性的内容を含む見出し、または他の制限トピックがあれば**スキップ**してください。"
        )
    else:  # English
        reporter_prompt = english_block
        reporter_header = "【📢 Headlines】\n"
        comment_prompt = (
            "Give short playful comments (≤150 words) for each headline. "
            "If a headline involves sexual content, explicit pornography, "
            "or other restricted topics, **skip that item**."
        )

    # news reports
    rep = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": reporter_prompt}],
        temperature=0,
    ).choices[0].message.content.strip()

    # --- 2‑2. insert links------------------------------------
    rep_lines = rep.splitlines()
    augmented = []
    for line in rep_lines:
        augmented.append(line)
        m = re.match(r"\s*(\d+)\.", line)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(arts):
                url = arts[idx].get("url") or ""
                if url:
                    augmented.append(url)          # 单独起一行
    rep_with_links = "\n".join(augmented)

    # --- GPT comments -------------------------------
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": CHARACTER_DESCRIPTION},
            {
                "role": "user",
                "content": rep_with_links + "\n\n" + comment_prompt,
            },
        ],
    ).choices[0].message.content.strip()

    # --- 4. final ---------------------------------------------
    return reporter_header + rep_with_links + "\n\n【comments:】\n" + resp

async def make_topic_headline_brief(
    topic_key: str,
    period: str,
    target_lang: str,
    max_articles: int = 5
) -> str:
    """
    period: "day" | "week" | "month"
    """
    # -------- 映射 --------
    query_en, display_zh, display_en = TOPIC_MAP[topic_key]

    # -------- 计算时间窗口 --------
    today = date.today()
    if period == "day":
        frm, to = today, today
    elif period == "week":
        frm = today - timedelta(days=today.weekday())  # 本周一
        to  = today
    else:  # "month"
        frm = today.replace(day=1)
        to  = today
    from_iso, to_iso = frm.isoformat(), to.isoformat()

    # -------- 抓新闻 --------
    if query_en == "__ASIA_BUCKET__":
        raw_articles = []
        for sub_key in ASIA_SUB_KEYS:
            sub_query = TOPIC_MAP[sub_key][0]
            raw_articles.extend(  # ← 没有 await
                fetch_general_news(
                    q=sub_query,
                    from_date=from_iso,
                    to_date=to_iso,
                    max_articles=max_articles,
                    lang="en"
                )
            )
        arts = deduplicate(raw_articles)[:max_articles]
    else:
        arts = fetch_general_news(
        q=query_en,
        from_date=from_iso,
        to_date=to_iso,
        max_articles=max_articles,
        lang="en")

    # -------- 无新闻兜底 --------
    if not arts:
        if target_lang == "zh":
            return f"{'今日' if period=='day' else '本周' if period=='week' else '本月'}没有重大{display_zh}新闻 🥲"
        elif target_lang == "ja":
            return f"{'本日' if period=='day' else '今週' if period=='week' else '今月'}の重要な{display_en}ニュースはありません 🥲"
        else:
            return f"No major {display_en} news for this {period} 🥲"

    # -------- 1. 英文标题块 --------
    english_block = "\n".join(
        f"{i+1}. {a['title']} ({a['source']['name']})"
        for i, a in enumerate(arts)
    )

    # -------- 2. 翻译 / 标题播报 --------
    if target_lang == "zh":
        reporter_prompt = "把下列英文新闻标题翻译成**简体中文**并保持序号：\n\n" + english_block
        reporter_header = f"【📢 { '今日' if period=='day' else '本周' if period=='week' else '本月' }{display_zh}头条】\n"
        comment_prompt  = "请逐条幽默点评（≤150字），若含受限内容请跳过。"
    elif target_lang == "ja":
        reporter_prompt = "以下の英語ニュース見出しを日本語に翻訳し、番号を保ったまま列挙してください：\n\n" + english_block
        reporter_header = f"【📢 { '本日' if period=='day' else '今週' if period=='week' else '今月' }の{display_zh}ヘッドライン】\n"
        comment_prompt  = "ユーモアを交え短く逐条コメント（150字以内）、制限トピックはスキップ。"
    else:
        reporter_prompt = english_block
        reporter_header = f"【📢 {display_en} Headlines】\n"
        comment_prompt  = ("Give playful comments (≤150 words) for each headline. "
                           "Skip any item that involves restricted topics.")

    # 2-1 翻译 / 播报
    rep = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": reporter_prompt}],
        temperature=0
    ).choices[0].message.content.strip()

    # -------- 3. 加链接 --------
    out = []
    for line in rep.splitlines():
        out.append(line)
        m = re.match(r"\s*(\d+)\.", line)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(arts):
                url = arts[idx].get("url") or ""
                if url:
                    out.append(url)
    rep_with_links = "\n".join(out)

    # -------- 4. 评论 --------
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": CHARACTER_DESCRIPTION},
            {"role": "user",   "content": rep_with_links + "\n\n" + comment_prompt}
        ]
    ).choices[0].message.content.strip()

    if target_lang == "zh":
        return reporter_header + rep_with_links + "\n\n【评】\n" + resp
    else:
        return reporter_header + rep_with_links + "\n\n【Comments】\n" + resp
