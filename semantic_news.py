# semantic_news.py — UTC date semantics, LLM generic/topical detection,
# LLM-chosen NewsAPI categories (no source-ID hardcoding), strict-source fetch,
# and auto-localized brief (no hard-coded language lists)

from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone, timedelta, date

from utils import openai_client, detect_lang
from news_service import fetch_news_safely, MAX_WINDOW_DAYS, get_provider_categories
from semantic_query import extract_news_query, build_boolean_query


# ---------- small render helpers ----------
def _english_headline_block(arts: List[Dict]) -> str:
    lines: List[str] = []
    for i, a in enumerate(arts or []):
        title = (a.get("title") or "").strip()
        src = ((a.get("source") or {}).get("name") or "").strip()
        if title:
            lines.append(f"{i+1}. {title}" + (f" ({src})" if src else ""))
    return "\n".join(lines)


def _attach_links_after_numbered_lines(numbered_text: str, arts: List[Dict]) -> str:
    out: List[str] = []
    for line in (numbered_text or "").splitlines():
        out.append(line)
        m = re.match(r"\s*(\d+)\.", line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(arts):
            url = arts[idx].get("url") or ""
            if url:
                out.append(url)
    return "\n".join(out)


# ---------- UTC helpers ----------
def _utc_today_date() -> date:
    return datetime.now(timezone.utc).date()


def _iso(d: date) -> str:
    return d.isoformat()


# ---------- date resolution (language-agnostic; no hard-coded keywords) ----------
async def _resolve_utc_window(user_text: str) -> Tuple[str, str, str]:
    """
    Any-language 'today/yesterday/5 days ago/last week/...' → concrete UTC dates.
    Returns (from_iso, to_iso, rationale). If parsing fails, defaults to UTC today.
    """
    today_iso = _iso(_utc_today_date())

    prompt = f"""
You are a date resolver. The current date in UTC is {today_iso}.
The user may speak any language. Infer the requested news time window.

Return STRICT JSON with keys:
{{
  "from": "YYYY-MM-DD",
  "to":   "YYYY-MM-DD",
  "explanation": "short reason in English"
}}

Rules:
- "today" → from=to=UTC today.
- "yesterday" → from=to=UTC today - 1 day.
- "N days ago" → from=to=UTC today - N days.
- "last week" → choose a sensible 7-day span immediately before the current week end (UTC).
- Explicit ranges ("from ... to ...") → use those dates (UTC).
- If ambiguous, choose the most recent reasonable window.
- Output ONLY the JSON. No extra text.

User request:
{user_text}
    """.strip()

    try:
        rep = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (rep.choices[0].message.content or "").strip()
        raw = raw.strip("` \n")
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip(":").strip()
        data = json.loads(raw)
        frm = str(data.get("from") or today_iso)
        to = str(data.get("to") or today_iso)
        why = str(data.get("explanation") or "")
        return frm, to, why
    except Exception as e:
        print("[semantic_news] date JSON parse failed; falling back to UTC today:", e)
        t = today_iso
        return t, t, "fallback: today (UTC)"


# ---------- generic vs topical (language-agnostic; decided by LLM) ----------
async def _is_generic_time_only(user_text: str, q_raw: str) -> bool:
    """
    Ask the model if the request is purely time-based/generic ('today's news')
    vs topical (mentions concrete subjects). Output: 'generic' or 'topical'.
    """
    try:
        prompt = (
            "Classify the user's request for news as either 'generic' or 'topical'.\n"
            "- generic: time-only or broad like 'today's news', 'yesterday's news', 'last week news', 'latest headlines'.\n"
            "- topical: mentions specific subjects (companies, people, places, tickers, policies, products, teams...).\n"
            "Return EXACTLY one word: generic or topical.\n\n"
            f"User text: {user_text}\n"
            f"Proposed boolean query (may be empty): {q_raw}\n"
        )
        rep = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        ans = (rep.choices[0].message.content or "").strip().lower().split()[0]
        return ans == "generic"
    except Exception as e:
        print("[semantic_news] generic-vs-topical check failed; default to generic:", e)
        return True


# ---------- category chooser (LLM; no source-ID hardcoding) ----------
async def _choose_categories(user_text: str, available: List[str], is_generic: bool) -> List[str]:
    """
    Pick NewsAPI categories (from provider's category list) appropriate for this query.
    Returns a JSON array of strings; fallback to ['general'] on failure.
    """
    if not available:
        return ["general"]
    # Make a stable, sorted list to present to the LLM
    cats = sorted(set(available))
    try:
        hint = (
            "If the request is very generic/time-only, prefer 'general' and only add others if clearly expected by most users.\n"
            "If the request is topical, include the relevant categories."
        )
        prompt = (
            "Choose the most appropriate NewsAPI categories for the user's news request.\n"
            f"Available categories: {cats}\n"
            f"Request type: {'generic' if is_generic else 'topical'}\n"
            f"Guidance: {hint}\n"
            "Return ONLY a JSON array, e.g., [\"general\",\"business\"].\n\n"
            f"User text: {user_text}\n"
        )
        rep = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (rep.choices[0].message.content or "").strip().strip("` \n")
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip(":").strip()
        arr = json.loads(raw)
        if isinstance(arr, list):
            out = [str(x).lower().strip() for x in arr if str(x).strip()]
            chosen = [c for c in out if c in cats]
            if chosen:
                print(f"[semantic_news] categories chosen by LLM: {chosen}")
                return chosen
    except Exception as e:
        print("[semantic_news] category selection failed; fallback to ['general']:", e)
    return ["general"]


# ---------- main entry ----------
async def make_semantic_news_brief(user_text: str, max_articles: int = 8) -> str:
    """
    Pipeline:
      1) semantic query plan (any language) → boolean query (English, canonical)
      2) resolve UTC date window from user text (no clamping)
      3) if window older than provider limit (MAX_WINDOW_DAYS), tell user it's too early
      4) fetch strictly from curated sources, filtered by LLM-chosen categories
      5) brief + auto-translate to user's language (no hard-coded language list)
    """
    user_lang = await detect_lang(user_text) or "en"

    # 1) Semantic plan & boolean query (canonical EN; output language handled later)
    plan = await extract_news_query(user_text)
    plan["language"] = "en"
    q_raw = build_boolean_query(plan) or (user_text or "").strip()

    # 2) Resolve UTC date window from the user's words (no clamping)
    frm, to, why = await _resolve_utc_window(user_text)
    print(f"[semantic_news] input={user_text!r} lang={user_lang} utc_window={frm}->{to} why={why}")
    print(f"[semantic_news] boolean_q_raw={q_raw!r}")

    # 3) Provider recency guard
    today = _utc_today_date()
    min_allowed = today - timedelta(days=int(MAX_WINDOW_DAYS))
    try:
        too_old = (date.fromisoformat(frm) < min_allowed) or (date.fromisoformat(to) < min_allowed)
    except Exception:
        too_old = False

    if too_old:
        earliest = _iso(min_allowed)
        base_msg = (
            f"Your requested date range is too old for my news provider. "
            f"The earliest I can fetch is {earliest} (UTC). "
            f"Try a more recent range within the last {MAX_WINDOW_DAYS} days."
        )
        if user_lang.startswith("en"):
            return base_msg
        try:
            tr = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": "Translate short system messages; keep numbers/dates intact."},
                    {"role": "user", "content": f"Translate to {user_lang}:\n{base_msg}"},
                ],
            )
            return (tr.choices[0].message.content or "").strip()
        except Exception:
            return base_msg

    # 4) Decide generic vs topical; for generic/time-only asks, drop keywords (q="")
    try:
        is_generic = await _is_generic_time_only(user_text, q_raw)
    except Exception as e:
        print("[semantic_news] generic-check failed:", e)
        is_generic = True
    q_effective = "" if is_generic else q_raw
    print(f"[semantic_news] classified={'generic' if is_generic else 'topical'} -> q_effective={q_effective!r}")

    # 4b) Ask provider for available categories, then let LLM choose subset
    available_cats = get_provider_categories()  # from provider metadata
    chosen_cats = await _choose_categories(user_text, available_cats, is_generic)
    print(f"[semantic_news] using categories={chosen_cats}")

    # 5) Fetch with strict sources, filtered by categories.
    strict = (frm == to)  # single-day: strict, no generic fallback
    try:
        arts: List[Dict] = fetch_news_safely(
            q=q_effective,
            from_date=frm,
            to_date=to,
            max_articles=max_articles,
            lang="en",
            strict_window=strict,
            allowed_categories=chosen_cats,
        )
    except Exception as e:
        print("[semantic_news] fetch error:", e)
        arts = []

    if not arts:
        msg_en = f"No articles were found from your configured sources for {frm} → {to} (UTC)."
        if user_lang.startswith("en"):
            return msg_en
        try:
            tr = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": "Translate short system messages; keep dates intact."},
                    {"role": "user", "content": f"Translate to {user_lang}:\n{msg_en}"},
                ],
            )
            return (tr.choices[0].message.content or "").strip()
        except Exception:
            return msg_en

    # 6) Build canonical EN headlines + URLs
    english_block = _english_headline_block(arts).strip()
    block_with_links_en = _attach_links_after_numbered_lines(english_block, arts)

    # Debug peek at a few titles
    try:
        sample = [ (a.get("title") or "")[:120] for a in arts[:3] ]
        print(f"[semantic_news] sample titles: {sample}")
    except Exception:
        pass

    # 7) Comments in user's language (no hard-coded language list)
    comments = ""
    if user_lang.startswith("en"):
        try:
            rep = await openai_client.chat.completions.create(
                model="gpt-4o",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": "You are concise, witty, and safe."},
                    {"role": "user",
                     "content": block_with_links_en
                                + f"\n\nWrite one short, witty, and safe comment (1–2 sentences) "
                                  f"for each numbered headline above, in {user_lang}. "
                                  f"Keep numbering if you use it; do not translate named entities."},
                ],
            )
            comments = (rep.choices[0].message.content or "").strip()
        except Exception as e:
            print("[semantic_news] comment generation failed:", e)
        final_headlines = block_with_links_en

    # 8) Translate headlines to user's language (preserve structure) only if needed
    else:
        try:
            tr = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system",
                     "content": "Translate while preserving numbering and URLs exactly. Do not translate proper names."},
                    {"role": "user", "content": f"Translate to {user_lang}:\n\n{block_with_links_en}"},
                ],
            )
            final_headlines = (tr.choices[0].message.content or "").strip()
        except Exception as e:
            print("[semantic_news] headline translation failed → keep EN:", e)

        try:
            rep2 = await openai_client.chat.completions.create(
                model="gpt-4o",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": "You are concise, witty, and safe."},
                    {"role": "user",
                     "content": final_headlines + (
                         f"\n\nWrite one short, witty, and safe comment (1–2 sentences) "
                         f"for each numbered headline above, in {user_lang}. "
                         f"Keep numbering if you use it; do not translate named entities."
                     )},
                ],
            )
            comments = (rep2.choices[0].message.content or "").strip()
        except Exception as e:
            print("[semantic_news] non-EN comment generation failed:", e)

    header = "【📢 News】\n"
    tail = ("\n\n【💬 Comments】\n" + comments) if comments else ""
    return header + final_headlines + tail


# Expose for tests if needed
__all__ = ["make_semantic_news_brief"]









