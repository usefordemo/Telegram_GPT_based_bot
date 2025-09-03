from __future__ import annotations

# news_service.py — strict trusted sources, UTC-day bounds, dynamic domains fallback,
# today-only top-headlines fallback, and category filtering driven by LLM (no source-ID hardcoding)

import os
import re
import logging
from logging.handlers import RotatingFileHandler
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

# Project config + utils (keys from API.py or env—no hard-coded secrets)
import API as _cfg
from API import PROXIES, NEWSAPI_KEY
from semantic_query import extract_news_query, build_boolean_query
from utils import openai_client
# Your curated reliable sources live here (your data.py)
from data import AUTHORITATIVE_SOURCES as DATA_SOURCES

# ------------------------------------------------------------------------------
# Tunables / Filters
# ------------------------------------------------------------------------------
TITLE_MIN_CHARS = int(os.getenv("NEWS_TITLE_MIN_CHARS", "20"))
TITLE_MIN_ALPHA = int(os.getenv("NEWS_TITLE_MIN_ALPHA", "5"))
URL_DENY_REGEX  = os.getenv("NEWS_URL_DENY_REGEX", r"/(programmes|schedule|live|video)/")

MAX_WINDOW_DAYS = int(os.getenv("NEWS_MAX_WINDOW_DAYS", "30"))
HTTP_TIMEOUT    = int(os.getenv("NEWS_HTTP_TIMEOUT", "12"))

_TOP_ENDPOINT        = "https://newsapi.org/v2/top-headlines"
_EVERYTHING_ENDPOINT = "https://newsapi.org/v2/everything"
_SOURCES_ENDPOINT    = "https://newsapi.org/v2/top-headlines/sources"

# NewsAPI fixed language set (provider constraint)
_SUPPORTED_LANGS = {"ar","de","en","es","fr","he","it","nl","no","pt","ru","sv","ud","zh"}

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
_LOG_DIR = os.getenv("NEWS_LOG_DIR", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
logger = logging.getLogger("news_service")
if not logger.handlers:
    logger.setLevel(logging.DEBUG if os.getenv("NEWS_DEBUG") else logging.INFO)
    fh = RotatingFileHandler(
        os.path.join(_LOG_DIR, "news_service.log"),
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(fh)
    if os.getenv("NEWS_LOG_TO_CONSOLE", "0") == "1":
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(ch)

_SUPPORTED_SOURCES_CACHE: List[str] = []
_SUPPORTED_SOURCE_META: Dict[str, Dict[str, str]] = {}  # id -> {"url": ..., "name": ..., "category": ...}
_PROVIDER_CATEGORIES: List[str] = []  # e.g., ['business','entertainment','general','health','science','sports','technology']

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _to_utc_bounds(from_date: str, to_date: str) -> Tuple[str, str]:
    """Expand ISO dates to full UTC day. If already has 'T', pass through."""
    def _start(d: str) -> str:
        return d if "T" in d else (d.strip() + "T00:00:00Z")
    def _end(d: str) -> str:
        return d if "T" in d else (d.strip() + "T23:59:59Z")
    return _start(from_date), _end(to_date)

def _redact(s: str) -> str:
    return f"…{s[-4:]}" if s and len(s) > 4 else "****"

def _get_newsapi_key() -> str:
    key = (getattr(_cfg, "NEWSAPI_KEY", "") or os.getenv("NEWSAPI_KEY", "")).strip()
    if not key:
        logger.error("NEWSAPI_KEY is empty. Set it in API.py or env.")
        raise RuntimeError("NEWSAPI_KEY is empty. Set it in API.py or env.")
    return key

try:
    tail = _redact(_get_newsapi_key())
    logger.info("NewsAPI key loaded | from=%s | tail=%s", getattr(_cfg, "__file__", "?"), tail)
except Exception as e:
    logger.exception("Failed to load NewsAPI key: %s", e)

def _normalize_lang(lang: str) -> str:
    l = (lang or "en").lower()
    return l if l in _SUPPORTED_LANGS else "en"

def _sanitize_sources(sources: List[str]) -> List[str]:
    seen, out = set(), []
    for raw in sources or []:
        if raw is None:
            continue
        for part in str(raw).split(","):
            s = part.strip().lower().replace(" ", "-")
            if s and s not in seen:
                out.append(s); seen.add(s)
    return out

def _request(endpoint: str, params: Dict[str, str]) -> requests.Response:
    safe = {k: v for k, v in params.items() if k != "apiKey"}
    print(f"[news_service] GET {endpoint} params={safe}")
    logger.info("HTTP GET %s | params=%s", endpoint, safe)
    r = requests.get(
        endpoint,
        params=params,
        headers={"X-Api-Key": _get_newsapi_key()},
        proxies=PROXIES,
        timeout=HTTP_TIMEOUT,
    )
    try:
        red = r.url.replace(_get_newsapi_key(), NEWSAPI_KEY)
    except Exception:
        red = "(url unavailable)"
    print(f"[news_service] <-- {r.status_code} {red}")
    logger.info("HTTP %s -> %s | %s", endpoint, r.status_code, red)
    return r

def _fetch_supported_sources_from_provider() -> List[str]:
    global _SUPPORTED_SOURCES_CACHE, _SUPPORTED_SOURCE_META, _PROVIDER_CATEGORIES
    if _SUPPORTED_SOURCES_CACHE:
        return _SUPPORTED_SOURCES_CACHE
    try:
        params = {"language": "en", "apiKey": _get_newsapi_key()}
        r = _request(_SOURCES_ENDPOINT, params)
        r.raise_for_status()
        srcs = (r.json().get("sources") or [])
        ids = []
        meta: Dict[str, Dict[str, str]] = {}
        cats = set()
        for s in srcs:
            sid = s.get("id", "")
            if not sid:
                continue
            ids.append(sid)
            meta[sid] = {
                "url": s.get("url") or "",
                "name": s.get("name") or "",
                "category": (s.get("category") or "").strip().lower(),
            }
            if s.get("category"):
                cats.add((s.get("category") or "").strip().lower())
        _SUPPORTED_SOURCES_CACHE = sorted(set(ids))
        _SUPPORTED_SOURCE_META = meta
        _PROVIDER_CATEGORIES = sorted(cats) if cats else ['general']
        logger.info("Provider supports %d sources (en), categories=%s", len(_SUPPORTED_SOURCES_CACHE), _PROVIDER_CATEGORIES)
    except Exception as e:
        logger.exception("fetch /sources failed: %s", e)
        _SUPPORTED_SOURCES_CACHE = []
        _SUPPORTED_SOURCE_META = {}
        _PROVIDER_CATEGORIES = ['general']
    return _SUPPORTED_SOURCES_CACHE

def get_provider_categories() -> List[str]:
    """Categories reported by provider (e.g., ['business','entertainment','general',...])."""
    _fetch_supported_sources_from_provider()
    return list(_PROVIDER_CATEGORIES)

def _get_sources(allowed_categories: Optional[List[str]] = None) -> List[str]:
    desired = _sanitize_sources(DATA_SOURCES)
    if not desired:
        logger.warning("AUTHORITATIVE_SOURCES empty; skipping fetches.")
        return []
    supported = set(_fetch_supported_sources_from_provider() or [])
    if not supported:
        logger.warning("Could not validate sources with provider; using configured list as-is.")
        return desired

    # Intersect configured sources with supported ones
    intersect = [s for s in desired if s in supported]

    # If caller provided categories, filter by provider's metadata
    if allowed_categories:
        ac = {c.strip().lower() for c in allowed_categories if c}
        if ac:
            filtered: List[str] = []
            for sid in intersect:
                cat = (_SUPPORTED_SOURCE_META.get(sid, {}).get("category") or "").strip().lower()
                if cat in ac:
                    filtered.append(sid)
            if filtered:
                allowed = filtered
            else:
                # If filter yields nothing, fall back to 'general' if possible, else keep intersect
                alt = [sid for sid in intersect if (_SUPPORTED_SOURCE_META.get(sid, {}).get("category") or "").strip().lower() == "general"]
                allowed = alt or intersect
        else:
            allowed = intersect
    else:
        allowed = intersect

    dropped = [s for s in desired if s not in allowed]
    if dropped:
        print(f"[news_service] dropped unsupported or filtered sources: {dropped}")
        logger.warning("Dropping unsupported/filtered sources: %s", dropped)
    if not allowed:
        logger.error("All configured sources unsupported after filtering; minimal fallback used.")
        return ["reuters", "bbc-news", "associated-press"]
    print(f"[news_service] resolved sources: {allowed}")
    logger.info("Resolved sources: %s", allowed)
    return allowed

def _get_allowed_domains(allowed_categories: Optional[List[str]] = None) -> List[str]:
    """
    Derive domains from provider metadata for the curated source IDs you configured (and category-filtered).
    """
    _fetch_supported_sources_from_provider()
    allowed_ids = set(_get_sources(allowed_categories))
    domains: List[str] = []
    seen = set()
    for sid, meta in _SUPPORTED_SOURCE_META.items():
        if sid not in allowed_ids:
            continue
        url = (meta.get("url") or "").strip()
        if not url:
            continue
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        if netloc and netloc not in seen:
            domains.append(netloc)
            seen.add(netloc)
    if domains:
        print(f"[news_service] resolved domains: {domains}")
        logger.info("Resolved domains: %s", domains)
    else:
        logger.warning("No domains resolved from provider metadata for configured sources.")
    return domains

# ---- date helpers ----
def _today_iso() -> str:
    return date.today().isoformat()

def _period_to_window(period: str) -> Tuple[str, str]:
    today = date.today()
    p = (period or "day").lower()
    if p == "day":
        return today.isoformat(), today.isoformat()
    if p == "week":
        start = today - timedelta(days=today.weekday())
        return start.isoformat(), today.isoformat()
    start = today.replace(day=1)
    return start.isoformat(), today.isoformat()

def _clamp_recent_window(
    frm_iso: str,
    to_iso: str,
    *,
    max_days: int = MAX_WINDOW_DAYS,
    provider_min_date: Optional[str] = None,
) -> Tuple[str, str]:
    today = date.today()
    def _parse(s: str) -> date:
        try:
            return date.fromisoformat(s)
        except Exception:
            return today
    o_from, o_to = _parse(frm_iso), _parse(to_iso)
    min_from = today - timedelta(days=max_days)
    if provider_min_date:
        try:
            pmin = date.fromisoformat(provider_min_date)
            if pmin > min_from:
                min_from = pmin
        except Exception:
            logger.debug("provider_min_date parse failed: %s", provider_min_date)
    start = max(o_from, min_from)
    end   = min(o_to, today)
    if end < start:
        end = start
    logger.debug("Clamped window: %s → %s (orig %s → %s, provider_min=%s)",
                 start.isoformat(), end.isoformat(), o_from, o_to, provider_min_date)
    return start.isoformat(), end.isoformat()

# ---- rendering & quality ----
def _deduplicate_articles(arts: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for a in arts or []:
        u = a.get("url") or ""
        if u and u not in seen:
            out.append(a); seen.add(u)
    return out

def _english_headline_block(arts: List[Dict]) -> str:
    lines: List[str] = []
    for i, a in enumerate(arts or []):
        title = (a.get("title") or "").strip()
        src   = ((a.get("source") or {}).get("name") or "").strip()
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
            if url: out.append(url)
    return "\n".join(out)

def _looks_like_timestamp(title: str) -> bool:
    t = (title or "").strip()
    if not t:
        return True
    if re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+\d{1,2}:\d{2}\s*(GMT|UTC)?", t, re.I):
        return True
    if sum(1 for c in t if c.isalpha()) < TITLE_MIN_ALPHA:
        return True
    return False

def _quality_filter_articles(arts: List[Dict]) -> List[Dict]:
    deny = re.compile(URL_DENY_REGEX, re.I) if URL_DENY_REGEX else None
    out: List[Dict] = []
    for a in arts or []:
        ttl = (a.get("title") or "").strip()
        url = a.get("url") or ""
        if _looks_like_timestamp(ttl):
            continue
        if len(ttl) < TITLE_MIN_CHARS:
            desc = (a.get("description") or "").strip()
            if len(desc) < 40:
                continue
        if deny and deny.search(url):
            continue
        out.append(a)
    return out

# ------------------------------------------------------------------------------
# Fetchers (semantic-first, category-filtered)
# ------------------------------------------------------------------------------
def fetch_newsapi_top_news(q: str = "", lang: str = "en", limit: int = 5, *, allowed_categories: Optional[List[str]] = None) -> List[Dict]:
    srcs = _get_sources(allowed_categories)
    if not srcs:
        return []
    try:
        params = {
            "language": _normalize_lang(lang),
            "pageSize": limit,
            "sources": ",".join(srcs),
            "apiKey": _get_newsapi_key(),
        }
        if (q or "").strip():
            params["q"] = q.strip()
        r = _request(_TOP_ENDPOINT, params); r.raise_for_status()
        arts = r.json().get("articles", [])
        return _quality_filter_articles(_deduplicate_articles(arts))
    except Exception as e:
        logger.exception("Top-headlines failed: %s", e)
        return []

def fetch_general_news(
    *,
    q: str,
    from_date: str,
    to_date: str,
    max_articles: int,
    lang: str = "en",
    strict_window: bool = False,
    allowed_categories: Optional[List[str]] = None,
) -> List[Dict]:
    srcs = _get_sources(allowed_categories)
    if not srcs:
        return []

    # Provider guard (keeps requests within recent limits)
    from_date, to_date = _clamp_recent_window(from_date, to_date)

    # Expand to full UTC day(s)
    from_ts, to_ts = _to_utc_bounds(from_date, to_date)

    try:
        params = {
            "q": (q or "").strip(),
            "from": from_ts,          # UTC timestamps
            "to": to_ts,              # UTC timestamps
            "language": _normalize_lang(lang),
            "pageSize": max_articles,
            "sortBy": "publishedAt",
            "sources": ",".join(srcs),
            "apiKey": _get_newsapi_key(),
        }
        r = _request(_EVERYTHING_ENDPOINT, params)
        if r.status_code >= 400:
            r.raise_for_status()

        j = {}
        try:
            j = r.json()
        except Exception:
            j = {}
        total = j.get("totalResults")
        if total is not None:
            print(f"[news_service] everything.totalResults={total}")
            logger.info("everything.totalResults=%s", total)
        arts = j.get("articles", []) or []

        if not arts:
            logger.info(
                "Everything empty for %s→%s (strict=%s). Trying domains fallback.",
                from_ts, to_ts, strict_window,
            )
            # Secondary attempt: retry 'everything' using domains= (derived dynamically)
            domains = _get_allowed_domains(allowed_categories)
            if domains:
                try:
                    params2 = {
                        "q": (q or "").strip(),
                        "from": from_ts,
                        "to": to_ts,
                        "language": _normalize_lang(lang),
                        "pageSize": max_articles,
                        "sortBy": "publishedAt",
                        "domains": ",".join(domains),
                        "apiKey": _get_newsapi_key(),
                    }
                    r2 = _request(_EVERYTHING_ENDPOINT, params2)
                    r2.raise_for_status()
                    j2 = r2.json()
                    total2 = j2.get("totalResults")
                    if total2 is not None:
                        print(f"[news_service] everything(domains).totalResults={total2}")
                        logger.info("everything(domains).totalResults=%s", total2)
                    arts2 = j2.get("articles", []) or []
                    if arts2:
                        return _quality_filter_articles(_deduplicate_articles(arts2))
                except Exception as e:
                    logger.exception("Everything(domains) failed: %s", e)

            # Today-only fallback: if specific-day (strict) and it's today (UTC), try top-headlines
            try:
                utc_today = date.today().isoformat()
                if strict_window and from_date.startswith(utc_today) and to_date.startswith(utc_today):
                    logger.info("Empty everything for today; falling back to top-headlines (same trusted sources).")
                    return fetch_newsapi_top_news(q="", lang=lang, limit=max_articles, allowed_categories=allowed_categories)
            except Exception:
                pass

            # Otherwise: strict → no fallback; non-strict → generic fallback is acceptable
            return [] if strict_window else fetch_newsapi_top_news(q="", lang=lang, limit=max_articles, allowed_categories=allowed_categories)

        return _quality_filter_articles(_deduplicate_articles(arts))
    except Exception as e:
        logger.exception("Everything network error: %s", e)
        # On hard errors, fallback is okay (network/server)
        return [] if strict_window else fetch_newsapi_top_news(q="", lang=lang, limit=max_articles, allowed_categories=allowed_categories)


def fetch_news_safely(
    *,
    q: str,
    from_date: str,
    to_date: str,
    max_articles: int,
    lang: str = "en",
    strict_window: bool = False,
    allowed_categories: Optional[List[str]] = None,
) -> List[Dict]:
    return fetch_general_news(
        q=q,
        from_date=from_date,
        to_date=to_date,
        max_articles=max_articles,
        lang=lang,
        strict_window=strict_window,
        allowed_categories=allowed_categories,
    )

# ------------------------------------------------------------------------------
# Public convenience: (kept for compatibility)
# ------------------------------------------------------------------------------


async def make_headline_brief(_: str = "en") -> str:
    # Minimal helper; not category-aware
    arts = fetch_newsapi_top_news(q="", lang="en", limit=5) or []
    arts = arts[:5]
    if not arts:
        return "No fresh headlines from your trusted sources right now. 🥲"
    lines = []
    for i, a in enumerate(arts):
        title = (a.get("title") or "").strip()
        src = ((a.get("source") or {}).get("name") or "").strip()
        url = a.get("url") or ""
        if title:
            lines.append(f"{i+1}. {title}" + (f" ({src})" if src else ""))
            if url:
                lines.append(url)
    return "【Headlines】\n" + "\n".join(lines)

# ------------------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------------------


async def make_topic_headline_brief(
    topic_key: str,
    period: str,
    target_lang: str,
    max_articles: int = 5
) -> str:
    """
    Minimal, dependency-light topic brief to keep backward compatibility with tg_bot.py.
    - Resolves date window from `period` (day/week/month) with recent clamp.
    - Builds a boolean query from `topic_key` via your semantic_query helpers (no hard-coded keywords).
    - Fetches strictly from curated sources (no category filter here—topic can be anything).
    - Returns headlines + URLs; auto-translates to target_lang if not English (no hard-coded language list).
    """
    # 1) Resolve date window & clamp to provider limits
    frm, to = _period_to_window(period)
    frm, to = _clamp_recent_window(frm, to)

    # 2) Build query from topic (semantic, no hard-coded terms)
    try:
        plan = await extract_news_query(topic_key)
        q = build_boolean_query(plan) or topic_key
    except Exception:
        q = topic_key

    # 3) Fetch (strict if single-day window)
    strict = (frm == to)
    arts = fetch_news_safely(
        q=q,
        from_date=frm,
        to_date=to,
        max_articles=max_articles,
        lang="en",
        strict_window=strict,
        allowed_categories=None,  # topic requests can span categories
    )
    arts = arts[:max_articles] if arts else []

    if not arts:
        base_msg = "No matching topic news from your trusted sources right now. 🥲"
        if (target_lang or "en").lower().startswith("en") or not openai_client:
            return base_msg
        try:
            tr = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": "Translate short UI strings; keep emojis unchanged."},
                    {"role": "user", "content": f"Translate to {target_lang}:\n{base_msg}"},
                ],
            )
            return (tr.choices[0].message.content or "").strip()
        except Exception:
            return base_msg

    # 4) Render headlines (EN canonical) + URLs
    lines: List[str] = []
    for i, a in enumerate(arts):
        title = (a.get("title") or "").strip()
        src = ((a.get("source") or {}).get("name") or "").strip()
        url = a.get("url") or ""
        if not title:
            continue
        lines.append(f"{i+1}. {title}" + (f" ({src})" if src else ""))
        if url:
            lines.append(url)
    out_en = "【Topic News】\n" + "\n".join(lines)

    # 5) Auto-translate if target_lang is not English
    tgt = (target_lang or "en").lower()
    if tgt.startswith("en") or not openai_client:
        return out_en
    try:
        tr = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system",
                 "content": "Translate while preserving numbering and URLs exactly. Do not translate proper names."},
                {"role": "user", "content": f"Translate to {target_lang}:\n\n{out_en}"},
            ],
        )
        return (tr.choices[0].message.content or "").strip()
    except Exception:
        return out_en


def news_health_check() -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        out["key_tail"] = _redact(_get_newsapi_key())
    except Exception as e:
        out["key_error"] = str(e)
        return out

    srcs = _get_sources()
    out["sources"] = ",".join(srcs) if srcs else "(none)"
    out["categories"] = ",".join(get_provider_categories())

    try:
        r1 = _request(
            _TOP_ENDPOINT,
            {"language": "en", "pageSize": 1, "sources": ",".join(srcs), "apiKey": _get_newsapi_key()},
        )
        out["top_status"] = str(r1.status_code)
    except Exception as e:
        out["top_error"] = str(e)

    try:
        today = _today_iso()
        r2 = _request(
            _EVERYTHING_ENDPOINT,
            {
                "q": "",
                "from": today,
                "to": today,
                "language": "en",
                "pageSize": 1,
                "sortBy": "publishedAt",
                "sources": ",".join(srcs),
                "apiKey": _get_newsapi_key(),
            },
        )
        out["everything_status"] = str(r2.status_code)
    except Exception as e:
        out["everything_error"] = str(e)

    out["log_file"] = os.path.join(_LOG_DIR, "news_service.log")
    return out

# Expose constants/functions used elsewhere
__all__ = [
    "MAX_WINDOW_DAYS",
    "fetch_news_safely",
    "get_provider_categories",
    "news_health_check",
    "make_topic_headline_brief",
]



