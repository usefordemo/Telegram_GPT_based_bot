
from __future__ import annotations
import json
import re
from typing import Dict, List, Tuple
from datetime import date, timedelta
from utils import openai_client, detect_lang

# ---------- helpers ----------

def _default_time_window(text: str) -> Tuple[str, str]:
    """
    Very small, English-only fallback for time window detection.
    Falls back to "today" if unclear.
    """
    t = (text or "").lower()
    today = date.today()
    if "yesterday" in t:
        d0 = today - timedelta(days=1)
        return d0.isoformat(), d0.isoformat()
    if "last week" in t:
        # last Monday .. last Sunday
        start = today - timedelta(days=today.weekday() + 7)
        end = start + timedelta(days=6)
        return start.isoformat(), end.isoformat()
    if "this week" in t:
        start = today - timedelta(days=today.weekday())
        return start.isoformat(), today.isoformat()
    if "this month" in t:
        start = today.replace(day=1)
        return start.isoformat(), today.isoformat()
    # default -> today
    return today.isoformat(), today.isoformat()


async def extract_news_query(user_text: str) -> Dict:
    """
    LLM-driven semantic parsing to a structured query plan.
    No hard-coded keyword lists: the model expands synonyms/entities.
    Output schema:
    {
      "language": "en",
      "time_window": {"from": "YYYY-MM-DD", "to": "YYYY-MM-DD"},
      "entities": {"locations": [...], "organizations": [...], "people": [...]},
      "categories": [...],
      "keywords":   [...]    # deduped concrete search terms (incl. synonyms)
    }
    """
    sys = (
        "Extract a structured news query plan from the user text.\n"
        "Return strict JSON with keys: language, time_window, entities, categories, keywords.\n"
        "• language: use 'en'\n"
        "• time_window: {from: YYYY-MM-DD, to: YYYY-MM-DD} (prefer today if asked for today)\n"
        "• entities.locations: country/region/city names if any\n"
        "• entities.organizations: organizations if any\n"
        "• entities.people: person names if any\n"
        "• categories: high-level topics in 1–3 words (e.g., technology, finance, sports)\n"
        "• keywords: 6–14 concrete search terms and synonyms, short phrases allowed; avoid stopwords.\n"
        "Do not add commentary; return only JSON."
    )
    user = (user_text or "").strip()
    # Ask the LLM first. If anything fails, build a tiny fallback.
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user}
            ],
            temperature=0
        )
        raw = resp.choices[0].message.content.strip()
        # Ensure it's JSON (some models may wrap in ```json ... ```)
        m = re.search(r'\{[\s\S]*\}', raw)
        if m:
            raw = m.group(0)
        plan = json.loads(raw)
    except Exception:
        # Fallback: minimal extraction
        lg = detect_lang(user_text) or "en"
        f, t = _default_time_window(user_text)
        # naive keyword split
        toks = re.findall(r"[\w\-]+", user_text or "", flags=re.UNICODE)
        # keep unique tokens longer than 2 chars
        seen = set()
        kw = []
        for tok in toks:
            tk = tok.strip(' _-').lower()
            if len(tk) > 2 and tk not in seen:
                kw.append(tok)
                seen.add(tk)
        plan = {
            "language": "en",
            "time_window": {"from": f, "to": t},
            "entities": {"locations": [], "organizations": [], "people": []},
            "categories": [],
            "keywords": kw[:12] or [user_text]
        }
    # Fill missing time_window
    tw = plan.get("time_window") or {}
    if not tw.get("from") or not tw.get("to"):
        f, t = _default_time_window(user_text)
        plan["time_window"] = {"from": f, "to": t}
    # Force language to English
    plan["language"] = "en"
    # Dedup keywords (case-insensitive)
    dedup = []
    seen = set()
    for k in plan.get("keywords", []):
        s = (k or "").strip()
        low = s.lower()
        if s and low not in seen:
            dedup.append(s)
            seen.add(low)
    plan["keywords"] = dedup
    return plan


def build_boolean_query(plan: Dict) -> str:
    """
    Build a boolean query string for news APIs using the plan.
    Combines keywords + entities + categories into quoted ORs.
    No hard-coded wordlists; it uses only the plan content.
    """
    kws: List[str] = list(plan.get("keywords", []) or [])
    ents = plan.get("entities", {}) or {}
    for loc in ents.get("locations", []) or []:
        kws.append(str(loc))
    for org in ents.get("organizations", []) or []:
        kws.append(str(org))
    for person in ents.get("people", []) or []:
        kws.append(str(person))
    for cat in plan.get("categories", []) or []:
        kws.append(str(cat))

    # Dedup
    uniq: List[str] = []
    seen = set()
    for k in kws:
        kk = (k or "").strip()
        low = kk.lower()
        if kk and low not in seen:
            uniq.append(kk)
            seen.add(low)

    if not uniq:
        return ""

    # Quote multi-word phrases; join with OR
    def quote(s: str) -> str:
        s = s.strip()
        return f'"{s}"' if " " in s else s

    return " OR ".join(quote(x) for x in uniq)
