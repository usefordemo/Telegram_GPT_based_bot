from pathlib import Path

# ══════════════════════════════
#  ❖  Static dictionaries
# ══════════════════════════════
message_counts: dict[int, int]           = {}
thresholds: dict[int, int]               = {}
conversation_histories: dict[int, list]  = {}

custom_responses = {
    "zh": "我不好说",
    "en": "I can't say that",
}

# ── Sector / ticker maps ──────────────────────────────────────────
stock_sectors = {
    "Technology": [
        "Apple", "Microsoft", "NVIDIA", "AMD", "Google",
        "Meta", "Amazon", "Tesla", "ASML", "TSMC",
    ],
    "Healthcare": [
        "Pfizer", "Moderna", "Johnson & Johnson", "UnitedHealth",
        "AbbVie", "Gilead", "Amgen", "Merck",
    ],
    "Defense": [
        "Lockheed Martin", "Raytheon", "Northrop Grumman",
        "General Dynamics", "Boeing",
    ],
    "Internet": [
        "Netflix", "Alibaba", "Tencent", "Baidu", "Snap", "Zoom",
    ],
}

company_to_ticker = {
    "Apple": "AAPL", "Microsoft": "MSFT", "NVIDIA": "NVDA", "AMD": "AMD",
    "Google": "GOOGL", "Meta": "META", "Amazon": "AMZN", "Tesla": "TSLA",
    "ASML": "ASML", "TSMC": "TSM", "Pfizer": "PFE", "Moderna": "MRNA",
    "Johnson & Johnson": "JNJ", "UnitedHealth": "UNH", "AbbVie": "ABBV",
    "Gilead": "GILD", "Amgen": "AMGN", "Merck": "MRK",
    "Lockheed Martin": "LMT", "Raytheon": "RTX", "Northrop Grumman": "NOC",
    "General Dynamics": "GD", "Boeing": "BA", "Netflix": "NFLX",
    "Alibaba": "BABA", "Tencent": "TCEHY", "Baidu": "BIDU",
    "Snap": "SNAP", "Zoom": "ZM",
}

crypto_keywords = [
    "Bitcoin", "Ethereum", "BNB", "Solana", "XRP",
    "Cardano", "Dogecoin", "Toncoin", "Polkadot", "Avalanche",
]
macro_topics = ["Federal Reserve", "tariff", "trade war"]

# ── Topic mapping helpers ─────────────────────────────────────────
TOPIC_MAP = {
    "中国": ("China", "中国", "China"),
    "亚洲": ("Asia", "亚洲", "Asia"),
    "东南亚": ("(Southeast Asia OR ASEAN)", "东南亚", "Southeast Asia"),
    "科技": ("technology OR tech", "科技", "Technology"),
    "军事": ("military OR defense OR war", "军事", "Military"),
    # English keys
    "china": ("China", "中国", "China"),
    "asia": ("Asia", "亚洲", "Asia"),
    "technology": ("technology OR tech", "科技", "Technology"),
    "military": ("military OR defense OR war", "军事", "Military"),
}

ASIA_SUB_KEYS = ["中国", "日本", "东南亚"]

AUTHORITATIVE_SOURCES = [
    "bbc-news", "the-new-york-times", "associated-press", "reuters", "vox",
    "bloomberg", "the-wall-street-journal", "financial-times", "cnbc",
    "the-verge", "techcrunch", "wired", "engadget",
    "cnn", "the-washington-post", "nbc-news", "abc-news",
]

# ── Prompts, enter your prompt for the character in txt file here ───────────────────────────────────────────────────────
CHARACTER_DESCRIPTION: str = (
    Path(__file__).with_name("character_prompt.txt")
    .read_text(encoding="utf-8")
)

QUANT_ANALYST_PROMPT = (
    "You are a rigorous quantitative analyst proficient in interpreting stock, "
    "cryptocurrency, and economic news data. "
    "Please provide concise insights and strategy recommendations "
    "based on the following market data."
)

sticker_mapping = {
    "happy": "CAACAgEAAxkBAAExrolnrG_lD3Yyw5L5lfrlS1rR8c6ZVwACDAADnf3SFUxbeMHoIqUoNgQ",
    "sad": "CAACAgEAAxkBAAExrotnrHAcNGNHJmNcoyfJR-E5C_cb9QACMwADnf3SFR16drSJKo_MNgQ",
    "neutral": "CAACAgIAAxkBAAEFneutralStickerID",
    "angry": "CAACAgEAAxkBAAExro9nrHA-3dn4LQa7zDv2kSojpjDMggACMQADnf3SFfX0oxGrpZqENgQ",
    "surprised": "CAACAgEAAxkBAAExr_dnrJb2vnTYx-iLqQI4tlDk4CB2FwACDQADnf3SFUv4sN9zB59TNgQ",
    "amused": "CAACAgEAAxkBAAExr_FnrJaqAAEP1ZQMliX4VPa2abyV7Q0AAhkAA5390hXBgbb3AfLucTYE",
    "playful": "CAACAgEAAxkBAAExsBNnrJl7qCeomM3vwe9frVroBr9KygACGwADnf3SFYKoncdDi6jVNgQ",
    "speechless": "CAACAgEAAxkBAAExsBlnrJnqbm0taX6EOPqSGN4Jd1Q4NQACJAADnf3SFboavMOqISuWNgQ",
    "excited": "CAACAgEAAxkBAAExsB1nrJojd2uwIPbUGBkEdd8G9TFM-QACJQADnf3SFb3t8m8-12SaNgQ",
    "shyness": "CAACAgEAAxkBAAExsCFnrJqs0fJ7IlRnX4ngBDVA2uEeewACIwADnf3SFTZiyK8Z11CXNgQ",
    "sarcastic": "CAACAgEAAxkBAAExr_FnrJaqAAEP1ZQMliX4VPa2abyV7Q0AAhkAA5390hXBgbb3AfLucTYE",
    "cynical": "CAACAgEAAxkBAAExsDVnrJyqHMn5TnOrhcre5ap8RUB3vgACDwADnf3SFQ0M5SOvI3bpNgQ",
    "frustrated": "CAACAgEAAxkBAAExsFFnrKAIfoMcpM5VRcjuBUVo6etSMAACLQADnf3SFRqTpZ2ItxhfNgQ",
    "concerned": "CAACAgEAAxkBAAExsF1nrKELucr0PRoXb26DdHrv2pJjnAACKgADnf3SFb0l2wOPA3vLNgQ",
    "ironic": "CAACAgEAAxkBAAExsNZnrLKwY1N15D0TNZRzldqFnoMkQQACKQADnf3SFXedkiIcbw5QNgQ",
    "pround": "CAACAgEAAxkBAAExvuVnryKjU9h7FgkDowo2zveKRzw8GgACGAADnf3SFe-GyK0JGzMqNgQ",
    #"apologetic":
}

QUANT_ANALYST_PROMPT = (
    "You are a rigorous quantitative analyst proficient in interpreting stock, cryptocurrency, and economic news data. "
    "Please make concise and logical assessments and strategy recommendations based on the following market data."
)

with open("C:/your_file_path/character_prompt.txt", "r",
          encoding="utf-8") as f:
          CHARACTER_DESCRIPTION = f.read()