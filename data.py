import os, json
# Global dictionaries to store message counts and conversation histories
message_counts = {}
thresholds = {}
conversation_histories = {}

custom_responses = {
    "en": "I can't say that"
}

stock_sectors = {
    "Technology": ["Apple", "Microsoft", "NVIDIA", "AMD", "Google", "Meta", "Amazon", "Tesla", "ASML", "TSMC"],
    "Healthcare": ["Pfizer", "Moderna", "Johnson & Johnson", "UnitedHealth", "AbbVie", "Gilead", "Amgen", "Merck"],
    "Defense": ["Lockheed Martin", "Raytheon", "Northrop Grumman", "General Dynamics", "Boeing"],
    "Internet": ["Netflix", "Alibaba", "Tencent", "Baidu", "Snap", "Zoom"]
}

company_to_ticker = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "AMD": "AMD",
    "Google": "GOOGL",
    "Meta": "META",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "ASML": "ASML",
    "TSMC": "TSM",
    "Pfizer": "PFE",
    "Moderna": "MRNA",
    "Johnson & Johnson": "JNJ",
    "UnitedHealth": "UNH",
    "AbbVie": "ABBV",
    "Gilead": "GILD",
    "Amgen": "AMGN",
    "Merck": "MRK",
    "Lockheed Martin": "LMT",
    "Raytheon": "RTX",
    "Northrop Grumman": "NOC",
    "General Dynamics": "GD",
    "Boeing": "BA",
    "Netflix": "NFLX",
    "Alibaba": "BABA",
    "Tencent": "TCEHY",
    "Baidu": "BIDU",
    "Snap": "SNAP",
    "Zoom": "ZM"
}

crypto_keywords = ["Bitcoin", "Ethereum", "BNB", "Solana", "XRP", "Cardano", "Dogecoin", "Toncoin", "Polkadot", "Avalanche"]

macro_topics = ["Federal Reserve rate hike", "Federal Reserve rate cut", "tariff", "trade war", "US inflation", "economic slowdown", "recession"]

crypto_list = ["bitcoin", "ethereum", "solana", "dogecoin"]
macro_topics = ["Federal Reserve", "tariff", "trade war"]

TOPIC_MAP = {
    "technology": ("technology OR tech","Technology"),
    "military":   ("military OR defense OR war","Military"),
}


AUTHORITATIVE_SOURCES = [
    # ① General / World
    "bbc-news", "the-new-york-times", "associated-press", "reuters", "vox",
    # ② Finance
    "bloomberg", "the-wall-street-journal", "financial-times", "cnbc",
    # ③ Technology
    "the-verge", "techcrunch", "wired", "engadget",
    # ④ Large-scale broadcasting/newspapers
    "cnn", "the-washington-post", "nbc-news", "abc-news",
    'the-new-york-times', 'vox', 'financial-times', 'cnbc',
]


BASE_DIR = os.path.dirname(__file__)
sticker_mapping_path = os.path.join(BASE_DIR, "sticker_mapping.txt")

with open(sticker_mapping_path, "r", encoding="utf-8") as f:
    sticker_mapping = json.load(f)

QUANT_ANALYST_PROMPT = (
    "You are a quantitative analyst proficient in interpreting stock, cryptocurrency, and economic news data. "
    "Please make concise and logical assessments and strategy recommendations based on the following market data."
)

PROMPT_FILE = os.path.join(BASE_DIR, "user_character_prompt.txt")

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    CHARACTER_DESCRIPTION = f.read()
