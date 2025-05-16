import re, os, datetime as _dt
from telegram import Update
from telegram.ext import ContextTypes
from openai import OpenAI
import openai
from config import OPENAI_API_KEY
from datetime import datetime
openai_client = OpenAI(api_key=OPENAI_API_KEY)
from data import TOPIC_MAP,CHARACTER_DESCRIPTION

def analyze_emotion_with_chatgpt(text: str) -> str:
    """
    Uses OpenAI's API to analyze the given text and returns a single emotion label
    (e.g., happy, sad, neutral, angry, surprised).
    """
    system_prompt = (
        "You are an expert in emotion analysis. Please analyze the following text and determine "
        "the emotion expressed. Return only one emotion label (e.g., happy, sad, neutral, angry, surprised) "
        "without any additional text."
    )
    user_prompt = f"Text: \"{text}\""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    emotion = response.choices[0].message.content.strip().lower()
    return emotion


def is_image_request_keyword(message: str) -> bool:
    keywords = ["generate image", "draw", "create image", "sketch", "picture"]
    return any(keyword in message.lower() for keyword in keywords)

def detect_lang(text: str) -> str:
    """
    超轻量 – 只分中文 / 日文 / 其它
    """
    if re.search(r'[\u4e00-\u9fff]', text):
        return "zh"
    if re.search(r'[\u3040-\u30ff]', text):
        return "ja"
    return "en"


def extract_image_size(prompt: str, default_size: str = "1024x1024") -> str:
    """
    Extracts an image size from the prompt. It will look for a pattern like
    '512x1024', '512 * 1024', or '512*1024' and returns it in the format "512x1024".
    If no valid size is found, returns the default size.
    """
    # This regex matches one or more digits, optional whitespace, an 'x' or '*',
    # optional whitespace, and one or more digits.
    match = re.search(r'(\d+\s*[x*]\s*\d+)', prompt)
    if match:
        size_str = match.group(1)
        # Remove any spaces and replace '*' with 'x'
        size_str = re.sub(r'\s+', '', size_str).replace('*', 'x')
        return size_str
    return default_size

async def is_image_request(message):
    if not message:
        return False
    # First try a direct keyword check
    if is_image_request_keyword(message):
        return True
    # Fallback to GPT detection
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Answer only 'yes' or 'no'. "
                                              "Is the following message a request to generate or describe an image?"},
                {"role": "user", "content": message}
            ]
        )
        reply = response.choices[0].message.content.strip().lower()
        print(f"Image request detection reply: '{reply}'")  # Debug output
        return "yes" in reply or "true" in reply
    except Exception as e:
        print(f"Failed to detect image request intent: {e}")
        return False

def is_direct_reply_or_mention(update: Update,
                               context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    ▸ True  ⇢  是“需要立即响应”的群聊消息
       1. 对 Bot 的直接回复
       2. 文本中显式出现 @BotUsername
    """
    if not update.message:
        return False

    text         = (update.message.text or "").lower()
    bot_username = f"@{(context.bot.username or '').lower()}"

    # 1️⃣ 回复 Bot
    if (update.message.reply_to_message and
        update.message.reply_to_message.from_user.id == context.bot.id):
        return True

    # 2️⃣ @提及
    return bot_username and bot_username in text

def parse_time_window(text: str) -> tuple[str, str]:
    """
    Returns (from_date, to_date) in ISO‑8601.
    • "today"  -> (yyyy‑mm‑dd, same)
    • "this week" -> (Mon, today)
    • fallback: yesterday
    """
    text = text.lower()
    today = date.today()
    if "week" in text or "这周" in text:
        start = today - timedelta(days=today.weekday())
    elif "yesterday" in text or "昨天" in text:
        start = today - timedelta(days=1)
        today = start
    elif "last week" in text or "上周" in text:
        start = today - timedelta(days=today.weekday()+7)
        today = start + timedelta(days=6)
    else:  # default -> today
        start = today
    return start.isoformat(), today.isoformat()


def parse_news_topic(msg: str) -> tuple[str, str] | None:
    """
    从用户消息里提取 (topic_key, period)
    period: "day" | "week" | "month"
    支持：
      • 中文：今日/今天 / 今周/本周 / 今月/本月 XX 新闻
      • 英文：today / this week / this month XX news
    """
    txt = msg.strip().lower()

    # ====== 定义 period 关键词 ======
    zh_map = {
        "今日": "day", "今天": "day",
        "今周": "week", "本周": "week", "这周": "week",
        "今月": "month", "本月": "month", "这个月": "month",
    }
    en_map = {
        "today": "day", "today's": "day",
        "this week": "week", "thisweek": "week",
        "this month": "month", "thismonth": "month",
    }

    # ① 中文匹配
    zh_pat = r"(" + "|".join(map(re.escape, zh_map.keys())) + r")\s*([\u4e00-\u9fff]+?)\s*新闻"
    m = re.search(zh_pat, msg, re.I)
    if m:
        period = zh_map[m.group(1)]
        topic  = m.group(2)
        return (topic, period) if topic in TOPIC_MAP else None

    # ② 英文匹配
    en_pat = r"(" + "|".join(en_map.keys()) + r")\s*([a-z ]+?)\s*news"
    m = re.search(en_pat, txt)
    if m:
        period = en_map[m.group(1)]
        topic  = m.group(2).strip()
        return (topic, period) if topic in TOPIC_MAP else None

    return None


#classify user intent
def classify_intent(question: str) -> str:
    prompt = (
        "You are a router. Decide whether the user's question "
        "should be answered directly (label: chat) or requires "
        "fetching FRESH NEWS (label: news).\n"
        "Return only 'news' or 'chat'.\n\n"
        "### Examples\n"
        "User: 今天科技新闻\nAssistant: news\n"
        "User: what happened in asia today?\nAssistant: news\n"
        "User: 请问你是谁\nAssistant: chat\n"
        "User: 今周军事新闻\nAssistant: news\n"
        "### End\n\n"
        f"User: {question}"
    )
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip().lower()