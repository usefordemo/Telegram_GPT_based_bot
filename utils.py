import re, os, datetime as _dt
from telegram import Update
from telegram.ext import ContextTypes
from openai import OpenAI
import openai
from API import OPENAI_API_KEY
from datetime import datetime
from data import TOPIC_MAP
from openai import AsyncOpenAI
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def analyze_emotion_with_chatgpt(text: str) -> str:
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
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    emotion = response.choices[0].message.content.strip().lower()
    return emotion

async def log_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.sticker:
        print("Sticker file ID:", update.message.sticker.file_id)

def is_image_request_keyword(message: str) -> bool:
    keywords = ["generate image", "draw", "create image", "sketch", "picture"]
    return any(keyword in message.lower() for keyword in keywords)

async def detect_lang(text: str) -> str:
    system_prompt = (
        "You are an expert in language analysis. Please analyze the following text and determine which language is"
        "the emotion expressed. Return only one language label"
        "without any additional text. eg 'ar','de','en','es','fr','he','it','nl','no','pt','ru','sv','ud','zh'"
    )
    user_prompt = f"Text: \"{text}\""
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    language = response.choices[0].message.content.strip().lower()
    return language

def log_message(user_id, username, text):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists("logs"):
        os.makedirs("logs")
    filename = username if username else f"Prof_{user_id}"
    filepath = f"logs/{filename}.txt"
    # Print the log file location
    print(f"Logging message to: {filepath}")

    try:
        with open(filepath, 'a', encoding='utf-8') as file:
            file.write(f"{current_time} - {text}\n")
    except Exception as e:
        print(f"Failed to write to log file: {e}")

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

async def is_image_request(message: str) -> bool:
    if not message:
        return False
    # First try a direct keyword check
    if is_image_request_keyword(message):
        return True
    # Fallback to GPT detection
    try:
        response = await openai_client.chat.completions.create(
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
    ▸ True ⇢ Group chat message that "requires immediate response"
    1. Direct reply to the bot
    2. @BotUsername explicitly appears in the text
    """

    if not update.message:
        return False

    text         = (update.message.text or "").lower()
    bot_username = f"@{(context.bot.username or '').lower()}"

    # reply Bot
    if (update.message.reply_to_message and
        update.message.reply_to_message.from_user.id == context.bot.id):
        return True

    # mention
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
    if "week" in text:
        start = today - timedelta(days=today.weekday())
    elif "yesterday" in text:
        start = today - timedelta(days=1)
        today = start
    elif "last week" in text:
        start = today - timedelta(days=today.weekday()+7)
        today = start + timedelta(days=6)
    else:  # default -> today
        start = today
    return start.isoformat(), today.isoformat()


def parse_news_topic(msg: str) -> tuple[str, str] | None:
    '''
    Extract (topic_key, period) from user messages
    period: "day" | "week" | "month"
    Supports:
    • English: Today / This week / This month XX news
    '''
    txt = msg.strip().lower()

    # period kew words
    en_map = {
        "today": "day", "today's": "day",
        "this week": "week", "thisweek": "week",
        "this month": "month", "thismonth": "month",
    }

    en_pat = r"(" + "|".join(en_map.keys()) + r")\s*([a-z ]+?)\s*news"
    m = re.search(en_pat, txt)
    if m:
        period = en_map[m.group(1)]
        topic  = m.group(2).strip()
        return (topic, period) if topic in TOPIC_MAP else None

    return None


#classify user intent
async def classify_intent(question: str) -> str:
    prompt = (
        "You are a router. Decide whether the user's question "
        "should be answered directly or directly comment news(label: chat) or requires "
        "fetching FRESH NEWS (label: news).\n"
        "Return only 'news' or 'chat'.\n\n"
        "### Examples\n"
        "User: Today's technology news\nAssistant: news\n"
        "User: what happened in asia today?\nAssistant: news\n"
        "User: may I ask who you are\nAssistant: chat\n"
        "User: This week's military news\nAssistant: news\n"
        "User: See the warning?\nAssistant: chat\n"
        "User: {Trump said he has already found a buyer for TikTok."
        "U.S. President Donald Trump told Fox News in an interview,"
        "that a buyer had been found for the short-video sharing platform TikTok."
        "He described them as a group of very wealthy people,"
        "and said he would announce the buyers' names in about two weeks."
        "Earlier this month, Trump signed an executive order,"
        "once again extending the deadline for China’s ByteDance"
        "to sell its TikTok U.S. operations by 90 days, until September 17."
        "TikTok at the time issued a statement thanking Trump"
        "for his leadership and support in ensuring that TikTok remained available."
        "In Beijing, China’s Ministry of Foreign Affairs responded last month"
        "to questions regarding TikTok, saying that China would handle the matter"
        "in accordance with its own laws and regulations,"
        "and that the U.S. should provide an open, fair, just,"
        "and non-discriminatory business environment for Chinese companies operating in the U.S."
        "— RTHK, Bloomberg, Forbes News}\nAssistant: chat\n"
        f"User: {question}"
    )
    resp = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip().lower()