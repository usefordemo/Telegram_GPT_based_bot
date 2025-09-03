from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Literal, Optional

import schedule
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

# Project imports (leave as-is to match your structure)
from image_cashe import download_telegram_photo_as_png
from API import TELEGRAM_BOT_TOKEN, openai_client, DAILY_DIGEST_ANALYST

# add (right below it):
try:
    from API import TELEGRAM_PROXY_URL  # optional, define in API.py if you use a proxy
except Exception:
    TELEGRAM_PROXY_URL = os.getenv("TELEGRAM_PROXY_URL", "").strip() or None

from news_service import make_headline_brief, make_topic_headline_brief
from semantic_news import make_semantic_news_brief
from finance_service import generate_investment_analysis
from data import (
    custom_responses,
    message_counts,
    thresholds,
    conversation_histories,
    sticker_mapping,
    TOPIC_MAP,
    CHARACTER_DESCRIPTION,
    QUANT_ANALYST_PROMPT,
)
from utils import (
    detect_lang,
    classify_intent,
    is_direct_reply_or_mention,
    parse_news_topic,
    is_image_request,           # available if you still want to use; router doesn't require it
    extract_image_size,
    analyze_emotion_with_chatgpt,
)
from news_service import news_health_check

# --------------------------------------------------------------------------------------
# Globals
# --------------------------------------------------------------------------------------

# Cache latest image per chat for edit flows
LAST_IMAGE: Dict[int, BytesIO] = {}

# Router intent types
RouterIntent = Literal["image_generate", "image_edit", "image_describe", "news", "chat"]

# When LLM answers with boilerplate refusal words, we retry (simple heuristic)
REFUSAL_KEYWORDS = ("sorry", "apologize", "cannot", "can't")

# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


async def chat_with_retries(
    model: str,
    messages: list[dict],
    max_retries: int = 5,
    backoff: float = 1.0,
) -> str:
    """
    Call Chat Completions with minimal backoff + simple refusal heuristic.
    """
    last_error: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = await openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
            content = (resp.choices[0].message.content or "").strip()
            if any(kw in content.lower() for kw in REFUSAL_KEYWORDS) and attempt < max_retries:
                await asyncio.sleep(backoff * attempt)
                continue
            return content or ""
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                await asyncio.sleep(backoff * attempt)
            else:
                break
    return last_error or "The request could not be completed."


async def safe_reply_text(msg, text: str, **kwargs):
    MAX = 4096
    if not text:
        return
    for i in range(0, len(text), MAX):
        await msg.reply_text(text[i:i+MAX], **kwargs)


def bytes_to_telegram_photo(b64_json: str, filename: str = "image.png") -> BytesIO:
    """
    Decode b64 JSON string from gpt-image-1 into a BytesIO suitable for Telegram send.
    """
    raw = base64.b64decode(b64_json)
    bio = BytesIO(raw)
    bio.name = filename
    bio.seek(0)
    return bio


def file_to_data_uri(bio: BytesIO, mime: str = "image/png") -> str:
    """
    Convert a BytesIO image to data URI for the vision chat pathway.
    """
    pos = bio.tell()
    bio.seek(0)
    data = bio.read()
    bio.seek(pos)
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"

# --------------------------------------------------------------------------------------
# Language-agnostic LLM Router
# --------------------------------------------------------------------------------------


async def detect_router_intent(
    text: str,
    has_prev_image: bool,
    has_attached_image: bool,
) -> RouterIntent:
    """
    Language-agnostic router. No hard-coded keyword lists.
    Returns exactly one of:
      image_generate | image_edit | image_describe | news | chat
    """
    text = (text or "").strip()
    context = json.dumps(
        {
            "has_prev_image": bool(has_prev_image),
            "has_attached_image": bool(has_attached_image),
        },
        ensure_ascii=False,
    )

    # Tiny, fast classification call
    resp = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a short-answer router. The user can speak ANY language.\n"
                    "Read the user's text and output EXACTLY one label from this set:\n"
                    "image_generate | image_edit | image_describe | news | chat\n"
                    "- image_generate: user asks to create an image (no source image required)\n"
                    "- image_edit: user wants to modify an existing/attached image\n"
                    "- image_describe: user wants a description/analysis of an image\n"
                    "- news: user wants headlines, a briefing/report, or topic/period news\n"
                    "- chat: all other general conversation or Q&A\n"
                    "Consider this JSON context about the message environment."
                ),
            },
            {"role": "user", "content": f"CONTEXT={context}\nTEXT={text}"},
        ],
    )
    label = (resp.choices[0].message.content or "").strip().lower()

    # Defensive normalization for near misses
    alias = {
        "image generation": "image_generate",
        "generate": "image_generate",
        "image-gen": "image_generate",
        "image edit": "image_edit",
        "edit": "image_edit",
        "describe": "image_describe",
        "analysis": "image_describe",
    }
    label = alias.get(label, label)
    if label not in {"image_generate", "image_edit", "image_describe", "news", "chat"}:
        # Reasonable fallback: if there is an attached image, describe; else chat.
        return "image_describe" if has_attached_image else "chat"
    return label  # type: ignore[return-value]

# --------------------------------------------------------------------------------------
# Core actions: describe / generate / edit
# --------------------------------------------------------------------------------------


async def describe_image_with_vision(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    prompt_text: str,
    source_png: BytesIO,
) -> None:
    """
    Describe an image via vision chat using data URI for reliability.
    """
    data_uri = file_to_data_uri(source_png, "image/png")
    messages = [
        {"role": "system", "content": CHARACTER_DESCRIPTION},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        },
    ]
    desc = await chat_with_retries("gpt-4o", messages)
    await update.message.reply_text(desc[:4000], parse_mode="Markdown")


async def generate_image_from_prompt(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    prompt: str,
) -> None:
    """
    Generate brand new image using gpt-image-1. Accepts size hints like 1024x1536.
    """
    size = extract_image_size(prompt) or "1024x1024"
    clean_prompt = re.sub(r"\d+\s*[x*]\s*\d+", "", prompt).strip()
    try:
        rsp = await openai_client.images.generate(
            model="gpt-image-1",
            prompt=clean_prompt,
            size=size,
        )
        photo = bytes_to_telegram_photo(rsp.data[0].b64_json, "gen.png")
        await update.message.reply_photo(photo=photo)
    except Exception as e:
        print("[ImageGenErr]", e)
        await update.message.reply_text("😿 Generate failed—please tweak your prompt or try a standard size.")


async def edit_image_with_prompt(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    source_png: BytesIO,
    prompt: str,
) -> None:
    """
    Edit the last/attached image with gpt-image-1.
    """
    source_png.seek(0)
    try:
        rsp = await openai_client.images.edit(
            model="gpt-image-1",
            image=source_png,
            prompt=prompt,
            size="1024x1024",
        )
        photo = bytes_to_telegram_photo(rsp.data[0].b64_json, "edit.png")
        await update.message.reply_photo(photo=photo)
    except Exception as e:
        print("[ImageEditErr]", e)
        await update.message.reply_text("😿 Edit failed—try rephrasing or simplify the instruction.")

# --------------------------------------------------------------------------------------
# High-level “user text” path (news vs regular chat)
# --------------------------------------------------------------------------------------


async def process_user_text(
    chat_id: int,
    user_id: int,
    username: str,
    chat_type: str,
    text: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> str:
    """
    Your original text processing, preserved and lightly cleaned.
    - Detects 'news' intent, then:
        • topic-style news -> make_topic_headline_brief
        • otherwise -> make_semantic_news_brief
    - Falls back to regular chat with CHARACTER_DESCRIPTION persona
    - Prints IDE breadcrumbs for debugging
    """
    key = user_id if chat_type == "private" else chat_id

    if key not in message_counts:
        message_counts[key] = 0
        thresholds[key] = random.randint(1000, 2000)
        conversation_histories[key] = [{"role": "system", "content": CHARACTER_DESCRIPTION}]

    conversation_histories[key].append({"role": "user", "content": text})
    chat_history = conversation_histories[key][-500:]

    # 1) Classify intent
    try:
        intent = await classify_intent(text)
    except Exception:
        intent = "chat"

    print(f"[tg_bot] router intent={intent} | prev_img={False} | text={text!r}")  # IDE breadcrumb

    # 2) NEWS route
    if intent == "news":
        try:
            # Try explicit 'topic' requests first (e.g., 'tech this week', 'markets today')
            tk = parse_news_topic(text)  # must return (topic_key, period) or None
            if tk:
                topic_key, period = tk
                lang = await detect_lang(text)
                print(f"[tg_bot] topic-news topic={topic_key!r} period={period!r} lang={lang}")  # IDE breadcrumb
                return await make_topic_headline_brief(topic_key, period, lang)

            # Otherwise do semantic news
            print(f"[tg_bot] calling make_semantic_news_brief for {text!r}")  # IDE breadcrumb
            return await make_semantic_news_brief(text)

        except Exception as e:
            import traceback
            print("[tg_bot] make_semantic_news_brief/topic crashed:", e)
            traceback.print_exc()
            return "News fetch crashed. Run /newshealth and check logs/news_service.log."

    # 3) Regular chat (fallback)
    rsp = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=chat_history,
    )
    reply = (rsp.choices[0].message.content or "").strip()
    conversation_histories[key].append({"role": "assistant", "content": reply})
    conversation_histories[key] = conversation_histories[key][-3000:]
    return reply


# --------------------------------------------------------------------------------------
# Telegram Handlers
# --------------------------------------------------------------------------------------


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Photo/doc image handler:
      - Cache the PNG in LAST_IMAGE
      - If caption looks like edit -> edit
      - Else -> describe (vision)
    """
    if update.effective_chat.type != "private" and not is_direct_reply_or_mention(update, context):
        return

    chat_id = update.effective_chat.id
    file_id = update.message.photo[-1].file_id if update.message.photo else None
    if file_id is None and update.message.document and update.message.document.mime_type.startswith("image/"):
        file_id = update.message.document.file_id

    if not file_id:
        return

    # Download & cache as PNG
    source = await download_telegram_photo_as_png(context.bot, file_id)
    LAST_IMAGE[chat_id] = source

    caption = (update.message.caption or "").strip()
    user_input = caption or ""

    # Decide: edit or describe (language-agnostic)
    intent = await detect_router_intent(
        text=user_input or "Describe this image.",
        has_prev_image=True,
        has_attached_image=True,
    )

    if intent == "image_edit":
        prompt = user_input or "Improve quality while keeping original content."
        await edit_image_with_prompt(update, context, source_png=source, prompt=prompt)
        return

    # Default: describe
    prompt_text = user_input or CHARACTER_DESCRIPTION
    await describe_image_with_vision(update, context, prompt_text, source)
    return


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Voice message handler (semantic-first):
      - Transcribe (tries gpt-4o-mini-transcribe → whisper-1 fallback)
      - Route through the same text pipeline (news/image/chat) via process_user_text
      - Reply with text + TTS (if possible), plus optional sticker
    """
    if update.effective_chat.type != "private" and not is_direct_reply_or_mention(update, context):
        return

    # Download OGG/Opus
    try:
        file = await context.bot.get_file(update.message.voice.file_id)
    except Exception as e:
        print("[voice] get_file failed:", e)
        await update.message.reply_text("Could not fetch the voice file from Telegram.")
        return

    # Save to temp & transcribe
    transcript = ""
    with tempfile.NamedTemporaryFile(suffix=".ogg") as tmp:
        try:
            await file.download_to_drive(tmp.name)
        except Exception as e:
            print("[voice] download failed:", e)
            await update.message.reply_text("Could not download the voice clip.")
            return

        # Try gpt-4o-mini-transcribe first
        try:
            with open(tmp.name, "rb") as f:
                r1 = await openai_client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=f,
                )
            transcript = (getattr(r1, "text", "") or "").strip()
        except Exception as e1:
            print("[voice] gpt-4o-mini-transcribe failed, trying whisper-1:", e1)
            try:
                with open(tmp.name, "rb") as f:
                    r2 = await openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                    )
                transcript = (getattr(r2, "text", "") or "").strip()
            except Exception as e2:
                print("[voice] whisper-1 failed:", e2)

    if not transcript:
        await update.message.reply_text("I couldn’t hear anything clear in that clip—mind trying again?")
        return

    # Route exactly like text
    try:
        reply_text = await process_user_text(
            chat_id=update.effective_chat.id,
            user_id=update.effective_user.id,
            username=update.effective_user.username or "",
            chat_type=update.effective_chat.type,
            text=transcript,
            update=update,
            context=context,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        await update.message.reply_text(f"Voice processing crashed: {e}")
        return

    # Send textual reply
    await safe_reply_text(update.message, reply_text)

    # TTS of the reply
    try:
        tts = await openai_client.audio.speech.create(
            model="gpt-4o-mini-tts",
            input=reply_text,
            voice="echo",
        )
        audio_bytes = await tts.aread()
        bio = BytesIO(audio_bytes)
        bio.name = "voice.ogg"
        await update.message.reply_voice(voice=bio)
    except Exception as e:
        print("[voice] TTS failed:", e)

    # Optional mood sticker
    try:
        emotion = await analyze_emotion_with_chatgpt(reply_text)
        sticker_id = sticker_mapping.get(emotion)
        if sticker_id:
            await update.message.reply_sticker(sticker_id)
    except Exception:
        pass


async def handle_edit_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /edit <instruction> command: edits the last cached image for the chat.
    """
    chat_id = update.effective_chat.id
    source = LAST_IMAGE.get(chat_id)
    if source is None:
        await update.message.reply_text("😿 Can't find a recent image—send one first, then /edit <what to change>.")
        return

    prompt = update.message.text.replace("/edit", "", 1).strip()
    if not prompt:
        await update.message.reply_text("Usage: reply /edit <what to change>")
        return

    await edit_image_with_prompt(update, context, source_png=source, prompt=prompt)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Text message handler:
      - Respect private/group mention logic
      - Analyze prompt first with router (language-agnostic)
      - Route: image_generate / image_edit / news / chat
      - If image_describe w/o attachment -> treat as chat request to describe missing image
    """
    # Ignore images here; photo/doc handler will catch them
    if update.message.photo or (
        getattr(update.message, "document", None) and str(update.message.document.mime_type or "").startswith("image/")
    ):
        return

    chat = update.effective_chat
    chat_type = chat.type  # "private" | "group" | "supergroup"
    chat_id = chat.id
    user_id = update.effective_user.id
    username = update.effective_user.username or ""

    # Gate: only respond privately, or when directly addressed in groups
    key = user_id if chat_type == "private" else chat_id
    triggered_by_reply = chat_type != "private" and is_direct_reply_or_mention(update, context)
    triggered_by_random = chat_type != "private" and message_counts.get(key, 0) >= thresholds.get(key, 0)
    triggered_private = chat_type == "private"
    should_respond = triggered_private or triggered_by_reply or triggered_by_random
    if not should_respond:
        return
    if triggered_by_random:
        thresholds[key] = random.randint(1000, 2000)
        message_counts[key] = 0

    # Gather user input (text only here)
    user_input = (update.message.text or "").strip()

    # ------------------ FIRST: LLM ROUTER ------------------
    has_prev_image = chat_id in LAST_IMAGE
    intent = await detect_router_intent(
        text=user_input,
        has_prev_image=has_prev_image,
        has_attached_image=False,
    )
    print(f"[tg_bot] router intent={intent} | prev_img={has_prev_image} | text={user_input[:80]!r}")

    # a) image_generate
    if intent == "image_generate":
        await generate_image_from_prompt(update, context, user_input)
        return

    # b) image_edit (requires LAST_IMAGE)
    if intent == "image_edit":
        if not has_prev_image:
            await update.message.reply_text("Send an image first, then tell me how to edit it.")
            return
        await edit_image_with_prompt(update, context, source_png=LAST_IMAGE[chat_id], prompt=user_input)
        return

    # c) image_describe without an attachment -> gently redirect
    if intent == "image_describe" and not has_prev_image:
        await update.message.reply_text("Please send the image you want me to describe. 🙂")
        return
    # -------------------------------------------------------

    # Track conversation (your original memory logic)
    message_counts.setdefault(key, 0)
    thresholds.setdefault(key, random.randint(1000, 2000))
    conversation_histories.setdefault(key, [{"role": "system", "content": CHARACTER_DESCRIPTION}])
    conversation_histories[key].append({"role": "user", "content": user_input})
    conversation_histories[key] = conversation_histories[key][-500:]
    message_counts[key] += 1

    # News or chat
    reply_text = await process_user_text(
        chat_id, user_id, username, chat_type, user_input, update, context
    )

    # Send reply
    await safe_reply_text(update.message, reply_text)

    # Optional mood sticker
    try:
        emotion = await analyze_emotion_with_chatgpt(reply_text)
        sticker_id = sticker_mapping.get(emotion)
        if sticker_id:
            await update.message.reply_sticker(sticker_id)
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------


async def cmd_advise(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📡 Analyzing…")
    try:
        analysis = await generate_investment_analysis()
        await update.message.reply_text(analysis[:4096])
    except Exception as exc:
        await update.message.reply_text(f"❌ Fetch error: {exc}")


async def cmd_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /voice <text> -> TTS with mood sticker
    """
    user_input = update.message.text.replace("/voice", "", 1).strip()
    if not user_input:
        await update.message.reply_text("Usage: `/voice hello there`", parse_mode="Markdown")
        return

    try:
        chat = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": CHARACTER_DESCRIPTION},
                {"role": "user", "content": user_input},
            ],
        )
        reply_text = (chat.choices[0].message.content or "").strip()

        # mood -> voice selection (you can expand this mapping)
        emotion = await analyze_emotion_with_chatgpt(reply_text)
        voice_map = {
            "happy": "echo",
            "sad": "echo",
            "angry": "echo",
            "surprised": "echo",
            "neutral": "echo",
        }
        voice_name = voice_map.get(emotion, "echo")

        tts = await openai_client.audio.speech.create(
            model="gpt-4o-mini-tts",
            input=reply_text,
            voice=voice_name,
        )
        audio_bytes = await tts.aread()
        bio = BytesIO(audio_bytes)
        bio.name = "voice.ogg"
        await update.message.reply_voice(voice=bio)

        sticker_id = sticker_mapping.get(emotion)
        if sticker_id:
            await update.message.reply_sticker(sticker_id)

    except Exception as e:
        await update.message.reply_text(f"❌ Voice generation failed: {e}")


async def cmd_ping(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Bot is running")


async def cmd_newshealth(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    info = news_health_check()
    pretty = json.dumps(info, ensure_ascii=False, indent=2)
    await update.message.reply_text(f"News health:\n```\n{pretty}\n```", parse_mode="Markdown")


async def mention_advise(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📈 Analyzing…")
    try:
        analysis = await generate_investment_analysis(lang="zh", role=QUANT_ANALYST_PROMPT)
        await update.message.reply_text(analysis[:4096])
    except Exception as exc:
        await update.message.reply_text(f"❌ Error: {exc}")

# --------------------------------------------------------------------------------------
# Scheduled tasks
# --------------------------------------------------------------------------------------


async def daily_digest(app):
    """
    Send both headline brief and investment analysis to DAILY_DIGEST_ANALYST.
    """
    try:
        b1 = await generate_investment_analysis(lang="en", role=QUANT_ANALYST_PROMPT)
        b2 = await make_headline_brief("en")
        await app.bot.send_message(chat_id=DAILY_DIGEST_ANALYST, text=b2[:4096])
        await app.bot.send_message(chat_id=DAILY_DIGEST_ANALYST, text=b1[:4096])
    except Exception as e:
        print("[DailyDigestErr]", e)


async def scheduler_loop(app):
    """
    Background scheduler loop for schedule.run_pending().
    """
    # Schedule at 08:30 every day (server time)
    schedule.every().day.at("08:30").do(lambda: asyncio.create_task(daily_digest(app)))

    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            print("[SchedulerErr]", e)
        await asyncio.sleep(30)

# --------------------------------------------------------------------------------------
# Error / fallback
# --------------------------------------------------------------------------------------


async def handle_error_response(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Language-aware fallback message using your custom_responses map.
    """
    try:
        user_language = update.message.from_user.language_code or "en"
        response = custom_responses.get(user_language[:2], custom_responses.get("en", "Sorry, something went wrong."))
        await update.message.reply_text(response)
    except Exception:
        pass

async def cmd_netdiag(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        me = await ctx.bot.get_me()
        await update.message.reply_text(
            f"Telegram OK ✅\nProxy: {TELEGRAM_PROXY_URL or '(none)'}\nBot: @{me.username}"
        )
    except Exception as e:
        await update.message.reply_text(
            f"Telegram FAIL ❌\nProxy: {TELEGRAM_PROXY_URL or '(none)'}\nError: {type(e).__name__}: {e}"
        )


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main():
    proxy_url = TELEGRAM_PROXY_URL  # None if not set

    request = HTTPXRequest(
        connect_timeout=40.0,
        read_timeout=40.0,
        write_timeout=40.0,
        pool_timeout=40.0,
        proxy_url=proxy_url,         # <-- key line
    )

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).request(request).build()

    # handlers (unchanged)
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_photo))
    app.add_handler(MessageHandler(filters.VOICE & ~filters.COMMAND, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT  & ~filters.COMMAND, handle_message))
    app.add_handler(CommandHandler("advise", cmd_advise))
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("edit", handle_edit_request))
    app.add_handler(CommandHandler("voice", cmd_voice))
    app.add_handler(CommandHandler("newshealth", cmd_newshealth))
    app.add_handler(MessageHandler(filters.Regex(r"@dingdongchicken_bot\s*/advise\b"), mention_advise))
    app.add_handler(CommandHandler("netdiag", cmd_netdiag))

    # start scheduler & polling (unchanged)
    asyncio.get_event_loop().create_task(scheduler_loop(app))
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()



