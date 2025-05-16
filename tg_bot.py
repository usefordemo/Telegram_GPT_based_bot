from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)
import asyncio, random, schedule
from config import TELEGRAM_BOT_TOKEN, openai_client, DAILY_DIGEST_ANALYST
from news_service import (make_headline_brief, make_topic_headline_brief)
from finance_service import generate_investment_analysis
from utils import ( detect_lang, classify_intent, is_direct_reply_or_mention,
                   parse_news_topic, is_image_request, extract_image_size)
from data import (custom_responses, message_counts, thresholds,
                  conversation_histories, sticker_mapping, TOPIC_MAP,
                 CHARACTER_DESCRIPTION,QUANT_ANALYST_PROMPT)
from utils import (openai_client, analyze_emotion_with_chatgpt)
import openai  # 用于捕获 openai.error.InvalidRequestError
from openai import OpenAIError
import re

async def handle_message(update: Update,
                         context: ContextTypes.DEFAULT_TYPE):
    # ---- 0️⃣ 过滤非文本消息 / 非消息类更新 ----------------------------
    if update.message is None:        # 频道贴文、编辑消息、回调按钮等
        return

    # ---------- 基本信息 ----------
    chat        = update.effective_chat
    chat_type   = chat.type                       # private | group | supergroup
    user_msg    = update.message.text or ""
    chat_id     = chat.id
    user_id     = update.effective_user.id
    username    = update.effective_user.username
    print(f"📩 收到消息：{user_msg} 来自 @{username or user_id}")

    # ---------- 统计结构初始化 ----------
    key = user_id if chat_type == "private" else chat_id
    if key not in message_counts:
        message_counts[key]          = 0
        thresholds[key]              = random.randint(1000, 2000)
        conversation_histories[key]  = [
            {"role": "system", "content": CHARACTER_DESCRIPTION}
        ]

    # 始终记入历史
    conversation_histories[key].append({"role": "user", "content": user_msg})
    message_counts[key] += 1

    # ---------- 触发判定 ----------
    triggered_by_reply  = (chat_type != "private" and
                           is_direct_reply_or_mention(update, context))
    triggered_by_random = (chat_type != "private" and
                           message_counts[key] >= thresholds[key])
    triggered_private   = (chat_type == "private")
    should_respond      = triggered_private or triggered_by_reply or triggered_by_random

    if not should_respond:
        return

#detect news topic
    topic_key = parse_news_topic(user_msg)
    if topic_key:
        topic_key, period = topic_key
        lang = detect_lang(user_msg)
        brief = await make_topic_headline_brief(topic_key, period, lang)
        await update.message.reply_text(brief[:4096])
        return
    # 随机触发后重置阈值
    if triggered_by_random:
        thresholds[key]     = random.randint(1000, 2000)
        message_counts[key] = 0

    # ---------- 1⃣ 语义分析 ----------
    intent = classify_intent(user_msg)    # "chat" | "news"

    # ---------- 2⃣ 图片请求 ----------
    if await is_image_request(user_msg):
        await generate_image_from_prompt(update, context, prompt=user_msg.strip())
        return

    # ---------- 3⃣ 新闻播报 ----------
    if intent == "news":
        lang  = detect_lang(user_msg)
        brief = await make_headline_brief(lang)
        await update.message.reply_text(brief[:4096])
        return

    # ---------- 4⃣ 普通聊天 ----------
    chat_history = conversation_histories[key][-500:]
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=chat_history
        )
        reply = response.choices[0].message.content.strip()

        # 防短&抱歉式回复
        for _ in range(3):
            if len(reply) >= 50 or not any(t in reply for t in
                 ["抱歉","無權","無法","对不起","error","sorry"]):
                break
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=chat_history
            )
            reply = response.choices[0].message.content.strip()
        else:
            await handle_error_response(update, context)
            return

        await update.message.reply_text(reply)

        # 更新状态
        conversation_histories[key].append({"role": "assistant","content": reply})
        conversation_histories[key] = conversation_histories[key][-3000:]

        emotion    = analyze_emotion_with_chatgpt(reply)
        print(emotion)
        sticker_id = sticker_mapping.get(emotion)
        if sticker_id:
            await update.message.reply_sticker(sticker_id)




    except OpenAIError as e:
        await notify_admin(update, context, e)
    except Exception as e:
        print(f"Failed to process message: {e}")
        await notify_admin(update, context, e)



async def generate_image_from_prompt(update, context, prompt):
    # Extract the image size from the prompt, if present.
    image_size = extract_image_size(prompt)
    # Optionally, remove the size specification from the prompt:
    prompt_clean = re.sub(r'\d+\s*[x*]\s*\d+', '', prompt).strip()
    try:
        response = openai.Image.create(
            model="dall-e-3",
            prompt=prompt_clean,
            n=1,
            size=image_size  # Use the extracted size
        )
        image_url = response['data'][0]['url']
        await update.message.reply_photo(image_url)
    except Exception as e:
        print(f"Failed to generate image: {e}")
        await update.message.reply_text("Sorry, I couldn't generate the image.")


async def cmd_advise(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📡 正在生成投资分析，请稍候……")
    try:
        analysis = await generate_investment_analysis()
        await update.message.reply_text(analysis[:4096])
    except Exception as exc:
        await update.message.reply_text(f"❌ 出错: {exc}")


async def cmd_ping(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Bot 正在运行")

async def mention_advise(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📈 正在生成中文投资分析，请稍候…")
    try:
        analysis = await generate_investment_analysis(lang="zh", role=QUANT_ANALYST_PROMPT)
        await update.message.reply_text(analysis[:4096])
        print("分析生成成功，长度：", len(analysis))
    except Exception as exc:
        await update.message.reply_text(f"❌ 出错: {exc}")
# 定时任务发送消息
async def daily_news_digest_task(context):
    analysis = await generate_investment_analysis()
    await context.bot.send_message(chat_id=DAILY_DIGEST_ANALYST, text=analysis[:4096])

# 调度器
async def schedule_loop(app):
    schedule.every().day.at("08:30").do(lambda: asyncio.create_task(daily_news_digest_task(app)))
    schedule.every().day.at("08:30").do(lambda: asyncio.create_task(daily_digest(None)))
    while True:
        schedule.run_pending()
        await asyncio.sleep(60)

async def handle_error_response(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # 獲取用戶的語言代碼
    user_language = update.message.from_user.language_code
    # 根據語言代碼選擇對應的回應
    response = custom_responses.get(user_language[:2], custom_responses["en"])  # 默認為英文
    # 發送回應給用戶
    await update.message.reply_text(response)

async def handle_advise_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📡 正在抓取市场数据并生成分析，请稍等...")
    try:
        analysis = await generate_investment_analysis()
        await update.message.reply_text(analysis[:4096])  # Telegram 字符限制
    except Exception as e:
        print(f"Failed to generate advise: {e}")
        await update.message.reply_text("❌ 生成投资建议时出错，请稍后重试。")


async def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("advise", cmd_advise))
    app.add_handler(CommandHandler("ping",   cmd_ping))
    pattern = r"@your_bot_username\s*/advise\b"
    app.add_handler(MessageHandler(filters.Regex(pattern), mention_advise))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))


    async def daily_digest(_):
        txt = await generate_investment_analysis(lang="zh", role=QUANT_ANALYST_PROMPT)
        await app.bot.send_message(chat_id=DAILY_DIGEST_ANALYST, text=txt[:4096])

    # schedule daily digest at 08:30 local time
    schedule.every().day.at("08:30").do(lambda: asyncio.create_task(daily_digest(None)))

    async def scheduler_loop():
        while True:
            schedule.run_pending()
            await asyncio.sleep(60)

    asyncio.create_task(scheduler_loop())
    await app.run_polling()