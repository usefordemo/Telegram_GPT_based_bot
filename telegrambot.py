import os
from io import BytesIO
import random
import hashlib
import openai
import re
import requests
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters
)
from openai.error import OpenAIError
from datetime import datetime

# Your character description from files
def load_system_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Call the character description from the desired file path:
SYSTEM_PROMPT = load_system_prompt("/yourusername/character_description.txt")

# Global variable
pending_custom_replies = {}
ADMIN_USER_ID = 123456789  #Your telegram number id

async def handle_error_response(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_language = update.message.from_user.language_code
    response = custom_responses.get(user_language[:2], custom_responses["en"])
    await update.message.reply_text(response)

# Your OpenAI API Key and Telegram Bot Token
OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'
TELEGRAM_BOT_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'

# Store chat history
message_counts = {}
thresholds = {}
conversation_histories = {}

def anonymize_user(user_id: int) -> str:
    # SHA-256 hashing algorithm to hashes the user ID and returns the first 10 characters as an anonymous ID.
    hash_object = hashlib.sha256(str(user_id).encode('utf-8'))
    return hash_object.hexdigest()[:10]

def log_message(user_id, username, text):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists("logs"):
        os.makedirs("logs")
    # Anonymous ID file name, use for later data training.
    anon_user = anonymize_user(user_id)
    filename = f"logs/Prof_{anon_user}.txt"
    print(f"Logging message to: {filename}")

    try:
        with open(filename, 'a', encoding='utf-8') as file:
            file.write(f"{current_time} - {text}\n")
    except Exception as e:
        print(f"Failed to write to log file: {e}")

def is_direct_reply_or_mention(update, context):
    text = update.message.text or ""
    if update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id:
        return True
    if '@your_bot_username' in text:  # your actual bot name
        return True
    return False

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    user_message = update.message.text
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    chat_type = update.message.chat.type
    key = user_id if chat_type == 'private' else chat_id
    openai.api_key = OPENAI_API_KEY

    # hint from system
    system_prompt = SYSTEM_PROMPT

    if key not in message_counts:
        message_counts[key] = 0
        thresholds[key] = random.randint(100, 200)  # conversation replies in a open group
        conversation_histories[key] = [{"role": "system", "content": system_prompt}]

    message_counts[key] += 1
    conversation_histories[key].append({"role": "user", "content": user_message})

    if await is_image_request(user_message):
        prompt = user_message.strip()
        await generate_image_from_prompt(update, context, prompt)
        return

    immediate_response = (
            is_direct_reply_or_mention(update, context)
            or (message_counts[key] >= thresholds[key])
            or chat_type == 'private'
    )

    if immediate_response:
        chat_history = conversation_histories[key][-500:]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o", #chose your model here
                messages=chat_history
            )
            reply = response.choices[0].message['content'].strip()
            times = 0
            # Filter out when bot cannot reply due to user issues such as sensitive topics
            while times < 3 and (len(reply) < 50 and any(keyword in reply for keyword in ["error", "sorry"])):
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=chat_history
                )
                reply = response.choices[0].message['content'].strip()
                times += 1
                if times == 3:
                    await handle_error_response(update, context)
                    return
            else:
                await update.message.reply_text(reply)

            conversation_histories[key].append({"role": "assistant", "content": reply})
            conversation_histories[key] = conversation_histories[key][-3000:]
            message_counts[key] = 0

            log_message(user_id, username, f"User: {user_message}")
            log_message(user_id, username, f"Bot: {reply}")

        except openai.error.InvalidRequestError as e:
            await notify_admin(update, context, e)
        except Exception as e:
            print(f"Failed to process message: {e}")
            await notify_admin(update, context, e)
    else:
        print(f"Accumulated message count for key {key}: {message_counts[key]}")

async def notify_admin(update: Update, context: ContextTypes.DEFAULT_TYPE, error: Exception):
    pending_custom_replies[ADMIN_USER_ID] = {
        "target_chat_id": update.message.chat_id,
        "target_message_id": update.message.message_id,
        "user_id": update.message.from_user.id,
        "username": update.message.from_user.username,
        "user_message": update.message.text,
        "error": str(error)
    }
    #notify admin when meet error, so can handle it as soon as possible
    admin_text = (
        f"Error triggered.\n"
        f"Error details:\n{str(error)}\n\n")

    await context.bot.send_message(chat_id=ADMIN_USER_ID, text=admin_text)

# Detect key word for image prompt
def is_image_request_keyword(message: str) -> bool:
    keywords = ["generate image", "draw", "create image", "sketch", "picture"]
    return any(keyword in message.lower() for keyword in keywords)

# Use semantics to understand whether images need to be generated
async def is_image_request(message):
    if not message:
        return False
    if is_image_request_keyword(message):
        return True
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": '''Answer only 'yes' or 'no'. 
                Is the following message a request to generate or describe an image?'''},
                {"role": "user", "content": message}
            ]
        )
        reply = response.choices[0].message['content'].strip().lower()
        print(f"Image request detection reply: '{reply}'")
        return "yes" in reply or "true" in reply
    except Exception as e:
        print(f"Failed to detect image request intent: {e}")
        return False


def extract_image_size(prompt: str, default_size: str = "1024x1024") -> str:
    """
    Look for a pattern like "512*512" in the prompt.
    If found, return that size, otherwise return the default size.
    """
    # Regex pattern for sizes like 512x512 or 1024x1024 (assuming numbers are 3-4 digits)
    match = re.search(r'(\d{3,4}x\d{3,4})', prompt)
    if match:
        return match.group(1)
    return default_size

async def generate_image_from_prompt(update, context, prompt):
    # Extract desired image size from the prompt if specified
    image_size = extract_image_size(prompt)
    # Remove the size instruction from the prompt
    prompt_clean = re.sub(r'\d{3,4}x\d{3,4}', '', prompt).strip()

    try:
        response = openai.Image.create(
            model="dall-e-3",
            prompt=prompt_clean,
            n=1,
            size=image_size  # use the extracted or default size
        )
        image_url = response['data'][0]['url']
        await update.message.reply_photo(image_url)
    except Exception as e:
        print(f"Failed to generate image: {e}")
        await update.message.reply_text("Sorry, I couldn't generate the image.")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    print(f'An error occurred: {context.error}')

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    app.add_handler(MessageHandler(filters.User(ADMIN_USER_ID) & filters.TEXT, handle_admin_response))
    app.run_polling()

if __name__ == '__main__':
    main()
