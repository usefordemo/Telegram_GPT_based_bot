# Telegram AI News & Finance Bot

A Telegram bot that
* summarises authoritative headlines,
* performs quantitative market analysis,
* generates images on demand.

## Features
- News briefings with humorous secondary comments
- Real-time stock / crypto snapshot + trade ideas
- Image generation via OpenAI DALL·E-3
- Role-play responses driven by a custom `CHARACTER_DESCRIPTION`,
  you need customize with your words here and save in a txt file
  and specify it's path.

## Installation

```bash
# 1. Clone
git clone https://github.com/<you>/tg-ai-bot.git
cd tg-ai-bot

# 2. Python virtualenv (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets
cp .env.example .env            # then edit .env with your keys
# Alternatively export env vars manually:
export OPENAI_API_KEY="sk-..."
export TELEGRAM_BOT_TOKEN="123456:ABC..."
export NEWSAPI_KEY="..."
export GNEWS_API_KEY="..."
# (Optional) proxy
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"

# 5. Run
python main.py
