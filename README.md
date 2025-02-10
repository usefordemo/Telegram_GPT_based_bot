# Telegram Bot with OpenAI Integration
Python-based Telegram bot that uses OpenAI’s API for generating text and images. 
It maintains conversation history, logs interactions, and notifies an admin on errors.

# Features
Text Generation: Uses GPT models (e.g., `gpt-4o`, `gpt-4-turbo`) for context-aware responses.
Image Generation: Generates images via DALL·E 3; supports user-requested sizes.
Conversation History: Tracks context for each group chat.
Error Handling: Notifies a designated admin when error occoured.
Configurable System Prompt: Loads character and system prompt from a local text file, which can be custmised.
Anonymized Logging: Can be extended to anonymize user logs for training.

# Requirements
Python 3.8+
Dependencies: `python-telegram-bot`, `openai`, `requests`
OpenAI API key and Telegram Bot token 
A text file (`character_description.txt`) for your system prompt

# Setup
Clone the repository:
   bash
   git clone https://github.com/yourusername/telegram-openai-bot.git
   cd telegram-openai-bot  
   python3 -m venv venv
   source venv/bin/activate  
   pip install -r requirements.txt
   
    # On Windows: python -m venv venv
    # venv\Scripts\activate
   
# Code Overview
handle_message – Processes incoming messages, maintains history, and calls OpenAI.
generate_image_from_prompt – Calls DALL·E 3 to generate images; extracts requested size from the prompt.
notify_admin & handle_admin_response – Notifies the admin on errors and forwards the admin’s reply to the user.
load_system_prompt – Reads the system prompt from a local text file.
log_message – Logs conversation details (can be modified to anonymize data).
