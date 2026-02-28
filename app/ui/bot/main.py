import os
import asyncio
import httpx
from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, BotCommand
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_URL = os.getenv("API_URL")

if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is not set in the .env file. Please check your configuration.")

bot = Bot(token=TOKEN)
dp = Dispatcher()

async def setup_commands(bot: Bot):
    """Creates a convenient command menu in the Telegram client."""
    commands = [
        BotCommand(command="start", description="Restart MonoMind"),
        BotCommand(command="help", description="See what you can ask")
    ]
    await bot.set_my_commands(commands)

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """Handles the /start command and introduces the AI."""
    welcome_text = (
        "🏦 **Welcome! I am MonoMind** — your private financial AI assistant.\n\n"
        "I operate completely locally and do not share your data with third parties. "
        "What would you like to know about your finances?"
    )
    await message.answer(welcome_text, parse_mode="Markdown")

@dp.message(Command("help"))
async def command_help_handler(message: Message) -> None:
    """Handles the /help command and gives user hints."""
    help_text = (
        "🤖 **What can I do?**\n\n"
        "I am connected to your local secure PostgreSQL ledger. "
        "I analyze your intent and perform deterministic math. "
        "Try asking me (in English):\n\n"
        "🔹 *\"What is my current balance?\"*\n"
        "🔹 *\"How much money did I spend?\"*\n"
        "🔹 *\"Analyze my runway. How long will my money last?\"*\n"
    )
    await message.answer(help_text, parse_mode="Markdown")

@dp.message()
async def handle_message(message: Message) -> None:
    """Intercepts all text messages and forwards them to the FastAPI backend."""
    
    # 1. Notify the user that processing has started (UX improvement)
    processing_msg = await message.answer("🔄 Generate response, please wait...")
    
    # 2. Send an HTTP POST request to our LangGraph backend
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            payload = {
                "user_id": 1, 
                "message": message.text
            }
            response = await client.post(API_URL, json=payload)
            response.raise_for_status() 
            
            data = response.json()
            
            reply_text = data.get("response", "Failed to generate a response.")
            intent = data.get("intent", "unknown")
            
            final_text = f"{reply_text}\n\n⚙️ *Intent:* `{intent}`"
            
        except httpx.ConnectError:
            final_text = "⚠️ Connection error: MonoMind Core (FastAPI) is offline."
        except Exception as e:
            final_text = f"⚠️ Unexpected error with MonoMind Core: {e}"

    # 3. Update the temporary message with the actual AI response
    await processing_msg.edit_text(final_text, parse_mode="Markdown")

async def main() -> None:
    print("🤖 MonoMind Telegram Client is running...")
    # Register command handlers and set up the bot commands for better UX
    await setup_commands(bot)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())