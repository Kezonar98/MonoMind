import os
import io
import asyncio
import httpx
import base64
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
    """
    Intercepts all user messages (text or photos with captions), 
    processes them, and forwards them to the FastAPI LangGraph backend.
    """
    
    # 1. Extract text (if standard message) or caption (if an image was sent)
    text_content = message.text or message.caption or ""
    image_b64 = None

    # 2. Check if the user sent a photo
    if message.photo:
        # Notify the user that we are processing a visual input
        processing_msg = await message.answer("📸 Processing image, please wait...")
        
        # Telegram sends multiple sizes of the photo. We take the last one (highest resolution).
        highest_res_photo = message.photo[-1]
        file_info = await bot.get_file(highest_res_photo.file_id)
        
        # Download the file directly into memory to avoid saving it to the disk
        file_bytes = io.BytesIO()
        await bot.download_file(file_info.file_path, destination=file_bytes)
        
        # Encode the image bytes to a Base64 string for HTTP JSON transmission
        image_b64 = base64.b64encode(file_bytes.getvalue()).decode('utf-8')
    else:
        # Standard text processing notification
        processing_msg = await message.answer("🔄 Generating response, please wait...")
    
    # 3. Send an HTTP POST request to our LangGraph backend
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            # Dynamically assign the real user ID so memory is isolated per user
            payload = {
                "user_id": message.from_user.id, 
                "message": text_content,
                "image_base64": image_b64
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

    # 4. Update the temporary UX message with the actual AI response
    await processing_msg.edit_text(final_text, parse_mode="Markdown")

async def main() -> None:
    print("🤖 MonoMind Telegram Client is running...")
    # Register command handlers and set up the bot commands for better UX
    await setup_commands(bot)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())