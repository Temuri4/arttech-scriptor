# bot.py
# Telegram-интерфейс для Arttech Scriptor

import logging
from telegram import Update, ForceReply
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from corpbooks.arttech_scriptor import ArttechScriptor
import os

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# Создаём экземпляр ассистента
assistant = ArttechScriptor()

# Команды Telegram
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я Arttech Scriptor 🤖\nЗадай вопрос командой /ask <вопрос>, или используй /list, /info, /help."
    )

async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = " ".join(context.args)
    if not question:
        await update.message.reply_text("Пожалуйста, укажи вопрос после команды /ask")
        return

    try:
        result = assistant.process_command(f"/ask {question}")
        await update.message.reply_text(result[:4000])  # Telegram ограничение
    except Exception as e:
        logger.error(f"Ошибка при обработке /ask: {e}")
        await update.message.reply_text("Произошла ошибка при обработке запроса. Попробуйте позже.")

async def list_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    result = assistant.process_command("/list")
    await update.message.reply_text(result[:4000])

async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Укажи код компании, например: /info NIIPH")
        return
    code = context.args[0]
    result = assistant.process_command(f"/info {code}")
    await update.message.reply_text(result[:4000])

async def fallback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Команда не распознана. Используй /ask, /list, /info")

# Запуск бота
if __name__ == "__main__":
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN в переменных окружения")

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("list", list_command))
    app.add_handler(CommandHandler("info", info))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, fallback))

    logger.info("🚀 Telegram-бот запущен")
    app.run_polling()
