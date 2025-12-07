import telebot
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from keras.utils import pad_sequences
import re

# Твой токен от @BotFather
TOKEN = "8275828988:AAEvoC1vldPuxBqy5As39J5Fo43YSOzScok"  # ← впиши свой
bot = telebot.TeleBot(TOKEN)

# Загружаем модель и всё нужное
model = load_model("fake_news_tg_bot_ready/fake_news_lstm_final.keras")

with open("fake_news_tg_bot_ready/preprocess_config.pkl", "rb") as f:
    config = pickle.load(f)

with open("fake_news_tg_bot_ready/preprocess_function.pkl", "rb") as f:
    preprocess = pickle.load(f)

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "Привет! Кидай новость — скажу, фейк или правда.")

@bot.message_handler(func=lambda message: True)
def check_news(message):
    text = message.text
    try:
        X = preprocess(text)
        pred = model.predict(X, verbose=0)[0][0]
        label = "ФЕЙК" if pred > config["threshold"] else "ПРАВДА"
        confidence = pred if pred > 0.5 else 1 - pred
        bot.reply_to(message, f"{label}\nУверенность: {confidence:.1%}")
    except:
        bot.reply_to(message, "Ошибка обработки. Попробуй другую новость.")

print("Бот запущен!")
bot.infinity_polling()