import telebot
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from keras.utils import pad_sequences
import re
import nltk
import os

if not os.path.exists('/tmp/nltk_data'):
    nltk.download('stopwords', download_dir='/tmp/nltk_data', quiet=True)
    nltk.data.path.append('/tmp/nltk_data')
else:
    nltk.data.path.append('/tmp/nltk_data')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

TOKEN = "8275828988:AAEvoC1vldPuxBqy5As39J5Fo43YS0zScok"

bot = telebot.TeleBot(TOKEN)

# Загрузка модели
model = load_model("fake_news_tg_bot_ready/fake_news_lstm_final.keras")

VOCAB_SIZE = 5000
MAX_LENGTH = 40
THRESHOLD = 0.4
TEXT_CLEANING_RE = r"\b0\S*|\b[^A-Za-z0-9]+"

def preprocess_text(text):
    text = re.sub(TEXT_CLEANING_RE, " ", str(text).lower().strip())
    text = " ".join([word for word in text.split() if word not in stop_words])
    encoded = one_hot(text, VOCAB_SIZE)
    padded = pad_sequences([encoded], maxlen=MAX_LENGTH, padding='pre')
    return padded

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "Привет! Я могу по заголовку понять новость фейк или правда")

@bot.message_handler(func=lambda m: True)
def check_news(message):
    try:
        X = preprocess_text(message.text)
        pred = model.predict(X, verbose=0)[0][0]
        label = "ФЕЙК" if pred > THRESHOLD else "ПРАВДА"
        confidence = pred if pred > 0.5 else 1 - pred
        bot.reply_to(message, f"{label}\nУверенность: {confidence:.1%}")
    except Exception as e:
        print("Ошибка:", e)
        bot.reply_to(message, "Не понял текст. Попробуй короче.")

print("Бот запущен и работает")
bot.infinity_polling()