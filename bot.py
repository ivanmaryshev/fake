import telebot
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from keras.utils import pad_sequences
import re
import nltk
from nltk.corpus import stopwords

# === ЗАГРУЗКА РЕСУРСОВ ===
nltk.download('stopwords', quiet=True)
stop_words = stopwords.words('english')

# Твой токен
TOKEN = "8275828988:AAEvoC1vldPuxBqy5As39J5Fo43YSOzScok"
bot = telebot.TeleBot(TOKEN)

# Загружаем модель
model = load_model("fake_news_tg_bot_ready/fake_news_lstm_final.keras")

# Параметры (из твоего Colab)
VOCAB_SIZE = 5000
MAX_LENGTH = 40          # ← у тебя в коде max_length = 40
THRESHOLD = 0.4
TEXT_CLEANING_RE = r"\b0\S*|\b[^A-Za-z0-9]+"


# === ПЕРЕНОС ТВОЕЙ ПРЕДОБРАБОТКИ ПРЯМО В БОТ ===
def preprocess_text(text):
    # 1. Очистка (точно как у тебя в Colab)
    text = re.sub(TEXT_CLEANING_RE, " ", str(text).lower().strip())
    # 2. Убираем стоп-слова
    text = " ".join([word for word in text.split() if word not in stop_words])
    # 3. One-hot encoding
    encoded = one_hot(text, VOCAB_SIZE)
    # 4. Padding
    padded = pad_sequences([encoded], maxlen=MAX_LENGTH, padding='pre')
    return padded


# === ОБРАБОТЧИКИ ===
@bot.message_handler(commands=['start', 'help'])
def start(message):
    bot.reply_to(message, "Привет! Отправь заголовок новости — я скажу, фейк или правда.")


@bot.message_handler(func=lambda m: True)
def check_news(message):
    try:
        X = preprocess_text(message.text)
        pred = model.predict(X, verbose=0)[0][0]

        if pred > THRESHOLD:
            label = "ФЕЙК"
            confidence = pred
        else:
            label = "ПРАВДА"
            confidence = 1 - pred

        bot.reply_to(message, f"{label}\nУверенность: {confidence:.1%}")

    except Exception as e:
        print("Ошибка:", e)  # видно в логах Render
        bot.reply_to(message, "Не смог обработать. Попробуй другой текст.")


print("Бот запущен и работает 24/7!")
bot.infinity_polling()