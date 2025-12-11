FROM python:3.11-slim

WORKDIR /app

COPY .env .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .
COPY fake_news_tg_bot_ready ./fake_news_tg_bot_ready

RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/nltk_data')"

ENV NLTK_DATA=/usr/local/nltk_data

CMD ["python", "bot.py"]