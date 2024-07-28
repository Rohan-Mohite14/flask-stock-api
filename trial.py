from flask import Flask, request, jsonify
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os
import joblib
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

API_KEY = 'b2ac3023973c4fdfaf6d0a5b8b0b5178'
newsapi = NewsApiClient(api_key=API_KEY)

# Load the fine-tuned sentiment model and tokenizer
MODEL_PATH_SENTIMENT = './models/fine-tuned-model'
TOKENIZER_PATH_SENTIMENT = './models/fine-tuned-tokenizer'
tokenizer_sentiment = DistilBertTokenizerFast.from_pretrained(TOKENIZER_PATH_SENTIMENT)
model_sentiment = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH_SENTIMENT)

# Define the sentiment class labels
class_labels_sentiment = ["neutral", "positive", "negative"]

# ARIMA and LSTM model directories
MODEL_DIR_ARIMA = 'models/ARIMA/ARIMA_Models/'  
MODEL_DIR_LSTM = 'models/LSTM/LSTM_Models'

nifty_50_stocks = {
    'Reliance Industries Ltd': ['RELIANCE.NS', 'Reliance'],
    'Tata Consultancy Services': ['TCS.NS', 'Tata Consultancy', 'Tata'],
    'HDFC Bank': ['HDFCBANK.NS', 'HDFC'],
    'Infosys': ['INFY.NS', 'Infosys'],
    'ICICI Bank': ['ICICIBANK.NS', 'ICICI'],
    'Hindustan Unilever': ['HINDUNILVR.NS', 'Hindustan Unilever', 'HUL'],
    'HDFC': ['HDFC.NS', 'HDFC'],
    'Kotak Mahindra Bank': ['KOTAKBANK.NS', 'Kotak Mahindra', 'Kotak'],
    'State Bank of India': ['SBIN.NS', 'State Bank of India', 'SBI'],
    'Bharti Airtel': ['BHARTIARTL.NS', 'Bharti Airtel', 'Airtel'],
    'Asian Paints': ['ASIANPAINT.NS', 'Asian Paints'],
    'ITC': ['ITC.NS', 'ITC'],
    'Larsen & Toubro': ['LT.NS', 'Larsen & Toubro', 'L&T'],
    'Axis Bank': ['AXISBANK.NS', 'Axis Bank'],
    'Maruti Suzuki': ['MARUTI.NS', 'Maruti Suzuki', 'Maruti'],
    'Wipro': ['WIPRO.NS', 'Wipro'],
    'Sun Pharmaceutical': ['SUNPHARMA.NS', 'Sun Pharmaceutical', 'Sun Pharma'],
    'UltraTech Cement': ['ULTRACEMCO.NS', 'UltraTech Cement', 'UltraTech'],
    'Nestle India': ['NESTLEIND.NS', 'Nestle India', 'Nestle'],
    'Tata Motors': ['TATAMOTORS.NS', 'Tata Motors', 'Tata'],
    'Mahindra & Mahindra': ['M&M.NS', 'Mahindra & Mahindra', 'Mahindra'],
    'Dr. Reddy\'s Laboratories': ['DRREDDY.NS', 'Dr. Reddy\'s Laboratories', 'Dr. Reddy'],
    'Tech Mahindra': ['TECHM.NS', 'Tech Mahindra'],
    'Cipla': ['CIPLA.NS', 'Cipla'],
    'Grasim Industries': ['GRASIM.NS', 'Grasim Industries', 'Grasim'],
    'IndusInd Bank': ['INDUSINDBK.NS', 'IndusInd Bank'],
    'Power Grid Corporation': ['POWERGRID.NS', 'Power Grid Corporation', 'Power Grid'],
    'Oil & Natural Gas Corporation': ['ONGC.NS', 'Oil & Natural Gas Corporation', 'ONGC'],
    'Bajaj Finance': ['BAJAJFINANCE.NS', 'Bajaj Finance'],
    'JSW Steel': ['JSWSTEEL.NS', 'JSW Steel', 'JSW'],
    'Bajaj Finserv': ['BAJAJFINSV.NS', 'Bajaj Finserv'],
    'HCL Technologies': ['HCLTECH.NS', 'HCL Technologies', 'HCL'],
    'Hindalco Industries': ['HINDALCO.NS', 'Hindalco Industries', 'Hindalco'],
    'NTPC': ['NTPC.NS', 'NTPC'],
    'Adani Ports': ['ADANIPORTS.NS', 'Adani Ports'],
    'Divi\'s Laboratories': ['DIVISLAB.NS', 'Divi\'s Laboratories', 'Divi\'s'],
    'Eicher Motors': ['EICHERMOT.NS', 'Eicher Motors'],
    'Hero MotoCorp': ['HEROMOTOCO.NS', 'Hero MotoCorp'],
    'Bharti Infratel': ['BHARTIINFR.NS', 'Bharti Infratel'],
    'Tata Steel': ['TATASTEEL.NS', 'Tata Steel'],
    'M&M Financial Services': ['MMFIN.NS', 'M&M Financial Services', 'M&M Financial'],
    'HDFC Life Insurance': ['HDFCLIFE.NS', 'HDFC Life Insurance'],
    'SBI Life Insurance': ['SBILIFE.NS', 'SBI Life Insurance'],
    'Bandhan Bank': ['BANDHANBNK.NS', 'Bandhan Bank'],
    'Gail (India)': ['GAIL.NS', 'Gail (India)', 'Gail'],
    'Tata Power': ['TATAPOWER.NS', 'Tata Power']
}

def get_article_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_content = ' '.join([para.get_text() for para in paragraphs])
        return article_content
    except Exception as e:
        return f"Failed to fetch content: {e}"

def predict_sentiment(text):
    inputs = tokenizer_sentiment(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_sentiment(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_label = class_labels_sentiment[predicted_class_id]
    return predicted_label

def load_arima_model(company):
    path = os.path.join(MODEL_DIR_ARIMA, f'{company}_arima_model.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}")

def load_lstm_model(company):
    path = os.path.join(MODEL_DIR_LSTM, f'{company}_lstm_model.h5')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}")

def fetch_latest_data(company):
    ticker = yf.Ticker(company)
    hist = ticker.history(period="5d")
    return hist['Close']

def fetch_latest_data_lstm(company):
    ticker = yf.Ticker(company)
    hist = ticker.history(period="3mo")
    return hist['Close']

@app.route('/')
def home():
    return "Welcome to the Stock Analysis API!"

@app.route('/predict/arima', methods=['POST'])
def predict_arima():
    data = request.json
    company = data.get('company')
    if not company:
        return jsonify({'error': 'Company name is required'}), 400

    try:
        arima_model = load_arima_model(company)
        if arima_model is None:
            return jsonify({'error': 'Model loading failed'}), 500
    except FileNotFoundError:
        return jsonify({'error': f'No model found for company: {company}'}), 404
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500

    try:
        latest_data = fetch_latest_data(company)
        history = list(arima_model.data.endog) + list(latest_data.values)
        predictions = []

        for _ in range(5):
            updated_model = ARIMA(history, order=arima_model.model.order).fit()
            prediction = updated_model.forecast(steps=1)[0]
            predictions.append(prediction)
            history.append(prediction)

        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/lstm', methods=['POST'])
def predict_lstm():
    data = request.json
    company = data.get('company')
    if not company:
        return jsonify({'error': 'Company name is required'}), 400

    try:
        lstm_model = load_lstm_model(company)
        if lstm_model is None:
            return jsonify({'error': 'Model loading failed'}), 500
    except FileNotFoundError:
        return jsonify({'error': f'No model found for company: {company}'}), 404
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500

    try:
        latest_data = fetch_latest_data_lstm(company)

        if latest_data.empty:
            return jsonify({'error': 'No data fetched for the given company'}), 400

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(np.array(latest_data[-60:]).reshape(-1, 1))

        history = scaled_prices.reshape(-1, 1)
        predictions = []

        seq_length = 30
        if len(history) < seq_length:
            return jsonify({'error': 'Not enough data to make a prediction'}), 400
        input_data = history[-seq_length:].reshape((1, seq_length, 1))
        prediction = lstm_model.predict(input_data)[0][0]
        prediction = scaler.inverse_transform([[prediction]])[0][0]

        return jsonify({'predictions': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/sentiment', methods=['POST'])
def sentiment_analysis():
    data = request.json
    company_name = data.get('company')
    
    if not company_name:
        return jsonify({'error': 'Company name is required'}), 400

    if company_name not in nifty_50_stocks:
        return jsonify({'error': 'Company not found in Nifty 50 stocks'}), 404

    # Get related terms for the company
    related_terms = [company_name] + nifty_50_stocks[company_name]
    related_terms = list(set(related_terms))  # Remove duplicates

    # Fetch news articles related to the company
    five_days_ago = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    all_articles = newsapi.get_everything(
        q=' OR '.join(related_terms),
        from_param=five_days_ago,
        language='en',
        sort_by='popularity',
        page_size=2  # Limit to 2 articles
    )

    # Check if any articles were found
    if not all_articles['articles']:
        return jsonify({'error': 'No news articles found for the company'}), 404

    # Iterate over the articles to predict sentiment
    sentiments = []
    for article in all_articles['articles']:
        article_content = get_article_content(article['url'])
        if article_content:
            sentiment = predict_sentiment(article_content)
            sentiments.append({
                'title': article['title'],
                'published_at': article['publishedAt'],
                'content': article_content,
                'sentiment': sentiment
            })

    return jsonify({'company': company_name, 'sentiments': sentiments})


if __name__ == '__main__':
    app.run(debug=True)
