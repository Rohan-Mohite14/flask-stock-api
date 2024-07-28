from flask import Flask, request, jsonify
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
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
    'Adani Enterprises':['ADANIENT.NS','Adani Enterprises'],
    'Adani Ports': ['ADANIPORTS.NS', 'Adani Ports'],
    'Apollo Hospitals':['APOLLOHOSP.NS','Apollo Hospitals'],
    'Asian Paints': ['ASIANPAINT.NS', 'Asian Paints'],
    'Axis Bank': ['AXISBANK.NS', 'Axis Bank'],
    'Bajaj Auto':['BAJAJ-AUTO.NS','Bajaj Auto'],
    'Bajaj Finance': ['BAJFINANCE.NS', 'Bajaj Finance'],
    'Bajaj Finserv': ['BAJAJFINSV.NS', 'Bajaj Finserv'],
    'Bharti Airtel': ['BHARTIARTL.NS', 'Bharti Airtel', 'Airtel'],
    'BPCL':['BPCL.NS','Bharat Petroleum Corporation'],
    'Britannia':['BRITANNIA.NS','britannia'],
    'Cipla': ['CIPLA.NS', 'Cipla'],
    'Coal India':['COALINDIA.NS','coal india'],
    'Divi\'s Laboratories': ['DIVISLAB.NS', 'Divi\'s Laboratories', 'Divi\'s'],
    'Dr. Reddy\'s Laboratories': ['DRREDDY.NS', 'Dr. Reddy\'s Laboratories', 'Dr. Reddy'],
    'Eicher Motors': ['EICHERMOT.NS', 'Eicher Motors'],
    'Grasim Industries': ['GRASIM.NS', 'Grasim Industries', 'Grasim'],
    'HCL Technologies': ['HCLTECH.NS', 'HCL Technologies', 'HCL'],
    'HDFC Bank': ['HDFCBANK.NS', 'HDFC Bank'],
    'HDFC Life Insurance': ['HDFCLIFE.NS', 'HDFC Life Insurance'],
    'Hero MotoCorp': ['HEROMOTOCO.NS', 'Hero MotoCorp'],
    'Hindalco Industries': ['HINDALCO.NS', 'Hindalco Industries', 'Hindalco'],
    'Hindustan Unilever': ['HINDUNILVR.NS', 'Hindustan Unilever', 'HUL'],
    'ICICI Bank': ['ICICIBANK.NS', 'ICICI'],
    'IndusInd Bank': ['INDUSINDBK.NS', 'IndusInd Bank'],
    'Infosys': ['INFY.NS', 'Infosys'],
    'ITC': ['ITC.NS', 'ITC'],
    'JSW Steel': ['JSWSTEEL.NS', 'JSW Steel', 'JSW'],
    'Kotak Mahindra Bank': ['KOTAKBANK.NS', 'Kotak Mahindra', 'Kotak'],
    'Larsen & Toubro': ['LT.NS', 'Larsen & Toubro', 'L&T'],
    'Mahindra & Mahindra': ['M&M.NS', 'Mahindra & Mahindra', 'Mahindra'],
    'LTIMINDTREE':['LTIM.NS','LTI','mindtree'],
    'Maruti Suzuki': ['MARUTI.NS', 'Maruti Suzuki', 'Maruti'],
    'Nestle India': ['NESTLEIND.NS', 'Nestle India', 'Nestle'],
    'NTPC': ['NTPC.NS', 'NTPC'],
    'Oil & Natural Gas Corporation': ['ONGC.NS', 'Oil & Natural Gas Corporation', 'ONGC'],
    'Power Grid Corporation': ['POWERGRID.NS', 'Power Grid Corporation', 'Power Grid'],
    'Reliance Industries Ltd': ['RELIANCE.NS', 'Reliance'],
    'SBI Life Insurance': ['SBILIFE.NS', 'SBI Life Insurance'],
    'State Bank of India': ['SBIN.NS', 'State Bank of India', 'SBI'],
    'Shriram Finance':['SHRIRAMFIN.NS','shriram finance'],
    'Sun Pharmaceutical': ['SUNPHARMA.NS', 'Sun Pharmaceutical', 'Sun Pharma'],
    'Tata Consultancy Services': ['TCS.NS', 'Tata Consultancy', 'Tata'],
    'Tata Motors': ['TATAMOTORS.NS', 'Tata Motors', 'Tata'],
    'Tata Consumer':['TATACONSUM.NS','tata consumer'],
    'Tata Steel': ['TATASTEEL.NS', 'Tata Steel'],
    'Tech Mahindra': ['TECHM.NS', 'Tech Mahindra'],
    'Titan':['TITAN.NS','Titan'],
    'UltraTech Cement': ['ULTRACEMCO.NS', 'UltraTech Cement', 'UltraTech'],
    'Wipro': ['WIPRO.NS', 'Wipro']
}


# below function to get news after getting url
def get_article_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_content = ' '.join([para.get_text() for para in paragraphs])
        return article_content
    except Exception as e:
        return f"Failed to fetch content: {e}"

# predicts the sentiment of the news
def predict_sentiment(text):
    inputs = tokenizer_sentiment(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_sentiment(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_label = class_labels_sentiment[predicted_class_id]
    return predicted_label,logits

# loads saved arima model
def load_arima_model(company):
    path = os.path.join(MODEL_DIR_ARIMA, f'{company}_arima_model.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}")

# loads lstm model
def load_lstm_model(company):
    path = os.path.join(MODEL_DIR_LSTM, f'{company}_lstm_model.h5')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}")

# fetching 
def fetch_latest_data(company, Period):
    ticker = yf.Ticker(company)
    hist = ticker.history(period=Period)
    return hist['Close']

@app.route('/')
def home():
    return "Welcome to the Stock Analysis API!"

def predict_arima(company):
    path = os.path.join(MODEL_DIR_ARIMA, f'{company}_arima_model.pkl')

    try:
        arima_model = load_arima_model(company)
        if arima_model is None:
            return 'Model loading failed'
    except FileNotFoundError:
        return f'No model found for company: {company}'
    except RuntimeError as e:
        return f'Error: {str(e)}'

    try:
        latest_data = fetch_latest_data(company, "1d")
        history = list(arima_model.data.endog) + list(latest_data.values)
        predictions = []
        returns = []

        for i in range(7):
            updated_model = ARIMA(history, order=arima_model.model.order).fit()
            if i == 0:  # Save the model on the first pass only
                with open(path, 'wb') as f:
                    joblib.dump(updated_model, f)

            prediction = updated_model.forecast(steps=1)[0]
            predictions.append(prediction)

            if len(predictions) > 1:
                previous_prediction = predictions[-2]
            else:
                latest_data_list=latest_data.tolist()
                previous_prediction = latest_data_list[0]  # Use latest actual value if no previous prediction

            current_return = ((prediction - previous_prediction) / previous_prediction) * 100
            returns.append(current_return)
            history.append(prediction)

        return predictions, returns
    except Exception as e:
        return f'Error: {str(e)}'

def predict_lstm(company):
    try:
        lstm_model = load_lstm_model(company)
        if lstm_model is None:
            return 'Model loading failed'
    except FileNotFoundError:
        return f'No model found for company: {company}'
    except RuntimeError as e:
        return f'Error: {str(e)}'

    try:
        latest_data = fetch_latest_data(company, "3mo")

        if latest_data.empty:
            return 'No data fetched for the given company'

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(np.array(latest_data[-60:]).reshape(-1, 1))

        history = scaled_prices.reshape(-1, 1)
        predictions = []
        returns = []

        seq_length = 30
        if len(history) < seq_length:
            return 'Not enough data to make a prediction'

        for _ in range(7):
            input_data = history[-seq_length:].reshape((1, seq_length, 1))
            prediction = lstm_model.predict(input_data)[0][0]
            prediction = scaler.inverse_transform([[prediction]])[0][0]
            predictions.append(prediction)

            if len(predictions) > 1:
                previous_prediction = predictions[-2]
            else:
                latest_data_oneday = fetch_latest_data(company, "1d")
                latest_data_list=latest_data_oneday.tolist()
                previous_prediction = latest_data_list[0]  # Use latest actual value if no previous prediction

            current_return = ((prediction - previous_prediction) / previous_prediction) * 100
            returns.append(current_return)
            history = np.append(history, [[prediction]], axis=0)

        return predictions, returns
    except Exception as e:
        return f'Error: {str(e)}'


def fetcher(related_terms):
    five_days_ago = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    all_articles = newsapi.get_everything(
        q=' OR '.join(related_terms),
        from_param=five_days_ago,
        language='en',
        sort_by='popularity',
        page_size=5#limit is 5

    )
    return all_articles

def sentiment_analysis(company_name):
    arr = []
    x, y, z = 0, 0, 0
    related_terms = [company_name] + nifty_50_stocks.get(company_name, [])
    related_terms = list(set(related_terms))

    todays_price = fetch_latest_data(nifty_50_stocks[company_name][0], "1d")
    todays_price_list = todays_price.tolist()  # Convert Series to list

    all_articles = fetcher(related_terms)

    if not all_articles['articles']:
        related_terms = ["Nifty 50", "nifty", "NIFTY", "Nifty50"]
        all_articles = fetcher(related_terms)

    num_articles = len(all_articles['articles'])
    if num_articles == 0:
        return [[0, 0, 0], todays_price_list[0]]

    for article in all_articles['articles']:
        article_content = get_article_content(article['url'])
        if article_content:
            sentiment, logits = predict_sentiment(article_content)
            logits_list = logits.squeeze().tolist()  # Convert tensor to list

            # Ensure logits_list is a list of floats or numbers
            if isinstance(logits_list, list) and all(isinstance(item, (int, float)) for item in logits_list):
                if len(logits_list) == 3:  # Assuming there are three classes: neutral, positive, negative
                    x += logits_list[0]
                    y += logits_list[1]
                    z += logits_list[2]
                else:
                    app.logger.error(f"Unexpected number of logits for article: {article['url']}")
            else:
                app.logger.error(f"Invalid logits format for article: {article['url']}")

    avgX, avgY, avgZ = x / num_articles, y / num_articles, z / num_articles

    return [[avgX, avgY, avgZ], todays_price_list[0]]




@app.route('/predict', methods=['GET'])
def predict():
    mpPred = {}
    mpBert = {}
    for stock in nifty_50_stocks.keys():
        ticker = nifty_50_stocks[stock][0]
        arr = sentiment_analysis(stock)
        # Predict with ARIMA model
        try:
            predA, returnsA = predict_arima(ticker)
            if isinstance(predA, str):
                app.logger.error(f"ARIMA error for {ticker}: {predA}")
                continue  # Skip this stock and continue with the next
        except Exception as e:
            app.logger.error(f"Exception in ARIMA for {ticker}: {str(e)}")
            continue

        # Predict with LSTM model
        try:
            predL, returnsL = predict_lstm(ticker)
            if isinstance(predL, str):
                app.logger.error(f"LSTM error for {ticker}: {predL}")
                continue
        except Exception as e:
            app.logger.error(f"Exception in LSTM for {ticker}: {str(e)}")
            continue

        # Ensure both predictions are lists and have the same length
        if not isinstance(predA, list) or not isinstance(predL, list) or len(predA) != len(predL):
            app.logger.error(f"Inconsistent prediction lengths or types for {ticker}")
            continue

        resultant = [predA[0]]  # Starting with the first ARIMA prediction
        avgReturns = [returnsA[0]]
        try:
            arr = sentiment_analysis(stock)
        except Exception as e:
            app.logger.error(f"Exception in sentiment analysis for {stock}: {str(e)}")
            arr = []  # Default to an empty array if sentiment analysis fails

        for i in range(1, 7):
            try:
                avg = (predA[i] + predL[i]) / 2
                ret_avg = (returnsA[i] + returnsL[i]) / 2
            except Exception as e:
                app.logger.error(f"Error calculating average for {stock}: {str(e)}")
                continue  # Skip this iteration and continue with the next

            resultant.append(avg)
            avgReturns.append(ret_avg)
        
        mpPred[stock] = [resultant, avgReturns]
        mpBert[stock] = arr
    
    return jsonify({"mppred": mpPred, "mpbert": mpBert}), 200



if __name__ == '__main__':
    app.run(debug=True)