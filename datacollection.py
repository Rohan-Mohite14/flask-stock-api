import yfinance as yf
import pandas as pd
import os

# Create a list of Nifty50 stocks (replace with your actual list)
nifty50 = ["ADANIPORTS.NS", "RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS"]  # Add more stocks

# Specify the data directory
data_directory = "data"

# Create the data directory if it doesn't exist
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

# Loop through Nifty50 stocks and save data to CSV
for ticker in nifty50:
    try:
        # Create a YFinance ticker object
        stock_data = yf.Ticker(ticker)

        # Fetch historical data
        data = stock_data.history(period="max")

        # Create the CSV file path
        csv_file = os.path.join(data_directory, f"{ticker}.csv")

        # Save data to CSV
        data.to_csv(csv_file, index=True)
        print(f"Data for {ticker} saved to {csv_file}")
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
