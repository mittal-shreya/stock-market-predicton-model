import requests
import pandas as pd
import joblib

def fetch_realtime_data(url):
    # API call
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        data = response.json()
 
        # Extracting the latest available week's data
        latest_date = list(data['Time Series (Daily)'].keys())[0]
        latest_data = data['Time Series (Daily)'][latest_date]

        # Converting to floats and handling types
        prev_close = float(latest_data['4. close'])
        prev_volume = float(latest_data['5. volume'])
        close_prices = [float(data['Time Series (Daily)'][date]['4. close']) for date in list(data['Time Series (Daily)'].keys())[:10]]

        # Convert the last 10 days' close prices into a DataFrame to calculate SMA and EMA
        df = pd.DataFrame(close_prices, columns=['close'])
        sma_5 = df['close'].rolling(window=5).mean().iloc[-1]
        ema_10 = df['close'].ewm(span=10, adjust=False).mean().iloc[-1]

        # Create a DataFrame in the same structure as your training data
        features = pd.DataFrame({
            'Prev_Close': [prev_close],
            'Prev_Volume': [prev_volume],
            'SMA_5': [sma_5],
            'EMA_10': [ema_10]
        })
    
        return features
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return None
    except Exception as e:
        print(f"Data processing error: {e}")
        return None

# Load the model from disk
rf_model = joblib.load('random_forest_stock_model.pkl')

# URL of the API that provides the real-time data
# Create your own API key
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=TCS&apikey=apikey'

# Fetching the real-time features
realtime_features = fetch_realtime_data(url)

if realtime_features is not None:
    # Predict the closing price using the model
    predicted_close_price = rf_model.predict(realtime_features)
    print(f"Predicted Close Price: {predicted_close_price[0]}")
else:
    print("Failed to fetch or process data.")
