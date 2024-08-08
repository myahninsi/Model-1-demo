import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st
from keras.callbacks import EarlyStopping
import os
import matplotlib.pyplot as plt

# Define technical indicators
def add_technical_indicators(df):
    df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
    df['EMA_200'] = ta.trend.EMAIndicator(df['Close'], window=200).ema_indicator()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    return df

# Title of the app
st.title("Stock Price Prediction")

# Load company data
company_data_path = "final_v2.csv"  # Use relative path

# Check if the file exists
if not os.path.exists(company_data_path):
    st.error(f"The file {company_data_path} does not exist. Please ensure the file is uploaded correctly.")
else:
    company_data = pd.read_csv(company_data_path)

    # Allow users to select sector and then company
    sector = st.selectbox("Select Sector", company_data['Sector'].unique())
    companies_in_sector = company_data[company_data['Sector'] == sector]['Company'].unique()
    ticker = st.selectbox("Select Company", companies_in_sector)

    # Select prediction window input
    prediction_window = st.number_input("Enter the prediction window (days):", min_value=1, max_value=60, value=30)

    # Select lookback period input
    lookback = st.number_input("Enter the lookback period:", min_value=1, max_value=100, value=60)

    # Select technical indicators input
    technical_indicators = ['EMA_50', 'EMA_200', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']
    selected_indicators = st.multiselect("Select technical indicators to include:", technical_indicators, default=technical_indicators)

    if st.button('Predict'):
        # Load the dataset from yfinance
        @st.cache_data
        def load_data(ticker):
            try:
                data = yf.download(ticker, period='5y', progress=False)
                data.reset_index(inplace=True)
                data = add_technical_indicators(data)
                return data.dropna()
            except Exception as e:
                st.error(f"Error loading data for ticker {ticker}: {e}")
                return pd.DataFrame()

        data = load_data(ticker)

        if not data.empty:
            # Check if data is loaded correctly
            st.write(f"Data loaded for ticker {ticker}:")
            st.write(data.head())

            # Prepare features based on selected indicators
            features = selected_indicators
            X = data[features].values
            Y = data['Close'].values

            # Normalize the dataset
            scaler_X = MinMaxScaler(feature_range=(0, 1))
            scaler_Y = MinMaxScaler(feature_range=(0, 1))

            # Check if X is empty
            if X.shape[0] == 0:
                st.error("No data available for the selected features and lookback period.")
            else:
                X_scaled = scaler_X.fit_transform(X)
                Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1))

                # Create sequences (lookback period)
                X_seq = []
                Y_seq = []

                for i in range(lookback, len(X_scaled)):
                    X_seq.append(X_scaled[i - lookback:i])
                    Y_seq.append(Y_scaled[i])

                X_seq, Y_seq = np.array(X_seq), np.array(Y_seq)

                # TimeSeriesSplit for train-test split
                tscv = TimeSeriesSplit(n_splits=2)
                for train_index, test_index in tscv.split(X_seq):
                    X_train, X_test = X_seq[train_index], X_seq[test_index]
                    Y_train, Y_test = Y_seq[train_index], Y_seq[test_index]

                # Define the LSTM model
                model = Sequential()
                model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
                model.add(Dropout(0.2))
                model.add(LSTM(units=50, return_sequences=False))
                model.add(Dropout(0.2))
                model.add(Dense(units=1))  # Predicting the 'Close' price

                # Compile the model
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Train the model
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                history = model.fit(X_train, Y_train, epochs=50, batch_size=16, validation_data=(X_test, Y_test), verbose=1, callbacks=[early_stopping])

                # Generate predictions for the next 30 days
                future_predictions = []
                last_sequence = X_scaled[-lookback:]

                for _ in range(30):
                    # Make prediction
                    next_prediction_scaled = model.predict(last_sequence.reshape(1, lookback, -1))
                    next_prediction = scaler_Y.inverse_transform(next_prediction_scaled)
                    future_predictions.append(next_prediction[0][0])

                    # Update the sequence by appending the prediction and removing the oldest entry
                    next_prediction_features = np.append(last_sequence[-1, :-1], next_prediction_scaled).reshape(1, -1)
                    last_sequence = np.append(last_sequence[1:], next_prediction_features, axis=0)

                # Prepare dates for plotting future predictions
                last_date = pd.to_datetime(company_data[company_data['Company'] == ticker]['Date'].max())
                future_dates = [last_date + pd.DateOffset(days=i) for i in range(1, 31)]

                # Display next 30 days prices with dates
                st.write("### Next 30 Days Stock Price Predictions")
                for date, price in zip(future_dates, future_predictions):
                    st.write(f'Date: {date}, Predicted Close Price: {price}')

                # Plot future predictions
                plt.figure(figsize=(12, 6))
                plt.plot(future_dates, future_predictions, marker='o', linestyle='-', color='b')
                plt.xlabel('Date')
                plt.ylabel('Predicted Stock Price')
                plt.title(f'Next {prediction_window} Days Stock Price Prediction')
                plt.grid()
                plt.xticks(rotation=45)
                st.pyplot(plt)
        else:
            st.error("No data available for the selected ticker.")
