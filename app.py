import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# title of the app
st.title("Stock Price Prediction")

# Load company data
company_data_path = r"C:\Users\Home\Downloads\Lambton\MHS\2nd Term\AI\Group Project\Demo\AI demo code\final_v2.csv"
company_data = pd.read_csv(company_data_path)

# Allow users to select sector and then company
sector = st.selectbox("Select Sector", company_data['Sector'].unique())
companies_in_sector = company_data[company_data['Sector'] == sector]['Company']
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
    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, period='5y')
        data.reset_index(inplace=True)
        data = add_technical_indicators(data)
        return data.dropna()

    data = load_data(ticker)

    # Check if data is loaded correctly
    st.write(f"Data loaded for ticker {ticker}:")
    st.write(data.head())

    # Prepare features based on selected indicators
    features = selected_indicators
    X = data[features].values
    Y = data['Close'].values

    # Debug statements to check feature selection
    st.write("Selected features:")
    st.write(features)
    st.write("Shape of X:", X.shape)
    st.write("Shape of Y:", Y.shape)

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
            X_seq.append(X_scaled[i-lookback:i])
            Y_seq.append(Y_scaled[i])

        X_seq, Y_seq = np.array(X_seq), np.array(Y_seq)

        # Split the data into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X_seq, Y_seq, test_size=0.2, random_state=42)

        # Define the LSTM model
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=100, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))  # Predicting the 'Close' price

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_data=(X_test, Y_test), verbose=1, callbacks=[early_stopping])

        # Evaluate the model
        loss = model.evaluate(X_test, Y_test, verbose=1)
        st.write(f'Test Loss: {loss}')

        # Make predictions
        Y_pred_scaled = model.predict(X_test)

        # Inverse transform the predictions and actual values
        Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
        Y_actual = scaler_Y.inverse_transform(Y_test)

        # Calculate MSE and RMSE
        mse = mean_squared_error(Y_actual, Y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(Y_actual, Y_pred)
        r2 = r2_score(Y_actual, Y_pred)

        # Display evaluation metrics in the app
        st.write("### Evaluation Metrics")
        st.write(f"MSE: {mse}")
        st.write(f"RMSE: {rmse}")
        st.write(f"MAE: {mae}")
        st.write(f"R2 Score: {r2}")

        # Predict the stock price
        future_data = data[selected_indicators].values[-lookback:]
        future_data_scaled = scaler_X.transform(future_data)  # Correctly reshape future_data
        future_data_seq = future_data_scaled.reshape(1, lookback, len(selected_indicators))
        prediction_scaled = model.predict(future_data_seq)
        prediction = scaler_Y.inverse_transform(prediction_scaled)

        st.write("### Predicted Stock Price")
        st.write(f"Predicted price for the next {prediction_window} days: {prediction[0][0]}")
