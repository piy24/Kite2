import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from datetime import datetime

# Check if GPU is available
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

# Function to download historical stock data
def download_stock_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

# Function to prepare data for models
def prepare_data(df, target_col, window_size, test_size):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[[target_col]].values)

    x, y = [], []
    for i in range(window_size, len(scaled_data) - test_size):
        x.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])

    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    return x, y, scaler

#  build LSTM model
def build_lstm_model(input_shape, n_units, dropout_rate):
    with tf.device(device):
        inputs = Input(shape=input_shape)
        x = Conv1D(filters=64, kernel_size=5, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = LSTM(units=n_units, return_sequences=True)(x)
        x = Dropout(dropout_rate)(x)
        x = LSTM(units=n_units, return_sequences=False)(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(units=100, activation='relu')(x)
        outputs = Dense(units=1)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# prepare data for sklearn models
def prepare_sklearn_data(df, window_size, test_size):
    x, y = [], []
    for i in range(window_size, len(df) - test_size):
        x.append(df[i-window_size:i])
        y.append(df[i])
    
    x, y = np.array(x), np.array(y)
    return x, y

# Streamlit app
st.title("Stock Price Predictor")

# User inputs
stock_symbol = st.text_input("Enter the stock symbol (e.g., AAPL for Apple):").upper()
prediction_days = st.number_input("Enter number of days to predict future prices:(max 5)", min_value=1, value=1)
end_date = st.date_input("Enter the end date for data:", datetime.today())
start_date = end_date - pd.DateOffset(days=365 * 5)  # 5 years of data

# Download data
if st.button("Predict"):
    with st.spinner("Predicting..."):
        data = download_stock_data(stock_symbol, start_date, end_date)
        if data is not None and not data.empty:
            window_size = 60
            x_train, y_train, scaler = prepare_data(data, 'Close', window_size, prediction_days)
            sklearn_x, sklearn_y = prepare_sklearn_data(data['Close'].values, window_size, prediction_days)
            x_train_sklearn, x_test_sklearn = sklearn_x[:-prediction_days], sklearn_x[-prediction_days:]
            y_train_sklearn, y_test_sklearn = sklearn_y[:-prediction_days], sklearn_y[-prediction_days:]

            # LSTM model
            input_shape = (x_train.shape[1], 1)
            n_units = 200  # Reduced number of units
            dropout_rate = 0.3

            model = build_lstm_model(input_shape, n_units, dropout_rate)

            # Early Stopping
            early_stop = EarlyStopping(monitor='val_loss', patience=10)

            with tf.device(device):
                model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.1, callbacks=[early_stop])

            x_test = x_train[-prediction_days:]
            predicted_prices_lstm = model.predict(x_test)
            predicted_prices_lstm = scaler.inverse_transform(predicted_prices_lstm).flatten()

            # Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(x_train_sklearn, y_train_sklearn)
            predicted_prices_lr = linear_model.predict(x_test_sklearn)

            # Random Forest model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(x_train_sklearn, y_train_sklearn)
            predicted_prices_rf = rf_model.predict(x_test_sklearn)

            y_test_actual = data['Close'][-prediction_days:].values

            mse_lstm = mean_squared_error(y_test_actual, predicted_prices_lstm)
            mae_lstm = mean_absolute_error(y_test_actual, predicted_prices_lstm)

            mse_lr = mean_squared_error(y_test_actual, predicted_prices_lr)
            mae_lr = mean_absolute_error(y_test_actual, predicted_prices_lr)

            mse_rf = mean_squared_error(y_test_actual, predicted_prices_rf)
            mae_rf = mean_absolute_error(y_test_actual, predicted_prices_rf)

            st.write(f"LSTM - Mean Squared Error (MSE): {mse_lstm:.2f}")
            st.write(f"LSTM - Mean Absolute Error (MAE): {mae_lstm:.2f}")
            st.write(f"Linear Regression - Mean Squared Error (MSE): {mse_lr:.2f}")
            st.write(f"Linear Regression - Mean Absolute Error (MAE): {mae_lr:.2f}")
            st.write(f"Random Forest - Mean Squared Error (MSE): {mse_rf:.2f}")
            st.write(f"Random Forest - Mean Absolute Error (MAE): {mae_rf:.2f}")

            # Ensure all arrays have the same length
            prediction_dates = pd.date_range(end=end_date, periods=prediction_days).to_list()
            min_len = min(len(prediction_dates), len(y_test_actual), len(predicted_prices_lstm), len(predicted_prices_lr), len(predicted_prices_rf))
            prediction_dates = prediction_dates[:min_len]
            y_test_actual = y_test_actual[:min_len]
            predicted_prices_lstm = predicted_prices_lstm[:min_len]
            predicted_prices_lr = predicted_prices_lr[:min_len]
            predicted_prices_rf = predicted_prices_rf[:min_len]

            predicted_df = pd.DataFrame({
                'Date': prediction_dates,
                'Actual Price': y_test_actual,
                'LSTM Predicted Price': predicted_prices_lstm,
            })
            st.write(predicted_df)

            # Plot results
            plt.figure(figsize=(14, 7))
            plt.plot(prediction_dates, y_test_actual, label='Actual Price', color='blue')
            plt.plot(prediction_dates, predicted_prices_lstm, label='LSTM Predicted Price', color='green')
            plt.plot(prediction_dates, predicted_prices_lr, label='Linear Regression Predicted Price', color='red')
            plt.plot(prediction_dates, predicted_prices_rf, label='Random Forest Predicted Price', color='orange')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title('Stock Price Prediction')
            plt.legend()
            st.pyplot(plt.gcf())
        else:
            st.write(f"No data found for {stock_symbol}. Please try again.")
