import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from joblib import load
import logging

# Function to validate data
def validate_data(data, step):
    if not np.all(np.isfinite(data)):
        print(f"Invalid data detected at {step}: {data}")
        raise ValueError("Data contains NaN or infinite values")

def predict_next_months(model, scaler, data, months=5, days_per_month=30, look_back=60):
    predictions = []
    last_60_days = data[-look_back:].reshape(-1, 1)

    for month in range(months):  # Predict for 5 months
        monthly_predictions = []
        for day in range(days_per_month):  # Predict each day of the month
            scaled_last_60_days = scaler.transform(last_60_days)
            validate_data(scaled_last_60_days, f"scaled input for Month {month + 1}, Day {day + 1}")
            X_test = scaled_last_60_days.reshape(1, look_back, 1)

            # Make prediction
            prediction = model.predict(X_test)
            predicted_price = scaler.inverse_transform(prediction)[0][0]

            # Handle invalid prediction
            if not np.isfinite(predicted_price):
                predicted_price = np.mean(last_60_days)  # Use the mean as fallback

            monthly_predictions.append(predicted_price)
            last_60_days = np.append(last_60_days[1:], [[predicted_price]], axis=0)
            validate_data(last_60_days, f"updated last_60_days for Month {month + 1}, Day {day + 1}")

        predictions.append(monthly_predictions)

    return predictions

def load_model_and_predict():
    # Load the trained model and scaler
    model = load_model('trained_model.h5')
    scaler = load('scaler.joblib')

    # Load the data
    df = pd.read_csv('reliance.csv', parse_dates=['Date'])
    df = df[df['Date'] <= '2024-07-31'].dropna()  # Filter data till July 31

    # Prepare data for prediction
    data = df['Close'].values

    # Predict for the next 5 months
    predictions = predict_next_months(model, scaler, data, months=5)

    return predictions

if __name__ == "__main__":
    predictions = load_model_and_predict()
    print("Predictions for the next 5 months:")
    for month_idx, month_predictions in enumerate(predictions, start=1):
        print(f"Month {month_idx}: {month_predictions}")
