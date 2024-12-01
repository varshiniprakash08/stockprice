import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from joblib import dump
import logging


def validate_data(data, step):
    """ Validate data to ensure there are no NaN or infinite values. """
    if not np.all(np.isfinite(data)):
        logging.error(f"Invalid data detected at {step}: {data}")
        raise ValueError(f"Data contains NaN or infinite values at {step}")


def train_model(file_name='reliance.csv', batch_size=32, epochs=50, look_back=60):
    """ Train the LSTM model using stock price data. """
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info("Loading and processing data...")

    # Load the dataset
    df = pd.read_csv(file_name, parse_dates=['Date'])
    df = df[df['Date'] <= '2024-07-31'].dropna()  # Filter data till July 31 and remove missing values

    # Extract the 'Close' price as the target variable
    data = df['Close'].values.reshape(-1, 1)

    # Scale the data using MinMaxScaler to fit the data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    validate_data(scaled_data, "scaled data")

    # Save the scaler for future use during predictions
    dump(scaler, 'scaler.joblib')

    # Prepare training data using a look-back period
    X_train, y_train = [], []
    for i in range(look_back, len(scaled_data)):
        X_train.append(scaled_data[i-look_back:i, 0])  # 60 previous days as input
        y_train.append(scaled_data[i, 0])  # Target is the next day's price
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    validate_data(X_train, "training input")
    validate_data(y_train, "training target")

    logging.info(f"Training data prepared. Shape: {X_train.shape}")

    # Build the LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model with early stopping
    logging.info("Starting model training...")
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stop]
    )

    # Save the trained model
    model.save('trained_model.h5')
    logging.info("Model trained and saved successfully!")

    # Save the training history for future analysis
    pd.DataFrame(history.history).to_csv('training_history.csv', index=False)
    logging.info("Training history saved!")

    # Evaluate the model (optional step)
    logging.info("Evaluating model performance...")
    loss = model.evaluate(X_train, y_train, verbose=0)
    logging.info(f"Model evaluation loss: {loss:.4f}")


if __name__ == '__main__':
    train_model()
