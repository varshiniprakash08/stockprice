import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from joblib import load

# Load the trained model and scaler
model = load_model('trained_model.h5')
scaler = load('scaler.joblib')

# Load the dataset
df = pd.read_csv('reliance.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True)  # Ensure data is sorted by date

# Handle missing values in the data
df['Close'].fillna(method='ffill', inplace=True)  # Forward fill to handle NaN values
df['Close'].fillna(method='bfill', inplace=True)  # Backward fill to handle NaN values

def validate_data(data, step):
    if not np.all(np.isfinite(data)):
        print(f"Invalid data detected at {step}: {data}")
        raise ValueError(f"Data contains NaN or infinite values at {step}")

def predict_august(data):
    predictions = []
    for day in range(150):  # Predict for the next 150 days (5 months)
        # Make prediction here for each day
        # Assuming you use a trained model for prediction
        # Example: predict the next day's closing price
        input_data = np.array(data[-60:]).reshape(1, 60, 1)  # Use last 60 days
        prediction = model.predict(input_data)
        predictions.append(prediction[0][0])  # Add predicted value to the list
        data = np.append(data, prediction)  # Append prediction to data for next prediction
    
    return predictions

# Generate predictions
august_predictions = predict_august(df['Close'].values)

def calculate_risk(predictions):
    if len(predictions) == 0 or not np.any(np.isfinite(predictions)):
        return 50  # Default risk if predictions are invalid

    volatility = np.std(predictions)
    mean_price = np.mean(predictions)

    if mean_price == 0:
        return 100  # High risk if mean price is zero

    risk_percentage = (volatility / mean_price) * 100
    risk_percentage = min(max(risk_percentage, 10), 90)  # Clamp risk between 10-90%

    return round(risk_percentage, 2)

risk_percentage = calculate_risk(august_predictions)

# Create the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Stock Price Prediction Dashboard", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Select Stock:"),
        dcc.Dropdown(
            id='stock-dropdown',
            options=[{'label': 'Reliance', 'value': 'reliance.csv'}],
            value='reliance.csv'
        ),
    ], style={'width': '50%', 'margin': 'auto'}),

    dcc.Graph(id='trend-graph'),
    dcc.Graph(id='last-month-graph'),
    dcc.Graph(id='august-graph'),

    dcc.Graph(
        id='risk-chart',
        figure={
            'data': [
                go.Pie(
                    labels=['Risk', 'Safe'],
                    values=[risk_percentage, 100 - risk_percentage],
                    hole=0.4
                )
            ],
            'layout': go.Layout(title=f'Investment Risk: {risk_percentage}% Risky')
        }
    ),
])

@app.callback(
    [Output('trend-graph', 'figure'),
     Output('last-month-graph', 'figure'),
     Output('august-graph', 'figure')],
    [Input('stock-dropdown', 'value')]
)
def update_graphs(selected_stock):
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Actual Prices'))
    trend_fig.update_layout(title="Complete Stock Trend", xaxis_title="Date", yaxis_title="Price")

    last_month_fig = go.Figure()
    last_month_fig.add_trace(go.Scatter(x=df[df['Date'] > '2024-06-30']['Date'],
                                       y=df[df['Date'] > '2024-06-30']['Close'],
                                       mode='lines', name='Last Month Prices'))
    last_month_fig.update_layout(title="Last Month (July) Prices", xaxis_title="Date", yaxis_title="Price")

    august_dates = pd.date_range(start='2024-08-01', end='2024-08-31')
    august_fig = go.Figure()
    august_fig.add_trace(go.Scatter(x=august_dates, y=august_predictions, mode='lines', name='Predicted Prices'))
    august_fig.update_layout(title="August 2024 Predicted Prices", xaxis_title="Date", yaxis_title="Price")

    return trend_fig, last_month_fig, august_fig

if __name__ == '__main__':
    app.run_server(debug=True)
