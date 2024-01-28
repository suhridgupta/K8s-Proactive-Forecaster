import numpy as np
import pandas as pd
import requests
import json
from datetime import date, datetime, timedelta
import os

from flask import Flask, Response
from prometheus_client import Gauge, generate_latest, REGISTRY

from scipy.signal import savgol_filter
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

CONTENT_TYPE = str('text/plain; charset=utf-8')
FORECAST_DELTA = int(os.environ.get('FORECAST_DELTA', '10'))

forecasted_cpu = Gauge(name='forecasted_cpu',
                       documentation='CPU usage forecasted for compose-post-service',
                       labelnames=['service']
                       )

df_forecast = pd.DataFrame(columns=['timestamp', 'forecast', 'value'])
df_forecast.set_index('timestamp')

last_training_time = datetime(1970, 1, 1)

@app.route('/metrics', methods=['GET'])
def metrics():
    global df_forecast
    global last_training_time

    closest_value = 0.0
    current_time = datetime.now()
    target_time = current_time + timedelta(minutes=FORECAST_DELTA)
    
    print(f'Current time: {current_time}    Requested time: {target_time}')
    
    tolerance = 5   # Time within the forecast dataframe to find a value
    closest_timestamp = min(df_forecast.index,
                            key=lambda df_time: abs(df_time - target_time),
                            default=datetime(1970, 1, 1)
                            )
    
    if abs((closest_timestamp - target_time).total_seconds()) < 5:
        closest_value = df_forecast.loc[df_forecast.index == closest_timestamp, 'forecast'].item()
    forecasted_cpu.labels(service='compose-post-service')\
            .set(closest_value)

    # If training has not been done in the last 1 hour, send GET request
    if(abs(current_time - last_training_time).total_seconds() > 60 * 60):
        print("Beginning model training.")
        os.system("curl http://localhost:8082/train&")
        last_training_time = current_time

    return Response(generate_latest(REGISTRY.restricted_registry(['forecasted_cpu'])), mimetype=CONTENT_TYPE)

@app.route('/train')
def train():
    global df_forecast
    query = 'round(sum by (namespace) (rate(container_cpu_usage_seconds_total{container="compose-post-service"}[120s])) / 0.03)'
    end = datetime.now().timestamp()
    start = end - (15 * 60 * 60)    # 15 hours in seconds
    step = 5
    url = f'http://172.26.128.130:30090/api/v1/query_range?query={query}&start={start}&end={end}&step={step}'
    #url = 'http://172.26.128.130:30090/api/v1/query_range?query=sum by (namespace) (rate(container_cpu_usage_seconds_total{container="compose-post-service"}[120s])) / 0.03&start=1705910469.323&end=1705928469.323&step=5'
    response = requests.get(url)
    output = json.loads(response.text)

    rows = output['data']['result'][0]['values']
    fields = ['timestamp', 'value']
    df = pd.DataFrame(np.array(rows), columns=fields)
    df.index = pd.to_datetime(df['timestamp'], unit='s')
    df.pop('timestamp')

    # Do not train and predict if there is less than 1500 data points
    if(len(df) < 1500):
        return f'Insufficient data, skipping training and prediction.'
    
    df['value'] = savgol_filter(df, window_length=75, polyorder=2, axis=0)

    n_lookback = 5
    n_forecast = 540

    X = []
    y = []

    for i in range(n_lookback, len(df) - n_forecast + 1):
        X.append(df[i - n_lookback: i])
        y.append(df[i: i + n_forecast])

    X = np.array(X)
    y = np.array(y)

    n = len(X)

    X_train, y_train = X[:int(n*0.7)], y[:int(n*0.7)]
    X_val, y_val = X[int(n*0.7):], y[int(n*0.7):]

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(n_forecast))

    early_stop = EarlyStopping(monitor = 'loss', patience = 8, restore_best_weights=True)

    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.02), metrics=[RootMeanSquaredError()])
    start_time = datetime.now()
    model_hist = model.fit(X, y,
                            validation_data=(X_val, y_val),
                            epochs=100,
                            batch_size=100,
                            callbacks=[early_stop]
                            )
    end_time = datetime.now()
    print(f'Training completed in {(end_time-start_time).seconds} seconds.')

    X_predict = df['value'][- n_lookback:] # Last input sequence
    X_predict = np.array(X_predict)
    X_predict = X_predict.reshape(1, n_lookback, 1)

    y_predict = model.predict(X_predict).reshape(-1, 1)

    # Plot actual and forecasted values
    df_past = df[['value']].reset_index()
    df_past['forecast'] = np.nan
    df_past['forecast'].iloc[-1] = df_past['value'].iloc[-1]

    df_future = pd.DataFrame(columns=['timestamp', 'forecast', 'value'])
    df_future['timestamp'] = pd.date_range(start=df_past['timestamp'].iloc[-1], freq='5s', periods=n_forecast)
    df_future['forecast'] = y_predict.flatten()
    df_future['value'] = np.nan

    df_past = df_past.set_index('timestamp')
    df_future = df_future.set_index('timestamp')
    results = pd.concat([df_past, df_future])
    df_forecast = df_future.copy()
    
    print(f'Model loss value: {model_hist.history["loss"][-1]}')
    return f'Model trained, final loss value: {model_hist.history["loss"][-1]}'


