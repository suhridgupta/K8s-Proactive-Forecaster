import numpy as np
import pandas as pd
import requests
import json
from datetime import date, datetime, timedelta
import os
import argparse

from flask import Flask, Response
from prometheus_client import Gauge, generate_latest, REGISTRY

from scipy.signal import savgol_filter
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

import warnings
#from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--app', dest='app', type=str)
parser.add_argument('run')
parser.add_argument('--host', dest='host', type=str)
parser.add_argument('--port', dest='port', type=str)
args = parser.parse_args()

app = Flask(__name__)

CONTENT_TYPE = str('text/plain; charset=utf-8')
FORECAST_DELTA_MINS = int(os.environ.get('FORECAST_DELTA_MINS', '10'))
DATA_QUERY = os.environ.get('DATA_QUERY', 'round(sum by (namespace) (rate(container_cpu_usage_seconds_total{container="compose-post-service"}[120s])) / 0.03)')
TRAINING_THRESHOLD = os.environ.get('TRAINING_THRESHOLD', '500')

forecasted_cpu = Gauge(name='forecasted_cpu',
                       documentation='CPU usage forecasted for compose-post-service',
                       labelnames=['service', 'forecast_delta_mins']
                       )

df_forecast = pd.DataFrame(columns=['timestamp', 'forecast', 'value'])
df_forecast.set_index('timestamp')

last_training_time = datetime(1970, 1, 1)

n_lookback = 10
n_forecast = 540

batch_size = 100
epochs = 50
learning_rate = 0.005

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
model.add(Dropout(0.1))
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
model.add(Dropout(0.1))
model.add(LSTM(units=50))
model.add(Dropout(0.1))
model.add(Dense(n_forecast))


best_loss = np.Infinity
best_model = None

@app.route('/metrics', methods=['GET'])
def metrics():
    global df_forecast
    global last_training_time

    closest_value = 0.0
    current_time = datetime.now()
    target_time = current_time + timedelta(minutes=FORECAST_DELTA_MINS)
    
    print(f'Current time: {current_time}    Requested time: {target_time}')
    
    tolerance = 5   # Time within the forecast dataframe to find a value
    closest_timestamp = min(df_forecast.index,
                            key=lambda df_time: abs(df_time - target_time),
                            default=datetime(1970, 1, 1)
                            )
    
    if abs((closest_timestamp - target_time).total_seconds()) < 5:
        closest_value = df_forecast.loc[df_forecast.index == closest_timestamp, 'forecast'].item()
        closest_value = max(closest_value, 0)
    forecasted_cpu.labels(service='compose-post-service', forecast_delta_mins=FORECAST_DELTA_MINS)\
            .set(closest_value)

    # If training has not been done in the last 15 mins, send GET request
#    if(abs(current_time - last_training_time).total_seconds() > 15 * 60):
#        os.system(f'curl http://{args.host}:{args.port}/train&')
#        last_training_time = current_time

    return Response(generate_latest(REGISTRY.restricted_registry(['forecasted_cpu'])), mimetype=CONTENT_TYPE)

@app.route('/train')
def train():
    global model
    global n_lookback
    global n_forecast
    global df_forecast
    global best_loss
    global best_model
    global batch_size
    global epochs
    global learning_rate
    global DATA_QUERY
    global TRAINING_THRESHOLD

    query = DATA_QUERY
    #query = 'sum by (namespace) (rate(container_cpu_usage_seconds_total{container="home-timeline-service"}[120s])) / 0.015'
    end = datetime.now().timestamp()
    start = end - (15 * 60 * 60)    # 15 hours in seconds
    step = 5
    url = f'http://172.26.128.130:30090/api/v1/query_range?query={query}&start={start}&end={end}&step={step}'
    #url = 'http://172.26.128.130:30090/api/v1/query_range?query=sum by (namespace) (rate(container_cpu_usage_seconds_total{container="compose-post-service"}[120s])) / 0.03&start=1705910469.323&end=1705928469.323&step=5'
    response = requests.get(url)
    output = json.loads(response.text)

    rows = []
    for row in output['data']['result']:
        rows = rows + row['values']

    fields = ['timestamp', 'value']
    df = pd.DataFrame(np.array(rows), columns=fields)
    df.index = pd.to_datetime(df['timestamp'], unit='s')
    df.pop('timestamp')

    # Do not train and predict if there is less than 2500 data points
    if(len(df) < 3500):
        return f'Insufficient data, skipping training and prediction.'
    
    df['value'] = savgol_filter(df, window_length=205, polyorder=7, axis=0)

    X = []
    y = []

    for i in range(n_lookback, len(df) - n_forecast + 1):
        X.append(df[i - n_lookback: i])
        y.append(df[i: i + n_forecast])

    X = np.array(X)
    y = np.array(y)

    n = len(X)

    X_train, y_train = X[:int(n*0.7)], y[:int(n*0.7)]
    X_val, y_val = X[int(n*0.7):int(n*0.9)], y[int(n*0.7):int(n*0.9)]
    X_test, y_test = X[int(n*0.9):], y[int(n*0.9):]

    early_stop = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights=True)

    # Find best parameters
    metric_query = f'count_over_time((wrk2_avg_api_latency>{TRAINING_THRESHOLD})[25m:])'
    metric_end =  datetime.now().timestamp()
    metric_start = metric_end - (25 * 60)
    metric_step = 5
    metric_url = f'http://172.26.128.130:30090/api/v1/query_range?query={metric_query}&start={metric_start}&end={metric_end}&step={metric_step}'
    metric_response = requests.get(metric_url)
    metric_output = json.loads(metric_response.text)

    if(len(metric_output['data']['result']) > 0):
        print("Latency threshold reached, tuning model accordingly")
        learning_rate = min(learning_rate-0.0002, 0.001)
        batch_size = max(150, batch_size+5)
        epochs = max(100, epochs+5)
    else:
        print("No latency threshold detected, reverting to default settings")
        learning_rate = 0.005
        batch_size = 100
        epochs = 50

    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])
    print("Starting training ...")
    start_time = datetime.now()
    model_hist = model.fit(X, y,
                            validation_data=(X_val, y_val),
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[early_stop],
                            verbose=0
                            )
    end_time = datetime.now()
    print(f'Training completed in {(end_time-start_time).seconds} seconds.')
    print(f'Model loss value: {model_hist.history["loss"][-1]}')

    X_predict = df['value'][- n_lookback:] # Last input sequence
    X_predict = np.array(X_predict)
    X_predict = X_predict.reshape(1, n_lookback, 1)

    validation_loss = model_hist.history["loss"][-1]

    y_test_hat = model.predict(X_test)
    test_mse = MeanSquaredError()
    current_loss = test_mse(y_test_hat.flatten(), y_test.flatten()).numpy()
    print(f'Test loss: {current_loss}')

    if (current_loss < best_loss):
        print("Current model is best performing, saving ...")
        best_loss = current_loss
        best_model = model
    else:
        print("Current model not best performing, using previous best model ...")

    y_predict = best_model.predict(X_predict).reshape(-1, 1)

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
    
    return f'{best_loss}'



