import numpy as np
import pandas as pd
import requests
import json
from datetime import date, datetime

from flask import Flask
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client import make_wsgi_app

from scipy.signal import savgol_filter
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

# Add prometheus wsgi middleware to route /metrics requests
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

@app.route('/train')
def train():
    url = 'http://172.26.128.130:30090/api/v1/query_range?query=sum by (namespace) (rate(container_cpu_usage_seconds_total{container="compose-post-service"}[120s])) / 0.03&start=1705910469.323&end=1705928469.323&step=5'
    response = requests.get(url)
    output = json.loads(response.text)

    rows = output['data']['result'][0]['values']
    fields = ['timestamp', 'value']
    df = pd.DataFrame(np.array(rows), columns=fields)
    df.index = pd.to_datetime(df['timestamp'], unit='s')
    df.pop('timestamp')
    
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
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dropout(0.3))
    model.add(Dense(n_forecast))

    early_stop = EarlyStopping(monitor = 'loss', patience = 10, restore_best_weights=True)

    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.02), metrics=[RootMeanSquaredError()])
    start_time = datetime.now()
    model_hist = model.fit(X, y,
                            validation_data=(X_val, y_val),
                            epochs=75,
                            batch_size=100,
                            callbacks=[early_stop]
                            )
    end_time = datetime.now()
    print(f'Training completed in {(end_time-start_time).seconds} seconds.')
    return f'Model trained, final loss value: {model_hist.history["loss"][-1]}'
