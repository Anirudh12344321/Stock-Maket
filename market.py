import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


data = yf.download("AAPL", start="2015-01-01", end="2024-01-01")

close_prices = data['Close'].values
close_prices = close_prices.reshape(-1,1)


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_prices)

X = []
y = []


for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)


split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(X_train, y_train, epochs=10, batch_size=32)


predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)


plt.plot(close_prices[split:])
plt.plot(predictions)
plt.show()
