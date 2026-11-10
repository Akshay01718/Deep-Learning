import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

csv_file_path = r"/home/cs-ai-33/aharon/DATA11/BRITANNIA.csv"  

# 2. Load and Prepare Data
try:
    df = pd.read_csv(csv_file_path)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Robust dynamic column mapping for Date and Close columns
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    close_cols = [col for col in df.columns if 'close' in col.lower()]

    if not date_cols or not close_cols:
        raise KeyError("Could not automatically locate 'Date' or 'Close' columns in the CSV header.")

    date_col = date_cols[0]
    close_col = close_cols[0]
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    print(f"Successfully loaded '{csv_file_path}' with {len(df)} records.")
    
except FileNotFoundError:
    print(f"Error: Could not find file at '{csv_file_path}'. Please verify the path.")
    exit()
except Exception as e:
    print(f"Data Loading Error: {e}")
    exit()

# Extract Closing prices and normalize between 0 and 1
close_prices = df[[close_col]].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# 3. Create Sliding Window Sequences
TIME_STEPS = 60  # Past 60 trading days used to predict the next day

X, y = [], []
for i in range(TIME_STEPS, len(scaled_data)):
    X.append(scaled_data[i - TIME_STEPS:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Reshape input to 3D for LSTM: [samples, time_steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Train-Test Split (80% Train, 20% Test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. Build LSTM Model Architecture
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. Train Model
print("\nTraining RNN/LSTM model...")
model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

# 6. Evaluate and Plot Results
predictions = model.predict(X_test)
predictions_unscaled = scaler.inverse_transform(predictions)
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(y_test_unscaled, label='Actual Close Price', color='blue')
plt.plot(predictions_unscaled, label='RNN Predicted Close Price', color='red', linestyle='--')
plt.title(f'Stock Price Time Series Forecasting for BRITANNIA')
plt.xlabel('Trading Days (Test Set)')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True)
plt.show()