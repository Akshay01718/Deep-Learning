import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load locally stored NIFTY-50 data
df = pd.read_csv("NIFTY50.csv")

data = df["Close"].values.reshape(-1, 1)

mn, mx = data.min(), data.max()
data = (data - mn) / (mx - mn)

X, y = [], []
for i in range(30, len(data)):
    X.append(data[i-30:i])
    y.append(data[i])

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)

# Train-Test Split (80% Train, 20% Test)
train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# LSTM Model
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


model = LSTM()
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)

# Train Model
print("\nTraining RNN/LSTM model...")

for epoch in range(20):
    model.train()

    opt.zero_grad()
    out = model(X_train)
    loss = loss_fn(out, y_train)
    loss.backward()
    opt.step()

    if (epoch + 1) % 5 == 0:
        print("Epoch:", epoch + 1, "Loss:", loss.item())


# Evaluate Model
model.eval()

with torch.no_grad():
    predictions = model(X_test)

predictions = predictions.numpy() * (mx - mn) + mn
actual = y_test.numpy() * (mx - mn) + mn

# Plot Results
plt.figure(figsize=(12, 6))

plt.plot(actual, label='Actual Close Price', color='blue')
plt.plot(predictions, label='RNN Predicted Close Price',
         color='red', linestyle='--')

plt.title('NIFTY-50 Stock Price Time Series Forecasting')
plt.xlabel('Trading Days (Test Set)')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True)
plt.show()