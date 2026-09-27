
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

data = yf.download("^NSEI", period="5y", auto_adjust=True,progress=False)["Close"].values
data = data.reshape(-1, 1)

mn, mx = data.min(), data.max()
data = (data - mn) / (mx - mn)

X, y = [], []
for i in range(30, len(data)):
    X.append(data[i-30:i])
    y.append(data[i])

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)

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

for epoch in range(20):
    opt.zero_grad()
    out = model(X)
    loss = loss_fn(out, y)
    loss.backward()
    opt.step()

    if (epoch + 1) % 5 == 0:
        print("Epoch:", epoch + 1, "Loss:", loss.item())

model.eval()
seq = torch.tensor(data[-30:], dtype=torch.float32).unsqueeze(0)
predictions = []

with torch.no_grad():
    for _ in range(10):
        p = model(seq)
        predictions.append(p.item())
        seq = torch.cat((seq[:, 1:, :], p.unsqueeze(1)), dim=1)

predictions = np.array(predictions) * (mx - mn) + mn

print("\nPredicted NIFTY-50 values:")
print(predictions)

plt.plot(data[-100:] * (mx - mn) + mn, label="Actual")
plt.plot(
    range(100, 110),
    predictions,
    marker="o",
    label="Forecast"
)

plt.xlabel("Days")
plt.ylabel("NIFTY-50")
plt.title("NIFTY-50 Time Series Forecasting")
plt.legend()
plt.show()