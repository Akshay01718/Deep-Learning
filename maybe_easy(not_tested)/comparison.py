
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Load dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Preprocessing
x_train = pad_sequences(x_train, maxlen=200)[:10000]
x_test = pad_sequences(x_test, maxlen=200)[:2000]

x_train = torch.tensor(x_train, dtype=torch.long)
y_train = torch.tensor(y_train[:10000], dtype=torch.float32)

x_test = torch.tensor(x_test, dtype=torch.long)
y_test = torch.tensor(y_test[:2000], dtype=torch.float32).to(device)

# Model
class Model(nn.Module):
    def __init__(self, typ):
        super().__init__()
        self.emb = nn.Embedding(10000, 128)

        if typ == "RNN":
            self.rnn = nn.RNN(128, 64, batch_first=True)
        elif typ == "GRU":
            self.rnn = nn.GRU(128, 64, batch_first=True)
        else:
            self.rnn = nn.LSTM(128, 64, batch_first=True)

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.emb(x)
        _, h = self.rnn(x)
        if isinstance(h, tuple):
            h = h[0]
        return self.fc(h[-1]).squeeze(-1)

# Store results
results = {}

# Train RNN, GRU and LSTM
for typ in ["RNN", "GRU", "LSTM"]:

    print("\nTraining:", typ)

    model = Model(typ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    train_x, val_x = x_train[:8000], x_train[8000:]
    train_y, val_y = y_train[:8000], y_train[8000:]

    acc_list, loss_list = [], []
    start = time.time()

    for epoch in range(5):
        model.train()

        for i in range(0, 8000, 128):
            x = train_x[i:i+128].to(device)
            y = train_y[i:i+128].to(device)

            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        with torch.no_grad():
            out = model(val_x.to(device))
            loss = loss_fn(out, val_y.to(device))
            pred = (torch.sigmoid(out) >= 0.5).float()
            acc = (pred == val_y.to(device)).float().mean()

        acc_list.append(acc.item())
        loss_list.append(loss.item())

        print(f"Epoch {epoch+1}: Val Accuracy={acc.item():.4f}")

    # Testing
    with torch.no_grad():
        out = model(x_test.to(device))
        pred = (torch.sigmoid(out) >= 0.5).float()
        accuracy = (pred == y_test.to(device)).float().mean()

    results[typ] = [
        accuracy.item(),
        time.time() - start,
        acc_list,
        loss_list
    ]

# Final results
print("\nFINAL RESULTS")

for typ, r in results.items():
    print(f"{typ}: Accuracy={r[0]:.4f}, Time={r[1]:.2f}s")

# Accuracy comparison
plt.bar(results.keys(), [r[0]*100 for r in results.values()])
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.show()

# Training time comparison
plt.bar(results.keys(), [r[1] for r in results.values()])
plt.title("Training Time Comparison")
plt.ylabel("Time (seconds)")
plt.show()

# Validation accuracy
for typ, r in results.items():
    plt.plot(r[2], label=typ)

plt.title("Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

# Validation loss
for typ, r in results.items():
    plt.plot(r[3], label=typ)

plt.title("Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()
