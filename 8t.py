import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

MAX_WORDS, MAX_LEN = 10000, 200

# Load data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)

x_train = pad_sequences(x_train, maxlen=MAX_LEN)[:10000]
y_train = torch.tensor(y_train[:10000], dtype=torch.float32)

x_test = torch.tensor(
    pad_sequences(x_test, maxlen=MAX_LEN)[:2000],
    dtype=torch.long
)

x_train = torch.tensor(x_train, dtype=torch.long)
y_test = torch.tensor(y_test[:2000], dtype=torch.float32)


# Model
class Model(nn.Module):
    def __init__(self, typ):
        super().__init__()
        self.emb = nn.Embedding(MAX_WORDS, 128)

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

        return torch.sigmoid(self.fc(h[-1])).squeeze()


results = {}

# Train RNN, GRU, LSTM
for typ in ["RNN", "GRU", "LSTM"]:

    print("\nTraining", typ)

    model = Model(typ).to(device)
    opt = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCELoss()

    # 80/20 split
    split = 8000
    train_x, val_x = x_train[:split], x_train[split:]
    train_y, val_y = y_train[:split], y_train[split:]

    train_acc, val_acc, val_loss = [], [], []

    start = time.time()

    for epoch in range(5):

        model.train()

        for i in range(0, len(train_x), 128):

            x = train_x[i:i+128].to(device)
            y = train_y[i:i+128].to(device)

            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

        # Validation
        model.eval()

        with torch.no_grad():
            out = model(val_x.to(device))
            loss = loss_fn(out, val_y.to(device))
            acc = ((out >= .5) == val_y.to(device)).float().mean()

        val_acc.append(acc.item())
        val_loss.append(loss.item())

        print(
            f"Epoch {epoch+1}: "
            f"Val Accuracy={acc.item():.4f}"
        )

    # Test
    with torch.no_grad():
        out = model(x_test.to(device))
        acc = ((out >= .5) == y_test.to(device)).float().mean()

    results[typ] = {
        "accuracy": acc.item(),
        "time": time.time() - start,
        "val_acc": val_acc,
        "val_loss": val_loss
    }


# Results
print("\n==============================")
print("FINAL RESULTS")
print("==============================")

for typ, r in results.items():
    print(
        f"{typ:<6} "
        f"Accuracy: {r['accuracy']:.4f} "
        f"Time: {r['time']:.2f}s"
    )


# Accuracy comparison
plt.bar(
    results.keys(),
    [r["accuracy"] * 100 for r in results.values()]
)
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison")
plt.show()


# Training time
plt.bar(
    results.keys(),
    [r["time"] for r in results.values()]
)
plt.ylabel("Seconds")
plt.title("Training Time Comparison")
plt.show()


# Validation accuracy
for typ, r in results.items():
    plt.plot(r["val_acc"], label=typ)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.grid()
plt.show()


# Validation loss
for typ, r in results.items():
    plt.plot(r["val_loss"], label=typ)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss")
plt.legend()
plt.grid()
plt.show()