
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Parameters
VOCAB_SIZE = 10000
MAX_LENGTH = 200

# Load dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

# Preprocessing
x_train = pad_sequences(x_train, maxlen=MAX_LENGTH, padding="post")
x_test = pad_sequences(x_test, maxlen=MAX_LENGTH, padding="post")

# Convert to tensors
train = TensorDataset(
    torch.tensor(x_train, dtype=torch.long),
    torch.tensor(y_train, dtype=torch.float32)
)

test = TensorDataset(
    torch.tensor(x_test, dtype=torch.long),
    torch.tensor(y_test, dtype=torch.float32)
)

# DataLoader
tr = DataLoader(train, batch_size=64, shuffle=True)
te = DataLoader(test, batch_size=64)

# RNN Model
class RNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, 64)
        self.rnn = nn.RNN(64, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.rnn(x)
        return self.fc(h[-1]).squeeze(-1)

model = RNN().to(device)

# Loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
opt = optim.Adam(model.parameters(), lr=0.001)

# Store accuracy and loss
train_acc, val_acc = [], []
train_loss, val_loss = [], []

# Training
for epoch in range(5):
    model.train()
    correct = total = loss_sum = 0

    for x, y in tr:
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        opt.step()

        loss_sum += loss.item()
        pred = (torch.sigmoid(output) >= 0.5).float()
        correct += (pred == y).sum().item()
        total += y.size(0)

    train_loss.append(loss_sum / len(tr))
    train_acc.append(correct / total)

    # Validation
    model.eval()
    correct = total = loss_sum = 0

    with torch.no_grad():
        for x, y in te:
            x, y = x.to(device), y.to(device)

            output = model(x)
            loss = loss_fn(output, y)

            loss_sum += loss.item()
            pred = (torch.sigmoid(output) >= 0.5).float()
            correct += (pred == y).sum().item()
            total += y.size(0)

    val_loss.append(loss_sum / len(te))
    val_acc.append(correct / total)

    print(f"Epoch {epoch+1}: Loss={train_loss[-1]:.4f}, "
          f"Accuracy={train_acc[-1]:.4f}, "
          f"Val Accuracy={val_acc[-1]:.4f}")

# Final accuracy
print("Test Accuracy:", val_acc[-1] * 100, "%")
print("Test Loss:", val_loss[-1])

# Accuracy Graph
plt.figure(figsize=(8, 5))
plt.plot(train_acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("RNN Accuracy")
plt.legend()
plt.show()

# Loss Graph
plt.figure(figsize=(8, 5))
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("RNN Loss")
plt.legend()
plt.show()

# Predictions
x, y = next(iter(te))

with torch.no_grad():
    output = model(x.to(device))
    pred = (torch.sigmoid(output) >= 0.5).int().cpu()

print("Actual:   ", y[:10].numpy())
print("Predicted:", pred[:10].numpy())
