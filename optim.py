import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Dataset
train = datasets.MNIST(
    root="./data", train=True, download=True,
    transform=transforms.ToTensor()
)
test = datasets.MNIST(
    root="./data", train=False, download=True,
    transform=transforms.ToTensor()
)

train_loader = DataLoader(train, batch_size=128, shuffle=True)
test_loader = DataLoader(test, batch_size=128)

# Model
class Net(nn.Module):
    def __init__(self, init="xavier", dropout=0):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 10)

        if init == "xavier":
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
        elif init == "kaiming":
            nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
            nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


def train_model(init, dropout=0, weight_decay=0):
    model = Net(init, dropout)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    losses, accuracies = [], []

    for epoch in range(5):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Test accuracy
        model.eval()
        correct = total = 0

        with torch.no_grad():
            for x, y in test_loader:
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        losses.append(total_loss / len(train_loader))
        accuracies.append(100 * correct / total)

    return losses, accuracies


# Experiments
results = {
    "Xavier": train_model("xavier"),
    "Kaiming": train_model("kaiming"),
    "Dropout": train_model("kaiming", dropout=0.5),
    "L2 Regularization": train_model("kaiming", weight_decay=0.0001)
}

# Visualization
for name, (loss, acc) in results.items():
    plt.plot(loss, label=name)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Effect of Initialization and Regularization")
plt.legend()
plt.show()

for name, (loss, acc) in results.items():
    plt.plot(acc, marker="o", label=name)

plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.title("Performance Comparison")
plt.legend()
plt.show()

# Final accuracy
for name, (_, acc) in results.items():
    print(name, "Final Accuracy:", round(acc[-1], 2), "%")
