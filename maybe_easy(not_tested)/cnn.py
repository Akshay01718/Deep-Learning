
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load existing dataset
train = torchvision.datasets.MNIST(
    "./data", train=True, download=False, transform=transform
)

test = torchvision.datasets.MNIST(
    "./data", train=False, download=False, transform=transform
)

# DataLoader
tr = torch.utils.data.DataLoader(
    train, batch_size=64, shuffle=True
)

te = torch.utils.data.DataLoader(
    test, batch_size=64
)

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = CNN().to(device)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5):
    model.train()

    for x, y in tr:
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)

        loss.backward()
        opt.step()

    print("Epoch:", epoch + 1)

# Testing
model.eval()
correct = total = 0

with torch.no_grad():
    for x, y in te:
        x, y = x.to(device), y.to(device)

        pred = model(x).argmax(1)

        correct += (pred == y).sum().item()
        total += y.size(0)

print("Test Accuracy:", correct / total * 100, "%")

# Display predictions
x, y = next(iter(te))

with torch.no_grad():
    pred = model(x.to(device)).argmax(1).cpu()

print("Actual:   ", y[:5].numpy())
print("Predicted:", pred[:5].numpy())

plt.figure(figsize=(10, 4))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x[i].squeeze() * 0.5 + 0.5, cmap="gray")
    plt.title(f"A:{y[i]} P:{pred[i]}")
    plt.axis("off")

plt.show()
