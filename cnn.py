import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train = datasets.MNIST("./data", train=True, download=True, transform=transform)
test = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader = DataLoader(test, batch_size=64)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    loss_sum = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    print(f"Epoch {epoch+1}/5, Loss: {loss_sum/len(train_loader):.4f}")

model.eval()
correct = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()

print("Accuracy:", 100 * correct / len(test), "%")

x, y = next(iter(test_loader))
with torch.no_grad():
    pred = model(x.to(device)).argmax(1).cpu()

plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x[i].squeeze() * 0.5 + 0.5, cmap="gray")
    plt.title(f"Predicted:{pred[i]}\nActual:{y[i]}")
    plt.axis("off")
plt.show()
'''
# Print predictions 
for i in range(5): 
    print( f"Image {i+1}: " f"Predicted = {pred[i].item()}, " f"Actual = {y[i].item()}" )
    '''