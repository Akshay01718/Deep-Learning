import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# MNIST -> 3 channels -> 32x32
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(3),
    transforms.ToTensor()
])

train = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
test = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)

tr = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
te = torch.utils.data.DataLoader(test, batch_size=128)

# VGG19
model = vgg19(weights=VGG19_Weights.DEFAULT)

for p in model.parameters():
    p.requires_grad = False

model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

# Training
for epoch in range(3):
    model.train()

    for x, y in tr:
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

    print("Epoch:", epoch + 1)

# Test
model.eval()
correct = total = 0

with torch.no_grad():
    for x, y in te:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

print("Test Accuracy:", correct / total)

# First 10 predictions
x, y = next(iter(te))
with torch.no_grad():
    pred = model(x.to(device)).argmax(1).cpu()

print("Actual:   ", y[:10].numpy())
print("Predicted:", pred[:10].numpy())

plt.figure(figsize=(10, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x[i][0], cmap="gray")
    plt.title(f"A:{y[i]} P:{pred[i]}")
    plt.axis("off")

plt.show()