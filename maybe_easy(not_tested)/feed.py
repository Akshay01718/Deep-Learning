import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
train_dataset = datasets.CIFAR10(
    root = './data',    
    train = True,
    download = True,
    transform = transform 
)
test_dataset = datasets.CIFAR10(
    root = './data',
    train = False,
    download = False,
    transform = transform 
)
train_loader = DataLoader(train_dataset,batch_size = 64, shuffle = True)
test_loader = DataLoader(test_dataset,batch_size = 64, shuffle = False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3072,512),
    nn.ReLU(),
    nn.Linear(512,256),
    nn.ReLU(),
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Linear(128,10)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
epochs = 10
loss_history = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"Epoch:{epoch+1}/{epochs},loss: {running_loss/len(train_loader):.4f}")
    
model.eval()
correct = 0
total =0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _,predicted = torch.max(outputs,1)
        total +=labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    
    print(f"\nTest accuracy = {accuracy:.2f}%")
    
    

plt.plot(range(1, epochs+1), loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.grid(True)
plt.show()

classes = train_dataset.classes
images, labels = next(iter(test_loader))
images = images.to(device)
with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
images = images.cpu()

plt.figure(figsize=(12, 6))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    img = images[i] / 2 + 0.5
    plt.imshow(img.permute(1, 2, 0))
    plt.title(f"Pred: {classes[preds[i].item()]}")
    plt.axis("off")
plt.show()