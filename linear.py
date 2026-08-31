import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("/home/cs-ai-31/akash/data.csv")

X = torch.tensor(data["X"].values, dtype=torch.float32).view(-1, 1)
Y = torch.tensor(data["Y"].values, dtype=torch.float32).view(-1, 1)

model = nn.Linear(1, 1)

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 1000

for epoch in range(epochs):

    prediction = model(X)

    loss = criterion(prediction, Y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss = {loss.item():.4f}")

weight = model.weight.item()
bias = model.bias.item()

print("\nWeight :", weight)
print("Bias :", bias)

x_test = torch.tensor([[6.0]])
y_pred = model(x_test)

print("Prediction for x = 6 :", y_pred.item())

x_np = X.numpy()
y_np = Y.numpy()

plt.scatter(x_np, y_np, color='blue', label='Actual Data Points')

x_line = np.linspace(x_np.min()-1, x_test.item()+1, 100)
y_line = weight * x_line + bias

plt.plot(
    x_line,
    y_line,
    color='red',
    label=f"Regression Line (Y={weight:.2f}X+{bias:.2f})"
)
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("/home/cs-ai-31/akash/data.csv")

X = torch.tensor(data["X"].values, dtype=torch.float32).view(-1, 1)
Y = torch.tensor(data["Y"].values, dtype=torch.float32).view(-1, 1)

model = nn.Linear(1, 1)

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 1000

for epoch in range(epochs):

    prediction = model(X)

    loss = criterion(prediction, Y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss = {loss.item():.4f}")

weight = model.weight.item()
bias = model.bias.item()

print("\nWeight :", weight)
print("Bias :", bias)

x_test = torch.tensor([[6.0]])
y_pred = model(x_test)

print("Prediction for x = 6 :", y_pred.item())

x_np = X.numpy()
y_np = Y.numpy()

plt.scatter(x_np, y_np, color='blue', label='Actual Data Points')

x_line = np.linspace(x_np.min()-1, x_test.item()+1, 100)
y_line = weight * x_line + bias

plt.plot(
    x_line,
    y_line,
    color='red',
    label=f"Regression Line (Y={weight:.2f}X+{bias:.2f})"
)

plt.scatter(
    x_test.item(),
    y_pred.item(),
    color='green',
    marker='x',
    s=120,
    linewidths=2,
    label=f"Prediction for x=6 ({y_pred.item():.2f})"
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression using PyTorch")
plt.legend()
plt.grid(True)
plt.show()


plt.scatter(
    x_test.item(),
    y_pred.item(),
    color='green',
    marker='x',
    s=120,
    linewidths=2,
    label=f"Prediction for x=6 ({y_pred.item():.2f})"
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression using PyTorch")
plt.legend()
plt.grid(True)
plt.show()

