
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data.csv")

X = torch.tensor(data["Hours_Studied"].values, dtype=torch.float32).view(-1, 1)
Y = torch.tensor(data["Exam_Score"].values, dtype=torch.float32).view(-1, 1)

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    prediction = model(X)
    loss = criterion(prediction, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

weight = model.weight.item()
bias = model.bias.item()

print("Weight:", weight)
print("Bias:", bias)

x_test = torch.tensor([[6.0]])
y_pred = model(x_test)
print("Predicted score:", y_pred.item())

plt.scatter(X.numpy(), Y.numpy(), color="blue", label="Actual Data")

x_line = np.linspace(0, 12, 100)
y_line = weight * x_line + bias

plt.plot(x_line, y_line, color="red", label="Regression Line")
plt.scatter(6, y_pred.item(), color="green", label="Prediction")

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.show()
