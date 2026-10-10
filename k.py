import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Load dataset
path = "/home/cs-ai-32/Ajay/translation/Dataset_English_Hindi.csv"
if not os.path.exists(path):
    raise FileNotFoundError("Dataset not found!")

df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip").dropna().head(3000)

en = df["English"].astype(str).tolist()
hi = df["Hindi"].astype(str).tolist()

# TF-IDF
N = 500
en_vec = TfidfVectorizer(max_features=N, stop_words="english")
hi_vec = TfidfVectorizer(max_features=N)

X = en_vec.fit_transform(en).toarray()
Y = hi_vec.fit_transform(hi).toarray()

X_train, _, Y_train, _ = train_test_split(
    X, Y, test_size=0.1, random_state=42
)

loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32)
    ),
    batch_size=32,
    shuffle=True
)

# Autoencoder
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N, 128), nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, N)
        )

    def forward(self, x):
        z = nn.functional.normalize(self.encoder(x), dim=1)
        return z, self.decoder(z)

model = Model()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(20):
    loss_sum = 0
    for x, y in loader:
        optimizer.zero_grad()
        _, output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss_sum/len(loader):.4f}")

# Encode all dataset sentences
model.eval()
with torch.no_grad():
    train_z, _ = model(torch.tensor(X, dtype=torch.float32))
    train_z = train_z.numpy()

# Interactive search
print("\nModel ready! Type 'exit' to quit.")

while True:
    text = input("\nEnglish: ").strip()

    if text.lower() == "exit":
        break
    if not text:
        continue

    x = torch.tensor(
        en_vec.transform([text]).toarray(),
        dtype=torch.float32
    )

    with torch.no_grad():
        z, _ = model(x)

    similarity = cosine_similarity(z.numpy(), train_z)[0]
    i = np.argmax(similarity)

    print("Closest English:", en[i])
    print("Hindi:", hi[i])

'''output 
Epoch 5: Loss = 0.0018
Epoch 10: Loss = 0.0016
Epoch 15: Loss = 0.0015
Epoch 20: Loss = 0.0014

Model ready! Type 'exit' to quit.

English: happy birthday
Closest English: Happy birthday!
Hindi: जन्मदिन मुबारक हो!

English: exit
'''