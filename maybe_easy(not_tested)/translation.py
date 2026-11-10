
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load dataset
path = "/home/cs-ai-32/Ajay/translation/Dataset_English_Hindi.csv"

df = pd.read_csv(path, encoding="utf-8").dropna().head(3000)

english = df["English"].astype(str).tolist()
hindi = df["Hindi"].astype(str).tolist()

# 2. Convert English sentences into TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
X = vectorizer.fit_transform(english).toarray()

# 3. Create Autoencoder
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(500, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 500)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# 4. Initialize model
model = Model()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

data = torch.tensor(X, dtype=torch.float32)

# 5. Train model
for epoch in range(20):
    optimizer.zero_grad()

    encoded, output = model(data)
    loss = loss_fn(output, data)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print("Epoch:", epoch + 1, "Loss:", round(loss.item(), 4))

# 6. Encode all English sentences
model.eval()

with torch.no_grad():
    encoded, _ = model(data)
    encoded = encoded.numpy()

# 7. Find Hindi translation
print("\nModel ready! Type 'exit' to quit.")

while True:
    text = input("\nEnglish: ").strip()

    if text.lower() == "exit":
        break

    if not text:
        continue

    query = vectorizer.transform([text]).toarray()
    query = torch.tensor(query, dtype=torch.float32)

    with torch.no_grad():
        query_encoded, _ = model(query)
        query_encoded = query_encoded.numpy()

    similarity = cosine_similarity(query_encoded, encoded)[0]
    index = np.argmax(similarity)

    print("Closest English:", english[index])
    print("Hindi Translation:", hindi[index])
