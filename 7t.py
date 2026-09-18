# Experiment 7
import torch
import torch.nn as nn
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

x_train = pad_sequences(x_train, maxlen=200)
x_test = pad_sequences(x_test, maxlen=200)

x_train = torch.tensor(x_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.float32)

x_test = torch.tensor(x_test, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.float32)


# Model
class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(10000, 32)

        self.lstm = nn.LSTM(
            32,
            64,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.embedding(x)

        _, (h, _) = self.lstm(x)

        x = self.fc(h[-1])

        return x.squeeze()


model = Model().to(device)

loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training
history = []

for epoch in range(5):

    model.train()

    correct = 0
    total = 0

    for i in range(0, len(x_train), 64):

        x = x_train[i:i+64].to(device)
        y = y_train[i:i+64].to(device)

        optimizer.zero_grad()

        output = model(x)

        loss = loss_fn(output, y)

        loss.backward()

        optimizer.step()

        correct += ((output >= 0.5) == y).sum().item()
        total += len(y)

    accuracy = correct / total
    history.append(accuracy)

    print(
        f"Epoch {epoch + 1}: "
        f"Accuracy = {accuracy:.4f}"
    )


# Testing
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for i in range(0, len(x_test), 64):

        x = x_test[i:i+64].to(device)
        y = y_test[i:i+64].to(device)

        output = model(x)

        correct += ((output >= 0.5) == y).sum().item()
        total += len(y)

accuracy = correct / total

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")


# Predict review
word_index = imdb.get_word_index()


def predict_review(text):

    encoded = [1]

    for word in text.lower().split():

        word_id = word_index.get(word)

        if word_id is None:
            encoded.append(2)

        else:

            word_id += 3

            if word_id >= 10000:
                word_id = 2

            encoded.append(word_id)

    encoded = pad_sequences(
        [encoded],
        maxlen=200
    )

    encoded = torch.tensor(
        encoded,
        dtype=torch.long
    ).to(device)

    model.eval()

    with torch.no_grad():

        prediction = model(encoded).item()

    print(f"\nReview: {text}")
    print(f"Prediction value: {prediction:.4f}")

    if prediction >= 0.5:
        print("Predicted Output: Positive Review")
    else:
        print("Predicted Output: Negative Review")


# Sample predictions
predict_review(
    "the movie was fantastic and amazing"
)

predict_review(
    "the movie was terrible and boring"
)


# User input
user_text = input(
    "\nEnter your own movie review: "
)

predict_review(user_text)


# Accuracy graph
plt.plot(
    history,
    marker="o",
    label="Training Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("IMDB Sentiment Training Progress")
plt.legend()
plt.grid()

plt.show()