# Experiment 10 - English to Hindi Translation

import torch
import torch.nn as nn
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Dataset
data = pd.read_csv(r"D:\DL\code\Dataset.csv").dropna().head(3000)

def vocab(sentences):
    v = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for w in set(" ".join(sentences).lower().split()):
        if len(v) < 5000:
            v[w] = len(v)
    return v

ev = vocab(data["English"])
hv = vocab(data["Hindi"])

def encode(s, v):
    return [v.get(w, 3) for w in s.lower().split()]

X, Y = [], []

for e, h in zip(data["English"], data["Hindi"]):
    a = encode(e, ev)[:15]
    b = [1] + encode(h, hv)[:14] + [2]
    X.append(a + [0] * (15-len(a)))
    Y.append(b + [0] * (16-len(b)))

X, Y = torch.tensor(X), torch.tensor(Y)


# Encoder-Decoder
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Embedding(len(ev), 64)
        self.e2 = nn.Embedding(len(hv), 64)
        self.enc = nn.GRU(64, 128, batch_first=True)
        self.dec = nn.GRU(64, 128, batch_first=True)
        self.fc = nn.Linear(128, len(hv))

    def forward(self, x, y):
        _, h = self.enc(self.e1(x))
        return self.fc(self.dec(self.e2(y[:, :-1]), h)[0])


model = Model().to(device)
opt = torch.optim.Adam(model.parameters(), lr=.001)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

# Training
for epoch in range(10):
    for i in range(0, len(X), 64):
        x, y = X[i:i+64].to(device), Y[i:i+64].to(device)

        opt.zero_grad()
        out = model(x, y)
        loss = loss_fn(out.reshape(-1, out.size(-1)), y[:, 1:].reshape(-1))
        loss.backward()
        opt.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")


# Translation
def translate(text):
    model.eval()
    a=encode(text,ev)[:15]
    x=torch.tensor([a+[0]*(15-len(a))]).to(device)

    with torch.no_grad():
        _,h=model.enc(model.e1(x))
        word=torch.tensor([[1]]).to(device)
        result=[]

        for _ in range(15):
            out,h=model.dec(model.e2(word),h)
            word=model.fc(out[:,-1]).argmax(1).unsqueeze(1)

            if word.item()==2:
                break

            result.append(next((w for w,i in hv.items() if i==word.item()),"<unk>"))

    print("English:",text)
    print("Hindi:"," ".join(result))


text = input("\nEnter an English sentence: ")
translate(text)