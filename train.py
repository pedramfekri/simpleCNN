import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def train_model(model, data, labels, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
