import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def infer(model, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model(data.to(device))
        _, predicted = torch.max(outputs, 1)
        return predicted
