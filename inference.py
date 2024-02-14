import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def infer(model, data):
    with torch.no_grad():
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        return predicted
