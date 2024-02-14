import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def data_gen():
    # Generate random tensors for training data
    num_samples = 10
    train_data = torch.randn(num_samples, 3, 64, 64)  # 10 images of size 3x64x64
    train_labels = torch.randint(0, 3, (num_samples,))  # Random labels for 3 classes
    return train_data, train_labels
