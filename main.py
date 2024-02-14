from model import SimpleCNN
from train import train_model
from data_generator import data_gen
from inference import infer
import torch


if __name__ == '__main__':

    # Check if GPU is available and use it; otherwise, use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("the active device is:", device)
    # Initialize the model
    model = SimpleCNN().to(device)

    # Generate data
    train_data, train_labels = data_gen()
    train_data.to(device)
    train_labels.to(device)

    # Train the model
    train_model(model, train_data, train_labels)

    # Inference on the trained model
    predictions = infer(model, train_data)
    print("Predictions:", predictions)


