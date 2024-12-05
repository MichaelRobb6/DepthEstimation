from data_processing import prepare_data
from train_test import val_model, train_model
from model_architecture import DepthModel
from dataset_download import *
import torch.nn as nn
import torch

if __name__ == "__main__":

    # Download Dataset if needed (make sure to install pip install kaggle)
    # download()

    # Create DataLoaders for the Train and Val set
    print("Create DataLoaders for the Train and Val set")
    train_loader, val_loader = prepare_data()

    # Initialize our model
    print("Initiate our model")
    MyModel = DepthModel()

    # Define the criterion(Loss) that we will use
    print("Definining loss")
    criterion = nn.MSELoss()

    # Define what optomizer we will use for our model
    print("Defining Optimizers")
    optomizer = torch.optim.Adam(MyModel.parameters(), lr=0.001)

    # Put our model through training and validation
    epochs = 3
    for epoch in range(epochs):
        print("Starting training")
        train_loss = train_model(MyModel, train_loader, criterion, optomizer)
        print("Starting validation")
        val_loss = val_model(MyModel, val_loader, criterion)
