from data_processing import prepare_data
from train_test import val_model, train_model
from model_architecture import DepthModel
import torch.nn as nn
import torch

if __name__ == "__main__":
    
    #Create DataLoaders for the Train and Val set
    train_loader, val_loader = prepare_data()
    
    #Initialize our model
    MyModel = DepthModel()
    
    #Define the criterion(Loss) that we will use
    criterion = nn.MSELoss()
    
    #Define what optomizer we will use for our model
    optomizer = torch.optim.Adam(MyModel.parameters(), lr=0.001)
    
    #Put our model through training and validation
    epochs = 3
    for epoch in range(epochs):
        train_loss = train_model(MyModel, train_loader, criterion, optomizer)
        val_loss = val_model(MyModel, val_loader, criterion)