from data_processing import prepare_data
from train_test import val_model, train_model
from model_architecture import DepthModel
import torch.nn as nn
import torch

if __name__ == "__main__":

    import os
    import subprocess

    # Define the folder name
    folder_name = "nyu-depth-v2"

    # Check if the folder exists
    if not os.path.exists(folder_name):
        print(f"The folder '{folder_name}' does not exist. Downloading dataset...")
        try:
            # Execute the Kaggle command to download the dataset
            subprocess.run(
                ["kaggle", "datasets", "download", "soumikrakshit/nyu-depth-v2"],
                    check=True
                )
            print("Dataset downloaded successfully.")
        except subprocess.CalledProcessError as e:
                print(f"Error occurred while downloading the dataset: {e}")
    else:
            print(f"The folder '{folder_name}' already exists.")

    
    import os
    import zipfile

    # Define the zip file and the target extraction folder
    zip_file = "nyu-depth-v2.zip"
    extract_folder = "nyu-depth-v2"

    # Check if the folder already exists
    if os.path.exists(extract_folder):
        print(f"The folder '{extract_folder}' already exists. Skipping extraction.")
    else:
        # Check if the zip file exists
        if os.path.exists(zip_file):
            print(f"Found zip file: {zip_file}. Extracting contents...")
            try:
                # Open the zip file
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    # Extract all files to the target folder
                    zip_ref.extractall(extract_folder)
                print(f"Extraction completed. Files extracted to: {extract_folder}")
            except zipfile.BadZipFile:
                print("Error: The file is not a valid zip file.")
        else:
            print(f"Zip file '{zip_file}' not found. Please ensure the file is downloaded.")


    
    #Create DataLoaders for the Train and Val set
    print("Create DataLoaders for the Train and Val set")
    train_loader, val_loader = prepare_data()
    
    #Initialize our model
    print("Initiate our model")
    MyModel = DepthModel()
    
    #Define the criterion(Loss) that we will use
    print("Definining loss")
    criterion = nn.MSELoss()
    
    #Define what optomizer we will use for our model
    print("Defining Optimizers")
    optomizer = torch.optim.Adam(MyModel.parameters(), lr=0.001)
    
    #Put our model through training and validation
    epochs = 3
    for epoch in range(epochs):
        print("Starting training")
        train_loss = train_model(MyModel, train_loader, criterion, optomizer)
        print("Starting validation")
        val_loss = val_model(MyModel, val_loader, criterion)