import torch

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training function
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    return loss

# Validating function
def val_model(model, val_loader, criterion):
    model.eval()
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

    return loss

# Testing Function
def test_model(model, test_loader, criterion):
    model.eval()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

    return loss