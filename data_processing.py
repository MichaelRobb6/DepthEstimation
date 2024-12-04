import pandas as pd
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class NYUDataset(Dataset):
    def __init__(self, data_frame, input_transform=None, depth_transform=None):
        self.data = data_frame
        self.base_path = Path('../DepthEstimation/nyu-depth-v2/nyu_data')
        self.input_transform = input_transform
        self.depth_transform = depth_transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get input and depth paths from the DataFrame
        input_path = self.base_path / self.data.iloc[idx, 0]
        depth_path = self.base_path / self.data.iloc[idx, 1]

        # Open images using `with` to ensure they are properly closed
        with Image.open(input_path).convert("RGB") as input_image:
            if self.input_transform:
                input_image = self.input_transform(input_image)

        with Image.open(depth_path).convert("I") as depth_image:
            if self.depth_transform:
                depth_image = self.depth_transform(depth_image)

        return input_image, depth_image


def prepare_data():
    # Define transformations to apply to the image (x) (Change these to whatever)
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
    ])
    
    # Define transformations to apply to the Depth Map (y) (Change these to whatever)
    depth_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x / 65535.0)  # Normalize to [0, 1] for 16-bit images
        ])
    
    df = pd.read_csv("./nyu-depth-v2/nyu_data/data/nyu2_train.csv", header=None)
    train, val = train_test_split(df, test_size=0.2)
    
    train_dataset = NYUDataset(
        data_frame=train,
        input_transform=image_transform,
        depth_transform=depth_transform
    )
    
    val_dataset = NYUDataset(
        data_frame=val,
        input_transform=image_transform,
        depth_transform=depth_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    
    return train_loader, val_loader
