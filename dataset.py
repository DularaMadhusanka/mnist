import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_dataloader(batch_size=128):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    return dataloader

if __name__ == "__main__":
    # Quick sanity check test
    loader = get_mnist_dataloader(batch_size=4)
    images, labels = next(iter(loader))
    print(f"Batch image shape: {images.shape}")  # Expected: [4, 1, 28, 28]
    print(f"Batch label shape: {labels.shape}")  # Expected: [4]
    print(f"Pixel value range: min={images.min().item()}, max={images.max().item()}")