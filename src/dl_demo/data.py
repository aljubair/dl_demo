from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloader(cfg):
    batch_size = cfg["data"]["batch_size"]

    if cfg["model"]["name"] == "PretrainedModel":
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                        transforms.Resize((96, 96)),
                                        transforms.ToTensor(),])
    else:
        transform = transforms.Compose([transforms.ToTensor(),])
    
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = datasets.MNIST(root ='data', train=False, download=True, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader