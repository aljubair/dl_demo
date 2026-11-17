import argparse
from dl_demo.utils import setup_logger, get_config  
from dl_demo.data import get_dataloader
import torch
import torch.nn as nn
from dl_demo.model import tiny_cnn
from dl_demo.pretrained import PretrainedModel
import os

def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--config', type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = get_config(args.config)
    logger = setup_logger("experiments/run_001/train.log")

    logger.info(f"Starting training with config: {cfg}")

    train_loader, val_loader = get_dataloader(cfg)
    logger.info(f"Total Batches: {len(train_loader)}")

    model_name = cfg["model"]["name"]
    logger.info(f"Building model: {model_name}")

    if model_name == "tiny_cnn":
        model = tiny_cnn()
    elif model_name == "PretrainedModel":
        model = PretrainedModel()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    logger.info("Starting training loop...")

    epochs = cfg["training"]["epochs"]

    for epoch in range(epochs):
        model.train()
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 500 == 0:
                logger.info(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")


        val_accuracy = evaluate(model, val_loader)
        logger.info(f" Epoch {epoch+1} Validation Accuracy: {val_accuracy:.4f}")

    torch.save(model.state_dict(), "experiments/model.pth")
    logger.info("Model saved to experiments/model.pth")


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

if __name__ == "__main__":
    main()