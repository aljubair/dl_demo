# DL Demo: Config-Driven Deep Learning Pipeline

A PyTorch project demonstrating a modular, config-driven training pipeline. 
It supports training a custom CNN from scratch (`TinyCNN`) or fine-tuning a pre-trained lightweight model (`MobileNetV3-Small`).

## 🚀 Features
- **Config-Driven:** Switch models and hyperparameters via YAML files (no code changes).
- **Modular Design:** Clean `src/` layout with separate files for data, models, and training loops.
- **Mixed Precision & Evaluation:** Includes validation loops and model checkpointing.
- **Fast Setup:** Uses `uv` for lightning-fast, reproducible environment management.

## 🛠️ Project Structure
- `configs/`: YAML files for experiment configurations (`base.yaml`, `tiny_cnn.yaml`).
- `src/dl_demo/`: 
  - `data.py`: DataLoaders and transforms.
  - `model.py`: Custom `TinyCNN` architecture.
  - `pretrained.py`: Transfer learning wrapper for MobileNetV3.
  - `train.py`: The main training loop and argument parser.
  - `utils.py`: Logging and config loading utilities.

## ⚙️ How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aljubair/dl_demo.git
   cd dl_demo