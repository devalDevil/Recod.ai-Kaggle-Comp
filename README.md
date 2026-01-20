```markdown
# Recod.ai - Scientific Image Forgery Detection

**Rank 11 Solution for the Recod.ai/LUC Kaggle Competition**

This repository contains the source code and solution methodology for the **Scientific Image Forgery Detection** competition. The objective was to develop computer vision models capable of detecting and segmenting forged regions (copy-move, splicing, etc.) within scientific biomedical images.

## Repository Overview

This project utilizes a semantic segmentation approach to identify pixel-level anomalies in scientific imagery. The codebase is structured as a flat directory for ease of access to training and inference scripts.

### Key Features

* **Custom Dataset Loading**: tailored for competition data formats.
* **Segmentation Models**: Implementation of U-Net/SegFormer based architectures (`model.py`, `model1.py`).
* **Custom Loss Functions**: Specialized losses (`losses.py`) to handle class imbalance between forged and authentic pixels.
* **Metric Tracking**: Custom evaluation metrics (`metrics.py`).

## Project Structure

```text
.
├── config.py                 # Configuration file for paths, hyperparameters, and model settings
├── Cor.py                    # Core utilities and correction modules
├── create_dataset_csv.py     # Script to generate CSV manifests from raw image directories
├── dataset.py                # PyTorch Dataset class and augmentation pipelines
├── losses.py                 # Custom loss functions (e.g., Dice Loss, BCE, Focal Loss)
├── main.py                   # Main entry point for inference or pipeline execution
├── metrics.py                # Evaluation metrics (IoU, Dice Score, Pixel Accuracy)
├── model.py                  # Primary model architecture definition
├── model1.py                 # Secondary/Experiment model architecture
├── train.py                  # Training loop script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Getting Started

### Prerequisites

* Python 3.10+
* CUDA-compatible GPU

### Installation

Clone the repository:

```bash
git clone https://github.com/devalDevil/Recod.ai-Kaggle-Comp.git
cd Recod.ai-Kaggle-Comp
```

Install Dependencies:

```bash
pip install -r requirements.txt
```

## Usage Pipeline

### 1. Data Preparation

Before training, generate the dataset CSV files that map images to their corresponding ground truth masks.

```bash
python create_dataset_csv.py
```

### 2. Configuration

Open `config.py` and ensure the following paths are set correctly for your environment:

* `DATA_ROOT`: Path to the raw competition data.
* `TRAIN_CSV`: Path to the generated training CSV.
* `VALID_CSV`: Path to the generated validation CSV.
* Hyperparameters: `BATCH_SIZE`, `LR` (Learning Rate), `EPOCHS`.

### 3. Training

Run the training script to fine-tune the model. Checkpoints will be saved based on the settings in `config.py`.

```bash
python train.py
```

### 4. Inference / Main Execution

To run the full pipeline or generate submissions on test data:

```bash
python main.py
```

## Methodology

The solution treats forgery detection as a binary segmentation problem.

* **Input**: Scientific images (often containing cloned regions).
* **Output**: Binary mask indicating forged pixels.
* **Strategy**: The model is trained to minimize a combination loss (defined in `losses.py`) that optimizes for both pixel accuracy and intersection-over-union, crucial for detecting small forged artifacts.

## License

Distributed under the MIT License. See LICENSE for more information.

## Credits

Developed by devalDevil as part of the Recod.ai Kaggle Competition.
```
