Recod.ai - Scientific Image Forgery Detection

📌 Overview

This repository contains the solution and codebase for the Recod.ai/LUC - Scientific Image Forgery Detection Kaggle competition.

The goal of this project is to build computer vision models capable of detecting and segmenting copy-move forgeries in scientific biomedical images.

📂 Project Structure

.
├── data/
│   ├── raw/                    # Original competition data (not tracked in git)
│   └── processed/              # Preprocessed tiles/masks
├── notebooks/                  # Jupyter notebooks for EDA and prototyping
├── src/                        # Source code
│   ├── models/                 # Model architectures (e.g., U-Net, SegFormer)
│   ├── data/                   # Data loaders and augmentation pipelines
│   └── utils/                  # Helper functions and metrics
├── submissions/                # Generated CSVs for Kaggle submission
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation


🚀 Getting Started

Prerequisites

Python 3.10+

CUDA-enabled GPU (recommended)

Installation

Clone the repository:

git clone [https://github.com/devalDevil/Recod.ai-Kaggle-Comp.git](https://github.com/devalDevil/Recod.ai-Kaggle-Comp.git)
cd Recod.ai-Kaggle-Comp


Install dependencies:

pip install -r requirements.txt


🛠️ Usage

1. Data Setup

Download the competition data from Kaggle and place it in the data/raw/ directory.

2. Training

Run the training script to fine-tune the model:

# Example command
python src/train.py --config configs/default.yaml


3. Inference

Generate predictions on the test set:

python src/inference.py --model_path checkpoints/best_model.pth


📊 Methodology

(Update this section with your specific approach)

Preprocessing: Image resizing/tiling and normalization.

Augmentation: Geometric transformations (flip, rotate) and forgery-specific artifacts.

Model: Semantic segmentation architecture (e.g., U-Net, DeepLabV3+).

Loss: Combo Loss (Dice + BCE).

🤝 Contributing

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📜 License

Distributed under the MIT License. See LICENSE for more information.
