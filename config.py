"""
This module defines hyperparameters and command-line arguments for model training.

It uses Python's `argparse.ArgumentParser` to provide a structured interface for 
configuring runtime parameters such as training epochs, device_name, and 
other customizable options. These arguments are parsed from the command line and 
used throughout the training pipeline.

Example:
    To train a model with 32 epochs and specify an available device:
        python main.py --epochs 32 --device_name 2
"""

import argparse

config_args = argparse.ArgumentParser()


config_args.add_argument('--dataset_dir', type = str, default = "./recodai-luc-scientific-image-forgery-detection", help = "The root directory of the dataset")
config_args.add_argument('--image_dir', type = str, default = "train_images", help = "The root directory of the image files")
config_args.add_argument('--mask_dir', type = str, default = "train_masks", help = "The root directory of the mask files")
config_args.add_argument('--csv_file', type = str, default = "dataset.csv", help = "CSV file name")


config_args.add_argument('--seed', type = int, default = 24, help = "seed for reproduciability")
config_args.add_argument('--image_size', type = int, default = 512, help = "Resize images to dimention image_size X image_size")
config_args.add_argument('--num_classes', type = int, default = 2, help = "# of classes")

config_args.add_argument('--epochs', type = int, default = 50, help = "# of epochs")
config_args.add_argument('--num_folds', type = int, default = 1, help = "# of folds")
config_args.add_argument('--batch', type = int, default = 2, help = "batch size")
config_args.add_argument('--patience', type = int, default = 5, help = "# of epochs before early stopping")
config_args.add_argument('--lr', type = float, default = 3e-4, help = "learning rate")
config_args.add_argument('--output_dir', type = str, default = "./outputs", help = "The root directory of the outputs")
config_args.add_argument('--device_name', type = str, default = "1", help = "The available gpu in the cluster, check with nvidia_smi")
config_args.add_argument('--device2_name', type = str, default = "2", help = "The available gpu in the cluster, check with nvidia_smi")
config_args.add_argument('--device3_name', type = str, default = "3", help = "The available gpu in the cluster, check with nvidia_smi")
config_args.add_argument('--weight_path', type = str, default = "/home/s25devalm/CS489/CS489_kaggle_comp/outputs/Model_18/models/best_model_1.pth", help = "The available gpu in the cluster, check with nvidia_smi")
config_args.add_argument('--version', type = str, default = "Model_18", help = "The name of the version run (creates a directory based on the name).")
