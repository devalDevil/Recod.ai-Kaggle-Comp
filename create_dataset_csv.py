"""
Script to create a CSV file listing all images in the dataset with their labels.
Run this once before training to generate dataset.csv
"""
import os
import pandas as pd

def create_dataset_csv(dataset_dir, output_csv="dataset.csv"):
    """
    Create a CSV file with image_id and label columns.
    
    Args:
        dataset_dir: Root directory containing train_images folder
        output_csv: Name of output CSV file
    """
    data = []
    
    # Path to train_images
    train_images_dir = os.path.join(dataset_dir, "train_images")
    
    # Process authentic images
    authentic_dir = os.path.join(train_images_dir, "authentic")
    if os.path.exists(authentic_dir):
        for filename in os.listdir(authentic_dir):
            if filename.endswith('.png'):
                image_id = filename.replace('.png', '')
                data.append({'image_id': image_id, 'label': 'authentic'})
    
    # Process forged images
    forged_dir = os.path.join(train_images_dir, "forged")
    if os.path.exists(forged_dir):
        for filename in os.listdir(forged_dir):
            if filename.endswith('.png'):
                image_id = filename.replace('.png', '')
                data.append({'image_id': image_id, 'label': 'forged'})
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    output_path = os.path.join(dataset_dir, output_csv)
    df.to_csv(output_path, index=False)
    
    print(f"Created {output_path}")
    print(f"Total images: {len(df)}")
    print(f"Authentic: {len(df[df['label'] == 'authentic'])}")
    print(f"Forged: {len(df[df['label'] == 'forged'])}")
    
    return df

if __name__ == "__main__":
    dataset_dir = "./recodai-luc-scientific-image-forgery-detection/"
    create_dataset_csv(dataset_dir)