import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pickle import FALSE
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, GroupShuffleSplit
from typing import Any
import tqdm
import gc

# --- IMPORTS ---
from config import config_args
from dataset import CSVDataset
from train import train_one_epoch, validate
from model1 import *
from model import *
from losses import *


def collate_fn(batch):
    images = []
    targets = []
    infos = []
    
    for img, tgt, info in batch:
        images.append(img)
        targets.append(tgt) 
        infos.append(info)
        
    images = torch.stack(images, dim=0)
    
    return images, targets, infos

def run_training(args: Any) -> None:

    DEVICE1 = torch.device(f"cuda:{args.device_name}" if torch.cuda.is_available() else "cpu")
    DEVICE2 = torch.device(f"cuda:{args.device2_name}" if torch.cuda.is_available() else "cpu")
    print(f"Main Config: {DEVICE1} | Offload Config: {DEVICE2}\n")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    csv_file_path = os.path.join(args.dataset_dir, args.csv_file)
    df = pd.read_csv(csv_file_path)
    groups = df['image_id'].values
    
    os.makedirs(os.path.join(args.output_dir, args.version, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.version, "models"), exist_ok=True) 
    
    fold_results = []

    if args.num_folds == 1:
        print("\n----- Performing 80-20 Split -----")
        gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=args.seed)
        train_idx, test_idx = next(gss.split(df, groups=groups))
        
        train_ds = CSVDataset(args, df.iloc[train_idx], augment=True)
        test_ds = CSVDataset(args, df.iloc[test_idx])

        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=1, collate_fn=collate_fn,drop_last=True)
        test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=1, collate_fn=collate_fn)
        
        model = UnetMitB51(
            device1=f"cuda:{args.device_name}", 
            device2=f"cuda:{args.device2_name}",
            device3=f"cuda:{args.device3_name}",
            model_path="/home/s25devalm/CS489/CS489_kaggle_comp/mit_b5_weights"
        )
        

        if hasattr(args, 'weight_path') and args.weight_path:
            if os.path.exists(args.weight_path):
                print(f"Load weights from: {args.weight_path}")
                checkpoint = torch.load(args.weight_path, map_location=DEVICE1)
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            else:
                print(f"Warning: Weight path {args.weight_path} does not exist. Starting from scratch.")


        criterion = HungarianLoss(MixedLoss())
            
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.005)
        
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, verbose=True
        )
        
        best_dice = -1
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Train
            train_loss = train_one_epoch(model, train_dl, criterion=criterion, optimizer=optimizer, device=DEVICE1)
            print(f"Train Loss: {train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.8f}")
            
           
            path = os.path.join(args.output_dir, args.version, "test_images")
            val_loss, acc, dice, iou, fp, fn = validate(model, test_dl, DEVICE1, path, criterion)
            
            print(f"Val Loss: {val_loss:.4f} | Val Dice: {dice:.4f} | Val IoU: {iou:.4f}")
            
            sched.step(iou)
        
            fold_results.append({
                "fold": 1,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss, 
                "test_dice": dice,
                "test_IOU": iou,
                "test_FP": fp,
                "test_FN": fn
            })
            
            pd.DataFrame(fold_results).to_csv(os.path.join(args.output_dir, args.version, "results.csv"), index=False)
            
            
            torch.save({'model_state_dict': model.state_dict()}, 
                        os.path.join(args.output_dir, args.version, "models", f"best_model_{epoch+1}.pth"))
            
            # Save best dice model
            if dice > best_dice:
                best_dice = dice
                torch.save({'model_state_dict': model.state_dict()}, 
                            os.path.join(args.output_dir, args.version, "models", f"best_dice_model.pth"))

        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()

    else:
        # K-Fold logic
        kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
        
        for fold, (train_i, test_i) in enumerate(kf.split(df)):
            print(f"\n----- Fold {fold+1}/{args.num_folds} -----")
            
            train_ds = CSVDataset(args, df.iloc[train_i], augment=True)
            test_ds = CSVDataset(args, df.iloc[test_i])
            train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, collate_fn=collate_fn)
            test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=collate_fn)
            
            model = UnetMitB51(
                device1=f"cuda:{args.device_name}", 
                device2=f"cuda:{args.device2_name}",
                device3=f"cuda:{args.device3_name}",
                model_path="/home/s25devalm/CS489/CS489_kaggle_comp/mit_b5_weights"
            )

            # Weight Loading for K-Fold
            if hasattr(args, 'weight_path') and args.weight_path:
                if os.path.exists(args.weight_path):
                    print(f"Load weights from: {args.weight_path}")
                    checkpoint = torch.load(args.weight_path, map_location=DEVICE1)
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                    model.load_state_dict(state_dict, strict=False)

            base_loss = MixedLoss(w=0.4, alpha=0.25) 
            criterion = HungarianLoss(base_loss_fn=base_loss, weight_fp=0.1, weight_diversity=0.2)
            
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
            
            best_fold_dice = -1

            for epoch in range(args.epochs):
                train_loss = train_one_epoch(model, train_dl, criterion=criterion, optimizer=optimizer, device=DEVICE1)
                

                path = os.path.join(args.output_dir, args.version, "test_images")
                val_loss, acc, dice, iou, fp, fn = validate(model, test_dl, DEVICE1, path, criterion)
                
                print(f"Ep {epoch+1} | Trn Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Dice: {dice:.4f}")
                sched.step(iou)
                
                fold_results.append({
                    "fold": fold + 1, 
                    "epoch": epoch + 1, 
                    "train_loss": train_loss, 
                    "val_loss": val_loss,
                    "test_dice": dice
                })
                pd.DataFrame(fold_results).to_csv(os.path.join(args.output_dir, args.version, "kfold_results.csv"), index=False)

                if dice > best_fold_dice:
                    best_fold_dice = dice
                    torch.save({'model_state_dict': model.state_dict()}, 
                               os.path.join(args.output_dir, args.version, "models", f"fold_{fold+1}_best.pth"))

            del model, optimizer
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    args = config_args.parse_args()
    run_training(args)