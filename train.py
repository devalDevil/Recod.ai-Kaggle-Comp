import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import cv2
from metrics import calculate_metrics 


def create_color_map(masks, height, width):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if len(masks) == 0: return canvas
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (128, 0, 255), (255, 128, 0)]
    for i, mask in enumerate(masks):
        if torch.is_tensor(mask): mask = mask.detach().cpu().numpy().astype(np.uint8)
        else: mask = mask.astype(np.uint8)
        color = colors[i % len(colors)]
        for c in range(3): canvas[:, :, c] = np.where(mask > 0, color[c], canvas[:, :, c])
    return canvas



def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
):
    model.train()
    total_train_loss = 0.0
    batches = 0
    
    target_batch_size = 64
    physical_batch_size = dataloader.batch_size

    accumulation_steps = max(1, target_batch_size // physical_batch_size)
    
    print(f"Gradient Accumulation: Physical BS={physical_batch_size}, Accumulate {accumulation_steps} steps -> Effective BS={physical_batch_size * accumulation_steps}")

    optimizer.zero_grad() # Reset grads before starting

    # Use enumerate to track step count (i)
    for i, (images, targets, _) in enumerate(tqdm(dataloader, desc="Training")):
        
  
        images = images.to(device)
        targets = [t.to(device) for t in targets]


        outputs = model(images)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        # --- Calculate Loss ---
        loss = criterion(logits, targets)

        if torch.isnan(loss):
            print("Warning: Loss is NaN. Skipping.")
            continue
            
        loss = loss / accumulation_steps
        

        loss.backward()


        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        

        total_train_loss += loss.item() * accumulation_steps
        batches += 1
        

    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        
    avg_loss = total_train_loss / batches if batches > 0 else 0.0
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_image_dir: str,
    criterion: nn.Module # <--- Ensure this is passed
):
    model.eval()
    
    total_val_loss = 0.0
    batches = 0
    
    
    all_logits = []

    all_gt_semantic = []
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    with torch.no_grad():
        for batch_idx, (images, targets, infos) in enumerate(tqdm(dataloader, desc="Validation")):
            images = images.to(device)
            
            
            loss_targets = [t.to(device) for t in targets]
            
          
            logits, _ = model(images) # (B, 5, H, W)
            
            
            loss = criterion(logits, loss_targets)
            if not torch.isnan(loss):
                total_val_loss += loss.item()
            batches += 1
            
       
            for i in range(len(images)):
                
                
                
                all_logits.append(logits[i].cpu().numpy()) 
                
                # Process GT
                gt_raw = infos[i]['orig_masks'] # (N, H, W)
                h, w = images[i].shape[-2:]
                
                if gt_raw is not None and gt_raw.ndim == 3 and gt_raw.shape[0] > 0:
                    gt_semantic = np.max(gt_raw, axis=0).astype(np.uint8)
                else:
                    gt_semantic = np.zeros((h, w), dtype=np.uint8)
                
                gt_semantic = (gt_semantic > 0).astype(np.uint8)
                all_gt_semantic.append(gt_semantic)

                
                img_id = str(infos[i]['image_id'])
                case_dir = os.path.join(save_image_dir, img_id)
                os.makedirs(case_dir, exist_ok=True)
                
                
                img_vis = images[i] * std + mean
                img_vis = img_vis.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                img_vis = (img_vis * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(case_dir, "0_input.png"), cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
                
                
                probs = torch.sigmoid(logits[i])
                flat_prob, _ = torch.max(probs, dim=0)
                heatmap = flat_prob.cpu().numpy()
                heatmap_uint8 = (heatmap * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(case_dir, "1_heatmap.png"), heatmap_color)
                
               
                gt_instances = []
                if gt_raw is not None and gt_raw.ndim == 3:
                    for k in range(gt_raw.shape[0]): gt_instances.append(gt_raw[k])
                gt_vis = create_color_map(gt_instances, h, w)
                cv2.imwrite(os.path.join(case_dir, "2_gt_multichannel.png"), cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR))

                
                pred_instances = []
                raw_channels = probs.cpu().numpy() 
                for ch in range(raw_channels.shape[0]):
                    mask = (raw_channels[ch] > 0.5).astype(np.uint8)
                    if mask.sum() > 0: 
                        pred_instances.append(mask)
                pred_vis = create_color_map(pred_instances, h, w)
                cv2.imwrite(os.path.join(case_dir, "3_pred_multichannel.png"), cv2.cvtColor(pred_vis, cv2.COLOR_RGB2BGR))

    
    avg_loss = total_val_loss / batches if batches > 0 else 0.0

    if len(all_gt_semantic) > 0:
        all_masks_np = np.array(all_gt_semantic)
        all_logits_np = np.array(all_logits)
        acc, dice, iou, fpr, fnr = calculate_metrics(all_masks_np, all_logits_np)
    else:
        acc, dice, iou, fpr, fnr = 0.0, 0.0, 0.0, 0.0, 0.0

   
    del all_logits
    del all_gt_semantic
    del all_masks_np
    del all_logits_np
    
    # Delete loop variables
    if 'logits' in locals(): del logits
    if 'images' in locals(): del images
    if 'loss' in locals(): del loss
    
    # Force GPU cleanup
    torch.cuda.empty_cache()


    return avg_loss, acc, dice, iou, fpr, fnr