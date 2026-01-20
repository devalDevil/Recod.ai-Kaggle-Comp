from typing import Sequence, Tuple
import torch
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

threshold = 0.5

def stable_sigmoid(x):
    x = np.array(x, dtype=np.float64)
    res = np.zeros_like(x)
    pos_mask = (x >= 0)
    res[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    neg_mask = ~pos_mask
    z = np.exp(x[neg_mask])
    res[neg_mask] = z / (1 + z)
    return res

def _to_numpy(data):
    if isinstance(data, (list, tuple)):
        if len(data) > 0 and torch.is_tensor(data[0]):
            try:
                data = torch.cat(data, dim=0)
            except:
                data = torch.stack(data)
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    return np.array(data)

def calculate_metrics(y_true, y_pred, threshold=0.5):

    y_true = _to_numpy(y_true) # (B, H, W)
    y_pred = _to_numpy(y_pred) # (B, 5, H, W) or (B, 1, H, W)
    

    y_true = y_true.squeeze()
    if y_true.ndim == 2:
        y_true = y_true[np.newaxis, ...]
  
    
 
    is_multichannel = (y_pred.ndim == 4 and y_pred.shape[1] > 1)
    
    if is_multichannel:
 
        y_pred_prob = stable_sigmoid(y_pred)
        y_pred_bin = (y_pred_prob > threshold).astype(np.uint8)
        
 
        y_pred_global = np.max(y_pred_bin, axis=1)
    else:

        y_pred = y_pred.squeeze()
        if y_pred.ndim == 2: y_pred = y_pred[np.newaxis, ...]
        
        max_val = y_pred.max()
        min_val = y_pred.min()
        if min_val >= 0 and max_val <= 1.0:
            y_pred_bin = (y_pred > threshold).astype(np.uint8)
        else:
            y_pred_prob = stable_sigmoid(y_pred)
            y_pred_bin = (y_pred_prob > threshold).astype(np.uint8)
        
        y_pred_global = y_pred_bin # (B, H, W)

    y_true_bin = y_true.astype(np.uint8)


    tn, fp, fn, tp = confusion_matrix(y_true_bin.flatten(), y_pred_global.flatten())
    
    total = tn + fp + fn + tp
    acc = (tn + tp) / total if total != 0 else 0.0
    
    intersection = np.sum(np.logical_and(y_true_bin, y_pred_global))
    union = np.sum(np.logical_or(y_true_bin, y_pred_global))
    iou = intersection / union if union != 0 else 0.0
    
    fpr = fp / float(tn + fp) if (tn + fp) != 0 else 0.0
    fnr = fn / float(fn + tp) if (fn + tp) != 0 else 0.0

 
    batch_of1s = []
    batch_size = y_true_bin.shape[0]

    for i in range(batch_size):
        
        gt_img = np.ascontiguousarray(y_true_bin[i])
        num_gt, labels_gt = cv2.connectedComponents(gt_img, connectivity=8)
        gt_instances = [(labels_gt == k).astype(np.uint8) for k in range(1, num_gt)]

 
        pred_instances = []
        
        if is_multichannel:
  
   
            num_channels = y_pred_bin.shape[1]
            for c in range(num_channels):
                mask_c = y_pred_bin[i, c]
                if np.sum(mask_c) > 0: # Only add non-empty channels
                    pred_instances.append(mask_c)
        else:
            
            pred_img = np.ascontiguousarray(y_pred_bin[i])
            num_pred, labels_pred = cv2.connectedComponents(pred_img, connectivity=8)
            pred_instances = [(labels_pred == k).astype(np.uint8) for k in range(1, num_pred)]

        # Score
        if len(gt_instances) == 0 and len(pred_instances) == 0:
            score = 1.0
        elif len(gt_instances) == 0:
            score = 0.0
        elif len(pred_instances) == 0:
            score = 0.0
        else:
            score = oF1_score(pred_instances, gt_instances)
        
        batch_of1s.append(score)

    mean_of1 = np.mean(batch_of1s)

    return acc, mean_of1, iou, fpr, fnr



def calculate_f1_score(pred_mask, gt_mask):
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    tp = np.sum((pred_flat == 1) & (gt_flat == 1))
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))
    fn = np.sum((pred_flat == 0) & (gt_flat == 1))
    if (2 * tp + fp + fn) > 0:
        return (2 * tp) / (2 * tp + fp + fn)
    else:
        return 0.0

def calculate_f1_matrix(pred_masks, gt_masks):
    num_pred = len(pred_masks)
    num_gt = len(gt_masks)
    f1_matrix = np.zeros((num_pred, num_gt))
    for i in range(num_pred):
        for j in range(num_gt):
            f1_matrix[i, j] = calculate_f1_score(pred_masks[i], gt_masks[j])
    if num_pred < num_gt:
        padding = np.zeros((num_gt - num_pred, num_gt))
        f1_matrix = np.vstack((f1_matrix, padding))
    return f1_matrix

def oF1_score(pred_masks, gt_masks):
    f1_matrix = calculate_f1_matrix(pred_masks, gt_masks)
    row_ind, col_ind = linear_sum_assignment(-f1_matrix)
    matched_scores = f1_matrix[row_ind, col_ind]
    
    if len(pred_masks) == 0: return 0.0
    
    # Penalize for over/under-segmentation
    excess_penalty = len(gt_masks) / max(len(pred_masks), len(gt_masks))
    return np.mean(f1_matrix[row_ind, col_ind]) * excess_penalty

def confusion_matrix(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tn = np.sum(~y_true & ~y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    tp = np.sum(y_true & y_pred)
    return tn, fp, fn, tp