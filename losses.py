import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [B, 1, H, W] (raw outputs from model)
        targets: [B, H, W]   (binary mask)
        """
        
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-7, max=1-1e-7)
        
        
        if probs.dim() > 3:
            probs = probs.squeeze(1)
        
        
        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1).float()
               
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
               
        return 1.0 - dice.mean()




class InstanceDiceLoss(nn.Module):
    """
    Approximates the competition's oF1 metric by penalizing false positive 
    'ghost' blobs more heavily than standard Dice.
    
    It computes Dice in two ways and averages them:
    1. Global Dice: Standard pixel-wise overlap.
    2. Precision-Weighted Dice: Heavily penalizes pixels predicted far  anyfrom GT.
    """
    def __init__(self, smooth=1e-6, penalty_weight=0.5):
        super(InstanceDiceLoss, self).__init__()
        self.smooth = smooth
        self.penalty_weight = penalty_weight

    def forward(self, logits, targets):
        """
        logits: [B, 1, H, W]
        targets: [B, H, W]
        """
        # 1. Sigmoid & Basic Flattening
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-7, max=1-1e-7)
        
        if probs.dim() > 3:
            probs = probs.squeeze(1) # [B, H, W]
            

        targets = targets.float()

 
        p_flat = probs.contiguous().view(probs.size(0), -1)
        t_flat = targets.contiguous().view(targets.size(0), -1)
        
        intersection = (p_flat * t_flat).sum(dim=1)
        union = p_flat.sum(dim=1) + t_flat.sum(dim=1)
        global_dice = (2. * intersection + self.smooth) / (union + self.smooth)
        global_loss = 1.0 - global_dice.mean()

        
        # Calculate standard FP (pixels predicted but not in target)
        fp_pixels = p_flat * (1 - t_flat)
        

        gt_area = t_flat.sum(dim=1) + self.smooth

        fp_penalty = fp_pixels.sum(dim=1) / gt_area
        

        batch_size = probs.size(0)
        penalty_loss = 0.0
        
        for i in range(batch_size):
            if t_flat[i].sum() < 1.0: 

                penalty_loss += p_flat[i].mean()
            else:

                penalty_loss += torch.log1p(fp_penalty[i])

        penalty_loss /= batch_size


        return (1 - self.penalty_weight) * global_loss + self.penalty_weight * penalty_loss


class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-7, max=1-1e-7)
        
        if probs.dim() > 3:
            probs = probs.squeeze(1)
        

        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1).float()
        

        intersection = (probs * targets).sum(dim=1)
        total = (probs + targets).sum(dim=1)
        union = total - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - iou.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        """
        logits: [B, 1, H, W]
        targets: [B, H, W]
        """
        
        if logits.dim() > 3 and targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-7, max=1-1e-7)
        
        
        pt = probs * targets + (1 - probs) * (1 - targets)
        
        
        focal_weight = (1 - pt) ** self.gamma
        
        
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * focal_weight * bce_loss
        else:
            loss = focal_weight * bce_loss
            
        return loss.mean()

class HungarianLoss(nn.Module):
    def __init__(self, base_loss_fn, weight_fp=3, weight_diversity=1.0):
        """
        Args:
            base_loss_fn: Loss function for matching (e.g., Dice/MixedLoss).
            weight_fp (float): Penalty weight for unmatched slots (False Positives).
            weight_diversity (float): Weight for the exclusion loss to prevent duplicate channels.
        """
        super().__init__()
        self.loss_fn = base_loss_fn
        self.weight_fp = weight_fp
        self.weight_diversity = weight_diversity

    def forward(self, preds, target_list):
        """
        preds: (B, N_Slots, H, W) - Raw logits
        target_list: List[(N_GT, H, W)] - Variable length GT masks per image
        """
        batch_size = preds.shape[0]
        n_slots = preds.shape[1]
        

        probs = torch.sigmoid(preds)  # [B, N, H, W]
        probs_flat = probs.view(batch_size, n_slots, -1) # [B, N, Pixels]
        

        overlap_matrix = torch.bmm(probs_flat, probs_flat.transpose(1, 2))
        

        identity = torch.eye(n_slots, device=preds.device).unsqueeze(0).expand(batch_size, -1, -1)
        

        diversity_loss = (overlap_matrix * (1 - identity)).sum() / (probs_flat.shape[-1] * batch_size)
        
       
        matching_loss = 0.0
        
        for b in range(batch_size):
            p_logits = preds[b]     # (N_Slots, H, W)
            t_masks = target_list[b].to(p_logits.device)
            n_gt = t_masks.shape[0]

            if n_gt == 0:
                # All slots are False Positives
                zeros = torch.zeros_like(p_logits)
                
     
                loss_fp = self.loss_fn(p_logits.unsqueeze(1), zeros.unsqueeze(1))
                matching_loss += (loss_fp * self.weight_fp)
                continue

            cost_matrix = torch.zeros((n_slots, n_gt), device=p_logits.device)

            for i in range(n_slots):
                for j in range(n_gt):
                    p = p_logits[i].unsqueeze(0).unsqueeze(0)
                    t = t_masks[j].unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        cost_matrix[i, j] = self.loss_fn(p, t)

            cost_matrix_np = cost_matrix.cpu().numpy()
            row_idx, col_idx = linear_sum_assignment(cost_matrix_np)
            
            # A. Matched Pairs Loss
            matched_p = p_logits[row_idx].unsqueeze(1)
            matched_t = t_masks[col_idx].unsqueeze(1)
            matching_loss += self.loss_fn(matched_p, matched_t)
            
            # B. Unmatched Slots Loss (False Positives)
            all_indices = set(range(n_slots))
            matched_indices = set(row_idx)
            unmatched_indices = list(all_indices - matched_indices)
            
            if len(unmatched_indices) > 0:
                unmatched_p = p_logits[unmatched_indices].unsqueeze(1)
                zeros = torch.zeros_like(unmatched_p)
                fp_loss = self.loss_fn(unmatched_p, zeros)
                matching_loss += (fp_loss * self.weight_fp)

        final_matching_loss = matching_loss / batch_size
        
        # Combine Hungarian Matching Loss with Diversity Penalty
        return final_matching_loss + (self.weight_diversity * diversity_loss)
    
    
class MixedLoss(nn.Module):
    def __init__(self, w=0.5, alpha=0.75, gamma=2.0):
        """
        w (float): Weight for Focal Loss (0.0 to 1.0). 
                   Loss = w * Focal + (1-w) * Dice
        """
        super(MixedLoss, self).__init__()
        self.w = w
        self.dice = InstanceDiceLoss()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        
    def forward(self, logits, targets):
        # Pass raw logits to both. They handle sigmoid/shapes internally.
        focal = self.focal(logits, targets)
        dice = self.dice(logits, targets)
        
        return self.w * focal + (1 - self.w) * dice

class LovaszHingeLoss(nn.Module):
    def __init__(self, per_image=True, ignore_index=None):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        if logits.dim() > 3:
            logits = logits.squeeze(1)
            
        if self.per_image:
            batch_size = logits.size(0)
            loss = 0
            for i in range(batch_size):
                loss += self._lovasz_hinge_flat(logits[i].view(-1), targets[i].view(-1))
            return loss / batch_size
        else:
            return self._lovasz_hinge_flat(logits.view(-1), targets.view(-1))

    def _lovasz_grad(self, gt_sorted):
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1: 
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def _lovasz_hinge_flat(self, logits, labels):
        if self.ignore_index is not None:
            valid = (labels != self.ignore_index)
            logits = logits[valid]
            labels = labels[valid]
            
        if len(labels) == 0:
            return logits.sum() * 0.

        signs = 2. * labels.float() - 1.
        errors = (1. - logits * signs)
        
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        
        gt_sorted = labels[perm]
        grad = self._lovasz_grad(gt_sorted)
        
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss