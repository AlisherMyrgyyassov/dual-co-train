import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """Focal Loss for binary segmentation to handle class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        # Apply sigmoid to get probabilities
        pred_prob = torch.sigmoid(pred)
        
        # Calculate focal loss
        # For positive class (target = 1)
        pos_loss = -self.alpha * (1 - pred_prob) ** self.gamma * target * torch.log(pred_prob + 1e-8)
        # For negative class (target = 0)
        neg_loss = -(1 - self.alpha) * pred_prob ** self.gamma * (1 - target) * torch.log(1 - pred_prob + 1e-8)
        
        loss = pos_loss + neg_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation."""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        # Return Dice loss (1 - Dice coefficient)
        return 1 - dice


class TverskyLoss(nn.Module):
    """Tversky Loss - generalizes Dice loss with controllable FP/FN trade-off."""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.smooth = smooth
    
    def forward(self, pred, target):
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True Positives, False Positives, False Negatives
        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky


class CombinedLoss(nn.Module):  
    """Combined loss with multiple options."""
    def __init__(self, loss_type='focal_dice', focal_weight=0.3, dice_weight=0.7, 
                 tversky_weight=0.0, focal_alpha=0.25, focal_gamma=2.0,
                 tversky_alpha=0.3, tversky_beta=0.7, smooth=1.0):
        super(CombinedLoss, self).__init__()
        self.loss_type = loss_type
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss(smooth=smooth)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss()
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
    
    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        
        if self.loss_type == 'focal_dice':
            # Recommended: Focal Loss + Dice Loss
            # Focal handles hard examples, Dice handles overlap
            focal_loss = self.focal(pred, target)
            dice_loss = self.dice(pred_sigmoid, target)
            return self.focal_weight * focal_loss + self.dice_weight * dice_loss
        
        elif self.loss_type == 'focal_tversky':
            # Alternative: Focal Loss + Tversky Loss
            # Tversky's beta > alpha penalizes false negatives more (encourages larger predictions)
            focal_loss = self.focal(pred, target)
            tversky_loss = self.tversky(pred_sigmoid, target)
            return self.focal_weight * focal_loss + self.tversky_weight * tversky_loss
        
        elif self.loss_type == 'bce_tversky':
            # Another option: BCE + Tversky
            bce_loss = self.bce(pred, target)
            tversky_loss = self.tversky(pred_sigmoid, target)
            return self.focal_weight * bce_loss + self.tversky_weight * tversky_loss
        
        else:  # 'bce_dice' - original
            bce_loss = self.bce(pred, target)
            dice_loss = self.dice(pred_sigmoid, target)
            return self.focal_weight * bce_loss + self.dice_weight * dice_loss


def calculate_dice_score(pred, target):
    """Calculate Dice Score with MASKS"""
    smooth = 1.0
    pred_binary = (torch.sigmoid(pred) > 0.5).float()
    target_binary = target.float()
    intersection = (pred_binary * target_binary).sum()
    dice = (2. * intersection + smooth) / (pred_binary.sum() + target_binary.sum() + smooth)
    return dice.item()


def calculate_iou(pred, target):
    """Calculate Intersection over Union (IoU)."""
    pred_binary = (torch.sigmoid(pred) > 0.5).float()
    target_binary = target.float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    if union == 0:
        return 1.0  # Perfect match if both are empty
    
    iou = intersection / union
    return iou.item()

def calculate_mae(pred, target):
    """Calculate Mean Absolute Error between predicted probabilities and target."""
    pred_probs = torch.sigmoid(pred)
    mae = torch.mean(torch.abs(pred_probs - target))
    return mae.item()