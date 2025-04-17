import torch.nn as nn
import torch

from . import base
from . import functional as F
from ..base.modules import Activation
#from skimage.metrics import structural_similarity as ssim


class HybridLoss(base.Loss):
    def __init__(self, weight_jaccard=0.5, weight_ce=0.5, eps=1., ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.weight_jaccard = weight_jaccard
        self.weight_ce = weight_ce
        self.eps = eps
        self.jaccard_loss = JaccardLoss(eps=self.eps, activation=nn.Softmax(dim=1, **params), ignore_channels=ignore_channels)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, y_pr, y_gt):
        # Apply softmax activation for Jaccard Loss
        y_pr_activated = F.softmax(y_pr, dim=1)
        jaccard = self.jaccard_loss(y_pr_activated, y_gt)
        # Use logits directly for Cross-Entropy Loss
        cross_entropy = self.cross_entropy_loss(y_pr, y_gt)
        return self.weight_jaccard * jaccard + self.weight_ce * cross_entropy

class JaccardLoss(base.Loss):

    def __init__(self, eps=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr, y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss(base.Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, ignore_index=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.ignore_index = ignore_index

    def forward(self, y_pr, y_gt):
       
        y_pr = self.activation(y_pr).squeeze(1)

        if self.ignore_index is not None:
            mask = (y_gt != self.ignore_index)
            y_pr = y_pr * mask
            y_gt = y_gt * mask

        return 1 - F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )

class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass


class TextureAwareDiceSSIMTVLoss(base.Loss):
    __name__ = "texture_aware_dice_ssim_tv_loss"  # Explicitly set the name
    def __init__(self, ssim_weight=0.5, tv_weight=0.1, dice_weight=0.4, eps=1., ignore_index=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.ssim_weight = ssim_weight
        self.tv_weight = tv_weight
        self.dice_weight = dice_weight
        self.eps = eps
        self.ignore_index = ignore_index
        self.ignore_channels = ignore_channels
        self.activation = Activation("sigmoid")  # Use sigmoid activation for binary segmentation

    def total_variation_loss(self, pred):
        if pred.dim() == 3:  # [C, H, W]
            tv_h = torch.mean(torch.abs(pred[:, 1:, :] - pred[:, :-1, :]))
            tv_w = torch.mean(torch.abs(pred[:, :, 1:] - pred[:, :, :-1]))
        elif pred.dim() == 4:  # [B, C, H, W]
            tv_h = torch.mean(torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]))
            tv_w = torch.mean(torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]))
        else:
            raise ValueError("Unexpected tensor dimension in total_variation_loss")
        
        return tv_h + tv_w

    def dice_loss(self, y_pr, y_gt):
        intersection = 2.0 * (y_pr * y_gt).sum() + self.eps
        return 1 - (intersection / (y_pr.sum() + y_gt.sum() + self.eps))

    def ssim_loss(self, y_pr, y_gt):
        y_pr_np = y_pr.detach().cpu().numpy()
        y_gt_np = y_gt.detach().cpu().numpy()
        ssim_value = ssim(y_pr_np, y_gt_np, data_range=y_pr_np.max() - y_pr_np.min())
        return 1 - ssim_value

    def forward(self, y_pr, y_gt):
        # Apply activation
        y_pr = self.activation(y_pr).squeeze(1)

        # Mask out ignored index values if specified
        if self.ignore_index is not None:
            mask = (y_gt != self.ignore_index)
            y_pr = y_pr * mask
            y_gt = y_gt * mask

        # Calculate Dice Loss
        dice = self.dice_loss(y_pr, y_gt)
        
        # Calculate SSIM Loss
        ssim_loss = self.ssim_loss(y_pr, y_gt)
        
        # Calculate Total Variation Loss
        tv_loss = self.total_variation_loss(y_pr)

        # Combine losses with specified weights
        total_loss = (
            self.dice_weight * dice + 
            self.ssim_weight * ssim_loss + 
            self.tv_weight * tv_loss
        )
        
        return total_loss

class DynamicWeightedConfidenceDiceLoss(base.Loss):
    __name__ = "dynamic_weighted_confidence_dice_loss"

    def __init__(self, eps=1., beta=1., amplification_factor=2.0, confidence_threshold=0.8, correctness_threshold=0.5, activation=None, ignore_channels=None, ignore_index=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.amplification_factor = amplification_factor  # Factor to amplify confident incorrect predictions
        self.confidence_threshold = confidence_threshold  # Confidence threshold for amplification
        self.correctness_threshold = correctness_threshold  # Correctness threshold for amplification
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.ignore_index = ignore_index

    def forward(self, y_pr, y_gt):
        # Apply activation to get probabilities
        y_pr = self.activation(y_pr).squeeze(1)
        
        # Mask out ignore_index pixels if specified
        if self.ignore_index is not None:
            mask = (y_gt != self.ignore_index).float()
            y_pr = y_pr * mask
            y_gt = y_gt * mask

        # Step 1: Calculate confidence as distance from 0.5
        confidence = torch.abs(y_pr - 0.5) * 2  # High values for high confidence

        # Step 2: Calculate correctness as agreement with ground truth
        correctness = torch.abs(y_pr - y_gt.float())  # High values for incorrect predictions

        # Step 3: Calculate initial weights
        weights = 1 - confidence * (1 - correctness)

        # Step 4: Amplify weights for confident, incorrect predictions
        weights = torch.where(
            (confidence > self.confidence_threshold) & (correctness > self.correctness_threshold),
            weights * self.amplification_factor,
            weights
        )

        # Normalize weights to avoid instability
        weights = weights / (weights.mean() + 1e-8)

        # Calculate the weighted Dice loss
        intersection = (weights * y_pr * y_gt).sum()
        denominator = (weights * (y_pr + y_gt)).sum()
        dice_loss = 1 - (2 * intersection + self.eps) / (denominator + self.eps)
        
        return dice_loss


class DynamicWeightedConfidenceMulticlassDiceLoss(base.Loss):
    __name__ = "dynamic_weighted_confidence_multiclass_dice_loss"

    def __init__(self, eps=1., beta=1., amplification_factor=2.0, confidence_threshold=0.8, 
                 correctness_threshold=0.5, activation="softmax", ignore_channels=None, 
                 ignore_index=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.amplification_factor = amplification_factor
        self.confidence_threshold = confidence_threshold
        self.correctness_threshold = correctness_threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.ignore_index = ignore_index

    def forward(self, y_pr, y_gt):
        """
        Args:
            y_pr: (B, C, H, W) - Model predictions (logits)
            y_gt: (B, H, W) - Ground truth labels (integer values per pixel)
        """
        # Apply activation to obtain probabilities
        y_pr = self.activation(y_pr)  # Shape: (B, C, H, W)

        #print("Unique values in y_gt:", torch.unique(y_gt))  # Debugging
        assert torch.all((y_gt >= 0) & (y_gt < y_pr.shape[1]) | (y_gt == self.ignore_index)), \
            f"y_gt contains invalid class indices! Expected [0-{y_pr.shape[1]-1}], but got {torch.unique(y_gt)}"

        # Mask out ignore_index pixels if specified
        if self.ignore_index is not None:
            valid_mask = (y_gt != self.ignore_index).float()  # Shape: (B, H, W), False for class 5
            valid_mask = valid_mask.unsqueeze(1)  # Reshape to (B, 1, H, W) for broadcasting
            y_pr = y_pr * valid_mask  # Ignore class 5 in predictions

        # Convert y_gt to class probabilities (without invalid indices)
        y_gt_clamped = y_gt.clone()
        y_gt_clamped[y_gt == self.ignore_index] = 0  # Set ignored pixels to class 0 for scatter

        y_gt_probs = torch.zeros_like(y_pr)  # (B, C, H, W)
        y_gt_probs.scatter_(1, y_gt_clamped.unsqueeze(1), 1)  # Convert indices to class-wise probabilities

        # Apply valid_mask to remove influence of ignored pixels
        y_gt_probs = y_gt_probs * valid_mask

        # Step 1: Compute confidence (distance from uniform probability)
        confidence = torch.abs(y_pr - 1 / y_pr.shape[1]) * 2

        # Step 2: Compute correctness using class probabilities
        correctness = torch.abs(y_pr - y_gt_probs)

        # Step 3: Compute initial weights
        weights = 1 - confidence * (1 - correctness)

        # Step 4: Amplify weights for confident, incorrect predictions
        weights = torch.where(
            (confidence > self.confidence_threshold) & (correctness > self.correctness_threshold),
            weights * self.amplification_factor,
            weights
        )

        # Normalize weights to avoid instability
        weights = weights / (weights.mean(dim=[2, 3], keepdim=True) + 1e-8)

        # Step 5: Compute per-class Dice loss
        intersection = (weights * y_pr * y_gt_probs).sum(dim=[2, 3])
        denominator = (weights * (y_pr + y_gt_probs)).sum(dim=[2, 3])
        dice_per_class = 1 - (2 * intersection + self.eps) / (denominator + self.eps)

        # Step 6: Average across classes
        dice_loss = dice_per_class.mean()

        return dice_loss

class MaskedBCEWithLogitsLoss(nn.Module):
    __name__ = "masked_bce_with_logits_loss"

    def __init__(self, ignore_index=2):
        super().__init__()
        self.ignore_index = ignore_index
        self.bce = nn.BCEWithLogitsLoss(reduction='none')  # no reduction yet

    def forward(self, y_pr, y_gt):
        y_pr = y_pr.squeeze(1)
        valid_mask = (y_gt != self.ignore_index)
        loss = self.bce(y_pr, y_gt.float())
        masked_loss = loss[valid_mask]
        return masked_loss.mean() if masked_loss.numel() > 0 else torch.tensor(0.0, device=y_gt.device)

class MaskedFocalLoss(nn.Module):
    __name__ = "masked_focal_loss"

    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, y_pr, y_gt):
        # y_pr: logits [B, 1, H, W], y_gt: [B, H, W]
        y_pr = y_pr.squeeze(1)
        valid_mask = (y_gt != self.ignore_index)
        y_gt = y_gt.float()

        # Apply sigmoid
        y_prob = torch.sigmoid(y_pr)

        # Compute Focal Loss manually
        pt = torch.where(y_gt == 1, y_prob, 1 - y_prob)
        focal_term = (1 - pt) ** self.gamma
        alpha_factor = torch.where(y_gt == 1, self.alpha, 1 - self.alpha)
        loss = -alpha_factor * focal_term * torch.log(pt + 1e-7)

        # Apply ignore mask
        loss = loss[valid_mask]

        return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=y_gt.device)

class MulticlassDiceLoss(base.Loss):
    __name__ = "multiclass_dice_loss"

    def __init__(self, eps=1., beta=1., activation="softmax", ignore_index=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_index = ignore_index

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)  # (B, C, H, W)

        if self.ignore_index is not None:
            mask = (y_gt != self.ignore_index)
            mask = mask.unsqueeze(1)  # (B, 1, H, W)
            y_pr = y_pr * mask
        
        # Convert y_gt to one-hot
        y_gt_onehot = torch.zeros_like(y_pr)
        y_gt_clamped = y_gt.clone()
        y_gt_clamped[y_gt == self.ignore_index] = 0
        y_gt_onehot.scatter_(1, y_gt_clamped.unsqueeze(1), 1)
        if self.ignore_index is not None:
            y_gt_onehot = y_gt_onehot * mask

        intersection = (y_pr * y_gt_onehot).sum(dim=(2, 3))
        denominator = (y_pr + y_gt_onehot).sum(dim=(2, 3))

        dice = (2 * intersection + self.eps) / (denominator + self.eps)
        return 1 - dice.mean()

class MulticlassFocalLoss(nn.Module):
    __name__ = "multiclass_focal_loss"

    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, y_pr, y_gt):
        # y_pr: (B, C, H, W), y_gt: (B, H, W)
        ce_loss = self.ce(y_pr, y_gt)  # shape (B, H, W)
        pt = torch.exp(-ce_loss)       # pt = softmax prob of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # Masked mean
        valid_mask = (y_gt != self.ignore_index)
        return focal_loss[valid_mask].mean() if valid_mask.sum() > 0 else torch.tensor(0.0, device=y_gt.device)
