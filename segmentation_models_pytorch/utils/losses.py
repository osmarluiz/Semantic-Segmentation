import torch.nn as nn
import torch

from . import base
from . import functional as F
from ..base.modules import Activation
from skimage.metrics import structural_similarity as ssim


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
        #y_pr = self.activation(y_pr[0]) #.squeeze(1)
        
        y_pr = self.activation(y_pr).squeeze(1)

        if self.ignore_index is not None:
            mask = (y_gt != self.ignore_index)
            y_pr = y_pr * mask
            y_gt = y_gt * mask
            
        #print(y_pr.shape)
        #print(y_gt.shape)
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
    

class ProgressiveGaussianFocalLoss(base.Loss):
    __name__ = "progressive_gaussian_focal_loss"

    def __init__(self, alpha=0.25, gamma=2.0, initial_sigma=1, max_sigma=5, total_epochs=100, ignore_index=None, activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.initial_sigma = initial_sigma
        self.max_sigma = max_sigma
        self.total_epochs = total_epochs
        self.current_epoch = 0  # Track the current epoch to progressively adjust sigma
        self.ignore_index = ignore_index
        self.activation = Activation(activation)

    def update_sigma(self):
        # Linearly increase sigma from initial_sigma to max_sigma over total_epochs
        sigma = self.initial_sigma + (self.max_sigma - self.initial_sigma) * min(self.current_epoch / self.total_epochs, 1)
        return sigma

    def gaussian_heatmap(self, mask, sigma):
        heatmap = torch.zeros_like(mask, dtype=torch.float32)
        labeled_points = torch.nonzero(mask == 1, as_tuple=False)
        
        for point in labeled_points:
            y, x = point[-2], point[-1]
            # Spread Gaussian around each labeled point
            for dy in range(-sigma, sigma + 1):
                for dx in range(-sigma, sigma + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < heatmap.shape[-2] and 0 <= nx < heatmap.shape[-1]:
                        dist = (dx**2 + dy**2) / (2 * sigma**2)
                        heatmap[..., ny, nx] += torch.exp(torch.tensor(-dist, dtype=torch.float32))
        
        return torch.clamp(heatmap, 0, 1)

    def forward(self, y_pr, y_gt):
        # Update sigma based on the current epoch
        sigma = int(self.update_sigma())
        
        # Apply activation
        y_pr = self.activation(y_pr).squeeze(1)

        # Create Gaussian-enhanced target heatmap with updated sigma
        target_heatmap = self.gaussian_heatmap(y_gt, sigma=sigma)
        
        # Ignore unlabeled pixels if ignore_index is specified
        if self.ignore_index is not None:
            mask = (y_gt != self.ignore_index)
            y_pr = y_pr[mask]
            target_heatmap = target_heatmap[mask]
        
        # Calculate Focal Loss on Gaussian-enhanced targets
        focal_loss = -self.alpha * ((1 - y_pr) ** self.gamma) * target_heatmap * torch.log(y_pr + 1e-7) \
                     - (1 - self.alpha) * (y_pr ** self.gamma) * (1 - target_heatmap) * torch.log(1 - y_pr + 1e-7)
        
        return focal_loss.mean()

    
class SoftGuidanceLoss(base.Loss):
    __name__ = "soft_guidance_loss"
    
    def __init__(self, alpha=0.25, gamma=2.0, initial_sigma=1, max_sigma=5, total_epochs=100, ignore_index=None, activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.initial_sigma = initial_sigma
        self.max_sigma = max_sigma
        self.total_epochs = total_epochs
        self.current_epoch = 0  # Track the current epoch to progressively adjust sigma
        self.ignore_index = ignore_index
        self.activation = Activation(activation)

    def update_sigma(self):
        # Linearly increase sigma from initial_sigma to max_sigma over total_epochs
        sigma = self.initial_sigma + (self.max_sigma - self.initial_sigma) * min(self.current_epoch / self.total_epochs, 1)
        return sigma

    def gaussian_heatmap(self, mask, sigma):
        # Create a Gaussian spread around labeled pixels
        heatmap = torch.zeros_like(mask, dtype=torch.float32)
        labeled_points = torch.nonzero(mask == 1, as_tuple=False)
        
        for point in labeled_points:
            y, x = point[-2], point[-1]
            for dy in range(-sigma, sigma + 1):
                for dx in range(-sigma, sigma + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < heatmap.shape[-2] and 0 <= nx < heatmap.shape[-1]:
                        dist = (dx**2 + dy**2) / (2 * sigma**2)
                        heatmap[..., ny, nx] += torch.exp(torch.tensor(-dist, dtype=torch.float32))
        
        return torch.clamp(heatmap, 0, 1)

    def forward(self, y_pr, y_gt):
        # Update sigma based on the current epoch
        sigma = int(self.update_sigma())
        
        # Apply activation
        y_pr = self.activation(y_pr).squeeze(1)

        # Generate the Gaussian guidance map around labeled points
        guidance_map = self.gaussian_heatmap(y_gt, sigma=sigma)
        
        # Mask out ignored pixels if specified
        if self.ignore_index is not None:
            mask = (y_gt != self.ignore_index)
            y_pr = y_pr[mask]
            guidance_map = guidance_map[mask]
        
        # Compute the soft guidance focal loss
        focal_loss = -self.alpha * ((1 - y_pr) ** self.gamma) * guidance_map * torch.log(y_pr + 1e-7) \
                     - (1 - self.alpha) * (y_pr ** self.gamma) * (1 - guidance_map) * torch.log(1 - y_pr + 1e-7)
        
        return focal_loss.mean()



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