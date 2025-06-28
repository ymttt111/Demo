import numpy as np
import torch
import scipy.ndimage as ndi
from scipy.spatial.distance import cdist

class MaskMetrics:
    def __init__(self):
        pass

    def target_registration_error(self, mask1, mask2):
        """TRE"""
        centroid1 = np.mean(np.argwhere(mask1), axis=0)
        centroid2 = np.mean(np.argwhere(mask2), axis=0)
        tre = np.linalg.norm(centroid1 - centroid2)
        return tre

    def dice_coefficient(self, mask1, mask2):
        """DSC"""
        intersection = np.logical_and(mask1, mask2).sum()
        denominator = mask1.sum() + mask2.sum()
        if denominator == 0:
            return 1.0  
        return 2. * intersection / denominator

    def __call__(self, mask1, mask2):

        tre = self.target_registration_error(mask1, mask2)
        dice = self.dice_coefficient(mask1, mask2)
        return {
            "tre": tre,
            "dice": dice
        }

def compute_surface_distances(pred, gt, spacing):

    pred_surface = pred ^ ndi.binary_erosion(pred)
    gt_surface = gt ^ ndi.binary_erosion(gt)

    pred_pts = np.argwhere(pred_surface)
    gt_pts = np.argwhere(gt_surface)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return None, None

    pred_pts = pred_pts * spacing
    gt_pts = gt_pts * spacing

    dist_pred_to_gt = cdist(pred_pts, gt_pts)
    dist_gt_to_pred = cdist(gt_pts, pred_pts)

    return dist_pred_to_gt, dist_gt_to_pred  
    
def multi_metric_tensor(seg_pred: torch.Tensor, seg_gt: torch.Tensor, resolution=[0.8,0.8,0.8]):
   
    seg_pred = seg_pred.squeeze()
    seg_gt = seg_gt.squeeze()

    seg_pred = (seg_pred > 0.5).float()
    seg_gt = (seg_gt > 0.5).float()

    # DSC
    intersection = torch.sum(seg_pred[0] * seg_gt[0])
    union = torch.sum(seg_pred[0]) + torch.sum(seg_gt[0])
    dice = (2. * intersection + 1e-5) / (union + 1e-5)

    # TRE
    tre = torch.zeros(1, seg_gt.shape[0], device=seg_gt.device)
    for i in range(0, seg_gt.shape[0]):
        idx1 = torch.nonzero(seg_gt[i], as_tuple=False).float()
        idx2 = torch.nonzero(seg_pred[i], as_tuple=False).float()

        if idx1.shape[0] == 0:
            tre[0][i] = 0
            continue
        centroid1 = idx1.mean(dim=0)
        centroid2 = idx2.mean(dim=0)
        spacing = torch.tensor(resolution, dtype=centroid1.dtype, device=centroid1.device)
        tre[0][i] = torch.norm((centroid1 - centroid2) * spacing)
    tre = tre[tre != 0]
    
    pred_np = seg_pred[0].detach().cpu().numpy().astype(np.uint8)
    gt_np = seg_gt[0].detach().cpu().numpy().astype(np.uint8)

    dists_pred_gt, dists_gt_pred = compute_surface_distances(pred_np, gt_np, resolution)

    if dists_pred_gt is None or dists_gt_pred is None:
        hd95 = torch.tensor(0.0)
        asd = torch.tensor(0.0)
    else:
        hd95 = max(np.percentile(dists_pred_gt.min(axis=1), 95),
                   np.percentile(dists_gt_pred.min(axis=1), 95))
        asd = (dists_pred_gt.min(axis=1).mean() + dists_gt_pred.min(axis=1).mean()) / 2
        hd95 = torch.tensor(hd95, device=seg_gt.device)
        asd = torch.tensor(asd, device=seg_gt.device)
    return {
        "tre": tre.mean().item(),
        "dice": dice.item(),
        "hd95":hd95.item(),
        "asd":asd.item()
    }
