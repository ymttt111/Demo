import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import random
import wandb
from PIL import Image
import torch
import torch.nn.functional as F

def center_pad_last3d_to_shape(volume: torch.Tensor, target_shape=(128, 128, 128), pad_value=0):
    """
    Center pads the last three dimensions (D, H, W) of a tensor to the specified target size,
    keeping all other dimensions unchanged.

    Args:
        volume (torch.Tensor): Tensor with arbitrary dimensions, as long as the last three are D, H, W
        target_shape (tuple): Target size (D, H, W)
        pad_value (int/float): Value to use for padding

    Returns:
        torch.Tensor: The padded tensor, with the front dimensions unchanged and the last 3 dims as target_shape
    """
    assert volume.ndim >= 3, "Input tensor must have at least 3 dimensions (D, H, W)"
    current_shape = volume.shape[-3:]
    padding = []

    for dim, tgt in zip(reversed(current_shape), reversed(target_shape)):
        total_pad = max(tgt - dim, 0)
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        padding.extend([pad_before, pad_after])

    padded = F.pad(volume, padding, mode='constant', value=pad_value)
    return padded

def load_nii_image(nii_file_path):
    img = nib.load(nii_file_path)
    data = np.asanyarray(img.dataobj)
    return data, img.affine, img.header

def get_dim(data, index, slice_dim):
    if slice_dim == 0:
        return data[index, :, :]
    elif slice_dim == 1:
        return data[:, index, :]
    elif slice_dim == 2:
        return data[:, :, index]
    else:
        raise ValueError("Invalid slice dimension. Please enter a valid slice dimension.")

def get_slice(data, index, slice_dim, slice=None):
    if slice is not None:
        if slice_dim == 0:
            data[index, :, :] = slice
            return data[index, :, :]
        elif slice_dim == 1:
            data[:, index, :] = slice
            return data[:, index, :]
        elif slice_dim == 2:
            data[:, :, index] = slice
            return data[:, :, index]
        else:
            raise ValueError("Invalid slice dimension. Please enter a valid slice dimension.")
    else:
        get_dim(data, index, slice_dim)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# def save_segmentation_plot(img_rgb, mask, bbox, save_path, figsize=(10, 10)):
#     """
#     Save segmentation results (without display), including original image, mask, and bbox.
#
#     Args:
#     - img_rgb: RGB image (PIL.Image or np.ndarray)
#     - mask: binary mask for highlighted region
#     - bbox: [x1, y1, x2, y2] or multiple bboxes as np.ndarray
#     - save_path: path to save the output image
#     - figsize: output image size
#     """
#     plt.figure(figsize=figsize)
#     plt.imshow(img_rgb, cmap='gray')
#     # show_mask(mask, plt.gca())
#     show_box(bbox, plt.gca())
#     plt.axis('off')
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0.)
#     plt.close()  # Key: release resources without opening a window

def save_segmentation_plot(img_rgb, mask, bbox, save_path, figsize=(10, 10)):
    """
    Save the segmentation result plot (without displaying), crop by bbox, and support SVG format.

    Args:
    - img_rgb: RGB image (PIL.Image or np.ndarray)
    - mask: binary mask for the segmented region
    - bbox: [x1, y1, x2, y2] or multiple bboxes as np.ndarray
    - save_path: path to save the image
    - figsize: figure size
    """
    # Convert PIL image to numpy array if needed
    if isinstance(img_rgb, Image.Image):
        img_rgb = np.array(img_rgb)

    # Ensure bbox is a proper numpy array
    if bbox is not None:
        if isinstance(bbox, list) and len(bbox) == 4:
            bbox = np.array([bbox])
        elif isinstance(bbox, np.ndarray):
            if bbox.ndim == 1:
                bbox = bbox.reshape(1, 4)

    # If bbox exists, crop the image to the bounding region
    if bbox is not None and len(bbox) > 0:
        x1 = int(np.min(bbox[:, 0]))
        y1 = int(np.min(bbox[:, 1]))
        x2 = int(np.max(bbox[:, 2]))
        y2 = int(np.max(bbox[:, 3]))

        # Clip to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_rgb.shape[1], x2)
        y2 = min(img_rgb.shape[0], y2)

        img_cropped = img_rgb[y1:y2, x1:x2]

        # Adjust bbox to cropped coordinates
        bbox_cropped = bbox.copy()
        bbox_cropped[:, 0] -= x1
        bbox_cropped[:, 1] -= y1
        bbox_cropped[:, 2] -= x1
        bbox_cropped[:, 3] -= y1
    else:
        img_cropped = img_rgb
        bbox_cropped = None

    plt.figure(figsize=figsize)
    plt.imshow(img_cropped, cmap='gray')

    # # Show mask (if available)
    # if mask is not None and np.sum(mask) > 0:
    #     if bbox is not None and len(bbox) > 0:
    #         mask_cropped = mask[y1:y2, x1:x2]
    #     else:
    #         mask_cropped = mask
    #     plt.imshow(mask_cropped, alpha=0.5, cmap='jet')

    # # Show cropped bboxes (if available)
    # if bbox_cropped is not None:
    #     for box in bbox_cropped:
    #         x1_crop, y1_crop, x2_crop, y2_crop = box
    #         rect = plt.Rectangle((x1_crop, y1_crop), x2_crop - x1_crop, y2_crop - y1_crop,
    #                              fill=False, edgecolor='red', linewidth=2)
    #         plt.gca().add_patch(rect)

    plt.axis('off')

    # Save based on extension
    if save_path.lower().endswith('.svg'):
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    plt.close()  # Release resources

def random_perturb_bbox(bbox_gt, img, shift=5):
    """
    Add random perturbation to a bounding box.

    Args:
    - bbox_gt: torch.tensor([x_min, y_min, x_max, y_max]) in float
    - img: the corresponding image slice (2D array)
    - shift: maximum pixel perturbation in any direction
    """
    H, W = img.shape
    bbox = bbox_gt.clone().float()

    dx_min = float(random.uniform(-shift, shift))
    dy_min = float(random.uniform(-shift, shift))
    dx_max = float(random.uniform(-shift, shift))
    dy_max = float(random.uniform(-shift, shift))

    bbox[0] += dx_min
    bbox[1] += dy_min
    bbox[2] += dx_max
    bbox[3] += dy_max

    # Ensure bbox is valid: x_min < x_max, y_min < y_max
    bbox[0] = min(bbox[0], bbox[2] - 1e-2)
    bbox[1] = min(bbox[1], bbox[3] - 1e-2)
    bbox[2] = max(bbox[2], bbox[0] + 1e-2)
    bbox[3] = max(bbox[3], bbox[1] + 1e-2)

    # Clip to image bounds
    bbox[0] = max(0.0, min(bbox[0], W - 1e-2))
    bbox[1] = max(0.0, min(bbox[1], H - 1e-2))
    bbox[2] = max(0.0, min(bbox[2], W - 1e-2))
    bbox[3] = max(0.0, min(bbox[3], H - 1e-2))

    return bbox

def log_wandb_image(vol, name, step, slice_idx=None):
    """
    Log a specific slice of a 3D volume to Weights & Biases as an image.

    Args:
    - vol: 3D torch tensor
    - name: logging name
    - step: training step or epoch
    - slice_idx: which slice to show (default is center slice)
    """
    if slice_idx is None:
        slice_idx = vol.shape[2] // 2
    img = vol[slice_idx, ...].cpu().numpy()
    wandb.log({name: wandb.Image(img)}, step=step)
