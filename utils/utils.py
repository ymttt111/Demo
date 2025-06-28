import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import random
import wandb
from PIL import Image
def load_nii_image(nii_file_path):
    img = nib.load(nii_file_path)
    data = np.asanyarray(img.dataobj)
    return data,img.affine,img.header
def get_dim(data,index,slice_dim):
    if slice_dim == 0:
        return data[index,:,:]
    elif slice_dim == 1:
        return data[:,index,:]
    elif slice_dim == 2:
        return data[:,:,index]   
    else:
        raise ValueError("输入切片维度不正确，请重新输入切片维度")
def get_slice(data,index,slice_dim,slice = None):
    if slice is not None:
        if slice_dim == 0:
            data[index,:,:] = slice
            return data[index,:,:]
        elif slice_dim == 1:
            data[:,index,:] = slice
            return data[:,index,:]
        elif slice_dim == 2:
            data[:,:,index] = slice
            return data[:,:,index]   
        else:
            raise ValueError("输入切片维度不正确，请重新输入切片维度")
    else:
        get_dim(data,index,slice_dim)
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
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
# def save_segmentation_plot(img_rgb, mask, bbox, save_path, figsize=(10, 10)):
#     """
#     保存分割结果图（不显示），包括原图、mask 和 bbox。

#     参数：
#     - img_rgb: RGB 图像（PIL.Image 或 np.ndarray）
#     - mask: 二值 mask，用于显示的区域
#     - bbox: [x1, y1, x2, y2] 或多个 bbox 的 np.ndarray
#     - save_path: 图像保存路径
#     - figsize: 图像尺寸
#     """
#     plt.figure(figsize=figsize)
#     plt.imshow(img_rgb,cmap='gray')
#     # show_mask(mask, plt.gca())
#     show_box(bbox, plt.gca())
#     plt.axis('off')
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0.)
#     plt.close()  # 关键：释放资源，不打开窗口

def save_segmentation_plot(img_rgb, mask, bbox, save_path, figsize=(10, 10)):
    """
    保存分割结果图（不显示），根据bbox裁剪并支持SVG格式。

    参数：
    - img_rgb: RGB 图像（PIL.Image 或 np.ndarray）
    - mask: 二值 mask，用于显示的区域
    - bbox: [x1, y1, x2, y2] 或多个 bbox 的 np.ndarray
    - save_path: 图像保存路径
    - figsize: 图像尺寸
    """
    # 将PIL图像转换为numpy数组
    if isinstance(img_rgb, Image.Image):
        img_rgb = np.array(img_rgb)
    
    # 确保bbox是numpy数组且形状正确
    if bbox is not None:
        if isinstance(bbox, list) and len(bbox) == 4:
            bbox = np.array([bbox])
        elif isinstance(bbox, np.ndarray):
            if bbox.ndim == 1:
                bbox = bbox.reshape(1, 4)
    
    # 如果有bbox，则根据bbox裁剪图像
    if bbox is not None and len(bbox) > 0:
        # 计算所有bbox的联合区域
        x1 = int(np.min(bbox[:, 0]))
        y1 = int(np.min(bbox[:, 1]))
        x2 = int(np.max(bbox[:, 2]))
        y2 = int(np.max(bbox[:, 3]))
        
        # 确保裁剪区域在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_rgb.shape[1], x2)
        y2 = min(img_rgb.shape[0], y2)
        
        # 裁剪图像
        img_cropped = img_rgb[y1:y2, x1:x2]
        
        # 调整bbox坐标为裁剪后的坐标系
        bbox_cropped = bbox.copy()
        bbox_cropped[:, 0] -= x1
        bbox_cropped[:, 1] -= y1
        bbox_cropped[:, 2] -= x1
        bbox_cropped[:, 3] -= y1
    else:
        img_cropped = img_rgb
        bbox_cropped = None
    
    # 创建图形并显示裁剪后的图像
    plt.figure(figsize=figsize)
    plt.imshow(img_cropped, cmap='gray')
    
    # # 显示mask（如果有）
    # if mask is not None and np.sum(mask) > 0:
    #     # 裁剪mask
    #     if bbox is not None and len(bbox) > 0:
    #         mask_cropped = mask[y1:y2, x1:x2]
    #     else:
    #         mask_cropped = mask
    #     plt.imshow(mask_cropped, alpha=0.5, cmap='jet')
    
    # # 显示裁剪后的bbox（如果有）
    # if bbox_cropped is not None:
    #     for box in bbox_cropped:
    #         x1_crop, y1_crop, x2_crop, y2_crop = box
    #         rect = plt.Rectangle((x1_crop, y1_crop), x2_crop - x1_crop, y2_crop - y1_crop,
    #                              fill=False, edgecolor='red', linewidth=2)
    #         plt.gca().add_patch(rect)
    
    plt.axis('off')
    
    # 根据文件扩展名设置保存参数
    if save_path.lower().endswith('.svg'):
        # SVG保存设置（透明背景）
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0,
                    transparent=True)
    else:
        # 其他格式保存
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    
    plt.close()  # 释放资源
def random_perturb_bbox(bbox_gt, img, shift=5):
    """
    bbox_gt: torch.tensor([x_min, y_min, x_max, y_max]) in float
    img_shape: (H, W) of the slice
    shift: max perturbation in any direction
    """
    H, W = img.shape
    bbox = bbox_gt.clone().float()

    # 为每个边生成 [-shift, shift] 的浮点扰动
    dx_min = float(random.uniform(-shift, shift))
    dy_min = float(random.uniform(-shift, shift))
    dx_max = float(random.uniform(-shift, shift))
    dy_max = float(random.uniform(-shift, shift))

    # 应用扰动
    bbox[0] += dx_min  # x_min
    bbox[1] += dy_min  # y_min
    bbox[2] += dx_max  # x_max
    bbox[3] += dy_max  # y_max

    # 保证顺序正确：x_min < x_max，y_min < y_max
    bbox[0] = min(bbox[0], bbox[2] - 1e-2)
    bbox[1] = min(bbox[1], bbox[3] - 1e-2)
    bbox[2] = max(bbox[2], bbox[0] + 1e-2)
    bbox[3] = max(bbox[3], bbox[1] + 1e-2)

    # 限制在图像边界范围内
    bbox[0] = max(0.0, min(bbox[0], W - 1e-2))
    bbox[1] = max(0.0, min(bbox[1], H - 1e-2))
    bbox[2] = max(0.0, min(bbox[2], W - 1e-2))
    bbox[3] = max(0.0, min(bbox[3], H - 1e-2))

    return bbox
def log_wandb_image(vol, name, step, slice_idx=None):
    if slice_idx is None:
        slice_idx = vol.shape[2] // 2
    img = vol[slice_idx,...].cpu().numpy()
    wandb.log({name: wandb.Image(img)}, step=step)