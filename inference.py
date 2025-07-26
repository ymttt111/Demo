import torch 
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from func_3d.utils import eval_seg, get_network
from func_3d.utils import MaskMetrics
import cfg
from torch.utils.data import DataLoader
import torch.nn.functional as F
from func_3d.dataset.prostate import PROSTATE, InterDataset, IntraDataset, IntraDataset_revice
import numpy as np
import torch
import nibabel as nib
import argparse
import cv2
import csv
from scipy.ndimage import label, find_objects
from region_correspondence.paired_regions import PairedRegions
from region_correspondence.utils import warp_by_ddf
from torch.utils.data import DataLoader
from mmdet.apis import DetInferencer
from mmengine.registry import init_default_scope
from PIL import Image
from utils import utils
import random
import os
import re
from utils.metric import bbox_metric, mask_metric

device = torch.device('cuda', 0)

mr_image_path = '/home/adminer/code/auto_reg/dataset/prostate1/test/mr_images/'
mr_label_path = '/home/adminer/code/auto_reg/dataset/prostate1/test/mr_labels/'
us_label_path = "/home/adminer/code/auto_reg/dataset/prostate1/test/us_labels/"
us_image_path = "/home/adminer/code/auto_reg/dataset/prostate1/test/us_images/"
mr_dino_checkpoint = "/home/adminer/code/auto_reg/checkpoints/MR-GD.pth"
us_dino_checkpoint = "/home/adminer/code/auto_reg/checkpoints/US-GD.pth"
temp_path = '/home/adminer/code/auto_reg/result/vis_result/output.jpg'
inferencer_mr = DetInferencer(model='grounding_dino_swin-t_finetune_16xb2_1x_coco', weights=mr_dino_checkpoint, device=device, show_progress=False)
inferencer_us = DetInferencer(model='grounding_dino_swin-t_finetune_16xb2_1x_coco', weights=us_dino_checkpoint, device=device, show_progress=False)
result_temp_path = '/home/adminer/code/auto_reg/result/vis_result/seg_result.png'
csv_path = '/home/adminer/code/auto_reg/test.csv'
args = cfg.parse_args()
test_set = IntraDataset_revice(mr_image_path, us_image_path, mr_label_path, us_label_path)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

ckpt1 = torch.load(args.mr_pretrain, map_location=device)
net1 = ckpt1['model']  # Model instance directly
net1 = net1.to(torch.float32)
print("Loaded epoch:", ckpt1['epoch'])
print("Best dice saved:", ckpt1['best_dice'])
weights = torch.load(args.mr_pretrain, map_location=device)
net1.load_state_dict(weights, strict=False)
net1.eval()

ckpt2 = torch.load(args.us_pretrain, map_location=device)
net2 = ckpt2['model']  # Model instance directly
net2 = net2.to(torch.float32)
print("Loaded epoch:", ckpt2['epoch'])
print("Best dice saved:", ckpt2['best_dice'])
weights = torch.load(args.us_pretrain, map_location=device)
net2.load_state_dict(weights, strict=False)
net2.eval()

n_val = len(test_loader)
mix_res = (0,) * 1 * 2
tot = 0
threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
prompt_freq = args.prompt_freq

bbox_err = bbox_metric.BBoxMetrics()
mask_err = mask_metric.MaskMetrics()
mask_multi = mask_metric.multi_metric_tensor
init_default_scope('mmdet')

prompt = args.prompt

def normalize(img):
    img = (img - torch.mean(img)) / (torch.std(img) + 1e-5)
    return img

def sam_with_box(pack, temp_path, slice_dim, sam_type):
    if sam_type == 'mr':
        net = net1
        inference = inferencer_mr
    if sam_type == 'us':
        net = net2
        inference = inferencer_us
    iou_total = 0.0
    giou_total = 0.0
    diou_total = 0.0
    ciou_total = 0.0
    count = 0
    imgs_tensor = pack['image']
    mask_dict = pack['label']
    bbox_dict = pack['bbox']  # bounding boxes
    start_index = pack['start_index']
    seg = pack['seg_data'].squeeze()
    if len(imgs_tensor.size()) == 5:
        imgs_tensor = imgs_tensor.squeeze(0)
    frame_id = list(range(imgs_tensor.size(0)))

    train_state = net.val_init_state(imgs_tensor=imgs_tensor)
    prompt_frame_id = list(range(0, len(frame_id), prompt_freq))
    obj_list = []
    for id in frame_id:
        obj_list += list(mask_dict[id].keys())
    obj_list = list(set(obj_list))
    with torch.no_grad():
        for id in prompt_frame_id:
            for ann_obj_id in obj_list:
                try:
                    slice_img = imgs_tensor[id, :, :, :]
                    slice_img = slice_img.cpu().numpy()
                    slice_img = slice_img.transpose(1, 2, 0)

                    result = inference({'img': slice_img, 'text': 'prostate'})
                    bounding_box = torch.tensor([result['predictions'][0]['bboxes']], device=device)[0]
                    bbox = bbox_dict[id][ann_obj_id]
                    bbox_result = bbox_err(bounding_box, bbox[0])
                    iou_total += bbox_result['iou']
                    giou_total += bbox_result['giou']
                    diou_total += bbox_result['diou']
                    ciou_total += bbox_result['ciou']
                    count += 1
                    _, _, _ = net.train_add_new_bbox(
                        inference_state=train_state,
                        frame_idx=id,
                        obj_id=ann_obj_id,
                        bbox=bounding_box.to(device=device),
                        clear_old_points=False,
                    )
                except KeyError:
                    _, _, _ = net.train_add_new_mask(
                        inference_state=train_state,
                        frame_idx=id,
                        obj_id=ann_obj_id,
                        mask=torch.zeros(imgs_tensor.shape[2:]).to(device=device),
                    )
        video_segments = {}  # Store per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in net.propagate_in_video(train_state, start_frame_idx=0):
            video_segments[out_frame_idx] = {
                out_obj_id: out_mask_logits[i]
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        for id in frame_id:
            for ann_obj_id in obj_list:
                pred = video_segments[id][ann_obj_id]
                pred = pred.unsqueeze(0)
                pred = (pred > 0.5).float()
                masks = F.interpolate(pred, size=seg[0, ...].shape, mode='nearest')

                for region_idx, mask in enumerate(masks):
                    if region_idx not in region_masks:
                        region_masks[region_idx] = np.zeros(seg.shape, dtype=np.uint8)
                    current_slice_mask = utils.get_dim(region_masks[region_idx], start_index + id, slice_dim)
                    combined_mask = np.where(mask[0].cpu().numpy() > 0, 1, current_slice_mask)
                    utils.get_slice(region_masks[region_idx], start_index + id, slice_dim, combined_mask)

    if count > 0:
        iou_avg = iou_total / count
        giou_avg = giou_total / count
        diou_avg = diou_total / count
        ciou_avg = ciou_total / count
    else:
        raise ValueError("No valid slices with segmentation found.")
    
    assert len(region_masks) == 1, "region_masks should contain only one entry"
    region_idx, mask_3d = next(iter(region_masks.items()))
    return torch.tensor(mask_3d, device=device), [iou_avg.item(), giou_avg.item(), diou_avg.item(), ciou_avg.item()]

csv_path = f'/home/adminer/code/auto_reg/test2.csv'
with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['dice_raw', 'tre_raw', 'dice', 'tre', 'hd95', 'asd'])  # CSV header

for fix_pack, mov_pack, us_seg, mr_seg, mr_img, us_img in tqdm(test_loader, desc="Inference", unit="batch", total=len(test_loader)):
    fix_seg = fix_pack['seg_data'].squeeze().to(device)
    mov_seg = mov_pack['seg_data'].squeeze().to(device)
    us_img = us_img.to(device)
    mr_img = mr_img.to(device)

    path = fix_pack['path'][0]
    _, affine, header = utils.load_nii_image(path)

    region_masks = {}
    mask_fix, fix_result = sam_with_box(fix_pack, temp_path, slice_dim=0, sam_type='us')
    region_masks = {}
    mask_mov, mov_result = sam_with_box(mov_pack, temp_path, slice_dim=0, sam_type='mr')

    bbox_err_avg = [(a + b) / 2 for a, b in zip(fix_result, mov_result)]
    masks_mov = torch.stack([mask_mov], dim=0).to(torch.bool)
    masks_fix = torch.stack([mask_fix], dim=0).to(torch.bool)
    mov_gt = torch.stack([mr_seg[i] for i in range(mr_seg.shape[0])], dim=0).to(torch.bool).squeeze()

    masks_mov, masks_fix, mov_gt, mr_img, us_img, us_seg, mr_seg, fix_seg, mov_seg, mask_fix, mask_mov = (
        utils.center_pad_last3d_to_shape(i) for i in [
            masks_mov, masks_fix, mov_gt, mr_img, us_img, us_seg,
            mr_seg, fix_seg, mov_seg, mask_fix, mask_mov])

    paired_rois = PairedRegions(masks_mov=masks_mov, masks_fix=masks_fix, images_mov=mr_img, images_fix=us_img, device=device)

    ddf = paired_rois.get_dense_correspondence(transform_type='ddf', max_iter=5000, lr=0.01,
                                               w_roi=1, w_ddf2=5, w_ddfb=0, w_img=0, verbose=True)

    print("ddf.shape:", ddf.shape)

    roi_warped = (warp_by_ddf(mov_gt.to(dtype=torch.float32, device=device), ddf) * 255).bool().int()
    img_warped = warp_by_ddf(mr_img.to(dtype=torch.float32, device=device), ddf, mode='bilinear')

    # if i == 0:
    #     print(path)
    #     i += 1
    #     nib.save(nib.Nifti1Image(img_warped.squeeze().cpu().detach().numpy(), affine, header),
    #              '/home/adminer/code/auto_reg/warped_img.nii.gz')

    roi_warped = roi_warped.squeeze()
    print(roi_warped.shape)
    print("img_warped.shape:", img_warped.shape)

    result_raw = mask_err(fix_seg.cpu().numpy(), mask_fix.cpu().numpy())
    us_seg = us_seg.to(device)
    result = mask_multi(us_seg.squeeze(), roi_warped)

    print('dice_raw:{:.2f},tre_raw:{:.2f},dice:{:.2f},tre:{:.2f},hd95:{:.2f},asd:{:.2f}'.format(
        result_raw['dice'], result_raw['tre'], result['dice'], result['tre'], result['hd95'], result['asd']))

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([result_raw['dice'], result_raw['tre'],
                         result['dice'], result['tre'], result['hd95'], result['asd']])
