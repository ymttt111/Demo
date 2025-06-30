import torch 
from tqdm import tqdm
import matplotlib.pyplot as plt
import cfg
from torch.utils.data import DataLoader
import torch.nn.functional as F
from func_3d.dataset.prostate import IntraDataset_revice,IntreDataset_revice
import numpy as np
import torch
import nibabel as nib
import argparse
import cv2
from region_correspondence.paired_regions import PairedRegions
from region_correspondence.utils import warp_by_ddf
from torch.utils.data import DataLoader
from mmdet.apis import DetInferencer
from mmengine.registry import init_default_scope
from PIL import Image
from utils import utils
from utils.metric import mask_metric

device = torch.device('cuda', 0)

mr_image_path = '/home/adminer/code/Det-SAMReg-demo/dataset/test/mr_images/'
mr_label_path = '/home/adminer/code/Det-SAMReg-demo/dataset/test/mr_labels/'
us_label_path = "/home/adminer/code/Det-SAMReg-demo/dataset/test/us_labels/"
us_image_path = "/home/adminer/code/Det-SAMReg-demo/dataset/test/us_images/"
mr_dino_checkpoint="/home/adminer/code/Det-SAMReg-demo/checkpoint/MR-G-DINO.pth"
us_dino_checkpoint="/home/adminer/code/Det-SAMReg-demo/checkpoint/US-G-DINO.pth"
temp_path = '//home/adminer/code/Det-SAMReg-demo/result/vis_result/output.jpg'
inferencer_mr = DetInferencer(model='grounding_dino_swin-t_finetune_16xb2_1x_coco', weights=mr_dino_checkpoint,device=device,show_progress=False)
inferencer_us = DetInferencer(model='grounding_dino_swin-t_finetune_16xb2_1x_coco', weights=us_dino_checkpoint,device=device,show_progress=False)
result_temp_path1 = '//home/adminer/code/Det-SAMReg-demo/result/vis_result/seg_result.svg'
result_temp_path = '//home/adminer/code/Det-SAMReg-demo/result/vis_result/seg_result.png'

csv_path = '//home/adminer/code/Det-SAMReg-demo/result/csv_result/test.csv'
args = cfg.parse_args()
test_set = IntraDataset_revice(mr_image_path,us_image_path,mr_label_path,us_label_path)
# test_set = IntreDataset_revice(mr_image_path,mr_label_path)

test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
ckpt1 = torch.load(args.mr_pretrain, map_location=device)
net1 = ckpt1['model']          
net1 = net1.to(torch.float32)
net1.load_state_dict(ckpt1,strict=False)
net1.eval()

ckpt2 = torch.load(args.us_pretrain, map_location=device)
net2 = ckpt2['model']          
net2 = net2.to(torch.float32)
net2.load_state_dict(ckpt2,strict=False)
net2.eval()

n_val = len(test_loader)  # the number of batch
mix_res = (0,)*1*2
tot = 0
threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
prompt_freq = args.prompt_freq
# prompt_freq = 1

# lossfunc = paper_loss
mask_err = mask_metric.MaskMetrics()
mask_multi = mask_metric.multi_metric_tensor
init_default_scope('mmdet')

prompt = args.prompt

def sam_with_box(pack,temp_path,slice_dim,sam_type,shift):
    if sam_type=='mr':
        net = net1
        inference = inferencer_mr
    if sam_type=='us':
        net = net2
        inference = inferencer_us
    imgs_tensor = pack['image']
    mask_dict = pack['label']
    bbox_dict = pack['bbox']   # boundingbox
    start_index  = pack['start_index']
    seg = pack['seg_data'].squeeze()
    if len(imgs_tensor.size()) == 5:
        imgs_tensor = imgs_tensor.squeeze(0)
    frame_id = list(range(imgs_tensor.size(0)))  # video_length
    
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
                    # slice_img = imgs_tensor[id, :, :, :]
                    # slice_img = slice_img.squeeze().cpu().numpy()[0]
                    # # print(slice_img.shape)
                    # img_rgb = Image.fromarray(slice_img).convert("RGB")
                    # img_rgb.save(temp_path) 
                    slice_img = imgs_tensor[id, :, :, :]         # (C, H, W)
                    slice_img = slice_img.cpu().numpy()           # (C, H, W)
                    slice_img = slice_img.transpose(1, 2, 0)      # (H, W, C)

                    result = inference({'img': slice_img, 'text': 'prostate'})
                    # optional
                    bounding_box = torch.tensor([result['predictions'][0]['bboxes']], device=device)[0]
                    bbox_gt = bbox_dict[id][ann_obj_id]
                    bbox = utils.random_perturb_bbox(bbox_gt[0],slice_img[...,0],shift)
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
                        mask = torch.zeros(imgs_tensor.shape[2:]).to(device=device),
                    )
        video_segments = {}  # video_segments contains the per-frame segmentation results
    
        for out_frame_idx, out_obj_ids, out_mask_logits in net.propagate_in_video(train_state, start_frame_idx=0):
            video_segments[out_frame_idx] = {
                out_obj_id: out_mask_logits[i]
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        for id in frame_id:
            for ann_obj_id in obj_list:
                pred = video_segments[id][ann_obj_id]
                pred = pred.unsqueeze(0)
                pred = (pred >0.5).float()
                # bbox = bbox_dict[id][ann_obj_id]
                # mask = mask_dict[id][ann_obj_id].to(dtype = torch.float32, device = device)
                # slice_img = imgs_tensor[id, :, :, :]
                # slice_img = slice_img.squeeze().cpu().numpy()[0]
                masks = F.interpolate(pred, size = seg[0,...].shape, mode='nearest')
            
            # utils.save_segmentation_comparison(
            #                             img_rgb=slice_img,                         # (3, H, W) tensor
            #                             pred_mask=pred.squeeze().detach().cpu().numpy(),         # (H, W)
            #                             true_mask=mask.squeeze().detach().cpu().numpy(),         # (H, W)
            #                             bbox_gt=bbox[0].cpu().numpy(),
            #                             bbox_dn=bounding_box.cpu().numpy(),
            #                             save_path=result_temp_path
            #                         )
            # stavk mask slice
            for region_idx, mask in enumerate(masks):
                if region_idx not in region_masks:
                    region_masks[region_idx] = np.zeros(seg.shape, dtype=np.uint8)
                    # print(region_masks[region_idx].shape)
                current_slice_mask = utils.get_dim(region_masks[region_idx], start_index+id, slice_dim)
                combined_mask = np.where(mask[0].cpu().numpy() > 0, 1, current_slice_mask)
                utils.get_slice(region_masks[region_idx], start_index+id, slice_dim, combined_mask)
    assert len(region_masks) == 1, "region_mask must be one"
    region_idx, mask_3d = next(iter(region_masks.items()))

    return torch.tensor(mask_3d,device=device)

for fix_pack,mov_pack,us_seg,mr_seg,mr_img,us_img in tqdm(test_loader, desc="Inference", unit="batch",total=len(test_loader)):

    fix_seg = fix_pack['seg_data'].squeeze().to(device)
    mov_seg = mov_pack['seg_data'].squeeze().to(device)
    path = fix_pack['path'][0]
    _,affine,header = utils.load_nii_image(path)

    region_masks = {}
    mask_fix = sam_with_box(fix_pack,temp_path,slice_dim=0,sam_type='us',shift=0)
    region_masks = {}

    mask_mov = sam_with_box(mov_pack,temp_path,slice_dim=0,sam_type='mr',shift=0)
    masks_mov = torch.stack([mask_mov], dim=0).to(torch.bool)
    masks_fix = torch.stack([mask_fix], dim=0).to(torch.bool)
    mov_gt = torch.stack([mr_seg[i] for i in range(mr_seg.shape[0])], dim=0).to(torch.bool).squeeze()

    paired_rois = PairedRegions(masks_mov=masks_mov, masks_fix=masks_fix, device=device)
    ddf = paired_rois.get_dense_correspondence(transform_type='ddf', max_iter=1000, lr=0.01, w_roi=1,w_ddf2=5.0, w_ddfb=0, verbose=True)
    print(ddf.shape)
    roi_warped = (warp_by_ddf(mov_gt.to(dtype=torch.float32, device=device), ddf) * 255).bool().int()
    # img_warped = (warp_by_ddf(mr_img.to(dtype=torch.float32, device=device), ddf,mode='bilinear'))
    roi_warped = roi_warped.squeeze()
    # print(roi_warped.shape)  
    result_raw_us = mask_err(fix_seg.cpu().numpy(), mask_fix.cpu().numpy())
    result_raw_mr = mask_err(mov_seg.cpu().numpy(), mask_mov.cpu().numpy())
    
    us_seg = us_seg.to(device)
    result = mask_multi(us_seg.squeeze(), roi_warped)
    print('dice_raw_us:{:.2f},dice_raw_mr:{:.2f},dice:{:.2f},tre_raw_us:{:.2f},tre_raw_mr:{:.2f},tre:{:.2f},hd95:{:.2f},asd:{:.2f}'.format(
    result_raw_us['dice'],result_raw_mr['dice'],result['dice'],result_raw_us['tre'],result_raw_mr['tre'],result['tre'],result['hd95'],result['asd']))
