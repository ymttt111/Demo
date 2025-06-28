import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import nibabel as nib
import glob
from itertools import permutations,combinations
from func_3d.utils import generate_bbox,random_perturb_bbox
from scipy.ndimage import zoom
import torch.nn.functional as F

class IntreDataset_revice(Dataset):
    def __init__(self,mr_image_dir,mr_label_dir,seed=None,variation=0):

        mr_image_set = sorted(glob.glob(mr_image_dir + '*.nii.gz'))
        mr_label_set = sorted(glob.glob(mr_label_dir + '*.nii.gz'))
 
        self.mr_image_paths = mr_image_set
        self.mr_label_paths = mr_label_set
        self.variation = variation
        self.img_size = 1024
        self.seed = seed
        self.noise = 0
        self.pairs = list(permutations(range(len(self.mr_image_paths)), 2))
        self.pairs = self.pairs[827:]
        print(len(self.pairs))
        
    def one_hot(self, seg):
        C = seg.shape[-1]
        out = np.zeros((C, seg.shape[0], seg.shape[1], seg.shape[2]))
        for i in range(C):
            out[i,...] = (seg[..., i] == 255).astype(np.uint8)
        return out

    def __getitem__(self, index):
        
        fix_idx,mov_idx = self.pairs[index]
        mr_img_path = self.mr_image_paths[mov_idx]
        us_img_path = self.mr_image_paths[fix_idx]
        mr_seg_path = self.mr_label_paths[mov_idx]
        us_seg_path = self.mr_label_paths[fix_idx]
        
        mr_img = nib.load(mr_img_path).get_fdata().squeeze()
        us_img = nib.load(us_img_path).get_fdata().squeeze()
        mr_seg = self.one_hot(nib.load(mr_seg_path).get_fdata())
        us_seg = self.one_hot(nib.load(us_seg_path).get_fdata())
        mr_img = np.ascontiguousarray(mr_img)
        us_img = np.ascontiguousarray(us_img)
        mr_seg = np.ascontiguousarray(mr_seg)
        us_seg = np.ascontiguousarray(us_seg)
        fix_info = self.return_dict(us_seg[0],us_img,us_seg_path)
        mov_info = self.return_dict(mr_seg[0],mr_img,mr_seg_path)
        
        mr_img,us_img = [torch.tensor(i,dtype=torch.float32) for i in [mr_img,us_img]]
        
        return fix_info,mov_info,us_seg,mr_seg,mr_img,us_img
    
    def return_dict(self,seg,img,path):
        seg = torch.tensor(seg, dtype=torch.float32)
        newsize = (self.img_size, self.img_size)
        """Get the images"""
        num_frame = seg.shape[0]
        data_seg_3d = seg.detach().cpu().numpy()
        image = img
        nonzero_slices = [i for i in range(data_seg_3d.shape[0]) if np.any(data_seg_3d[i, ...] != 0)]
        starting_frame_nonzero = nonzero_slices[0] if nonzero_slices else None
        data_seg_3d = data_seg_3d[nonzero_slices, :, :] if nonzero_slices else np.zeros((0, *data_seg_3d.shape[1:]))
        num_frame = data_seg_3d.shape[0]
        starting_frame = 0
        video_length = num_frame
        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        mask_dict = {}
        bbox_dict = {}
        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size) 

        for frame_index in range(starting_frame, starting_frame + video_length):
            if frame_index + starting_frame_nonzero >= image.shape[0]:
                raise ValueError("Frame index out of bounds!!!")
            img = image[frame_index + starting_frame_nonzero,...]
            img = Image.fromarray(img).convert('RGB')  
            mask = data_seg_3d[frame_index,...]
            obj_list = np.unique(mask[mask > 0])
            diff_obj_mask_dict = {}

            diff_obj_bbox_dict = {}

            for obj in obj_list:
                obj_mask = mask == obj
                obj_mask = Image.fromarray(obj_mask)
                obj_mask = obj_mask.resize(newsize)
                obj_mask = torch.tensor(np.array(obj_mask)).unsqueeze(0).int()
                diff_obj_mask_dict[obj] = obj_mask
                diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed)
                if self.noise != 0:
                        diff_obj_bbox_dict[obj] = random_perturb_bbox(torch.tensor(diff_obj_bbox_dict[obj]),obj_mask.squeeze(0),self.noise)
            img = img.resize(newsize)
            # img = resize_data(img, (3,1024,1024))
            # img = img.resize(newsize)
            img = torch.tensor(np.array(img)).permute(2, 0, 1)
            # print(img.shape) #torch.Size([3, 1024, 1024])
            # print(img_tensor[frame_index - starting_frame,:, :, :].shape) #torch.Size([3,1024,1024])
            # print(img_tensor.shape)
            img_tensor[frame_index - starting_frame,:, :, :] = img
            # print(img_tensor.shape)
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
            bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict

        return {
            'start_index':starting_frame_nonzero,
            'image':img_tensor,
            'label': mask_dict,
            'bbox': bbox_dict,
            'seg_data':seg,
            'path':path
            
        }
    def __len__(self):
        return len(self.pairs)


class IntraDataset_revice(Dataset):
    def __init__(self,mr_image_dir,us_image_dir,mr_label_dir,us_label_dir,seed=None,variation=0):
        mr_image_set = sorted(glob.glob(mr_image_dir + '*.nii.gz'))
        us_image_set = sorted(glob.glob(us_image_dir + '*.nii.gz'))
        mr_label_set = sorted(glob.glob(mr_label_dir + '*.nii.gz'))
        us_label_set = sorted(glob.glob(us_label_dir + '*.nii.gz'))
        
        self.mr_image_paths = mr_image_set
        self.mr_label_paths = mr_label_set
        self.us_image_paths = us_image_set
        self.us_label_paths = us_label_set
        self.variation = variation
        self.img_size = 1024
        self.seed = seed
        self.noise = 0
        
    def one_hot(self, seg):
        C = seg.shape[-1]
        out = np.zeros((C, seg.shape[0], seg.shape[1], seg.shape[2]))
        for i in range(C):
            out[i,...] = (seg[..., i] == 255).astype(np.uint8)
        return out

    def __getitem__(self, index):
        
        mr_img_path = self.mr_image_paths[index]
        us_img_path = self.us_image_paths[index]
        mr_seg_path = self.mr_label_paths[index]
        us_seg_path = self.us_label_paths[index]
        
        mr_img = nib.load(mr_img_path).get_fdata().squeeze()
        us_img = nib.load(us_img_path).get_fdata().squeeze()
        mr_seg = self.one_hot(nib.load(mr_seg_path).get_fdata())
        us_seg = self.one_hot(nib.load(us_seg_path).get_fdata())
        mr_img = np.ascontiguousarray(mr_img)
        us_img = np.ascontiguousarray(us_img)
        mr_seg = np.ascontiguousarray(mr_seg)
        us_seg = np.ascontiguousarray(us_seg)
        fix_info = self.return_dict(us_seg[0],us_img,us_seg_path)
        mov_info = self.return_dict(mr_seg[0],mr_img,mr_seg_path)
        
        mr_img,us_img = [torch.tensor(i,dtype=torch.float32) for i in [mr_img,us_img]]

        return fix_info,mov_info,us_seg,mr_seg,mr_img,us_img
    
    def return_dict(self,seg,img,path):
        seg = torch.tensor(seg, dtype=torch.float32)
        newsize = (self.img_size, self.img_size)
        """Get the images"""
        num_frame = seg.shape[0]
        data_seg_3d = seg.detach().cpu().numpy()
        image = img
        nonzero_slices = [i for i in range(data_seg_3d.shape[0]) if np.any(data_seg_3d[i, ...] != 0)]
        starting_frame_nonzero = nonzero_slices[0] if nonzero_slices else None
        data_seg_3d = data_seg_3d[nonzero_slices, :, :] if nonzero_slices else np.zeros((0, *data_seg_3d.shape[1:]))
        num_frame = data_seg_3d.shape[0]
        starting_frame = 0
        video_length = num_frame
        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size) 
        mask_dict = {}
        bbox_dict = {}
        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size) 

        for frame_index in range(starting_frame, starting_frame + video_length):
            if frame_index + starting_frame_nonzero >= image.shape[0]:
                raise ValueError("Frame index out of bounds!!!")
            img = image[frame_index + starting_frame_nonzero,...]
            
            img = Image.fromarray(img).convert('RGB')
            mask = data_seg_3d[frame_index,...]
            obj_list = np.unique(mask[mask > 0])
            diff_obj_mask_dict = {}

            diff_obj_bbox_dict = {}

            for obj in obj_list:
                obj_mask = mask == obj
                obj_mask = Image.fromarray(obj_mask)
                obj_mask = obj_mask.resize(newsize)
                obj_mask = torch.tensor(np.array(obj_mask)).unsqueeze(0).int()
                diff_obj_mask_dict[obj] = obj_mask
                diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed)
                if self.noise != 0:
                        diff_obj_bbox_dict[obj] = random_perturb_bbox(torch.tensor(diff_obj_bbox_dict[obj]),obj_mask.squeeze(0),self.noise)
            img = img.resize(newsize)
            img = torch.tensor(np.array(img)).permute(2, 0, 1)
            img_tensor[frame_index - starting_frame,:, :, :] = img
            # print(img_tensor.shape)
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
            bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict

        return {
            'start_index':starting_frame_nonzero,
            'image':img_tensor,
            'label': mask_dict,
            'bbox': bbox_dict,
            'seg_data':seg,
            'path':path
            
        }
    def __len__(self):
        return len(self.mr_image_paths)
