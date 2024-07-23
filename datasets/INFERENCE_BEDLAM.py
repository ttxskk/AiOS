import os
import os.path as osp
from glob import glob

import pickle
import cv2
import torch
import numpy as np

from config.config import cfg
from util.preprocessing import load_img, augmentation_keep_size
from util.formatting import DefaultFormatBundle
from detrsmpl.data.datasets.pipelines.transforms import Normalize
from detrsmpl.core.conventions.keypoints_mapping import convert_kps


class INFERENCE_BEDLAM(torch.utils.data.Dataset):
    def __init__(self, img_dir=None, out_path=None):
        self.img_dir = os.path.join(img_dir, '**/*.png')
        self.out_path = out_path
        self.img_paths = sorted(glob(self.img_dir,recursive=True)) 

        self.score_threshold = cfg.threshold if 'threshold' in cfg else 0.85 
        self.resolution = [720,1280]  
        self.format = DefaultFormatBundle()
        self.normalize = Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
       
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = load_img(self.img_paths[idx],'BGR')
        img_whole_bbox = np.array([0, 0, img.shape[1],img.shape[0]])
        img, img2bb_trans, bb2img_trans, _, _ = \
            augmentation_keep_size(img, img_whole_bbox, 'test')

        cropped_img_shape=img.shape[:2]
        img = (img.astype(np.float32)) 
        

        inputs = {'img': img}
        targets = {
            'body_bbox_center': np.array(img_whole_bbox[None]),
            'body_bbox_size': np.array(img_whole_bbox[None])}
        meta_info = {
            'ori_shape':np.array(self.resolution),
            'img_shape': np.array(img.shape[:2]),
            'img2bb_trans': img2bb_trans,
            'bb2img_trans': bb2img_trans,
            'ann_idx': idx}
        result = {**inputs, **targets, **meta_info}
        
        result = self.normalize(result)
        result = self.format(result)
            
        return result
        
    def inference(self, outs):
        img_paths = self.img_paths
        output = {}
        for out in outs:
            ann_idx = out['image_idx']
            scores = out['scores'].clone().cpu().numpy()
            img_shape = out['img_shape'].cpu().numpy()[::-1] # w, h
            
            img = cv2.imread(img_paths[ann_idx]) # h, w
            img = cv2.imread(img_paths[ann_idx]) # h, w
            # os.makedirs(osp.join(self.out_path, 'vis'), exist_ok=True)
            # os.makedirs(osp.join(self.out_path, 'failed'), exist_ok=True)

                
            img = cv2.imread(img_paths[ann_idx]) # h, w      
            # os.makedirs(osp.join(self.out_path, 'vis'), exist_ok=True)
            # os.makedirs(osp.join(self.out_path, 'failed'), exist_ok=True)

                
            joint_proj = out['smplx_joint_proj'].clone().cpu().numpy()
            smpl_kp3d, _ = convert_kps(
                out['smpl_kp3d'].clone().cpu().numpy(),
                src='smplx',dst='agora', approximate=True)
            
            smpl_verts = out['smpl_verts'].clone().cpu().numpy()
            scale = img.shape[1]/img_shape[0]
            joint_proj = joint_proj * scale

            for i, score in enumerate(scores):
                if score < self.score_threshold:
                    break
                save_name = img_paths[ann_idx].split('/')[-4] + '_frameID_' + img_paths[ann_idx].split('/')[-1]
                save_name = save_name.replace('.png','')
                save_dict = {
                    'verts': smpl_verts[i].reshape(-1, 3),
                    'joints': joint_proj[i].reshape(-1, 2)[:24],
                    'allSmplJoints3d':smpl_kp3d[i].reshape(-1, 3), 
                    }
                # save
                exist_result_path = glob(osp.join(self.out_path, 'predictions', save_name + '*'))
                if len(exist_result_path) == 0:
                    person_idx = 0
                else:
                    last_person_idx = max([
                        int(name.split('personId_')[1].split('.pkl')[0])
                        for name in exist_result_path
                    ])
                    person_idx = last_person_idx + 1

                save_name += '_personId_' + str(person_idx) + '.pkl'
                os.makedirs(osp.join(self.out_path, 'predictions'), exist_ok=True)
                with open(osp.join(self.out_path, 'predictions', save_name),'wb') as f:
                    pickle.dump(save_dict, f)
            
        return output

