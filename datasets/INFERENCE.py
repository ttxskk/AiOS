import os
import os.path as osp
from glob import glob
import numpy as np
from config.config import cfg
import copy
import json
import pickle
import cv2
import torch
from pycocotools.coco import COCO
from util.human_models import smpl_x
from util.preprocessing import load_img, sanitize_bbox, process_bbox,augmentation_keep_size, load_ply, load_obj
from util.transforms import rigid_align, rigid_align_batch
import tqdm
import random
from util.formatting import DefaultFormatBundle
from detrsmpl.data.datasets.pipelines.transforms import Normalize
from humandata import HumanDataset
from detrsmpl.utils.demo_utils import xywh2xyxy, xyxy2xywh, box2cs
from detrsmpl.core.conventions.keypoints_mapping import convert_kps
import mmcv
import cv2
import numpy as np
from detrsmpl.core.visualization.visualize_keypoints2d import visualize_kp2d
from detrsmpl.core.visualization.visualize_smpl import visualize_smpl_hmr,render_smpl
from detrsmpl.models.body_models.builder import build_body_model
from detrsmpl.core.visualization.visualize_keypoints3d import visualize_kp3d
from detrsmpl.data.data_structures.multi_human_data import MultiHumanData
from detrsmpl.utils.ffmpeg_utils import video_to_images
from mmcv.runner import get_dist_info
from config.config import cfg
import torch.distributed as dist
import shutil

class INFERENCE(torch.utils.data.Dataset):
    def __init__(self, img_dir=None,out_path=None):
        
        self.output_path = out_path

        self.img_dir = img_dir

        self.is_vid = False
        
        rank, _ = get_dist_info()
        if self.img_dir.endswith('.mp4'):
            self.is_vid = True
            img_name = self.img_dir.split('/')[-1][:-4]
            # self.img_dir = self.img_dir[:-4]
        else:
            img_name = self.img_dir.split('/')[-1]
        self.img_name = img_name+'_out'
        self.output_path = os.path.join(self.output_path,self.img_name)
        os.makedirs(self.output_path, exist_ok=True)
        self.tmp_dir = os.path.join(self.output_path, 'temp_img')
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.result_img_dir = os.path.join(self.output_path, 'res_img')
        
        
        if not self.is_vid:
            if rank == 0:
                image_files = sorted(glob(self.img_dir + '/*.jpg') + glob(self.img_dir + '/*.png'))
                for i, image_file in enumerate(image_files):
                    new_name = os.path.join(self.tmp_dir, '%06d.png'%i)
                    shutil.copy(image_file, new_name)
            dist.barrier()
        else:
            if rank == 0:
                video_to_images(self.img_dir, self.tmp_dir)
            dist.barrier()
        self.img_paths = sorted(glob(self.tmp_dir+'/*',recursive=True)) 
        self.score_threshold = 0.2
        self.resolution = [720 ,1280] # AGORA test     
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
        sample_num = len(outs)
        output = {}
        
        for out in outs:
            ann_idx = out['image_idx']
            img_cropped = mmcv.imdenormalize(
                img=(out['img'].cpu().numpy()).transpose(1, 2, 0), 
                mean=np.array([123.675, 116.28, 103.53]), 
                std=np.array([58.395, 57.12, 57.375]),
                to_bgr=True).astype(np.uint8)
            # bb2img_trans = out['bb2img_trans']
            # img2bb_trans = out['img2bb_trans']
            scores = out['scores'].clone().cpu().numpy()
            img_shape = out['img_shape'].cpu().numpy()[::-1] # w, h
            width,height = img_shape
            width += width % 2
            height += height % 2
            img_shape = np.array([width, height])
            img = cv2.imread(img_paths[ann_idx]) # h, w

                
            joint_proj = out['smplx_joint_proj'].clone().cpu().numpy()
            joint_vis = out['smplx_joint_proj'].clone().cpu().numpy()
            joint_coco = out['keypoints_coco'].clone().cpu().numpy()
            joint_coco_raw = joint_coco.copy()
            smpl_kp3d_coco, _ = convert_kps(out['smpl_kp3d'].clone().cpu().numpy(),src='smplx',dst='coco', approximate=True)
            
            
            
            body_bbox = out['body_bbox'].clone().cpu().numpy()
            lhand_bbox = out['lhand_bbox'].clone().cpu().numpy()
            rhand_bbox = out['rhand_bbox'].clone().cpu().numpy()
            face_bbox = out['face_bbox'].clone().cpu().numpy()

            if self.resolution == [720, 1280]:
                joint_proj[:, :, 0] = joint_proj[:, :, 0] / img_shape[0] * 3840
                joint_proj[:, :, 1] = joint_proj[:, :, 1] / img_shape[1] * 2160
                joint_vis[:, :, 0] = joint_vis[:, :, 0] / img_shape[0] * img.shape[1]
                joint_vis[:, :, 1] = joint_vis[:, :, 1]/ img_shape[1] * img.shape[0]        
                
                joint_coco[:, :, 0] = joint_coco[:, :, 0] / img_shape[0] * img.shape[1]
                joint_coco[:, :, 1] = joint_coco[:, :, 1]/ img_shape[1] * img.shape[0] 
                scale = np.array([
                    img.shape[1]/img_shape[0],
                    img.shape[1]/img_shape[0], 
                    img.shape[1]/img_shape[0], 
                    img.shape[1]/img_shape[0], 
                    ])
                body_bbox_raw = body_bbox.copy()
                body_bbox = body_bbox * scale
                lhand_bbox = lhand_bbox * scale
                rhand_bbox = rhand_bbox * scale
                face_bbox = face_bbox * scale
            elif self.resolution == [1200, 1600]:
                
                joint_proj[:, :, 0] = joint_proj[:, :, 0] * (1200 / 800)
                joint_proj[:, :, 1] = joint_proj[:, :, 1] * (1600 / 1066)

                joint_vis[:, :, 0] = joint_vis[:, :, 0] * (1200 / 800)
                joint_vis[:, :, 1] = joint_vis[:, :, 1] * (1600 / 1066)             
                
                scale = np.array([1600/1066, 1200/800, 1600/1066, 1200/800])[None]
                body_bbox = body_bbox * scale
                lhand_bbox = lhand_bbox * scale
                rhand_bbox = rhand_bbox * scale
                face_bbox = face_bbox * scale
                
            for i, score in enumerate(scores):
                if score < self.score_threshold:
                    break

                save_name = img_paths[ann_idx].split('/')[-1][:-4] # if not crop should be -4
                if self.resolution == (2160, 3840):
                    save_name = save_name.split('_ann_id')[0]
                else:
                    save_name = save_name.split('_1280x720')[0] 


                
                save_dict = {
                    'params': {
                        'transl': out['cam_trans'][i].reshape(1, -1).cpu().numpy(),
                        'global_orient': out['smplx_root_pose'][i].reshape(1, -1).cpu().numpy(),
                        'body_pose': out['smplx_body_pose'][i].reshape(1, -1).cpu().numpy(),
                        'left_hand_pose': out['smplx_lhand_pose'][i].reshape(1, -1).cpu().numpy(),
                        'right_hand_pose': out['smplx_rhand_pose'][i].reshape(1, -1).cpu().numpy(),
                        'reye_pose': np.zeros((1, 3)),
                        'leye_pose': np.zeros((1, 3)),
                        'jaw_pose': out['smplx_jaw_pose'][i].reshape(1, -1).cpu().numpy(),
                        'expression': out['smplx_expr'][i].reshape(1, -1).cpu().numpy(),
                        'betas': out['smplx_shape'][i].reshape(1, -1).cpu().numpy()},
                    
                    'joints': joint_proj[i].reshape(1, -1, 2)[0,:24]}
                
                # save
                exist_result_path = glob(osp.join(self.output_path, 'predictions', save_name + '*'))
                if len(exist_result_path) == 0:
                    person_idx = 0
                else:
                    last_person_idx = max([
                        int(name.split('personId_')[1].split('.pkl')[0])
                        for name in exist_result_path
                    ])
                    person_idx = last_person_idx + 1

                save_name += '_personId_' + str(person_idx) + '.pkl'
                os.makedirs(osp.join(self.output_path, 'predictions'), exist_ok=True)
                with open(osp.join(self.output_path, 'predictions', save_name),'wb') as f:
                    pickle.dump(save_dict, f)
            # mesh
            # bbox

            
            if i == 0:
                save_name = img_paths[ann_idx].split('/')[-1][:-4]
                cv2.imwrite(os.path.join(self.result_img_dir,img_paths[ann_idx].split('/')[-1]), img)
            else:
                # dump bbox
                body_xywh = xyxy2xywh(body_bbox[:i])
                score = scores[:i]
                out_value = [{'bbox': b, 'score': s} for b, s in zip(body_xywh, score)]
                out_key = img_paths[ann_idx].split('/')[-1]
                output.update({out_key: out_value})
                
                # show bbox 
                img = mmcv.imshow_bboxes(img, body_bbox[:i], show=False, colors='green')
                img = mmcv.imshow_bboxes(img, lhand_bbox[:i], show=False, colors='blue')
                img = mmcv.imshow_bboxes(img, rhand_bbox[:i], show=False, colors='yellow')
                img = mmcv.imshow_bboxes(img, face_bbox[:i], show=False, colors='red')
                
                verts = out['smpl_verts'][:i] + out['cam_trans'][:i][:, None]
                body_model_cfg = dict(
                    type='smplx',
                    keypoint_src='smplx',
                    num_expression_coeffs=10,
                    num_betas=10,
                    gender='neutral',
                    keypoint_dst='smplx_137',
                    model_path='data/body_models/smplx',
                    use_pca=False,
                    use_face_contour=True)
                body_model = build_body_model(body_model_cfg).to('cuda')
                # for n, v in enumerate(verts):
                #     save_obj(
                #         osp.join(self.out_path, 'vis', img_paths[ann_idx].split('/')[-1].rjust(5+4,'0')).replace('.jpg',f'_{n}_.obj'),
                #         verts = v,
                #         faces=torch.tensor(body_model.faces.astype(np.int32))
                #     )
                # print(osp.join(self.out_path, 'vis', img_paths[ann_idx].split('/')[-1]))
                
                render_smpl(
                    verts=verts[None],
                    body_model=body_model,
                    # K= np.array(
                    #     [[img_shape[0]/2, 0, img_shape[0]/2],
                    #      [0, img_shape[0]/2, img_shape[1]/2],
                    #      [0, 0, 1]]),
                    K= np.array(
                        [[5000, 0, img_shape[0]/2],
                         [0, 5000, img_shape[1]/2],
                         [0, 0, 1]]),
                    R=None,
                    T=None,
                    # output_path=osp.join(self.out_path, 'vis', img_paths[ann_idx].split('/')[-1].rjust(5+4,'0')),
                    output_path=os.path.join(self.result_img_dir,img_paths[ann_idx].split('/')[-1]),
                    image_array=cv2.resize(img, (img_shape[0],img_shape[1]), cv2.INTER_CUBIC),
                    in_ndc=False,
                    alpha=0.9,
                    convention='opencv',
                    projection='perspective',
                    overwrite=True,
                    no_grad=True,
                    device='cuda',
                    resolution=[img_shape[0],img_shape[1]],
                    render_choice='hq',    
                )
        return output

