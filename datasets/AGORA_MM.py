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
from util.preprocessing import load_img, sanitize_bbox, process_bbox, load_ply, load_obj
from util.transforms import rigid_align, rigid_align_batch
import tqdm
import random
from util.formatting import DefaultFormatBundle
from detrsmpl.data.datasets.pipelines.transforms import Normalize
import time
from util.preprocessing import (
    load_img, process_bbox, augmentation_instance_sample
    ,process_human_model_output_batch_simplify,process_db_coord_batch_no_valid)
# from util.human_models import smpl_x
from .humandata import HumanDataset
import csv
KPS2D_KEYS = [
    'keypoints2d_ori', 'keypoints2d_smplx', 'keypoints2d_smpl',
    'keypoints2d_original','keypoints2d_gta'
]
KPS3D_KEYS = [
    'keypoints3d_cam', 'keypoints3d', 'keypoints3d_smplx', 'keypoints3d_smpl',
    'keypoints3d_original', 'keypoints3d_gta'
]
class AGORA_MM(HumanDataset):
    def __init__(self, transform, data_split):
        super(AGORA_MM, self).__init__(transform, data_split)
        self.img_shape = [2160,3840]
        pre_prc_file_train = 'spec_train_smpl.npz'
        pre_prc_file_test = 'spec_test_smpl.npz'
        self.save_idx = 0
        if self.data_split == 'train':
            filename = getattr(cfg, 'filename', pre_prc_file_train)
        else:
            self.test_set = 'val'
        
        self.img_dir = './data/datasets/agora'


        if data_split == 'train':
            if self.img_shape == [2160,3840]:
                self.annot_path = 'data/multihuman_data/agora_train_3840_w_occ_multi_2010.npz' # has 'is_kid'
                self.annot_path_cache = 'data/cache/agora_train_3840_w_occ_cache_2010.npz'
                # self.annot_path = 'data/preprocessed_npz/multihuman_data/agora_train_focal.npz'
                # self.annot_path_cache = 'data/preprocessed_npz/cache/agora_train_focal_cache.npz'
            elif self.img_shape == [720,1280]:
                self.annot_path = 'data/preprocessed_npz/multihuman_data/agora_train_1280_multi_1010.npz'
                self.annot_path_cache = 'data/preprocessed_npz/cache/agora_train_cache_1280_1010.npz'
                
            # self.annot_path_cache = 'data/preprocessed_npz/cache/agora_train_multi_1280_cache_all.npz'
            # self.annot_path_cache = 'data/preprocessed_npz/cache/agora_train_cache_1010.npz'

            # self.annot_path = 'data/preprocessed_npz/multihuman_data/agora_train_1280_multi_1010_occ.npz'
            # # self.annot_path_cache = 'data/preprocessed_npz/cache/agora_train_multi_1280_cache_all.npz'
            # self.annot_path_cache = 'data/preprocessed_npz/cache/agora_train_cache_0928.npz'
            # # self.annot_path_cache = 'data/preprocessed_npz/cache/agora_train_cache_1010_occ.npz'
        elif data_split == 'test':
            if self.img_shape == [2160,3840]:
                self.annot_path = 'data/multihuman_data/agora_validation_multi_3840_1010.npz'
                self.annot_path_cache = 'data/cache/agora_validation_cache_3840_1010_occ_cache_balance.npz'
                
                # self.annot_path = 'data/preprocessed_npz/multihuman_data/agora_val_3840_focal_wo_occ.npz'
                # self.annot_path_cache = 'data/preprocessed_npz/cache/agora_val_3840_focal_wo_occ.npz'
            elif self.img_shape == [720,1280]:
                self.annot_path = 'data/preprocessed_npz/multihuman_data/agora_validation_1280_1010_occ.npz'
                self.annot_path_cache = 'data/preprocessed_npz/cache/agora_validation_cache_1280_1010_occ.npz'
        
        self.use_cache = getattr(cfg, 'use_cache', False)
        self.cam_param = {}
        # self.img_shape = [720,1280]
        
        # check image shape
        # img_path = osp.join(self.img_dir, np.load(self.annot_path)['image_path'][0])
        # img_shape = cv2.imread(img_path).shape[:2]
        # assert self.img_shape == img_shape, 'image shape is incorrect: {} vs {}'.format(self.img_shape, img_shape)

        # load data or cache
        if self.use_cache and osp.isfile(self.annot_path_cache):
            print(f'[{self.__class__.__name__}] loading cache from {self.annot_path_cache}')
            self.datalist = self.load_cache(self.annot_path_cache)
        else:
            if self.use_cache:
                print(f'[{self.__class__.__name__}] Cache not found, generating cache...')
            self.datalist = self.load_data(
                train_sample_interval=getattr(cfg, f'{self.__class__.__name__}_train_sample_interval', 1))
            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)


    def load_data(self, train_sample_interval=1):

        content = np.load(self.annot_path, allow_pickle=True)
        # import ipdb;ipdb.set_trace()
        try:
            frame_range = content['frame_range']
        except KeyError:
            frame_range = \
                np.array([[i, i + 1] for i in range(self.num_data)])

        num_examples = len(frame_range)
        # import ipdb;ipdb.set_trace()
        if 'meta' in content:
            meta = content['meta'].item()
            print('meta keys:', meta.keys())
        else:
            meta = None
            print(
                'No meta info provided! Please give height and width manually')

        print(
            f'Start loading humandata {self.annot_path} into memory...\nDataset includes: {content.files}'
        )
        tic = time.time()
        image_path = content['image_path']

        if meta is not None and 'height' in meta:
            height = np.array(meta['height'])
            width = np.array(meta['width'])
            image_shape = np.stack([height, width], axis=-1)
        else:
            image_shape = None

        if meta is not None and 'gender' in meta and len(meta['gender']) != 0:
            gender = meta['gender']
        else:
            gender = None
        
        if meta is not None and 'is_kid' in meta and len(meta['is_kid']) != 0:
            is_kid = meta['is_kid']
        else:
            is_kid = None
                    
        bbox_xywh = content['bbox_xywh']

        if 'smplx' in content:
            smplx = content['smplx'].item()
            as_smplx = 'smplx'
        elif 'smpl' in content:
            smplx = content['smpl'].item()
            as_smplx = 'smpl'
        elif 'smplh' in content:
            smplx = content['smplh'].item()
            as_smplx = 'smplh'
        # TODO: temp solution, should be more general. But SHAPY is very special
        elif self.__class__.__name__ == 'SHAPY':
            smplx = {}
        else:
            raise KeyError('No SMPL for SMPLX available, please check keys:\n'
                           f'{content.files}')

        print('Smplx param', smplx.keys())

        if 'lhand_bbox_xywh' in content and 'rhand_bbox_xywh' in content:
            lhand_bbox_xywh = content['lhand_bbox_xywh']
            rhand_bbox_xywh = content['rhand_bbox_xywh']
        else:
            lhand_bbox_xywh = np.zeros_like(bbox_xywh)
            rhand_bbox_xywh = np.zeros_like(bbox_xywh)

        if 'face_bbox_xywh' in content:
            face_bbox_xywh = content['face_bbox_xywh']
        else:
            face_bbox_xywh = np.zeros_like(bbox_xywh)

        decompressed = False
        if content['__keypoints_compressed__']:
            decompressed_kps = self.decompress_keypoints(content)
            decompressed = True

        keypoints3d = None
        valid_kps3d = False
        keypoints3d_mask = None
        valid_kps3d_mask = False
        # import ipdb;ipdb.set_trace()
        
        # processing keypoints
        for kps3d_key in KPS3D_KEYS:
            if kps3d_key in content:
                keypoints3d = decompressed_kps[kps3d_key][:, self.SMPLX_137_MAPPING, :] if decompressed \
                else content[kps3d_key][:, self.SMPLX_137_MAPPING, :]
                valid_kps3d = True
                if keypoints3d.shape[-1] == 4:
                    valid_kps3d_mask = True
                break
        if self.keypoints2d is not None:
            keypoints2d = decompressed_kps[self.keypoints2d][:, self.SMPLX_137_MAPPING, :] if decompressed \
                else content[self.keypoints2d][:, self.SMPLX_137_MAPPING, :]
            

        else:
            for kps2d_key in KPS2D_KEYS:
                if kps2d_key in content:
                    keypoints2d = decompressed_kps[kps2d_key][:, self.SMPLX_137_MAPPING, :] if decompressed \
                        else content[kps2d_key][:, self.SMPLX_137_MAPPING, :]
                    
        if keypoints2d.shape[-1] == 3:
            valid_kps3d_mask = True
        occlusion = content['meta'][()]['occ'] if 'occ' in content['meta'][()] and len(content['meta'][()]['occ'])>0 else None
        # import ipdb;ipdb.set_trace()
        print('Done. Time: {:.2f}s'.format(time.time() - tic))

        datalist = []
        # num_examples

        # processing each image, filter according to bbox valid
        for i in tqdm.tqdm(range(int(num_examples))):
            if self.data_split == 'train' and i % train_sample_interval != 0:
                continue
            frame_start, frame_end = frame_range[i]
            img_path = osp.join(self.img_dir, image_path[frame_start])
            # im_shape = cv2.imread(img_path).shape[:2]
            img_shape = image_shape[
                frame_start] if image_shape is not None else self.img_shape
            

            bbox_list = bbox_xywh[frame_start:frame_end, :4]
            # import ipdb;ipdb.set_trace()
            valid_idx = []
            body_bbox_list = []
            
            if hasattr(cfg, 'bbox_ratio'):
                bbox_ratio = cfg.bbox_ratio * 0.833  # preprocess body bbox is giving 1.2 box padding
            else:
                bbox_ratio = 1.25
            # import ipdb;ipdb.set_trace()
            for bbox_i, bbox in enumerate(bbox_list):
                # import ipdb;ipdb.set_trace()
                bbox = process_bbox(bbox,
                                    img_width=img_shape[1],
                                    img_height=img_shape[0],
                                    ratio=bbox_ratio)
                if bbox is None:
                    continue
                else:
                    valid_idx.append(frame_start + bbox_i)
                    bbox[2:] += bbox[:2]
                    body_bbox_list.append(bbox)
            if len(valid_idx) == 0:
                continue
            valid_num = len(valid_idx)
            # hand/face bbox
            lhand_bbox_list = []
            rhand_bbox_list = []
            face_bbox_list = []
            
            for bbox_i in valid_idx:
                lhand_bbox = lhand_bbox_xywh[bbox_i]
                
                rhand_bbox = rhand_bbox_xywh[bbox_i]
                face_bbox = face_bbox_xywh[bbox_i]
                if lhand_bbox[-1] > 0:  # conf > 0
                    lhand_bbox = lhand_bbox[:4]
                    if hasattr(cfg, 'bbox_ratio'):
                        lhand_bbox = process_bbox(lhand_bbox,
                                                  img_width=img_shape[1],
                                                  img_height=img_shape[0],
                                                  ratio=bbox_ratio)
                    if lhand_bbox is not None:
                        lhand_bbox[2:] += lhand_bbox[:2]  # xywh -> xyxy
                else:
                    lhand_bbox = None
                if rhand_bbox[-1] > 0:
                    rhand_bbox = rhand_bbox[:4]
                    if hasattr(cfg, 'bbox_ratio'):
                        rhand_bbox = process_bbox(rhand_bbox,
                                                  img_width=img_shape[1],
                                                  img_height=img_shape[0],
                                                  ratio=bbox_ratio)
                    if rhand_bbox is not None:
                        rhand_bbox[2:] += rhand_bbox[:2]  # xywh -> xyxy
                else:
                    rhand_bbox = None
                if face_bbox[-1] > 0:
                    face_bbox = face_bbox[:4]
                    if hasattr(cfg, 'bbox_ratio'):
                        face_bbox = process_bbox(face_bbox,
                                                 img_width=img_shape[1],
                                                 img_height=img_shape[0],
                                                 ratio=bbox_ratio)
                    if face_bbox is not None:
                        face_bbox[2:] += face_bbox[:2]  # xywh -> xyxy
                else:
                    face_bbox = None
                lhand_bbox_list.append(lhand_bbox)
                rhand_bbox_list.append(rhand_bbox)
                face_bbox_list.append(face_bbox)
            # import ipdb;ipdb.set_trace()
            # lhand_bbox = np.stack(lhand_bbox_list,axis=0)
            # rhand_bbox = np.stack(rhand_bbox_list,axis=0)
            # face_bbox = np.stack(face_bbox_list,axis=0)
            joint_img = keypoints2d[valid_idx]
            
            # num_joints = joint_cam.shape[0]
            # joint_valid = np.ones((num_joints, 1))
            if valid_kps3d:
                joint_cam = keypoints3d[valid_idx]
            else:
                joint_cam = None
            # import ipdb;ipdb.set_trace()
            if 'leye_pose_0' in smplx.keys():
                smplx.pop('leye_pose_0')
            if 'leye_pose_1' in smplx.keys():
                smplx.pop('leye_pose_1')
            if 'leye_pose' in smplx.keys():
                smplx.pop('leye_pose')
            if 'reye_pose_0' in smplx.keys():
                smplx.pop('reye_pose_0')
            if 'reye_pose_1' in smplx.keys():
                smplx.pop('reye_pose_1')
            if 'reye_pose' in smplx.keys():
                smplx.pop('reye_pose')
            
            occlusion_frame = occlusion[valid_idx] \
                if occlusion is not None else np.array([1]*(valid_num))

            smplx_param = {k: v[valid_idx] for k, v in smplx.items()}
            gender_ = gender[valid_idx] \
                if gender is not None else np.array(['neutral']*(valid_num))
                
            is_kid_ = is_kid[valid_idx] \
                if is_kid is not None else np.array([1]*(valid_num))
            lhand_bbox_valid = lhand_bbox_xywh[valid_idx,4]
            rhand_bbox_valid = rhand_bbox_xywh[valid_idx,4]
            face_bbox_valid = face_bbox_xywh[valid_idx,4]
            
            # TODO: set invalid if None?
            smplx_param['root_pose'] = smplx_param.pop('global_orient', None)
            smplx_param['shape'] = smplx_param.pop('betas', None)
            smplx_param['trans'] = smplx_param.pop('transl', np.zeros(3))
            smplx_param['lhand_pose'] = smplx_param.pop('left_hand_pose', None)
            smplx_param['rhand_pose'] = smplx_param.pop(
                'right_hand_pose', None)
            smplx_param['expr'] = smplx_param.pop('expression', None)

            # TODO do not fix betas, give up shape supervision
            if 'betas_neutral' in smplx_param and self.data_split == 'train':
                smplx_param['shape'] = smplx_param.pop('betas_neutral')
                # smplx_param['shape'] = np.zeros(10, dtype=np.float32)

            # # TODO fix shape of poses
            if self.__class__.__name__ == 'Talkshow':
                smplx_param['body_pose'] = smplx_param['body_pose'].reshape(
                    -1, 21, 3)
                smplx_param['lhand_pose'] = smplx_param['lhand_pose'].reshape(
                    -1, 15, 3)
                smplx_param['rhand_pose'] = smplx_param['lhand_pose'].reshape(
                    -1, 15, 3)
                smplx_param['expr'] = smplx_param['expr'][:, :10]

            if self.__class__.__name__ == 'BEDLAM':
                smplx_param['shape'] = smplx_param['shape'][:, :10]
            if self.__class__.__name__ == 'COCO_NA':
                smplx_param['expr'] = None

            if as_smplx == 'smpl':
                smplx_param['shape'] = np.zeros(
                    [valid_num, 10],
                    dtype=np.float32)  # drop smpl betas for smplx
                smplx_param['body_pose'] = smplx_param[
                    'body_pose'][:, :21, :]  # use smpl body_pose on smplx
            if as_smplx == 'smplh':
                smplx_param['shape'] = np.zeros(
                    [valid_num, 10],
                    dtype=np.float32)  # drop smpl betas for smplx

            if smplx_param['lhand_pose'] is None or self.body_only == True:
                smplx_param['lhand_valid'] = np.zeros(valid_num, dtype=np.bool8)
            else:
                smplx_param['lhand_valid'] = lhand_bbox_valid.astype(np.bool8)
            if smplx_param['rhand_pose'] is None or self.body_only == True:
                smplx_param['rhand_valid'] = np.zeros(valid_num, dtype=np.bool8)
            else:
                smplx_param['rhand_valid'] = rhand_bbox_valid.astype(np.bool8)
            if smplx_param['expr'] is None or self.body_only == True:
                smplx_param['face_valid'] = np.zeros(valid_num, dtype=np.bool8)
            else:
                smplx_param['face_valid'] = face_bbox_valid.astype(np.bool8)

            if joint_cam is not None and np.any(np.isnan(joint_cam)):
                continue
            
            
            datalist.append({
                'img_path': img_path,
                'img_shape': img_shape,
                'bbox': body_bbox_list,
                'lhand_bbox': lhand_bbox_list,
                'rhand_bbox': rhand_bbox_list,
                'face_bbox': face_bbox_list,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'smplx_param': smplx_param,
                'as_smplx': as_smplx,
                'gender': gender_,
                'occlusion': occlusion_frame,
                'is_kid': is_kid_,
            })

        # save memory
        del content, image_path, bbox_xywh, lhand_bbox_xywh, rhand_bbox_xywh, face_bbox_xywh, keypoints3d, keypoints2d

        if self.data_split == 'train':
            print(f'[{self.__class__.__name__} train] original size:',
                  int(num_examples), '. Sample interval:',
                  train_sample_interval, '. Sampled size:', len(datalist))

        if getattr(cfg, 'data_strategy',
                   None) == 'balance' and self.data_split == 'train':
            print(
                f'[{self.__class__.__name__}] Using [balance] strategy with datalist shuffled...'
            )
            random.shuffle(datalist)

        return datalist
   
    def __getitem__(self, idx):
        try:
            data = copy.deepcopy(self.datalist[idx])
        except Exception as e:
            print(f'[{self.__class__.__name__}] Error loading data {idx}')
            print(e)
            exit(0)

        img_path, img_shape, bbox = \
            data['img_path'], data['img_shape'], data['bbox']
        as_smplx = data['as_smplx']
        gender = data['gender'].copy()
        for gender_str, gender_num in {
            'neutral': -1, 'male': 0, 'female': 1}.items():
            gender[gender==gender_str]=gender_num
        gender = gender.astype(int)    
        
        img_whole_bbox = np.array([0, 0, img_shape[1], img_shape[0]])
        # img.shape: h,w,c, e.g. 1080,1920
        # img_shape: h,w
        # bbox: x,y,w,h
        # cropped_img_shape=np.array([img_whole_bbox[3],img_whole_bbox[2]])

        # self.normalize will convert the order
        # for ida in range(100):
        #     if os.path.exists('path%d.txt'%ida):
        #         continue
        #     else:
        #         with open('path%d.txt'%ida,'w') as f:
        #             f.writelines(img_path)
        #         break
        
        img = load_img(img_path, order='BGR')
        # if self.data_split == 'test':
        #     cv2.imwrite('temp.png',img)
        num_person = len(data['bbox'])
        data_name = self.__class__.__name__
        img, img2bb_trans, bb2img_trans, rot, do_flip = \
            augmentation_instance_sample(img, img_whole_bbox, self.data_split,data,data_name)
        cropped_img_shape=img.shape[:2]
        
        num_person = len(data['bbox'])
        if self.data_split == 'train':
            joint_cam = data['joint_cam']  # num, 137,4
            if joint_cam is not None:
                dummy_cord = False
                joint_cam[:,:,:3] = \
                    joint_cam[:,:,:3] - joint_cam[:, self.joint_set['root_joint_idx'], None, :3]  # root-relative
            else:
                # dummy cord as joint_cam
                dummy_cord = True
                joint_cam = np.zeros(
                    (num_person, self.joint_set['joint_num'], 4),
                    dtype=np.float32)

            joint_img = data['joint_img']
            # TODO
            # do rotation on keypoints
            joint_img_aug, joint_cam_wo_ra, joint_cam_ra, joint_trunc = \
                process_db_coord_batch_no_valid(
                    joint_img, joint_cam, do_flip, img_shape,
                    self.joint_set['flip_pairs'], img2bb_trans, rot,
                    self.joint_set['joints_name'], smpl_x.joints_name,
                    cropped_img_shape)
            # smplx coordinates and parameters
            smplx_param = data['smplx_param']
            smplx_pose, smplx_shape, smplx_expr, smplx_pose_valid, \
            smplx_joint_valid, smplx_expr_valid, smplx_shape_valid = \
                process_human_model_output_batch_simplify(
                    smplx_param, do_flip, rot, as_smplx)
            # if cam not provided, we take joint_img as smplx joint 2d, 
            # which is commonly the case for our processed humandata
            
            # TODO temp fix keypoints3d for renbody
            
            # change smplx_shape if use_betas_neutral
            # processing follows that in process_human_model_output
            
            if self.use_betas_neutral:
                smplx_shape = smplx_param['betas_neutral'].reshape(
                    num_person, -1)
                smplx_shape[(np.abs(smplx_shape) > 3).any(axis=1)] = 0.
                smplx_shape = smplx_shape.reshape(num_person, -1)

            # SMPLX pose parameter validity
            # for name in ('L_Ankle', 'R_Ankle', 'L_Wrist', 'R_Wrist'):
            #     smplx_pose_valid[smpl_x.orig_joints_name.index(name)] = 0
            #import ipdb;ipdb.set_trace()

            smplx_pose_valid = np.tile(smplx_pose_valid[:,:, None], (1, 3)).reshape(num_person,-1)
            
            # SMPLX joint coordinate validity
            # for name in ('L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel'):
            #     smplx_joint_valid[smpl_x.joints_name.index(name)] = 0
            smplx_joint_valid = smplx_joint_valid[:, :, None]
            # import ipdb;ipdb.set_trace()

                
            # import ipdb;ipdb.set_trace()
            # TODO: check here
            # if not (smplx_shape == 0).all():
            #     smplx_shape_valid = True
            # else:
            #     smplx_shape_valid = False
            lhand_bbox_center_list = []
            lhand_bbox_valid_list = []
            lhand_bbox_size_list = []
            lhand_bbox_list = []
            face_bbox_center_list = []
            face_bbox_size_list = []
            face_bbox_valid_list = []
            face_bbox_list = []
            rhand_bbox_center_list = []
            rhand_bbox_valid_list = []
            rhand_bbox_size_list = []
            rhand_bbox_list = []
            body_bbox_center_list = []
            body_bbox_size_list = []
            body_bbox_valid_list = []
            body_bbox_list = []
            # hand and face bbox transform
            # import ipdb;ipdb.set_trace()

            for i in range(num_person):
                lhand_bbox, lhand_bbox_valid = self.process_hand_face_bbox(
                    data['lhand_bbox'][i], do_flip, img_shape, img2bb_trans,
                    cropped_img_shape)
                rhand_bbox, rhand_bbox_valid = self.process_hand_face_bbox(
                    data['rhand_bbox'][i], do_flip, img_shape, img2bb_trans,
                    cropped_img_shape)
                face_bbox, face_bbox_valid = self.process_hand_face_bbox(
                    data['face_bbox'][i], do_flip, img_shape, img2bb_trans,
                    cropped_img_shape)
                # import ipdb;ipdb.set_trace()
                body_bbox, body_bbox_valid = self.process_hand_face_bbox(
                    data['bbox'][i], do_flip, img_shape, img2bb_trans,
                    cropped_img_shape)
                
                if do_flip:
                    lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
                    lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid
                    
                body_bbox_list.append(body_bbox)
                lhand_bbox_list.append(lhand_bbox)
                rhand_bbox_list.append(rhand_bbox)
                face_bbox_list.append(face_bbox)
                
                lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1]) / 2.
                rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1]) / 2.
                face_bbox_center = (face_bbox[0] + face_bbox[1]) / 2.
                body_bbox_center = (body_bbox[0] + body_bbox[1]) / 2.
                lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0]
                rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0]

                face_bbox_size = face_bbox[1] - face_bbox[0]
                body_bbox_size = body_bbox[1] - body_bbox[0]
                lhand_bbox_center_list.append(lhand_bbox_center)
                lhand_bbox_valid_list.append(lhand_bbox_valid)
                lhand_bbox_size_list.append(lhand_bbox_size)
                face_bbox_center_list.append(face_bbox_center)
                face_bbox_size_list.append(face_bbox_size)
                face_bbox_valid_list.append(face_bbox_valid)
                rhand_bbox_center_list.append(rhand_bbox_center)
                rhand_bbox_valid_list.append(rhand_bbox_valid)
                rhand_bbox_size_list.append(rhand_bbox_size)
                body_bbox_center_list.append(body_bbox_center)
                body_bbox_size_list.append(body_bbox_size)
                body_bbox_valid_list.append(body_bbox_valid)
            # import ipdb;ipdb.set_trace()
            
            body_bbox = np.stack(body_bbox_list, axis=0)
            lhand_bbox = np.stack(lhand_bbox_list, axis=0)
            rhand_bbox = np.stack(rhand_bbox_list, axis=0)
            face_bbox = np.stack(face_bbox_list, axis=0)
            lhand_bbox_center = np.stack(lhand_bbox_center_list, axis=0)
            lhand_bbox_valid = np.stack(lhand_bbox_valid_list, axis=0)
            lhand_bbox_size = np.stack(lhand_bbox_size_list, axis=0)
            face_bbox_center = np.stack(face_bbox_center_list, axis=0)
            face_bbox_size = np.stack(face_bbox_size_list, axis=0)
            face_bbox_valid = np.stack(face_bbox_valid_list, axis=0)
            body_bbox_center = np.stack(body_bbox_center_list, axis=0)
            body_bbox_size = np.stack(body_bbox_size_list, axis=0)
            body_bbox_valid = np.stack(body_bbox_valid_list, axis=0)
            rhand_bbox_center = np.stack(rhand_bbox_center_list, axis=0)
            rhand_bbox_valid = np.stack(rhand_bbox_valid_list, axis=0)
            rhand_bbox_size = np.stack(rhand_bbox_size_list, axis=0)


            if 'occlusion' in data:
                occlusion = data['occlusion']
                occ_mask = occlusion<97
                # import ipdb;ipdb.set_trace()
                joint_img_aug[:,:,2] = joint_img_aug[:,:,2]*occ_mask[:,None]
                joint_cam_wo_ra[:,:,3] = joint_cam_wo_ra[:,:,3]*occ_mask[:,None]
                joint_trunc = joint_trunc*occ_mask[:,None,None]
                smplx_pose_valid = smplx_pose_valid*occ_mask[:,None]
                smplx_joint_valid = smplx_joint_valid*occ_mask[:,None,None]
                smplx_expr_valid = smplx_expr_valid*occ_mask
                smplx_shape_valid = smplx_shape_valid*occ_mask
                rhand_bbox_valid = rhand_bbox_valid*occ_mask
                lhand_bbox_valid = lhand_bbox_valid*occ_mask
                face_bbox_valid = face_bbox_valid*occ_mask
                
            
            if 'is_kid' in data:
                is_kid = data['is_kid'].copy()
                smplx_shape_valid = smplx_shape_valid * (is_kid==0)
                
                
            inputs = {'img': img}

            joint_img_aug[:,:,2] = joint_img_aug[:,:,2]*joint_trunc[:,:,0]*body_bbox_valid[:,None]
            
            is_3D = float(False) if dummy_cord else float(True)
            if self.__class__.__name__ == 'COCO_NA':
                is_3D = False
            if self.__class__.__name__ == 'GTA_Human2':
                smplx_shape_valid = smplx_shape_valid*0
            # if body_bbox_valid.sum() > 0:
            # import ipdb;ipdb.set_trace()
            # import ipdb;ipdb.set_trace()
            targets = {
                # keypoints2d, [0,img_w],[0,img_h] -> [0,1] -> [0,output_hm_shape]
                'joint_img': joint_img_aug[body_bbox_valid>0], 
                # joint_cam, kp3d wo ra # raw kps3d probably without ra
                'joint_cam': joint_cam_wo_ra[body_bbox_valid>0], 
                # kps3d with body, face, hand ra
                'smplx_joint_cam': joint_cam_ra[body_bbox_valid>0], 
                'smplx_pose': smplx_pose[body_bbox_valid>0],
                'smplx_shape': smplx_shape[body_bbox_valid>0],
                'smplx_expr': smplx_expr[body_bbox_valid>0],
                'lhand_bbox_center': lhand_bbox_center[body_bbox_valid>0], 
                'lhand_bbox_size': lhand_bbox_size[body_bbox_valid>0],
                'rhand_bbox_center': rhand_bbox_center[body_bbox_valid>0], 
                'rhand_bbox_size': rhand_bbox_size[body_bbox_valid>0],
                'face_bbox_center': face_bbox_center[body_bbox_valid>0], 
                'face_bbox_size': face_bbox_size[body_bbox_valid>0],
                'body_bbox_center': body_bbox_center[body_bbox_valid>0], 
                'body_bbox_size': body_bbox_size[body_bbox_valid>0],
                'body_bbox': body_bbox.reshape(-1,4)[body_bbox_valid>0],
                'lhand_bbox': lhand_bbox.reshape(-1,4)[body_bbox_valid>0],
                'rhand_bbox': rhand_bbox.reshape(-1,4)[body_bbox_valid>0],
                'face_bbox': face_bbox.reshape(-1,4)[body_bbox_valid>0],
                'gender': gender[body_bbox_valid>0]}
            
            meta_info = {
                'joint_trunc': joint_trunc[body_bbox_valid>0],
                'smplx_pose_valid': smplx_pose_valid[body_bbox_valid>0],
                'smplx_shape_valid': smplx_shape_valid[body_bbox_valid>0],
                'smplx_expr_valid': smplx_expr_valid[body_bbox_valid>0],
                'is_3D': is_3D, 
                'lhand_bbox_valid': lhand_bbox_valid[body_bbox_valid>0],
                'rhand_bbox_valid': rhand_bbox_valid[body_bbox_valid>0], 
                'face_bbox_valid': face_bbox_valid[body_bbox_valid>0],
                'body_bbox_valid': body_bbox_valid[body_bbox_valid>0],
                'img_shape': np.array(img.shape[:2]), 
                'ori_shape':data['img_shape'],
                'idx': idx
               
            }
            # else:
            #     # no person in image, take only one
            #     targets = {
            #         # keypoints2d, [0,img_w],[0,img_h] -> [0,1] -> [0,output_hm_shape]
            #         'joint_img': joint_img_aug[:1], 
            #         # joint_cam, kp3d wo ra # raw kps3d probably without ra
            #         'joint_cam': joint_cam_wo_ra[:1], 
            #         # kps3d with body, face, hand ra
            #         'smplx_joint_cam': joint_cam_ra[:1], 
            #         'smplx_pose': smplx_pose[:1],
            #         'smplx_shape': smplx_shape[:1],
            #         'smplx_expr': smplx_expr[:1],
            #         'lhand_bbox_center': lhand_bbox_center[:1], 
            #         'lhand_bbox_size': lhand_bbox_size[:1],
            #         'rhand_bbox_center': rhand_bbox_center[:1], 
            #         'rhand_bbox_size': rhand_bbox_size[:1],
            #         'face_bbox_center': face_bbox_center[:1], 
            #         'face_bbox_size': face_bbox_size[:1],
            #         'body_bbox_center': body_bbox_center[:1], 
            #         'body_bbox_size': body_bbox_size[:1],
            #         'body_bbox': body_bbox.reshape(-1,4)[:1],
            #         'lhand_bbox': body_bbox.reshape(-1,4)[:1],
            #         'rhand_bbox': body_bbox.reshape(-1,4)[:1],
            #         'face_bbox': body_bbox.reshape(-1,4)[:1],
            #         'gender': gender[:1]}

            #     meta_info = {
            #         'joint_trunc': joint_trunc[:1]*0,
            #         'smplx_pose_valid': smplx_pose_valid[:1]*0,
            #         'smplx_shape_valid': float(smplx_shape_valid)*0,
            #         'smplx_expr_valid': smplx_expr_valid[:1]*0,
            #         'is_3D': is_3D*0, 
            #         'lhand_bbox_valid': lhand_bbox_valid[:1]*0,
            #         'rhand_bbox_valid': rhand_bbox_valid[:1]*0, 
            #         'face_bbox_valid': face_bbox_valid[:1]*0,
            #         'body_bbox_valid': body_bbox_valid[:1]*0,
            #         'img_shape': np.array(img.shape[:2]), 
            #         'ori_shape':data['img_shape']
            #     }
            # if self.body_only:
            #     meta_info = {
            #         'joint_trunc': joint_trunc,
            #         'smplx_pose_valid': smplx_pose_valid,
            #         'smplx_shape_valid': float(smplx_shape_valid),
            #         'smplx_expr_valid': smplx_expr_valid,
            #         'is_3D': is_3D, 
            #         'lhand_bbox_valid': lhand_bbox_valid,
            #         'rhand_bbox_valid': rhand_bbox_valid, 
            #         'face_bbox_valid': face_bbox_valid,
            #         'body_bbox_valid': body_bbox_valid,
            #         'img_shape': np.array(img.shape[:2]), 
            #         'ori_shape':data['img_shape'],
            #     }
            # else:
                # meta_info = {
                #     'joint_trunc': joint_trunc,
                #     'smplx_pose_valid': smplx_pose_valid,
                #     'smplx_shape_valid': float(smplx_shape_valid),
                #     'smplx_expr_valid': smplx_expr_valid,
                #     'is_3D': is_3D, 
                #     'lhand_bbox_valid': lhand_bbox_valid,
                #     'rhand_bbox_valid': rhand_bbox_valid, 
                #     'face_bbox_valid': face_bbox_valid,
                #     'body_bbox_valid': body_bbox_valid,
                #     'img_shape': np.array(img.shape[:2]), 
                #     'ori_shape':data['img_shape']
                # }
            
            result = {**inputs, **targets, **meta_info}
            
            result = self.normalize(result)
            result = self.format(result)
            return result

        

        if self.data_split == 'test':
            self.cam_param = {}
            joint_cam = data['joint_cam']
            # import ipdb;ipdb.set_trace()
            if joint_cam is not None:
                dummy_cord = False
                joint_cam[:,:,:3] = joint_cam[:,:,:3] - joint_cam[
                    :, self.joint_set['root_joint_idx'], None, :3]  # root-relative
            else:
                # dummy cord as joint_cam
                dummy_cord = True
                joint_cam = np.zeros(
                    (num_person, self.joint_set['joint_num'], 3),
                                     dtype=np.float32)

            joint_img = data['joint_img']
            
            
            joint_img_aug, joint_cam_wo_ra, joint_cam_ra, joint_trunc = \
                process_db_coord_batch_no_valid(
                    joint_img, joint_cam, do_flip, img_shape,
                    self.joint_set['flip_pairs'], img2bb_trans, rot,
                    self.joint_set['joints_name'], smpl_x.joints_name,
                    cropped_img_shape)
            
            

            # smplx coordinates and parameters
            smplx_param = data['smplx_param']
            # smplx_cam_trans = np.array(
            #     smplx_param['trans']) if 'trans' in smplx_param else None
            # TODO: remove this, seperate smpl and smplx
            smplx_pose, smplx_shape, smplx_expr, smplx_pose_valid, \
            smplx_joint_valid, smplx_expr_valid, smplx_shape_valid = \
                process_human_model_output_batch_simplify(
                    smplx_param, do_flip, rot, as_smplx)
            
            # if cam not provided, we take joint_img as smplx joint 2d, 
            # which is commonly the case for our processed humandata
            if self.use_betas_neutral:
                smplx_shape = smplx_param['betas_neutral'].reshape(
                    num_person, -1)
                smplx_shape[(np.abs(smplx_shape) > 3).any(axis=1)] = 0.
                smplx_shape = smplx_shape.reshape(num_person, -1)
            smplx_pose_valid = np.tile(smplx_pose_valid[:,:, None], (1, 3)).reshape(num_person,-1)
            smplx_joint_valid = smplx_joint_valid[:, :, None]

            # if not (smplx_shape == 0).all():
            #     smplx_shape_valid = True
            # else:
            #     smplx_shape_valid = False
            lhand_bbox_center_list = []
            lhand_bbox_valid_list = []
            lhand_bbox_size_list = []
            lhand_bbox_list = []
            face_bbox_center_list = []
            face_bbox_size_list = []
            face_bbox_valid_list = []
            face_bbox_list = []
            rhand_bbox_center_list = []
            rhand_bbox_valid_list = []
            rhand_bbox_size_list = []
            rhand_bbox_list = []
            body_bbox_center_list = []
            body_bbox_size_list = []
            body_bbox_valid_list = []
            body_bbox_list = []
                        
            for i in range(num_person):
                lhand_bbox, lhand_bbox_valid = self.process_hand_face_bbox(
                    data['lhand_bbox'][i], do_flip, img_shape, img2bb_trans,
                    cropped_img_shape)
                rhand_bbox, rhand_bbox_valid = self.process_hand_face_bbox(
                    data['rhand_bbox'][i], do_flip, img_shape, img2bb_trans,
                    cropped_img_shape)
                face_bbox, face_bbox_valid = self.process_hand_face_bbox(
                    data['face_bbox'][i], do_flip, img_shape, img2bb_trans,
                    cropped_img_shape)
                # import ipdb;ipdb.set_trace()
                body_bbox, body_bbox_valid = self.process_hand_face_bbox(
                    data['bbox'][i], do_flip, img_shape, img2bb_trans,
                    cropped_img_shape)                

                if do_flip:
                    lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
                    lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid            

                body_bbox_list.append(body_bbox)
                lhand_bbox_list.append(lhand_bbox)
                rhand_bbox_list.append(rhand_bbox)
                face_bbox_list.append(face_bbox)

                lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1]) / 2.
                rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1]) / 2.
                face_bbox_center = (face_bbox[0] + face_bbox[1]) / 2.
                body_bbox_center = (body_bbox[0] + body_bbox[1]) / 2.
                lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0]
                rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0]

                face_bbox_size = face_bbox[1] - face_bbox[0]
                body_bbox_size = body_bbox[1] - body_bbox[0]
                lhand_bbox_center_list.append(lhand_bbox_center)
                lhand_bbox_valid_list.append(lhand_bbox_valid)
                lhand_bbox_size_list.append(lhand_bbox_size)
                face_bbox_center_list.append(face_bbox_center)
                face_bbox_size_list.append(face_bbox_size)
                face_bbox_valid_list.append(face_bbox_valid)
                rhand_bbox_center_list.append(rhand_bbox_center)
                rhand_bbox_valid_list.append(rhand_bbox_valid)
                rhand_bbox_size_list.append(rhand_bbox_size)
                body_bbox_center_list.append(body_bbox_center)
                body_bbox_size_list.append(body_bbox_size)
                body_bbox_valid_list.append(body_bbox_valid)

            body_bbox = np.stack(body_bbox_list, axis=0)
            lhand_bbox = np.stack(lhand_bbox_list, axis=0)
            rhand_bbox = np.stack(rhand_bbox_list, axis=0)
            face_bbox = np.stack(face_bbox_list, axis=0)
            lhand_bbox_center = np.stack(lhand_bbox_center_list, axis=0)
            lhand_bbox_valid = np.stack(lhand_bbox_valid_list, axis=0)
            lhand_bbox_size = np.stack(lhand_bbox_size_list, axis=0)
            face_bbox_center = np.stack(face_bbox_center_list, axis=0)
            face_bbox_size = np.stack(face_bbox_size_list, axis=0)
            face_bbox_valid = np.stack(face_bbox_valid_list, axis=0)
            body_bbox_center = np.stack(body_bbox_center_list, axis=0)
            body_bbox_size = np.stack(body_bbox_size_list, axis=0)
            body_bbox_valid = np.stack(body_bbox_valid_list, axis=0)
            rhand_bbox_center = np.stack(rhand_bbox_center_list, axis=0)
            rhand_bbox_valid = np.stack(rhand_bbox_valid_list, axis=0)
            rhand_bbox_size = np.stack(rhand_bbox_size_list, axis=0)
                                            
                            
            inputs = {'img': img}
            
            targets = {
                # keypoints2d, [0,img_w],[0,img_h] -> [0,1] -> [0,output_hm_shape]
                'joint_img': joint_img_aug, 
                # projected smplx if valid cam_param, else same as keypoints2d
                # joint_cam, kp3d wo ra # raw kps3d probably without ra
                'joint_cam': joint_cam_wo_ra, 
                'ann_idx': idx,
                # kps3d with body, face, hand ra
                'smplx_joint_cam': joint_cam_ra,
                'smplx_pose': smplx_pose,
                'smplx_shape': smplx_shape,
                'smplx_expr': smplx_expr,
                'lhand_bbox_center': lhand_bbox_center, 
                'lhand_bbox_size': lhand_bbox_size,
                'rhand_bbox_center': rhand_bbox_center, 
                'rhand_bbox_size': rhand_bbox_size,
                'face_bbox_center': face_bbox_center, 
                'face_bbox_size': face_bbox_size,
                'body_bbox_center': body_bbox_center, 
                'body_bbox_size': body_bbox_size,
                'body_bbox': body_bbox.reshape(-1,4),
                'lhand_bbox': lhand_bbox.reshape(-1,4),
                'rhand_bbox': rhand_bbox.reshape(-1,4),
                'face_bbox': face_bbox.reshape(-1,4),
                'gender': gender,
                'bb2img_trans': bb2img_trans,
            }
            
            if self.body_only:
                meta_info = {
                    'joint_trunc': joint_trunc,
                    'smplx_pose_valid': smplx_pose_valid,
                    'smplx_shape_valid': float(smplx_shape_valid),
                    'smplx_expr_valid': smplx_expr_valid,
                    'is_3D': float(False) if dummy_cord else float(True), 
                    'lhand_bbox_valid': lhand_bbox_valid,
                    'rhand_bbox_valid': rhand_bbox_valid, 
                    'face_bbox_valid': face_bbox_valid,
                    'body_bbox_valid': body_bbox_valid,
                    'img_shape': np.array(img.shape[:2]), 
                    'ori_shape':data['img_shape'],
                    'idx': idx
                }
            else:
                meta_info = {
                    'joint_trunc': joint_trunc,
                    'smplx_pose_valid': smplx_pose_valid,
                    'smplx_shape_valid': smplx_shape_valid,
                    'smplx_expr_valid': smplx_expr_valid,
                    'is_3D': float(False) if dummy_cord else float(True), 
                    'lhand_bbox_valid': lhand_bbox_valid,
                    'rhand_bbox_valid': rhand_bbox_valid, 
                    'face_bbox_valid': face_bbox_valid,
                    'body_bbox_valid': body_bbox_valid,
                    'img_shape': np.array(img.shape[:2]), 
                    'ori_shape':data['img_shape'],
                    'idx': idx
                   }
            
            result = {**inputs, **targets, **meta_info}
            result = self.normalize(result)
            result = self.format(result)
            return result
        
    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {
            'pa_mpvpe_all': [],
            'pa_mpvpe_l_hand': [],
            'pa_mpvpe_r_hand': [],
            'pa_mpvpe_hand': [],
            'pa_mpvpe_face': [],
            'mpvpe_all': [],
            'mpvpe_l_hand': [],
            'mpvpe_r_hand': [],
            'mpvpe_hand': [],
            'mpvpe_face': []
        }

        vis = getattr(cfg, 'vis', False)
        vis_save_dir = cfg.vis_dir
        
        csv_file = f'{cfg.result_dir}/agora_smplx_error.csv'
        file = open(csv_file, 'a', newline='')
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]
            mesh_gt = out['smplx_mesh_cam_target']
            mesh_out = out['smplx_mesh_cam']
            
            # print('zzz',mesh_gt.shape,mesh_out.shape)
            # from pytorch3d.io import save_obj
            # for m_i,(mesh_gt_i,mesh_out_i) in enumerate(zip(mesh_gt,mesh_out)):
            #     save_obj('temp_gt_%d.obj'%m_i,verts=torch.Tensor(mesh_gt_i),faces=torch.tensor([]))
            #     save_obj('temp_pred_%d.obj'%m_i,verts=torch.Tensor(mesh_out_i),faces=torch.tensor([]))
            
            ann_idx = out['gt_ann_idx']
            img_path = []
            for ann_id in ann_idx:
                img_path.append(annots[ann_id]['img_path'])
            eval_result['img_path'] = img_path
            eval_result['ann_idx'] = ann_idx
            # MPVPE from all vertices
            mesh_out_align = \
                mesh_out - np.dot(
                    smpl_x.J_regressor, mesh_out).transpose(1,0,2)[:, smpl_x.J_regressor_idx['pelvis'], None, :] + \
                    np.dot(smpl_x.J_regressor, mesh_gt).transpose(1,0,2)[:, smpl_x.J_regressor_idx['pelvis'], None, :]
            
            eval_result['mpvpe_all'].extend(
                np.sqrt(np.sum(
                    (mesh_out_align - mesh_gt)**2, -1)).mean(-1) * 1000)
            mesh_out_align = rigid_align_batch(mesh_out, mesh_gt)
            eval_result['pa_mpvpe_all'].extend(
                np.sqrt(np.sum(
                    (mesh_out_align - mesh_gt)**2, -1)).mean(-1) * 1000)

            # MPVPE from hand vertices
            mesh_gt_lhand = mesh_gt[:, smpl_x.hand_vertex_idx['left_hand'], :]
            mesh_out_lhand = mesh_out[:, smpl_x.hand_vertex_idx['left_hand'], :]
            mesh_gt_rhand = mesh_gt[:, smpl_x.hand_vertex_idx['right_hand'], :]
            mesh_out_rhand = mesh_out[:, smpl_x.hand_vertex_idx['right_hand'], :]
            mesh_out_lhand_align = \
                mesh_out_lhand - \
                np.dot(smpl_x.J_regressor, mesh_out).transpose(1,0,2)[:, smpl_x.J_regressor_idx['lwrist'], None, :] + \
                np.dot(smpl_x.J_regressor, mesh_gt).transpose(1,0,2)[:, smpl_x.J_regressor_idx['lwrist'], None, :]
                    
            mesh_out_rhand_align = \
                mesh_out_rhand - \
                np.dot(smpl_x.J_regressor, mesh_out).transpose(1,0,2)[:, smpl_x.J_regressor_idx['rwrist'], None, :] + \
                np.dot(smpl_x.J_regressor, mesh_gt).transpose(1,0,2)[:, smpl_x.J_regressor_idx['rwrist'], None, :]
            
            eval_result['mpvpe_l_hand'].extend(
                np.sqrt(np.sum(
                    (mesh_out_lhand_align - mesh_gt_lhand)**2, -1)).mean(-1) *
                1000)
            eval_result['mpvpe_r_hand'].extend(
                np.sqrt(np.sum(
                    (mesh_out_rhand_align - mesh_gt_rhand)**2, -1)).mean(-1) *
                1000)
            eval_result['mpvpe_hand'].extend(
                (np.sqrt(np.sum(
                    (mesh_out_lhand_align - mesh_gt_lhand)**2, -1)).mean(-1) *
                 1000 +
                 np.sqrt(np.sum(
                     (mesh_out_rhand_align - mesh_gt_rhand)**2, -1)).mean(-1) *
                 1000) / 2.)
            mesh_out_lhand_align = rigid_align_batch(mesh_out_lhand, mesh_gt_lhand)
            mesh_out_rhand_align = rigid_align_batch(mesh_out_rhand, mesh_gt_rhand)
            eval_result['pa_mpvpe_l_hand'].extend(
                np.sqrt(np.sum(
                    (mesh_out_lhand_align - mesh_gt_lhand)**2, -1)).mean(-1) *
                1000)
            eval_result['pa_mpvpe_r_hand'].extend(
                np.sqrt(np.sum(
                    (mesh_out_rhand_align - mesh_gt_rhand)**2, -1)).mean(-1) *
                1000)
            eval_result['pa_mpvpe_hand'].extend(
                (np.sqrt(np.sum(
                    (mesh_out_lhand_align - mesh_gt_lhand)**2, -1)).mean(-1) *
                 1000 +
                 np.sqrt(np.sum(
                     (mesh_out_rhand_align - mesh_gt_rhand)**2, -1)).mean(-1) *
                 1000) / 2.)
            
            
            save_error=True
            if save_error:
                writer = csv.writer(file)
                new_line = [ann_idx[n],img_path[n], eval_result['mpvpe_all'][-1], eval_result['pa_mpvpe_all'][-1]]
                writer.writerow(new_line)
                self.save_idx += 1
            
            
        return eval_result


    def print_eval_result(self, eval_result):

        print('AGORA test results are dumped at: ' +
              osp.join(cfg.result_dir, 'predictions'))

        if self.data_split == 'test' and self.test_set == 'test':  # do not print. just submit the results to the official evaluation server
            return

        print('======AGORA-val======')
        print('PA MPVPE (All): %.2f mm' % np.mean(eval_result['pa_mpvpe_all']))
        print('PA MPVPE (L-Hands): %.2f mm' %
              np.mean(eval_result['pa_mpvpe_l_hand']))
        print('PA MPVPE (R-Hands): %.2f mm' %
              np.mean(eval_result['pa_mpvpe_r_hand']))
        print('PA MPVPE (Hands): %.2f mm' %
              np.mean(eval_result['pa_mpvpe_hand']))
        print('PA MPVPE (Face): %.2f mm' %
              np.mean(eval_result['pa_mpvpe_face']))
        print()

        print('MPVPE (All): %.2f mm' % np.mean(eval_result['mpvpe_all']))
        print('MPVPE (L-Hands): %.2f mm' %
              np.mean(eval_result['mpvpe_l_hand']))
        print('MPVPE (R-Hands): %.2f mm' %
              np.mean(eval_result['mpvpe_r_hand']))
        print('MPVPE (Hands): %.2f mm' % np.mean(eval_result['mpvpe_hand']))
        print('MPVPE (Face): %.2f mm' % np.mean(eval_result['mpvpe_face']))
        
        out_file = osp.join(cfg.result_dir,'agora_val.txt')
        if os.path.exists(out_file):
            f = open(out_file, 'a+')
        else:
            f = open(out_file, 'w', encoding="utf-8")
            
        f.write('\n')
        f.write(f'{cfg.exp_name}\n')            
        f.write(f'AGORA-val dataset: \n')
        f.write('PA MPVPE (All): %.2f mm\n' %
                np.mean(eval_result['pa_mpvpe_all']))
        f.write('PA MPVPE (L-Hands): %.2f mm\n' %
                np.mean(eval_result['pa_mpvpe_l_hand']))
        f.write('PA MPVPE (R-Hands): %.2f mm\n' %
                np.mean(eval_result['pa_mpvpe_r_hand']))
        f.write('PA MPVPE (Hands): %.2f mm\n' %
                np.mean(eval_result['pa_mpvpe_hand']))
        f.write('PA MPVPE (Face): %.2f mm\n' %
                np.mean(eval_result['pa_mpvpe_face']))
        f.write('MPVPE (All): %.2f mm\n' % np.mean(eval_result['mpvpe_all']))
        f.write('MPVPE (L-Hands): %.2f mm\n' %
                np.mean(eval_result['mpvpe_l_hand']))
        f.write('MPVPE (R-Hands): %.2f mm\n' %
                np.mean(eval_result['mpvpe_r_hand']))
        f.write('MPVPE (Hands): %.2f mm\n' % np.mean(eval_result['mpvpe_hand']))
        f.write('MPVPE (Face): %.2f mm\n' % np.mean(eval_result['mpvpe_face']))
