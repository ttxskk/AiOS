import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
from config.config import cfg
from util.human_models import smpl_x
from util.preprocessing import (
    load_img, process_bbox, augmentation_instance_sample, process_human_model_output_batch_simplify,process_db_coord_batch_no_valid)
from util.transforms import world2cam, cam2pixel, rigid_align
from detrsmpl.utils.geometry import batch_rodrigues, project_points_new, weak_perspective_projection, perspective_projection
import tqdm
import time
import random
from detrsmpl.utils.demo_utils import box2cs, xywh2xyxy, xyxy2xywh
import torch.distributed as dist

KPS2D_KEYS = [
    'keypoints2d_ori', 'keypoints2d_smplx', 'keypoints2d_smpl',
    'keypoints2d_original','keypoints2d_gta','keypoints2d'
]
KPS3D_KEYS = [
    'keypoints3d_cam', 'keypoints3d', 'keypoints3d_smplx', 'keypoints3d_smpl',
    'keypoints3d_original', 'keypoints3d_gta','keypoints3d'
]
# keypoints3d_cam with root-align has higher priority, followed by old version key keypoints3d
# when there is keypoints3d_smplx, use this rather than keypoints3d_original

from util.formatting import DefaultFormatBundle
from detrsmpl.data.datasets.pipelines.transforms import Normalize

class Cache():
    """A custom implementation for OSX pipeline."""
    def __init__(self, load_path=None):
        if load_path is not None:
            self.load(load_path)

    def load(self, load_path):
        self.load_path = load_path
        self.cache = np.load(load_path, allow_pickle=True)
        self.data_len = self.cache['data_len']
        self.data_strategy = self.cache['data_strategy']
        assert self.data_len == len(self.cache) - 2  # data_len, data_strategy
        self.cache = None

    @classmethod
    def save(cls, save_path, data_list, data_strategy):
        assert save_path is not None, 'save_path is None'
        data_len = len(data_list)
        cache = {}
        for i, data in enumerate(data_list):
            cache[str(i)] = data
        assert len(cache) == data_len
        # update meta
        cache.update({'data_len': data_len, 'data_strategy': data_strategy})
        # import pdb; pdb.set_trace()
        np.savez_compressed(save_path, **cache)
        print(f'Cache saved to {save_path}.')

    # def shuffle(self):
    #     random.shuffle(self.mapping)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if self.cache is None:
            self.cache = np.load(self.load_path, allow_pickle=True)
        # mapped_idx = self.mapping[idx]
        # cache_data = self.cache[str(mapped_idx)]
        # print(self.cache.files)
        cache_data = self.cache[str(idx)]
        data = cache_data.item()
        return data


class HumanDataset(torch.utils.data.Dataset):

    # same mapping for 144->137 and 190->137
    SMPLX_137_MAPPING = [
        0, 1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21, 60, 61, 62, 63, 64,
        65, 59, 58, 57, 56, 55, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68,
        34, 35, 36, 69, 31, 32, 33, 70, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44,
        45, 73, 49, 50, 51, 74, 46, 47, 48, 75, 22, 15, 56, 57, 76, 77, 78, 79,
        80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
        98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
        126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
        140, 141, 142, 143
    ]

    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split

        # dataset information, to be filled by child class
        self.img_dir = None
        self.annot_path = None
        self.annot_path_cache = None
        self.use_cache = False
        self.img_shape = None  # (h, w)
        self.cam_param = None  # {'focal_length': (fx, fy), 'princpt': (cx, cy)}
        self.use_betas_neutral = False
        self.body_only = False
        self.joint_set = {
            'joint_num': smpl_x.joint_num,
            'joints_name': smpl_x.joints_name,
            'flip_pairs': smpl_x.flip_pairs
        }
        self.joint_set['root_joint_idx'] = self.joint_set['joints_name'].index(
            'Pelvis')
        self.format = DefaultFormatBundle()
        self.normalize = Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        self.keypoints2d = None
        # self.rank = dist.get_rank()
        self.lhand_mean = smpl_x.layer['neutral'].left_hand_mean.reshape(15, 3).cpu().numpy()
        self.rhand_mean = smpl_x.layer['neutral'].right_hand_mean.reshape(15, 3).cpu().numpy()
        # self.log_file_path = f'indices_node{rank}.txt'
    def load_cache(self, annot_path_cache):
        datalist = Cache(annot_path_cache)
        # assert datalist.data_strategy == getattr(cfg, 'data_strategy', None), \
        #     f'Cache data strategy {datalist.data_strategy} does not match current data strategy ' \
        #     f'{getattr(cfg, "data_strategy", None)}'
        return datalist

    def save_cache(self, annot_path_cache, datalist):
        print(
            f'[{self.__class__.__name__}] Caching datalist to {self.annot_path_cache}...'
        )
        Cache.save(annot_path_cache,
                   datalist,
                   data_strategy=getattr(cfg, 'data_strategy', None))

    def load_data(self, train_sample_interval=1,
                  hand_bbox_ratio=1, body_bbox_ratio=1):

        content = np.load(self.annot_path, allow_pickle=True)
        try:
            frame_range = content['frame_range']
        except KeyError:
            self.num_data = len(content['image_path'])
            frame_range = \
                np.array([[i, i + 1] for i in range(self.num_data)])

        num_examples = len(frame_range)
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
        if meta is not None and 'height' in meta and len(meta['height'])>0:
            height = np.array(meta['height'])
            width = np.array(meta['width'])
            image_shape = np.stack([height, width], axis=-1)
        else:
            image_shape = None

        if meta is not None and 'gender' in meta and len(meta['gender']) != 0:
            gender = np.array(meta['gender'])
        else:
            gender = None
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
        
        if meta is not None and 'smplx_valid' in meta:
            smplx_valid = meta['smplx_valid']
        else:
            smplx_valid = np.ones(len(bbox_xywh))
            
        decompressed = False
        if content['__keypoints_compressed__']:
            decompressed_kps = self.decompress_keypoints(content)
            decompressed = True

        keypoints3d = None
        valid_kps3d = False
        keypoints3d_mask = None
        valid_kps3d_mask = False

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
                    break
        if keypoints2d.shape[-1] == 3:
            valid_kps3d_mask = True
        
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
            
            valid_idx = []
            body_bbox_list = []
            
            # if hasattr(cfg, 'bbox_ratio'):
            #     bbox_ratio = cfg.bbox_ratio * 0.833  # preprocess body bbox is giving 1.2 box padding
            # else:
            #     bbox_ratio = 1.25
            # if self.__class__.__name__ == 'SPEC':
            #     bbox_ratio = 1.25
            
            for bbox_i, bbox in enumerate(bbox_list):
                
                bbox = process_bbox(bbox,
                                    img_width=img_shape[1],
                                    img_height=img_shape[0],
                                    ratio=body_bbox_ratio)
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
            smplx_valid_list = []
            for bbox_i in valid_idx:
                smplx_valid_list.append(smplx_valid[bbox_i])
                lhand_bbox = lhand_bbox_xywh[bbox_i]
                rhand_bbox = rhand_bbox_xywh[bbox_i]
                face_bbox = face_bbox_xywh[bbox_i]
                if lhand_bbox[-1] > 0:  # conf > 0
                    lhand_bbox = lhand_bbox[:4]
                    # if hasattr(cfg, 'bbox_ratio'):
                    lhand_bbox = process_bbox(lhand_bbox,
                                                img_width=img_shape[1],
                                                img_height=img_shape[0],
                                                ratio=hand_bbox_ratio)
                    if lhand_bbox is not None:
                        lhand_bbox[2:] += lhand_bbox[:2]  # xywh -> xyxy
                else:
                    lhand_bbox = None
                if rhand_bbox[-1] > 0:
                    rhand_bbox = rhand_bbox[:4]
                    # if hasattr(cfg, 'bbox_ratio'):
                    rhand_bbox = process_bbox(rhand_bbox,
                                                img_width=img_shape[1],
                                                img_height=img_shape[0],
                                                ratio=hand_bbox_ratio)
                    if rhand_bbox is not None:
                        rhand_bbox[2:] += rhand_bbox[:2]  # xywh -> xyxy
                else:
                    rhand_bbox = None
                if face_bbox[-1] > 0:
                    face_bbox = face_bbox[:4]
                    # if hasattr(cfg, 'bbox_ratio'):
                    face_bbox = process_bbox(face_bbox,
                                                img_width=img_shape[1],
                                                img_height=img_shape[0],
                                                ratio=hand_bbox_ratio)
                    if face_bbox is not None:
                        face_bbox[2:] += face_bbox[:2]  # xywh -> xyxy
                else:
                    face_bbox = None
                lhand_bbox_list.append(lhand_bbox)
                rhand_bbox_list.append(rhand_bbox)
                face_bbox_list.append(face_bbox)
            
            joint_img = keypoints2d[valid_idx]
            
            if valid_kps3d:
                joint_cam = keypoints3d[valid_idx]
            else:
                joint_cam = None
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
            

            smplx_param = {k: v[valid_idx] for k, v in smplx.items()}
            gender_ = gender[valid_idx] \
                if gender is not None else np.array(['neutral']*(valid_num))
            lhand_bbox_valid = lhand_bbox_xywh[valid_idx,4]
            rhand_bbox_valid = rhand_bbox_xywh[valid_idx,4]
            face_bbox_valid = face_bbox_xywh[valid_idx,4]
            
            # TODO: set invalid if None?
            smplx_param['root_pose'] = smplx_param.pop('global_orient', None)
            smplx_param['shape'] = smplx_param.pop('betas', None)
            smplx_param['trans'] = smplx_param.pop('transl', np.zeros([len(valid_idx),3]))
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
                # smplx_param['expr'] = None
            if self.__class__.__name__ == 'GTA':
                smplx_param['shape'] = np.zeros(
                    [valid_num, 10],
                    dtype=np.float32)
            if self.__class__.__name__ == 'COCO_NA':
                # smplx_param['expr'] = None
                smplx_param['body_pose'] = smplx_param['body_pose'].reshape(
                    -1, 21, 3)
                smplx_param['lhand_pose'] = smplx_param['lhand_pose'].reshape(
                    -1, 15, 3)
                smplx_param['rhand_pose'] = smplx_param['rhand_pose'].reshape(
                    -1, 15, 3)
            if as_smplx == 'smpl':
                smplx_param['shape'] = np.zeros(
                    [valid_num, 10],
                    dtype=np.float32)  # drop smpl betas for smplx
                smplx_param['body_pose'] = smplx_param[
                    'body_pose'].reshape(-1,23,3)[:, :21, :]  # use smpl body_pose on smplx
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

            smplx_param['smplx_valid'] = np.array(smplx_valid_list).astype(np.bool8)
            if joint_cam is not None and np.any(np.isnan(joint_cam)):
                continue
            
            
            if self.__class__.__name__ == 'SPEC':
                joint_img[:,:,2] = joint_img[:,:,2]>0
                joint_cam[:,:,3] = joint_cam[:,:,0]!=0
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
                'gender': gender_
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

    def __len__(self):
        return len(self.datalist)
    # 19493
    def __getitem__(self, idx):
        # rank = self.rank
        # local_rank = rank % torch.cuda.device_count()
        # with open(f'index_log_{rank}.txt', 'a') as f:
        #     f.write(f'{rank}-{local_rank}-{idx}\n')
        try:
            data = copy.deepcopy(self.datalist[idx])
        except Exception as e:
            print(f'[{self.__class__.__name__}] Error loading data {idx}')
            print(e)
            exit(0)
        # data/datasets/coco_2017/train2017/000000029582.jpg' 45680
        img_path, img_shape, bbox = \
            data['img_path'], data['img_shape'], data['bbox']
        as_smplx = data['as_smplx']
        gender = data['gender'].copy()
        for gender_str, gender_num in {
            'neutral': -1, 'male': 0, 'female': 1}.items():
            gender[gender==gender_str]=gender_num
        gender = gender.astype(int)    
        
        img_whole_bbox = np.array([0, 0, img_shape[1], img_shape[0]])
        img = load_img(img_path, order='BGR')

        num_person = len(data['bbox'])
        data_name = self.__class__.__name__
        try:
            # dist.barrier()
            img, img2bb_trans, bb2img_trans, rot, do_flip = \
                augmentation_instance_sample(img, img_whole_bbox, self.data_split, data, data_name)
        except Exception as e:
            rank = self.rank
            local_rank = rank % torch.cuda.device_count()
            with open(f'index_log_{rank}.txt', 'a') as f:
                f.write(f'{rank}-{local_rank}-{idx}\n')
                f.write(f'[{self.__class__.__name__}] Error loading data {idx}\n')
                f.write(f'Error in augmentation_instance_sample for {img_path}\n')
            # print(f'[{self.__class__.__name__}] Error loading data {idx}')
            # print(f'Error in augmentation_instance_sample for {img_path}')
            raise e
        cropped_img_shape = img.shape[:2]
        
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
            # do rotation on keypoints
            joint_img_aug, joint_cam_wo_ra, joint_cam_ra, joint_trunc = \
                process_db_coord_batch_no_valid(
                    joint_img, joint_cam, do_flip, img_shape,
                    self.joint_set['flip_pairs'], img2bb_trans, rot,
                    self.joint_set['joints_name'], smpl_x.joints_name,
                    cropped_img_shape)
            joint_img_aug[:,:,2:] = joint_img_aug[:,:,2:] * joint_trunc
            
            # smplx coordinates and parameters
            smplx_param = data['smplx_param']
            # smplx_param
            smplx_pose, smplx_shape, smplx_expr, smplx_pose_valid, \
            smplx_joint_valid, smplx_expr_valid, smplx_shape_valid = \
                process_human_model_output_batch_simplify(
                    smplx_param, do_flip, rot, as_smplx, data_name)
            smplx_joint_valid = smplx_joint_valid[:, :, None]
            # if cam not provided, we take joint_img as smplx joint 2d, 
            # which is commonly the case for our processed humandata
            # change smplx_shape if use_betas_neutral
            # processing follows that in process_human_model_output
            if self.use_betas_neutral:
                smplx_shape = smplx_param['betas_neutral'].reshape(
                    num_person, -1)
                smplx_shape[(np.abs(smplx_shape) > 3).any(axis=1)] = 0.
                smplx_shape = smplx_shape.reshape(num_person, -1)
            
            if self.__class__.__name__ == 'MPII_MM' :
                for name in ('L_Ankle', 'R_Ankle', 'L_Wrist', 'R_Wrist'):
                    smplx_pose_valid[:, smpl_x.orig_joints_name.index(name)] = 0
                for name in ('L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel'):
                     smplx_joint_valid[:,smpl_x.joints_name.index(name)] = 0
    
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
            

            for i in range(num_person):
                body_bbox, body_bbox_valid = self.process_hand_face_bbox(
                    data['bbox'][i], do_flip, img_shape, img2bb_trans,
                    cropped_img_shape)
                
                lhand_bbox, lhand_bbox_valid = self.process_hand_face_bbox(
                    data['lhand_bbox'][i], do_flip, img_shape, img2bb_trans,
                    cropped_img_shape)
                lhand_bbox_valid *= smplx_param['lhand_valid'][i]
                
                rhand_bbox, rhand_bbox_valid = self.process_hand_face_bbox(
                    data['rhand_bbox'][i], do_flip, img_shape, img2bb_trans,
                    cropped_img_shape)
                rhand_bbox_valid *= smplx_param['rhand_valid'][i]
                
                face_bbox, face_bbox_valid = self.process_hand_face_bbox(
                    data['face_bbox'][i], do_flip, img_shape, img2bb_trans,
                    cropped_img_shape)
                face_bbox_valid *= smplx_param['face_valid'][i]
                
                # BEDLAM and COCO_NA do not have face expression
                # if self.__class__.__name__ != 'BEDLAM':
                #     face_bbox_valid *= smplx_param['face_valid'][i]

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

            # joint_img_aug[:,:,2] = joint_img_aug[:,:,2] * body_bbox_valid[:,None]
            
            is_3D = float(False) if dummy_cord else float(True)
            if self.__class__.__name__ == 'COCO_NA':
                is_3D = False
            if self.__class__.__name__ == 'GTA_Human2':
                smplx_shape_valid = smplx_shape_valid * 0
            if self.__class__.__name__ == 'PoseTrack' or self.__class__.__name__ == 'MPII_MM' \
            or self.__class__.__name__ == 'CrowdPose'  or self.__class__.__name__ == 'UBody_MM' \
            or self.__class__.__name__ == 'COCO_NA':
                joint_cam_ra[...,-1] = joint_cam_ra[...,-1] * smplx_joint_valid[...,0]
                joint_cam_wo_ra[...,-1] = joint_cam_wo_ra[...,-1] * smplx_joint_valid[...,0]
                joint_img_aug[...,-1] = joint_img_aug[...,-1] * smplx_joint_valid[...,0]
            # if body_bbox_valid.sum() > 0:
            
            
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

            result = {**inputs, **targets, **meta_info}
            
            result = self.normalize(result)
            result = self.format(result)
            return result

        

        if self.data_split == 'test':
            self.cam_param = {}
            joint_cam = data['joint_cam']
            
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
            # smplx_pose_valid = np.tile(smplx_pose_valid[:,:, None], (1, 3)).reshape(num_person,-1)
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

    def process_hand_face_bbox(self, bbox, do_flip, img_shape, img2bb_trans,
                               input_img_shape):
        if bbox is None:
            bbox = np.array([0, 0, 1, 1],
                            dtype=np.float32).reshape(2, 2)  # dummy value
            bbox_valid = float(False)  # dummy value
        else:
            # reshape to top-left (x,y) and bottom-right (x,y)
            bbox = bbox.reshape(2, 2)

            # flip augmentation
            if do_flip:
                bbox[:, 0] = img_shape[1] - bbox[:, 0] - 1
                bbox[0, 0], bbox[1, 0] = bbox[1, 0].copy(), bbox[
                    0, 0].copy()  # xmin <-> xmax swap

            # make four points of the bbox
            bbox = bbox.reshape(4).tolist()
            xmin, ymin, xmax, ymax = bbox
            bbox = np.array(
                [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
                dtype=np.float32).reshape(4, 2)

            # affine transformation (crop, rotation, scale)
            bbox_xy1 = np.concatenate((bbox, np.ones_like(bbox[:, :1])), 1)
            bbox = np.dot(img2bb_trans,
                          bbox_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            
            # print(bbox)
            # bbox[:, 0] = bbox[:, 0] / input_img_shape[1] * cfg.output_hm_shape[2]
            # bbox[:, 1] = bbox[:, 1] / input_img_shape[0] * cfg.output_hm_shape[1]
            bbox[:, 0] /= input_img_shape[1]
            bbox[:, 1] /= input_img_shape[0]

            
            # make box a rectangle without rotation
            if np.max(bbox[:,0])<=0 or np.min(bbox[:,0])>=1 or np.max(bbox[:,1])<=0 or np.min(bbox[:,1])>=1:
                bbox_valid = float(False)
                bbox = np.array([0, 0, 1, 1], dtype=np.float32)
            else:
                xmin = np.max([np.min(bbox[:, 0]), 0])
                xmax = np.min([np.max(bbox[:, 0]), 1])
                ymin = np.max([np.min(bbox[:, 1]), 0])
                ymax = np.min([np.max(bbox[:, 1]), 1])
                bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

                bbox = np.clip(bbox,0,1)
                bbox_valid = float(True)
            bbox = bbox.reshape(2, 2)

        return bbox, bbox_valid

    def evaluate(self, outs, cur_sample_idx=None):
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
            'mpvpe_face': [],
            'pa_mpjpe_body': [],
            'pa_mpjpe_l_hand': [],
            'pa_mpjpe_r_hand': [],
            'pa_mpjpe_hand': []
        }
        
        for n in range(sample_num):
            out = outs[n]
            ann_idx = out['gt_ann_idx']
            mesh_gt = out['smplx_mesh_cam_pseudo_gt']
            mesh_out = out['smplx_mesh_cam']
            cam_trans = out['cam_trans']
            ann_idx = out['gt_ann_idx']
            img_path = []
            for ann_id in ann_idx:
                img_path.append(annots[ann_id]['img_path'])
            eval_result['img_path'] = img_path
            eval_result['ann_idx'] = ann_idx
            
            img = out['img']
            # MPVPE from all vertices
            mesh_out_align = mesh_out - np.dot(
                smpl_x.J_regressor,
                mesh_out)[smpl_x.J_regressor_idx['pelvis'], None, :] + np.dot(
                    smpl_x.J_regressor,
                    mesh_gt)[smpl_x.J_regressor_idx['pelvis'], None, :]
            eval_result['mpvpe_all'].append(
                np.sqrt(np.sum(
                    (mesh_out_align - mesh_gt)**2, 1)).mean() * 1000)
            mesh_out_align = rigid_align(mesh_out, mesh_gt)
            eval_result['pa_mpvpe_all'].append(
                np.sqrt(np.sum(
                    (mesh_out_align - mesh_gt)**2, 1)).mean() * 1000)
            # MPVPE from hand vertices
            mesh_gt_lhand = mesh_gt[smpl_x.hand_vertex_idx['left_hand'], :]
            mesh_out_lhand = mesh_out[smpl_x.hand_vertex_idx['left_hand'], :]
            mesh_gt_rhand = mesh_gt[smpl_x.hand_vertex_idx['right_hand'], :]
            mesh_out_rhand = mesh_out[smpl_x.hand_vertex_idx['right_hand'], :]
            mesh_out_lhand_align = mesh_out_lhand - np.dot(
                smpl_x.J_regressor,
                mesh_out)[smpl_x.J_regressor_idx['lwrist'], None, :] + np.dot(
                    smpl_x.J_regressor,
                    mesh_gt)[smpl_x.J_regressor_idx['lwrist'], None, :]
            mesh_out_rhand_align = mesh_out_rhand - np.dot(
                smpl_x.J_regressor,
                mesh_out)[smpl_x.J_regressor_idx['rwrist'], None, :] + np.dot(
                    smpl_x.J_regressor,
                    mesh_gt)[smpl_x.J_regressor_idx['rwrist'], None, :]
            eval_result['mpvpe_l_hand'].append(
                np.sqrt(np.sum(
                    (mesh_out_lhand_align - mesh_gt_lhand)**2, 1)).mean() *
                1000)
            eval_result['mpvpe_r_hand'].append(
                np.sqrt(np.sum(
                    (mesh_out_rhand_align - mesh_gt_rhand)**2, 1)).mean() *
                1000)
            eval_result['mpvpe_hand'].append(
                (np.sqrt(np.sum(
                    (mesh_out_lhand_align - mesh_gt_lhand)**2, 1)).mean() *
                 1000 +
                 np.sqrt(np.sum(
                     (mesh_out_rhand_align - mesh_gt_rhand)**2, 1)).mean() *
                 1000) / 2.)
            mesh_out_lhand_align = rigid_align(mesh_out_lhand, mesh_gt_lhand)
            mesh_out_rhand_align = rigid_align(mesh_out_rhand, mesh_gt_rhand)
            eval_result['pa_mpvpe_l_hand'].append(
                np.sqrt(np.sum(
                    (mesh_out_lhand_align - mesh_gt_lhand)**2, 1)).mean() *
                1000)
            eval_result['pa_mpvpe_r_hand'].append(
                np.sqrt(np.sum(
                    (mesh_out_rhand_align - mesh_gt_rhand)**2, 1)).mean() *
                1000)
            eval_result['pa_mpvpe_hand'].append(
                (np.sqrt(np.sum(
                    (mesh_out_lhand_align - mesh_gt_lhand)**2, 1)).mean() *
                 1000 +
                 np.sqrt(np.sum(
                     (mesh_out_rhand_align - mesh_gt_rhand)**2, 1)).mean() *
                 1000) / 2.)
            
            if self.__class__.__name__ == 'UBody':
                joint_gt_body_wo_trans = np.dot(smpl_x.j14_regressor,
                                            mesh_gt)
                import ipdb;ipdb.set_trace()
                img_wh = out['gt_img_shape'].flip(-1)
                joint_gt_body_proj = project_points_new(
                    points_3d=joint_gt_body_wo_trans,
                    pred_cam=cam_trans,
                    focal_length=5000,
                    camera_center=img_wh/2
                )  # origin image space
                joint_gt_lhand_wo_trans = np.dot(
                    smpl_x.orig_hand_regressor['left'], mesh_gt)
                joint_gt_lhand_proj = project_points_new(
                    points_3d=joint_gt_lhand_wo_trans,
                    pred_cam=cam_trans,
                    focal_length=5000,
                    camera_center=img_wh/2
                )  # origin image space
                joint_gt_rhand_wo_trans = np.dot(
                    smpl_x.orig_hand_regressor['left'], mesh_gt)
                joint_gt_rhand_proj = project_points_new(
                    points_3d=joint_gt_rhand_wo_trans,
                    pred_cam=cam_trans,
                    focal_length=5000,
                    camera_center=img_wh/2
                )  # origin image space
                mesh_gt_proj = project_points_new(
                    points_3d=mesh_gt,
                    pred_cam=cam_trans,
                    focal_length=5000,
                    camera_center=img_wh/2)
                joint_gt_body_valid = self.validate_within_img(
                    img, joint_gt_body_proj)
                joint_gt_lhand_valid = self.validate_within_img(
                    img, joint_gt_lhand_proj)
                joint_gt_rhand_valid = self.validate_within_img(
                    img, joint_gt_rhand_proj)
                mesh_valid = self.validate_within_img(img, mesh_gt_proj)
                mesh_lhand_valid = mesh_valid[smpl_x.hand_vertex_idx['left_hand']]
                mesh_rhand_valid = mesh_valid[smpl_x.hand_vertex_idx['right_hand']]
                mesh_face_valid = mesh_valid[smpl_x.face_vertex_idx]

            # MPVPE from face vertices
            mesh_gt_face = mesh_gt[smpl_x.face_vertex_idx, :]
            mesh_out_face = mesh_out[smpl_x.face_vertex_idx, :]
            mesh_out_face_align = mesh_out_face - np.dot(
                smpl_x.J_regressor,
                mesh_out)[smpl_x.J_regressor_idx['neck'], None, :] + np.dot(
                    smpl_x.J_regressor,
                    mesh_gt)[smpl_x.J_regressor_idx['neck'], None, :]
            eval_result['mpvpe_face'].append(
                np.sqrt(np.sum(
                    (mesh_out_face_align - mesh_gt_face)**2, 1)).mean() * 1000)
            mesh_out_face_align = rigid_align(mesh_out_face, mesh_gt_face)
            eval_result['pa_mpvpe_face'].append(
                np.sqrt(np.sum(
                    (mesh_out_face_align - mesh_gt_face)**2, 1)).mean() * 1000)

            # MPJPE from body joints
            joint_gt_body = np.dot(smpl_x.j14_regressor, mesh_gt)
            joint_out_body = np.dot(smpl_x.j14_regressor, mesh_out)
            joint_out_body_align = rigid_align(joint_out_body, joint_gt_body)
            eval_result['pa_mpjpe_body'].append(
                np.sqrt(np.sum((joint_out_body_align - joint_gt_body)**2,
                               1))[joint_gt_body_valid].mean() * 1000)

            # eval_result['pa_mpjpe_body'].append(
            #     np.sqrt(np.sum(
            #         (joint_out_body_align - joint_gt_body)**2, 1)).mean() *
            #     1000)

            # MPJPE from hand joints
            joint_gt_lhand = np.dot(smpl_x.orig_hand_regressor['left'],
                                    mesh_gt)
            joint_out_lhand = np.dot(smpl_x.orig_hand_regressor['left'],
                                     mesh_out)
            joint_out_lhand_align = rigid_align(joint_out_lhand,
                                                joint_gt_lhand)
            joint_gt_rhand = np.dot(smpl_x.orig_hand_regressor['right'],
                                    mesh_gt)
            joint_out_rhand = np.dot(smpl_x.orig_hand_regressor['right'],
                                     mesh_out)
            joint_out_rhand_align = rigid_align(joint_out_rhand,
                                                joint_gt_rhand)
            # if self.__class__.__name__ == 'UBody':
            if sum(joint_gt_lhand_valid) != 0:
                pa_mpjpe_lhand = np.sqrt(
                    np.sum((joint_out_lhand_align - joint_gt_lhand)**2,
                           1))[joint_gt_lhand_valid].mean() * 1000
                pa_mpjpe_hand.append(pa_mpjpe_lhand)
                eval_result['pa_mpjpe_l_hand'].append(pa_mpjpe_lhand)
            if sum(joint_gt_rhand_valid) != 0:
                pa_mpjpe_rhand = np.sqrt(
                    np.sum((joint_out_rhand_align - joint_gt_rhand)**2,
                           1))[joint_gt_rhand_valid].mean() * 1000
                pa_mpjpe_hand.append(pa_mpjpe_rhand)
                eval_result['pa_mpjpe_r_hand'].append(pa_mpjpe_rhand)
            if len(pa_mpjpe_hand) > 0:
                eval_result['pa_mpjpe_hand'].append(np.mean(pa_mpjpe_hand))

            eval_result['pa_mpjpe_l_hand'].append(
                np.sqrt(np.sum(
                    (joint_out_lhand_align - joint_gt_lhand)**2, 1)).mean() *
                1000)
            eval_result['pa_mpjpe_r_hand'].append(
                np.sqrt(np.sum(
                    (joint_out_rhand_align - joint_gt_rhand)**2, 1)).mean() *
                1000)
            eval_result['pa_mpjpe_hand'].append(
                (np.sqrt(np.sum(
                    (joint_out_lhand_align - joint_gt_lhand)**2, 1)).mean() *
                 1000 +
                 np.sqrt(np.sum(
                     (joint_out_rhand_align - joint_gt_rhand)**2, 1)).mean() *
                 1000) / 2.)
        return eval_result

    def print_eval_result(self, eval_result):
        print(f'======{cfg.testset}======')
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
        print()

        print('PA MPJPE (Body): %.2f mm' %
              np.mean(eval_result['pa_mpjpe_body']))
        print('PA MPJPE (L-Hands): %.2f mm' %
              np.mean(eval_result['pa_mpjpe_l_hand']))
        print('PA MPJPE (R-Hands): %.2f mm' %
              np.mean(eval_result['pa_mpjpe_r_hand']))
        print('PA MPJPE (Hands): %.2f mm' %
              np.mean(eval_result['pa_mpjpe_hand']))

        f = open(os.path.join(cfg.result_dir, 'result.txt'), 'w')
        f.write(f'{cfg.testset} dataset \n')
        f.write('PA MPVPE (All): %.2f mm\n' %
                np.mean(eval_result['pa_mpvpe_all']))
        f.write('PA MPVPE (L-Hands): %.2f mm' %
                np.mean(eval_result['pa_mpvpe_l_hand']))
        f.write('PA MPVPE (R-Hands): %.2f mm' %
                np.mean(eval_result['pa_mpvpe_r_hand']))
        f.write('PA MPVPE (Hands): %.2f mm\n' %
                np.mean(eval_result['pa_mpvpe_hand']))
        f.write('PA MPVPE (Face): %.2f mm\n' %
                np.mean(eval_result['pa_mpvpe_face']))
        f.write('MPVPE (All): %.2f mm\n' % np.mean(eval_result['mpvpe_all']))
        f.write('MPVPE (L-Hands): %.2f mm' %
                np.mean(eval_result['mpvpe_l_hand']))
        f.write('MPVPE (R-Hands): %.2f mm' %
                np.mean(eval_result['mpvpe_r_hand']))
        f.write('MPVPE (Hands): %.2f mm' % np.mean(eval_result['mpvpe_hand']))
        f.write('MPVPE (Face): %.2f mm\n' % np.mean(eval_result['mpvpe_face']))
        f.write('PA MPJPE (Body): %.2f mm\n' %
                np.mean(eval_result['pa_mpjpe_body']))
        f.write('PA MPJPE (L-Hands): %.2f mm' %
                np.mean(eval_result['pa_mpjpe_l_hand']))
        f.write('PA MPJPE (R-Hands): %.2f mm' %
                np.mean(eval_result['pa_mpjpe_r_hand']))
        f.write('PA MPJPE (Hands): %.2f mm\n' %
                np.mean(eval_result['pa_mpjpe_hand']))
    def validate_within_img_batch(
            self, img_wh, points):  # check whether the points is within the image
        # img: (h, w, c), points: (num_points, 2)
        
        valid_mask = np.logical_and((points-img_wh[:,None])<0,points>0)
        valid_mask = np.logical_and(valid_mask[:,:,0],valid_mask[:,:,1])
        
        return valid_mask
    def decompress_keypoints(self, humandata) -> None:
        """If a key contains 'keypoints', and f'{key}_mask' is in self.keys(),
        invalid zeros will be inserted to the right places and f'{key}_mask'
        will be unlocked.

        Raises:
            KeyError:
                A key contains 'keypoints' has been found
                but its corresponding mask is missing.
        """
        assert bool(humandata['__keypoints_compressed__']) is True
        key_pairs = []
        for key in humandata.files:
            if key not in KPS2D_KEYS + KPS3D_KEYS:
                continue
            mask_key = f'{key}_mask'
            if mask_key in humandata.files:
                print(f'Decompress {key}...')
                key_pairs.append([key, mask_key])
        decompressed_dict = {}
        for kpt_key, mask_key in key_pairs:
            mask_array = np.asarray(humandata[mask_key])
            compressed_kpt = humandata[kpt_key]
            kpt_array = \
                self.add_zero_pad(compressed_kpt, mask_array)
            decompressed_dict[kpt_key] = kpt_array
        del humandata
        return decompressed_dict

    def add_zero_pad(self, compressed_array: np.ndarray,
                     mask_array: np.ndarray) -> np.ndarray:
        """Pad zeros to a compressed keypoints array.

        Args:
            compressed_array (np.ndarray):
                A compressed keypoints array.
            mask_array (np.ndarray):
                The mask records compression relationship.

        Returns:
            np.ndarray:
                A keypoints array in full-size.
        """
        assert mask_array.sum() == compressed_array.shape[1]
        data_len, _, dim = compressed_array.shape
        mask_len = mask_array.shape[0]
        ret_value = np.zeros(shape=[data_len, mask_len, dim],
                             dtype=compressed_array.dtype)
        valid_mask_index = np.where(mask_array == 1)[0]
        ret_value[:, valid_mask_index, :] = compressed_array
        return ret_value
