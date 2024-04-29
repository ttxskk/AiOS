import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
import csv
from pycocotools.coco import COCO
from config.config import cfg
from util.human_models import smpl_x

from util.transforms import world2cam, cam2pixel, rigid_align
from humandata import HumanDataset
from util.transforms import rigid_align, rigid_align_batch



class EgoBody_Egocentric(HumanDataset):
    def __init__(self, transform, data_split):
        super(EgoBody_Egocentric, self).__init__(transform, data_split)

        if self.data_split == 'train':
            filename = 'data/preprocessed_npz_old/egobody_egocentric_train_cache_ds2_0211.npz'
            self.annot_path_cache = 'data/preprocessed_npz_old/egobody_egocentric_test_230425_043_fix_betas.npz'
        else:
            filename = 'data/multihuman_data/egobody_egocentric_test_230425_043_fix_betas.npz'
            self.annot_path_cache = 'data/cache/egobody_egocentric_test_cache_0412.npz'
        self.use_betas_neutral = getattr(cfg, 'egobody_fix_betas', False)

        self.img_dir = 'data/osx_data/EgoBody'
        self.annot_path = filename
        # self.annot_path_cache = 'data/preprocessed_npz/egobody_egocentric_test_230425_043_cache.npz'
        self.use_cache = getattr(cfg, 'use_cache', False)
        self.img_shape = (1080, 1920)  # (h, w)
        self.cam_param = {}

        # check image shape
        img_path = osp.join(self.img_dir,
                            np.load(self.annot_path)['image_path'][0])
        img_shape = cv2.imread(img_path).shape[:2]
        assert self.img_shape == img_shape, 'image shape is incorrect: {} vs {}'.format(
            self.img_shape, img_shape)

        # load data or cache
        if self.use_cache and osp.isfile(self.annot_path_cache):
            print(
                f'[{self.__class__.__name__}] loading cache from {self.annot_path_cache}'
            )
            self.datalist = self.load_cache(self.annot_path_cache)
        else:
            if self.use_cache:
                print(
                    f'[{self.__class__.__name__}] Cache not found, generating cache...'
                )
            self.datalist = self.load_data(train_sample_interval=getattr(
                cfg, f'{self.__class__.__name__}_train_sample_interval', 2))
            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)
                
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
        csv_file = f'{cfg.result_dir}/egobody_smplx_error.csv'
        file = open(csv_file, 'a', newline='')
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]
            mesh_gt = out['smplx_mesh_cam_target']
            mesh_out = out['smplx_mesh_cam']
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
                new_line = [ann_idx[n], img_path[n], eval_result['mpvpe_all'][-1], eval_result['pa_mpvpe_all'][-1]]
                writer.writerow(new_line)
                # self.save_idx += 1
            vis = False
            if vis:
                import mmcv
                img = (out['img']).transpose(0,2,3,1)
                img = mmcv.imdenormalize(
                    img=img[0], 
                    mean=np.array([123.675, 116.28, 103.53]), 
                    std=np.array([58.395, 57.12, 57.375]),
                    to_bgr=True).astype(np.uint8)
                from detrsmpl.core.visualization.visualize_keypoints2d import visualize_kp2d
                import ipdb;ipdb.set_trace()
                visualize_kp2d(
                    out['smplx_joint_proj'][0][None],
                    image_array=img[None].copy(),
                    disable_limbs=True,
                    overwrite=True,
                    output_path='./figs/pred2d'
                )
                from pytorch3d.io import save_obj
                save_obj('temp.obj',verts=out['smplx_mesh_cam'][0],faces=torch.tensor([]))
            # MPVPE from face vertices
            mesh_gt_face = mesh_gt[:, smpl_x.face_vertex_idx, :]
            mesh_out_face = mesh_out[:, smpl_x.face_vertex_idx, :]
            mesh_out_face_align = \
                mesh_out_face - \
                np.dot(smpl_x.J_regressor, mesh_out).transpose(1,0,2)[:, smpl_x.J_regressor_idx['neck'], None, :] + \
                np.dot(smpl_x.J_regressor, mesh_gt).transpose(1,0,2)[:, smpl_x.J_regressor_idx['neck'], None, :]
            eval_result['mpvpe_face'].extend(
                np.sqrt(np.sum(
                    (mesh_out_face_align - mesh_gt_face)**2, -1)).mean(-1) * 1000)
            mesh_out_face_align = rigid_align_batch(mesh_out_face, mesh_gt_face)
            eval_result['pa_mpvpe_face'].extend(
                np.sqrt(np.sum(
                    (mesh_out_face_align - mesh_gt_face)**2, -1)).mean(-1) * 1000)
            
        # for k,v in eval_result.items():
        #     if k != 'img_path' and k != 'ann_idx':
        #         # import ipdb;ipdb.set_trace()
        #         if len(v)>1:
        #             eval_result[k] = np.concatenate(v,axis=0)
        #         else:
        #             eval_result[k] = np.array(v)

        return eval_result


    def print_eval_result(self, eval_result):

        print('======Egocentric======')
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
        
        out_file = osp.join(cfg.result_dir,'Egocentric_val.txt')
        if os.path.exists(out_file):
            f = open(out_file, 'a+')
        else:
            f = open(out_file, 'w', encoding="utf-8")
            
        f.write('\n')
        f.write(f'{cfg.exp_name}\n')            
        f.write(f'Egocentric dataset: \n')
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
