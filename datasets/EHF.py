import os
import os.path as osp
from glob import glob
import numpy as np
from config.config import cfg
import copy
import json
import cv2
import torch
from pycocotools.coco import COCO
from util.human_models import smpl_x
from util.preprocessing import load_img, process_bbox, load_ply
from util.transforms import rigid_align, rigid_align_batch
from humandata import HumanDataset
import csv

class EHF(HumanDataset):
    def __init__(self, transform, data_split):
        super(EHF, self).__init__(transform, data_split)

        self.transform = transform
        self.data_split = data_split
        self.save_idx = 0
        # self.cam_param = {'R': [-2.98747896, 0.01172457, -0.05704687]}
        # self.cam_param['R'], _ = cv2.Rodrigues(np.array(self.cam_param['R']))
        self.cam_param = {}
        self.img_dir = 'data/data_weichen/ehf'
        self.img_shape = [1200, 1600]
        
        self.annot_path = 'data_tmp/multihuman_data/ehf_val_230908_100.npz'
        self.annot_path_cache = 'data_tmp/cache/ehf_val_cache_230908_100.npz'
        
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
            'mpvpe_face': [],
            'pa_mpjpe_body': [],
            'pa_mpjpe_l_hand': [],
            'pa_mpjpe_r_hand': [],
            'pa_mpjpe_hand': []
        }
        
        csv_file = f'{cfg.result_dir}/ehf_smplx_error.csv'
        file = open(csv_file, 'a', newline='')
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            ann_id = annot['img_path'].split('/')[-1].split('_')[0]
            out = outs[n]
            ann_idx = out['gt_ann_idx']
            img_path = []
            for ann_id in ann_idx:
                img_path.append(annots[ann_id]['img_path'])
            eval_result['img_path'] = img_path
            eval_result['ann_idx'] = ann_idx
            # MPVPE from all vertices np.dot(self.cam_param['R'], out['smplx_mesh_cam_target'].transpose(0,2,1)).transpose(1,2,0)
            # mesh_gt = np.dot(
            #     self.cam_param['R'], 
            #     out['smplx_mesh_cam_target'].transpose(0,2,1)
            #     ).transpose(1,2,0)
            mesh_gt = out['smplx_mesh_cam_target']
            mesh_out = out['smplx_mesh_cam']

            # mesh_gt_align = rigid_align(mesh_gt, mesh_out)

            # print(mesh_out.shape)
            mesh_out_align = rigid_align_batch(mesh_out, mesh_gt)
            eval_result['pa_mpvpe_all'].append(
                np.sqrt(np.sum(
                    (mesh_out_align - mesh_gt)**2, -1)).mean() * 1000)
            mesh_out_align = mesh_out - np.dot(
                smpl_x.J_regressor,
                mesh_out).transpose(1,0,2)[:, smpl_x.J_regressor_idx['pelvis'], None, :] + np.dot(
                    smpl_x.J_regressor,
                    mesh_gt).transpose(1,0,2)[:, smpl_x.J_regressor_idx['pelvis'], None, :]
            eval_result['mpvpe_all'].append(
                np.sqrt(np.sum(
                    (mesh_out_align - mesh_gt)**2, -1)).mean() * 1000)

            # MPVPE from hand vertices
            mesh_gt_lhand = mesh_gt[:, smpl_x.hand_vertex_idx['left_hand'], :]
            mesh_out_lhand = mesh_out[:, smpl_x.hand_vertex_idx['left_hand'], :]
            mesh_out_lhand_align = rigid_align_batch(mesh_out_lhand, mesh_gt_lhand)
            mesh_gt_rhand = mesh_gt[:, smpl_x.hand_vertex_idx['right_hand'], :]
            mesh_out_rhand = mesh_out[:, smpl_x.hand_vertex_idx['right_hand'], :]
            mesh_out_rhand_align = rigid_align_batch(mesh_out_rhand, mesh_gt_rhand)
            eval_result['pa_mpvpe_l_hand'].append(
                np.sqrt(np.sum(
                    (mesh_out_lhand_align - mesh_gt_lhand)**2, -1)).mean() *
                1000)
            eval_result['pa_mpvpe_r_hand'].append(
                np.sqrt(np.sum(
                    (mesh_out_rhand_align - mesh_gt_rhand)**2, -1)).mean() *
                1000)
            eval_result['pa_mpvpe_hand'].append(
                (np.sqrt(np.sum(
                    (mesh_out_lhand_align - mesh_gt_lhand)**2, -1)).mean() *
                 1000 +
                 np.sqrt(np.sum(
                     (mesh_out_rhand_align - mesh_gt_rhand)**2, -1)).mean() *
                 1000) / 2.)

            mesh_out_lhand_align = mesh_out_lhand - np.dot(
                smpl_x.J_regressor,
                mesh_out).transpose(1,0,2)[:, smpl_x.J_regressor_idx['lwrist'], None, :] + np.dot(
                    smpl_x.J_regressor,
                    mesh_gt).transpose(1,0,2)[:, smpl_x.J_regressor_idx['lwrist'], None, :]
            mesh_out_rhand_align = mesh_out_rhand - np.dot(
                smpl_x.J_regressor,
                mesh_out).transpose(1,0,2)[:, smpl_x.J_regressor_idx['rwrist'], None, :] + np.dot(
                    smpl_x.J_regressor,
                    mesh_gt).transpose(1,0,2)[:, smpl_x.J_regressor_idx['rwrist'], None, :]

            eval_result['mpvpe_l_hand'].append(
                np.sqrt(np.sum(
                    (mesh_out_lhand_align - mesh_gt_lhand)**2, -1)).mean() *
                1000)
            eval_result['mpvpe_r_hand'].append(
                np.sqrt(np.sum(
                    (mesh_out_rhand_align - mesh_gt_rhand)**2, -1)).mean() *
                1000)
            eval_result['mpvpe_hand'].append(
                (np.sqrt(np.sum(
                    (mesh_out_lhand_align - mesh_gt_lhand)**2, -1)).mean() *
                 1000 +
                 np.sqrt(np.sum(
                     (mesh_out_rhand_align - mesh_gt_rhand)**2, -1)).mean() *
                 1000) / 2.)

            # MPVPE from face vertices
            mesh_gt_face = mesh_gt[:, smpl_x.face_vertex_idx, :]
            mesh_out_face = mesh_out[:, smpl_x.face_vertex_idx, :]
            mesh_out_face_align = rigid_align_batch(mesh_out_face, mesh_gt_face)
            eval_result['pa_mpvpe_face'].append(
                np.sqrt(np.sum(
                    (mesh_out_face_align - mesh_gt_face)**2, -1)).mean() * 1000)
            mesh_out_face_align = mesh_out_face - np.dot(
                smpl_x.J_regressor,
                mesh_out).transpose(1,0,2)[:, smpl_x.J_regressor_idx['neck'], None, :] + np.dot(
                    smpl_x.J_regressor,
                    mesh_gt).transpose(1,0,2)[:, smpl_x.J_regressor_idx['neck'], None, :]
            eval_result['mpvpe_face'].append(
                np.sqrt(np.sum(
                    (mesh_out_face_align - mesh_gt_face)**2, -1)).mean() * 1000)

            # MPJPE from body joints
            joint_gt_body = np.dot(smpl_x.j14_regressor, mesh_gt).transpose(1,0,2)
            joint_out_body = np.dot(smpl_x.j14_regressor, mesh_out).transpose(1,0,2)
            joint_out_body_align = rigid_align_batch(joint_out_body, joint_gt_body)
            eval_result['pa_mpjpe_body'].append(
                np.sqrt(np.sum(
                    (joint_out_body_align - joint_gt_body)**2, -1)).mean() *
                1000)

            # MPJPE from hand joints
            joint_gt_lhand = np.dot(smpl_x.orig_hand_regressor['left'],
                                    mesh_gt).transpose(1,0,2)
            joint_out_lhand = np.dot(smpl_x.orig_hand_regressor['left'],
                                     mesh_out).transpose(1,0,2)
            joint_out_lhand_align = rigid_align_batch(joint_out_lhand,
                                                joint_gt_lhand)
            joint_gt_rhand = np.dot(smpl_x.orig_hand_regressor['right'],
                                    mesh_gt).transpose(1,0,2)
            joint_out_rhand = np.dot(smpl_x.orig_hand_regressor['right'],
                                     mesh_out).transpose(1,0,2)
            joint_out_rhand_align = rigid_align_batch(joint_out_rhand,
                                                joint_gt_rhand)
            eval_result['pa_mpjpe_l_hand'].append(
                np.sqrt(np.sum(
                    (joint_out_lhand_align - joint_gt_lhand)**2, -1)).mean() *
                1000)
            eval_result['pa_mpjpe_r_hand'].append(
                np.sqrt(np.sum(
                    (joint_out_rhand_align - joint_gt_rhand)**2, 1)).mean() *
                1000)
            eval_result['pa_mpjpe_hand'].append(
                (np.sqrt(np.sum(
                    (joint_out_lhand_align - joint_gt_lhand)**2, -1)).mean() *
                 1000 +
                 np.sqrt(np.sum(
                     (joint_out_rhand_align - joint_gt_rhand)**2, -1)).mean() *
                 1000) / 2.)
            save_error=True
            if save_error:
                writer = csv.writer(file)
                new_line = [ann_idx[n],img_path[n], eval_result['mpvpe_all'][-1], eval_result['pa_mpvpe_all'][-1]]
                writer.writerow(new_line)
                self.save_idx += 1
                
            # vis = cfg.vis
            

        for k,v in eval_result.items():
            if k != 'img_path' and k != 'ann_idx':
                
                if len(v)>1:
                    eval_result[k] = np.concatenate(v,axis=0)
                else:
                    eval_result[k] = np.array(v)
        return eval_result

    def print_eval_result(self, eval_result):
        print('======EHF======')
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
        out_file = osp.join(cfg.result_dir,'ehf_test.txt')
        if os.path.exists(out_file):
            f = open(out_file, 'a+')
        else:
            f = open(out_file, 'w', encoding="utf-8")
        
        f.write('\n')
        f.write(f'{cfg.exp_name}\n')
        f.write(f'EHF dataset: \n')
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
        f.write('PA MPJPE (Body): %.2f mm\n' %
                np.mean(eval_result['pa_mpjpe_body']))
        f.write('PA MPJPE (L-Hands): %.2f mm\n' %
                np.mean(eval_result['pa_mpjpe_l_hand']))
        f.write('PA MPJPE (R-Hands): %.2f mm\n' %
                np.mean(eval_result['pa_mpjpe_r_hand']))
        f.write('PA MPJPE (Hands): %.2f mm\n' %
                np.mean(eval_result['pa_mpjpe_hand']))
        
        f.close()

