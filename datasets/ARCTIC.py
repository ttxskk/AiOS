import os
import os.path as osp
from glob import glob
import numpy as np
from config.config import cfg

import csv

from util.human_models import smpl_x

from util.transforms import rigid_align_batch

from humandata import HumanDataset

class ARCTIC(HumanDataset):
    def __init__(self, transform, data_split):
        super(ARCTIC, self).__init__(transform, data_split)

        self.img_dir = 'data/osx_data/ARCTIC'


        if data_split == 'train':
            self.annot_path = 'data/preprocessed_npz/multihuman_data/p1_train_multi.npz'
            self.annot_path_cache = 'data/preprocessed_npz/cache/p1_train_cache_sample1000_080824.npz'
            self.sample_interval = 1000  
        elif data_split == 'test':
            self.annot_path = 'data/preprocessed_npz_old/multihuman_data/p1_val_multi.npz'
            self.annot_path_cache = 'data/preprocessed_npz_old/cache/p1_val_cache_30.npz'
            self.sample_interval = 30
        
        
        self.use_cache = getattr(cfg, 'use_cache', False)
        self.img_shape = None  #1024, 1024)  # (h, w)
        self.cam_param = {}
        self.use_cache=True
        # load data
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
                cfg, f'{self.__class__.__name__}_train_sample_interval', self.sample_interval))
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
        csv_file = f'{cfg.result_dir}/arctic_smplx_error.csv'
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
            # MPVPE from all vertices
            mesh_out_align = \
                mesh_out - np.dot(
                    smpl_x.J_regressor, mesh_out).transpose(1,0,2)[:, smpl_x.J_regressor_idx['pelvis'], None, :] + \
                    np.dot(smpl_x.J_regressor, mesh_gt).transpose(1,0,2)[:, smpl_x.J_regressor_idx['pelvis'], None, :]
            
            eval_result['mpvpe_all'].append(
                np.sqrt(np.sum(
                    (mesh_out_align - mesh_gt)**2, -1)).mean() * 1000)
            mesh_out_align = rigid_align_batch(mesh_out, mesh_gt)
            eval_result['pa_mpvpe_all'].append(
                np.sqrt(np.sum(
                    (mesh_out_align - mesh_gt)**2, -1)).mean() * 1000)

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
            mesh_out_lhand_align = rigid_align_batch(mesh_out_lhand, mesh_gt_lhand)
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

            # MPVPE from face vertices
            mesh_gt_face = mesh_gt[:, smpl_x.face_vertex_idx, :]
            mesh_out_face = mesh_out[:, smpl_x.face_vertex_idx, :]
            mesh_out_face_align = \
                mesh_out_face - \
                np.dot(smpl_x.J_regressor, mesh_out).transpose(1,0,2)[:, smpl_x.J_regressor_idx['neck'], None, :] + \
                np.dot(smpl_x.J_regressor, mesh_gt).transpose(1,0,2)[:, smpl_x.J_regressor_idx['neck'], None, :]
            eval_result['mpvpe_face'].append(
                np.sqrt(np.sum(
                    (mesh_out_face_align - mesh_gt_face)**2, -1)).mean() * 1000)
            mesh_out_face_align = rigid_align_batch(mesh_out_face, mesh_gt_face)
            eval_result['pa_mpvpe_face'].append(
                np.sqrt(np.sum(
                    (mesh_out_face_align - mesh_gt_face)**2, -1)).mean() * 1000)
        
            save_error=True
            if save_error:
                writer = csv.writer(file)
                new_line = [ann_idx[n], img_path[n], eval_result['mpvpe_all'][-1], eval_result['pa_mpvpe_all'][-1]]
                writer.writerow(new_line)
                # self.save_idx += 1
        return eval_result

    def print_eval_result(self, eval_result):

        print('======ARCTIC-val======')
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

        out_file = osp.join(cfg.result_dir,'arctic_val.txt')
        if os.path.exists(out_file):
            f = open(out_file, 'a+')
        else:
            f = open(out_file, 'w', encoding="utf-8")
        f.write('\n')
        f.write(f'{cfg.exp_name}\n')   
        f.write(f'ARCTIC-val dataset: \n')
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
