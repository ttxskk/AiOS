import json
import os
import os.path
from abc import ABCMeta
from collections import OrderedDict
from typing import Any, List, Optional, Union

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from detrsmpl.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_num,
    get_mapping,
)

from detrsmpl.core.evaluation import (
    keypoint_3d_auc,
    keypoint_3d_pck,
    keypoint_mpjpe,
    vertice_pve,
)

from detrsmpl.data.data_structures.multi_human_data import MultiHumanData
from detrsmpl.models.body_models.builder import build_body_model
from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class MultiHumanImageDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self,
                 data_prefix: str,
                 pipeline: list,
                 body_model: Optional[Union[dict, None]] = None,
                 ann_file: Optional[Union[str, None]] = None,
                 convention: Optional[str] = 'human_data',
                 test_mode: Optional[bool] = False,
                 dataset_name: Optional[Union[str, None]] = None):
        self.num_keypoints = get_keypoint_num(convention)
        self.convention = convention
        super(MultiHumanImageDataset,
              self).__init__(data_prefix, pipeline, ann_file, test_mode,
                             dataset_name)

        if body_model is not None:
            self.body_model = build_body_model(body_model)
        else:
            self.body_model = None

    def get_annotation_file(self):
        """Get path of the annotation file."""
        ann_prefix = os.path.join(self.data_prefix, 'preprocessed_datasets')
        self.ann_file = os.path.join(ann_prefix, self.ann_file)

    def load_annotations(self):
        """Load annotations."""
        self.get_annotation_file()
        self.human_data = MultiHumanData()
        self.human_data.load(self.ann_file)

        self.instance_num = self.human_data.instance_num
        self.image_path = self.human_data['image_path']
        self.num_data = self.human_data.data_len

        try:
            self.frame_range = self.human_data['frame_range']
        except KeyError:
            self.frame_range = \
                np.array([[i, i + 1] for i in range(self.num_data)])

        self.num_data = self.frame_range.shape[0]
        if self.human_data.check_keypoints_compressed():
            self.human_data.decompress_keypoints()

        # change keypoint from 'human_data' to the given convention
        if 'keypoints3d_ori' in self.human_data:
            keypoints3d_ori = self.human_data['keypoints3d_ori']
            assert 'keypoints3d_ori_mask' in self.human_data
            keypoints3d_ori_mask = self.human_data['keypoints3d_ori_mask']
            keypoints3d_ori, keypoints3d_ori_mask = \
                convert_kps(
                    keypoints3d_ori,
                    src='human_data',
                    dst=self.convention,
                    mask=keypoints3d_ori_mask)
            self.human_data.__setitem__('keypoints3d_ori', keypoints3d_ori)
            self.human_data.__setitem__('keypoints3d_ori_convention',
                                        self.convention)
            self.human_data.__setitem__('keypoints3d_ori_mask',
                                        keypoints3d_ori_mask)
        elif 'keypoints3d' in self.human_data:
            keypoints3d_ori = self.human_data['keypoints3d']
            assert 'keypoints3d_mask' in self.human_data
            keypoints3d_ori_mask = self.human_data['keypoints3d_mask']
            keypoints3d_ori, keypoints3d_ori_mask = \
                convert_kps(
                    keypoints3d_ori,
                    src='human_data',
                    dst=self.convention,
                    mask=keypoints3d_ori_mask)
            self.human_data.__setitem__('keypoints3d_ori', keypoints3d_ori)
            self.human_data.__setitem__('keypoints3d_ori_convention',
                                        self.convention)
            self.human_data.__setitem__('keypoints3d_ori_mask',
                                        keypoints3d_ori_mask)

        if 'keypoints2d_ori' in self.human_data:
            keypoints2d_ori = self.human_data['keypoints2d_ori']
            assert 'keypoints2d_ori_mask' in self.human_data
            keypoints2d_ori_mask = self.human_data['keypoints2d_ori_mask']
            keypoints2d_ori, keypoints2d_ori_mask = \
                convert_kps(
                    keypoints2d_ori,
                    src='human_data',
                    dst=self.convention,
                    mask=keypoints2d_ori_mask)
            self.human_data.__setitem__('keypoints2d_ori', keypoints2d_ori)
            self.human_data.__setitem__('keypoints2d_ori_convention',
                                        self.convention)
            self.human_data.__setitem__('keypoints2d_ori_mask',
                                        keypoints2d_ori_mask)
            ori_mask = keypoints2d_ori[:, :, 2]
        elif 'keypoints2d' in self.human_data:
            keypoints2d_ori = self.human_data['keypoints2d']
            assert 'keypoints2d_mask' in self.human_data
            keypoints2d_ori_mask = self.human_data['keypoints2d_mask']
            keypoints2d_ori, keypoints2d_ori_mask = \
                convert_kps(
                    keypoints2d_ori,
                    src='human_data',
                    dst=self.convention,
                    mask=keypoints2d_ori_mask)
            self.human_data.__setitem__('keypoints2d_ori', keypoints2d_ori)
            self.human_data.__setitem__('keypoints2d_ori_convention',
                                        self.convention)
            self.human_data.__setitem__('keypoints2d_ori_mask',
                                        keypoints2d_ori_mask)

        # if 'has_smpl' in self.human_data:
        #     index = ori_mask.sum(-1)>=8
        #     self.human_data['has_smpl']=self.human_data['has_smpl'][:147270]*index
        # change keypoint from 'human_data' to the given convention
        if 'keypoints3d_smpl' in self.human_data:
            keypoints3d_smpl = self.human_data['keypoints3d_smpl']
            assert 'keypoints3d_smpl_mask' in self.human_data
            keypoints3d_smpl_mask = self.human_data['keypoints3d_smpl_mask']
            keypoints3d_smpl, keypoints3d_smpl_mask = \
                convert_kps(
                    keypoints3d_smpl,
                    src='human_data',
                    dst=self.convention,
                    mask=keypoints3d_smpl_mask)
            # index = ori_mask.sum(-1)<8
            # index = ori_mask.sum(-1)<8
            # keypoints3d_smpl[index]=np.concatenate(
            #     [keypoints3d_smpl[index][:,:,:3],
            #     keypoints2d_ori[index][:,:,[2]]],
            #     -1)
            self.human_data.__setitem__('keypoints3d_smpl', keypoints3d_smpl)
            self.human_data.__setitem__('keypoints3d_smpl_convention',
                                        self.convention)
            self.human_data.__setitem__('keypoints3d_smpl_mask',
                                        keypoints3d_smpl_mask)

        if 'keypoints2d_smpl' in self.human_data:
            keypoints2d_smpl = self.human_data['keypoints2d_smpl']
            assert 'keypoints2d_smpl_mask' in self.human_data
            keypoints2d_smpl_mask = self.human_data['keypoints2d_smpl_mask']
            keypoints2d_smpl, keypoints2d_smpl_mask = \
                convert_kps(
                    keypoints2d_smpl,
                    src='human_data',
                    dst=self.convention,
                    mask=keypoints2d_smpl_mask)
            # index = ori_mask.sum(-1)<8
            # keypoints2d_smpl[index]=np.concatenate(
            #     [keypoints2d_smpl[index][:,:,:2],
            #     keypoints2d_ori[index][:,:,[2]]],
            #     -1)
            # keypoints2d_smpl[index][:,:,2]=keypoints2d_ori[index][:, :,2]
            self.human_data.__setitem__('keypoints2d_smpl', keypoints2d_smpl)
            self.human_data.__setitem__('keypoints2d_smpl_convention',
                                        self.convention)
            self.human_data.__setitem__('keypoints2d_smpl_mask',
                                        keypoints2d_smpl_mask)
        self.human_data.compress_keypoints_by_mask()

    

    def prepare_raw_data(self, idx: int):
        """Get item from self.human_data."""
        sample_idx = idx
        frame_start, frame_end = self.frame_range[idx]
        frame_num = frame_end - frame_start
        # TODO: Support cache_reader?
        info = {}
        info['img_prefix'] = None
        image_path = self.human_data['image_path'][frame_start]
        info['image_path'] = os.path.join(self.data_prefix, 'datasets',
                                          self.dataset_name, image_path)
        # TODO: Support smc?
        info['dataset_name'] = self.dataset_name
        info['sample_idx'] = sample_idx
        if 'bbox_xywh' in self.human_data:
            info['bbox_xywh'] = self.human_data['bbox_xywh'][
                frame_start:frame_end]
            center, scale = [], []
            for bbox in info['bbox_xywh']:
                x, y, w, h, s = bbox
                cx = x + w / 2
                cy = y + h / 2
                # TODO: verify if we should keep w = h = max(w, h) for multi human data
                w = h = max(w, h)
                center.append([cx, cy])
                scale.append([w, h])
            info['center'] = np.array(center)
            info['scale'] = np.array(scale)
        else:
            info['bbox_xywh'] = np.zeros((frame_num, 5))
            info['center'] = np.zeros((frame_num, 2))
            info['scale'] = np.zeros((frame_num, 2))

        if 'keypoints2d_ori' in self.human_data:
            info['keypoints2d_ori'] = self.human_data['keypoints2d_ori'][
                frame_start:frame_end]
            conf = info['keypoints2d_ori'][..., -1].sum(-1) > 0
            info['has_keypoints2d_ori'] = np.ones(
                (frame_num, 1)) * conf[..., None]
        else:
            info['keypoints2d_ori'] = np.zeros(
                (frame_num, self.num_keypoints, 3))
            info['has_keypoints2d_ori'] = np.zeros((frame_num, 1))

        if 'keypoints3d_ori' in self.human_data:
            info['keypoints3d_ori'] = self.human_data['keypoints3d_ori'][
                frame_start:frame_end]
            conf = info['keypoints3d_ori'][..., -1].sum(-1) > 0
            info['has_keypoints3d_ori'] = np.ones(
                (frame_num, 1)) * conf[..., None]
        else:
            info['keypoints3d_ori'] = np.zeros(
                (frame_num, self.num_keypoints, 4))
            info['has_keypoints3d_ori'] = np.zeros((frame_num, 1))

        if 'keypoints2d_smpl' in self.human_data:
            info['keypoints2d_smpl'] = self.human_data['keypoints2d_smpl'][
                frame_start:frame_end]
            conf = info['keypoints2d_smpl'][..., -1].sum(-1) > 0
            info['has_keypoints2d_smpl'] = np.ones(
                (frame_num, 1)) * conf[..., None]
        else:
            info['keypoints2d_smpl'] = np.zeros(
                (frame_num, self.num_keypoints, 3))
            info['has_keypoints2d_smpl'] = np.zeros((frame_num, 1))

        if 'keypoints3d_smpl' in self.human_data:
            info['keypoints3d_smpl'] = self.human_data['keypoints3d_smpl'][
                frame_start:frame_end]
            conf = info['keypoints3d_smpl'][..., -1].sum(-1) > 0
            info['has_keypoints3d_smpl'] = np.ones(
                (frame_num, 1)) * conf[..., None]
        else:
            info['keypoints3d_smpl'] = np.zeros(
                (frame_num, self.num_keypoints, 4))
            info['has_keypoints3d_smpl'] = np.zeros((frame_num, 1))

        if 'smpl' in self.human_data:
            if 'has_smpl' in self.human_data:
                info['has_smpl'] = \
                    self.human_data['has_smpl'][frame_start:frame_end]
            else:
                info['has_smpl'] = np.ones((frame_num, 1))
            smpl_dict = self.human_data['smpl']
        else:
            info['has_smpl'] = np.zeros((frame_num, 1))
            smpl_dict = {}

        if 'body_pose' in smpl_dict:
            info['smpl_body_pose'] = smpl_dict['body_pose'][
                frame_start:frame_end]
        else:
            info['smpl_body_pose'] = np.zeros((frame_num, 23, 3))

        if 'global_orient' in smpl_dict:
            info['smpl_global_orient'] = smpl_dict['global_orient'][
                frame_start:frame_end]
        else:
            info['smpl_global_orient'] = np.zeros((frame_num, 3))

        if 'betas' in smpl_dict:
            info['smpl_betas'] = smpl_dict['betas'][frame_start:frame_end]
        else:
            info['smpl_betas'] = np.zeros((frame_num, 10))

        if 'transl' in smpl_dict:
            info['smpl_transl'] = smpl_dict['transl'][frame_start:frame_end]
        else:
            info['smpl_transl'] = np.zeros((frame_num, 3))

        if 'area' in self.human_data:
            info['area'] = self.human_data['area'][frame_start:frame_end]
        else:
            info['area'] = np.zeros((frame_num, 0))


        return info

    def prepare_data(self, idx: int):
        """Generate and transform data."""
        info = self.prepare_raw_data(idx)
        return self.pipeline(info)

    def evaluate(self,
                 outputs: list,
                 res_folder: str,
                 metric: Optional[Union[str, List[str]]] = 'pa-mpjpe',
                 **kwargs: dict):
        """Evaluate 3D keypoint results.

        Args:
            outputs (list): results from model inference.
            res_folder (str): path to store results.
            metric (Optional[Union[str, List(str)]]):
                the type of metric. Default: 'pa-mpjpe'
            kwargs (dict): other arguments.
        Returns:
            dict:
                A dict of all evaluation results.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        for metric in metrics:
            if metric not in self.ALLOWED_METRICS:
                raise KeyError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')
        # for keeping correctness during multi-gpu test, we sort all results
        res_dict = {}
        # 'scores', 'labels', 'boxes', 'keypoints', 'pred_smpl_pose',
        # 'pred_smpl_beta', 'pred_smpl_cam', 'pred_smpl_kp3d',
        # 'gt_smpl_pose', 'gt_smpl_beta', 'gt_smpl_kp3d', 'gt_boxes',
        # 'gt_keypoints', 'image_idx'
        for out in outputs:
            target_id = out['image_idx']
            batch_size = len(out['pred_smpl_kp3d'])
            for i in range(batch_size):
                res_dict[int(target_id[i])] = dict(
                    keypoints=out['pred_smpl_kp3d'][i],
                    gt_poses=out['gt_smpl_pose'][i],
                    gt_betas=out['gt_smpl_beta'][i],
                    pred_poses=out['pred_smpl_pose'][i],
                    pred_betas=out['pred_smpl_beta'][i])
        keypoints, gt_poses, gt_betas, pred_poses, pred_betas = \
            [], [], [], [], []
        # print(self.num_data)
        for i in range(self.num_data):
            keypoints.append(res_dict[i]['keypoints'])
            gt_poses.append(res_dict[i]['gt_poses'])
            gt_betas.append(res_dict[i]['gt_betas'])
            pred_poses.append(res_dict[i]['pred_poses'])
            pred_betas.append(res_dict[i]['pred_betas'])

        res = dict(keypoints=keypoints,
                   gt_poses=gt_poses,
                   gt_betas=gt_betas,
                   pred_poses=pred_poses,
                   pred_betas=pred_betas)
        # mmcv.dump(res, res_file)
        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                _nv_tuples = self._report_mpjpe(res)
            elif _metric == 'pa-mpjpe':
                _nv_tuples = self._report_mpjpe(res, metric='pa-mpjpe')
                print(_nv_tuples)
            elif _metric == '3dpck':
                _nv_tuples = self._report_3d_pck(res)
            elif _metric == 'pa-3dpck':
                _nv_tuples = self._report_3d_pck(res, metric='pa-3dpck')
            elif _metric == '3dauc':
                _nv_tuples = self._report_3d_auc(res)
            elif _metric == 'pa-3dauc':
                _nv_tuples = self._report_3d_auc(res, metric='pa-3dauc')
            elif _metric == 'pve':
                _nv_tuples = self._report_pve(res)
            elif _metric == 'ihmr':
                _nv_tuples = self._report_ihmr(res)
            else:
                raise NotImplementedError
            name_value_tuples.extend(_nv_tuples)

        name_value = OrderedDict(name_value_tuples)
        return name_value

    @staticmethod
    def _write_keypoint_results(keypoints: Any, res_file: str):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _parse_result(self, res, mode='keypoint', body_part=None):
        """Parse results."""

        if mode == 'vertice':
            # gt
            gt_beta, gt_pose, gt_global_orient, gender = [], [], [], []
            gt_smpl_dict = self.human_data['smpl']
            for idx in range(self.num_data):
                gt_beta.append(gt_smpl_dict['betas'][idx])
                gt_pose.append(gt_smpl_dict['body_pose'][idx])
                gt_global_orient.append(gt_smpl_dict['global_orient'][idx])
                if self.human_data['meta']['gender'][idx] == 'm':
                    gender.append(0)
                else:
                    gender.append(1)
            gt_beta = torch.FloatTensor(gt_beta)
            gt_pose = torch.FloatTensor(gt_pose).view(-1, 69)
            gt_global_orient = torch.FloatTensor(gt_global_orient)
            gender = torch.Tensor(gender)
            gt_output = self.body_model(betas=gt_beta,
                                        body_pose=gt_pose,
                                        global_orient=gt_global_orient,
                                        gender=gender)
            gt_vertices = gt_output['vertices'].detach().cpu().numpy() * 1000.
            gt_mask = np.ones(gt_vertices.shape[:-1])
            # pred
            pred_pose = torch.FloatTensor(res['pred_poses'])
            pred_beta = torch.FloatTensor(res['pred_betas'])
            pred_output = self.body_model(
                betas=pred_beta[:, 0],
                body_pose=pred_pose[:, 0, 1:],
                global_orient=pred_pose[:, 0, 0].unsqueeze(1),
                pose2rot=False)
            pred_vertices = pred_output['vertices'].detach().cpu().numpy(
            ) * 1000.

            assert len(pred_vertices) == self.num_data

            return pred_vertices, gt_vertices, gt_mask
        elif mode == 'keypoint':
            pred_keypoints3d = res['keypoints']
            assert len(pred_keypoints3d) == self.num_data
            # (B, 17, 3)
            pred_keypoints3d = np.array(pred_keypoints3d).reshape(
                len(pred_keypoints3d), -1, 3)
            # pred_keypoints3d,_ = convert_kps(
            #     pred_keypoints3d,
            #     src='smpl_54',
            #     dst='h36m',
            # )

            gt_smpl_pose = np.array(res['gt_poses'])
            gt_body_pose = gt_smpl_pose[..., 1:, :]
            gt_global_orient = gt_smpl_pose[..., 0, :]
            gt_betas = np.array(res['gt_betas'])
            gender = np.zeros([gt_betas.shape[0], gt_betas.shape[1]])
            if self.dataset_name == 'pw3d':
                # betas = []
                # body_pose = []
                # global_orient = []
                # gender = []
                # smpl_dict = self.human_data['smpl']

                # for idx in range(self.num_data):
                #     betas.append(smpl_dict['betas'][idx])
                #     body_pose.append(smpl_dict['body_pose'][idx])
                #     global_orient.append(smpl_dict['global_orient'][idx])
                #     if self.human_data['meta']['gender'][idx] == 'm':
                #         gender.append(0)
                #     else:
                #         gender.append(1)
                betas = torch.FloatTensor(gt_betas).view(-1, 10)
                body_pose = torch.FloatTensor(gt_body_pose).view(-1, 69)
                global_orient = torch.FloatTensor(gt_global_orient).view(-1, 3)
                gender = torch.Tensor(gender).view(-1)
                gt_output = self.body_model(betas=betas,
                                            body_pose=body_pose,
                                            global_orient=global_orient,
                                            gender=gender)
                gt_keypoints3d = gt_output['joints'].detach().cpu().numpy()
                # gt_keypoints3d,_ = convert_kps(
                # gt_keypoints3d,
                # src='smpl_54',
                # dst='h36m')
                gt_keypoints3d_mask = np.ones((len(pred_keypoints3d), 17))
            elif self.dataset_name == 'h36m':
                _, h36m_idxs, _ = get_mapping('human_data', 'h36m')
                gt_keypoints3d = \
                    self.human_data['keypoints3d'][:, h36m_idxs, :3]
                gt_keypoints3d_mask = np.ones((len(pred_keypoints3d), 17))
            elif self.dataset_name == 'humman':
                betas = []
                body_pose = []
                global_orient = []
                smpl_dict = self.human_data['smpl']
                for idx in range(self.num_data):
                    betas.append(smpl_dict['betas'][idx])
                    body_pose.append(smpl_dict['body_pose'][idx])
                    global_orient.append(smpl_dict['global_orient'][idx])
                betas = torch.FloatTensor(betas)
                body_pose = torch.FloatTensor(body_pose).view(-1, 69)
                global_orient = torch.FloatTensor(global_orient)
                gt_output = self.body_model(betas=betas,
                                            body_pose=body_pose,
                                            global_orient=global_orient)
                gt_keypoints3d = gt_output['joints'].detach().cpu().numpy()
                gt_keypoints3d_mask = np.ones((len(pred_keypoints3d), 24))
            else:
                raise NotImplementedError()

            # SMPL_49 only!
            if gt_keypoints3d.shape[1] == 49:
                assert pred_keypoints3d.shape[1] == 49

                gt_keypoints3d = gt_keypoints3d[:, 25:, :]
                pred_keypoints3d = pred_keypoints3d[:, 25:, :]

                joint_mapper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
                gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
                pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]

                # we only evaluate on 14 lsp joints
                pred_pelvis = (pred_keypoints3d[:, 2] +
                               pred_keypoints3d[:, 3]) / 2
                gt_pelvis = (gt_keypoints3d[:, 2] + gt_keypoints3d[:, 3]) / 2

            # H36M for testing!
            elif gt_keypoints3d.shape[1] == 17:
                assert pred_keypoints3d.shape[-2] == 17

                H36M_TO_J17 = [
                    6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9
                ]
                H36M_TO_J14 = H36M_TO_J17[:14]
                joint_mapper = H36M_TO_J14

                pred_pelvis = pred_keypoints3d[:, 0]
                gt_pelvis = gt_keypoints3d[:, 0]

                gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
                pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]

            # keypoint 24
            elif gt_keypoints3d.shape[1] == 24:
                assert pred_keypoints3d.shape[1] == 24

                joint_mapper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
                gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
                pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]

                # we only evaluate on 14 lsp joints
                pred_pelvis = (pred_keypoints3d[:, 2] +
                               pred_keypoints3d[:, 3]) / 2
                gt_pelvis = (gt_keypoints3d[:, 2] + gt_keypoints3d[:, 3]) / 2

            else:
                pass

            pred_keypoints3d = (pred_keypoints3d -
                                pred_pelvis[:, None, :]) * 1000
            gt_keypoints3d = (gt_keypoints3d - gt_pelvis[:, None, :]) * 1000

            gt_keypoints3d_mask = gt_keypoints3d_mask[:, joint_mapper] > 0

            return pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask

    def _report_mpjpe(self, res_file, metric='mpjpe', body_part=''):
        """Cauculate mean per joint position error (MPJPE) or its variants PA-
        MPJPE.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (PA-MPJPE)
        """
        pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask = \
            self._parse_result(res_file, mode='keypoint', body_part=body_part)

        err_name = metric.upper()
        if body_part != '':
            err_name = body_part.upper() + ' ' + err_name

        if metric == 'mpjpe':
            alignment = 'none'
        elif metric == 'pa-mpjpe':
            alignment = 'procrustes'
        else:
            raise ValueError(f'Invalid metric: {metric}')

        error = keypoint_mpjpe(pred_keypoints3d, gt_keypoints3d,
                               gt_keypoints3d_mask, alignment)
        info_str = [(err_name, error)]

        return info_str

    def _report_3d_pck(self, res_file, metric='3dpck'):
        """Cauculate Percentage of Correct Keypoints (3DPCK) w. or w/o
        Procrustes alignment.
        Args:
            keypoint_results (list): Keypoint predictions. See
                'Body3DMpiInf3dhpDataset.evaluate' for details.
            metric (str): Specify mpjpe variants. Supported options are:
                - ``'3dpck'``: Standard 3DPCK.
                - ``'pa-3dpck'``:
                    3DPCK after aligning prediction to groundtruth
                    via a rigid transformation (scale, rotation and
                    translation).
        """

        pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask = \
            self._parse_result(res_file)

        err_name = metric.upper()
        if metric == '3dpck':
            alignment = 'none'
        elif metric == 'pa-3dpck':
            alignment = 'procrustes'
        else:
            raise ValueError(f'Invalid metric: {metric}')

        error = keypoint_3d_pck(pred_keypoints3d, gt_keypoints3d,
                                gt_keypoints3d_mask, alignment)
        name_value_tuples = [(err_name, error)]

        return name_value_tuples

    def _report_3d_auc(self, res_file, metric='3dauc'):
        """Cauculate the Area Under the Curve (AUC) computed for a range of
        3DPCK thresholds.
        Args:
            keypoint_results (list): Keypoint predictions. See
                'Body3DMpiInf3dhpDataset.evaluate' for details.
            metric (str): Specify mpjpe variants. Supported options are:
                - ``'3dauc'``: Standard 3DAUC.
                - ``'pa-3dauc'``: 3DAUC after aligning prediction to
                    groundtruth via a rigid transformation (scale, rotation and
                    translation).
        """

        pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask = \
            self._parse_result(res_file)

        err_name = metric.upper()
        if metric == '3dauc':
            alignment = 'none'
        elif metric == 'pa-3dauc':
            alignment = 'procrustes'
        else:
            raise ValueError(f'Invalid metric: {metric}')

        error = keypoint_3d_auc(pred_keypoints3d, gt_keypoints3d,
                                gt_keypoints3d_mask, alignment)
        name_value_tuples = [(err_name, error)]

        return name_value_tuples

    def _report_pve(self, res_file, metric='pve', body_part=''):
        """Cauculate per vertex error."""
        pred_verts, gt_verts, _ = \
            self._parse_result(res_file, mode='vertice', body_part=body_part)
        err_name = metric.upper()
        if body_part != '':
            err_name = body_part.upper() + ' ' + err_name

        if metric == 'pve':
            alignment = 'none'
        elif metric == 'pa-pve':
            alignment = 'procrustes'
        else:
            raise ValueError(f'Invalid metric: {metric}')
        error = vertice_pve(pred_verts, gt_verts, alignment)
        return [(err_name, error)]

    def _report_ihmr(self, res_file):
        """Calculate IHMR metric.

        https://arxiv.org/abs/2203.16427
        """
        pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask = \
            self._parse_result(res_file, mode='keypoint')

        pred_verts, gt_verts, _ = \
            self._parse_result(res_file, mode='vertice')

        from detrsmpl.utils.geometry import rot6d_to_rotmat
        mean_param_path = 'data/body_models/smpl_mean_params.npz'
        mean_params = np.load(mean_param_path)
        mean_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        mean_shape = torch.from_numpy(
            mean_params['shape'][:].astype('float32')).unsqueeze(0)
        mean_pose = rot6d_to_rotmat(mean_pose).view(1, 24, 3, 3)
        mean_output = self.body_model(betas=mean_shape,
                                      body_pose=mean_pose[:, 1:],
                                      global_orient=mean_pose[:, :1],
                                      pose2rot=False)
        mean_verts = mean_output['vertices'].detach().cpu().numpy() * 1000.
        dis = (gt_verts - mean_verts) * (gt_verts - mean_verts)
        dis = np.sqrt(dis.sum(axis=-1)).mean(axis=-1)
        # from the most remote one to the nearest one
        idx_order = np.argsort(dis)[::-1]
        num_data = idx_order.shape[0]

        def report_ihmr_idx(idx):
            mpvpe = vertice_pve(pred_verts[idx], gt_verts[idx])
            mpjpe = keypoint_mpjpe(pred_keypoints3d[idx], gt_keypoints3d[idx],
                                   gt_keypoints3d_mask[idx], 'none')
            pampjpe = keypoint_mpjpe(pred_keypoints3d[idx],
                                     gt_keypoints3d[idx],
                                     gt_keypoints3d_mask[idx], 'procrustes')
            return (mpvpe, mpjpe, pampjpe)

        def report_ihmr_tail(percentage):
            cur_data = int(num_data * percentage / 100.0)
            idx = idx_order[:cur_data]
            mpvpe, mpjpe, pampjpe = report_ihmr_idx(idx)
            res_mpvpe = ('bMPVPE Tail ' + str(percentage) + '%', mpvpe)
            res_mpjpe = ('bMPJPE Tail ' + str(percentage) + '%', mpjpe)
            res_pampjpe = ('bPA-MPJPE Tail ' + str(percentage) + '%', pampjpe)
            return [res_mpvpe, res_mpjpe, res_pampjpe]

        def report_ihmr_all(num_bin):
            num_per_bin = np.array([0 for _ in range(num_bin)
                                    ]).astype(np.float32)
            sum_mpvpe = np.array([0
                                  for _ in range(num_bin)]).astype(np.float32)
            sum_mpjpe = np.array([0
                                  for _ in range(num_bin)]).astype(np.float32)
            sum_pampjpe = np.array([0 for _ in range(num_bin)
                                    ]).astype(np.float32)
            max_dis = dis[idx_order[0]]
            min_dis = dis[idx_order[-1]]
            delta = (max_dis - min_dis) / num_bin
            for i in range(num_data):
                idx = int((dis[i] - min_dis) / delta - 0.001)
                res_mpvpe, res_mpjpe, res_pampjpe = report_ihmr_idx([i])
                num_per_bin[idx] += 1
                sum_mpvpe[idx] += res_mpvpe
                sum_mpjpe[idx] += res_mpjpe
                sum_pampjpe[idx] += res_pampjpe
            for i in range(num_bin):
                if num_per_bin[i] > 0:
                    sum_mpvpe[i] = sum_mpvpe[i] / num_per_bin[i]
                    sum_mpjpe[i] = sum_mpjpe[i] / num_per_bin[i]
                    sum_pampjpe[i] = sum_pampjpe[i] / num_per_bin[i]
            valid_idx = np.where(num_per_bin > 0)
            res_mpvpe = ('bMPVPE All', sum_mpvpe[valid_idx].mean())
            res_mpjpe = ('bMPJPE All', sum_mpjpe[valid_idx].mean())
            res_pampjpe = ('bPA-MPJPE All', sum_pampjpe[valid_idx].mean())
            return [res_mpvpe, res_mpjpe, res_pampjpe]

        metrics = []
        metrics.extend(report_ihmr_all(num_bin=100))
        metrics.extend(report_ihmr_tail(percentage=10))
        metrics.extend(report_ihmr_tail(percentage=5))
        return metrics
