# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import Optional, Union

import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
from detrsmpl.core.post_processing.bbox.assigners import build_assigner
from detrsmpl.core.post_processing.bbox.samplers import build_sampler
from detrsmpl.core.conventions.keypoints_mapping import (get_keypoint_idx,
                                                          convert_kps)
from detrsmpl.utils.geometry import batch_rodrigues
from detrsmpl.utils.geometry import project_points
from detrsmpl.utils.misc import multi_apply
from ..backbones.builder import build_backbone
from ..body_models.builder import build_body_model
from ..heads.builder import build_head
from ..losses.builder import build_loss
from ..necks.builder import build_neck
from .base_architecture import BaseArchitecture

# from mmdet.core import bbox2result


class MultiBodyEstimator(BaseArchitecture, metaclass=ABCMeta):
    def __init__(
            self,
            backbone: Optional[Union[dict, None]] = None,
            neck: Optional[Union[dict, None]] = None,
            head: Optional[Union[dict, None]] = None,
            disc: Optional[Union[dict, None]] = None,
            registration: Optional[Union[dict, None]] = None,
            body_model_train: Optional[Union[dict, None]] = None,
            body_model_test: Optional[Union[dict, None]] = None,
            convention: Optional[str] = 'human_data',
            loss_keypoints2d: Optional[Union[dict, None]] = None,
            loss_keypoints3d: Optional[Union[dict, None]] = None,
            loss_vertex: Optional[Union[dict, None]] = None,
            loss_smpl_pose: Optional[Union[dict, None]] = None,
            loss_smpl_betas: Optional[Union[dict, None]] = None,
            loss_camera: Optional[Union[dict, None]] = None,
            loss_cls: Optional[Union[dict,
                                     None]] = dict(type='CrossEntropyLoss',
                                                   bg_cls_weight=0.1,
                                                   use_sigmoid=False,
                                                   loss_weight=1.0,
                                                   class_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0),
            init_cfg: Optional[Union[list, dict, None]] = None,
            train_cfg:
        Optional[Union[dict, None]] = dict(assigner=dict(
            type='HungarianAssigner',
            kp3d_cost=dict(
                type='Keypoints3DCost', convention='smpl_54', weight=5.0),
            kp2d_cost=dict(
                type='Keypoints2DCost', convention='smpl_54', weight=5.0),
            # cls_cost=dict(type='ClassificationCost', weight=1.),
            # reg_cost=dict(type='BBoxL1Cost', weight=5.0),
            # iou_cost=dict(
            #     type='IoUCost', iou_mode='giou', weight=2.0))
        )),
            test_cfg: Optional[Union[dict, None]] = None):

        super(MultiBodyEstimator, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        head.update(train_cfg=train_cfg)
        head.update(test_cfg=test_cfg)
        self.head = build_head(head)
        # class_weight = loss_cls.get('class_weight', None)
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            # TODO: update these
            # assert loss_cls['loss_weight'] == assigner['kp3d_cost']['weight'], \
            #     'The classification weight for loss and matcher should be' \
            #     'exactly the same.'
            # assert loss_bbox['loss_weight'] == assigner['kp3d_cost'][
            #     'weight'], 'The regression L1 weight for loss and matcher ' \
            #     'should be exactly the same.'
            # assert loss_iou['loss_weight'] == assigner['kp3d_cost']['weight'], \
            #     'The regression iou weight for loss and matcher should be' \
            #     'exactly the same.'
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # build loss
        self.loss_keypoints2d = build_loss(loss_keypoints2d)
        self.loss_keypoints3d = build_loss(loss_keypoints3d)
        self.loss_vertex = build_loss(loss_vertex)
        self.loss_smpl_pose = build_loss(loss_smpl_pose)
        self.loss_smpl_betas = build_loss(loss_smpl_betas)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        self.body_model_train = build_body_model(body_model_train)
        self.body_model_test = build_body_model(body_model_test)
        self.convention = convention

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self, img, img_metas, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # super(SingleStageDetector, self).forward_train(img, img_metas)
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.

        has_smpl = kwargs['has_smpl']
        gt_smpl_body_pose = kwargs[
            'smpl_body_pose']  # [bs_0: [ins_num, 23, 3]]
        gt_smpl_global_orient = kwargs['smpl_global_orient']
        gt_smpl_body_pose = \
            [torch.cat((gt_smpl_global_orient[i].view(-1, 1, 3),
                        gt_smpl_body_pose[i]), dim=1).float()
             for i in range(len(gt_smpl_body_pose))]
        gt_smpl_betas = kwargs['smpl_betas']
        gt_smpl_transl = kwargs['smpl_transl']
        gt_keypoints2d = kwargs['keypoints2d']
        gt_keypoints3d = kwargs['keypoints3d']  # [bs_0: [N. K, D], ...]

        if 'has_keypoints3d' in kwargs:
            has_keypoints3d = kwargs['has_keypoints3d']
        else:
            has_keypoints3d = None

        if 'has_keypoints2d' in kwargs:
            has_keypoints2d = kwargs['has_keypoints2d']
        else:
            has_keypoints2d = None

        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        # features = self.extract_feat(img)
        features = self.backbone(img)

        if self.neck is not None:
            features = self.neck(features)

        # outputs_classes, outputs_coords,
        pred_pose, \
            pred_betas, pred_cameras, _, _  = self.head(features, img_metas)

        L, B, N = pred_pose.shape[:3]
        if self.body_model_train is not None:
            pred_output = self.body_model_train(
                betas=pred_betas.reshape(L * B * N, 10),
                body_pose=pred_pose.reshape(L * B * N, 24, 3, 3)[:, 1:],
                global_orient=pred_pose.reshape(L * B * N, 24, 3,
                                                3)[:, 0].unsqueeze(1),
                pose2rot=False,
                num_joints=gt_keypoints2d[0].shape[1])
            pred_keypoints3d = pred_output['joints'].reshape(L, B, N, -1, 3)
            pred_vertices = pred_output['vertices'].reshape(L, B, N, 6890, 3)
        # loss
        num_dec_layers = pred_pose.shape[0]

        all_gt_smpl_body_pose_list = [
            gt_smpl_body_pose for _ in range(num_dec_layers)
        ]
        all_gt_smpl_global_orient_list = [
            gt_smpl_global_orient for _ in range(num_dec_layers)
        ]
        all_gt_smpl_betas_list = [gt_smpl_betas for _ in range(num_dec_layers)]
        all_gt_smpl_transl_list = [
            gt_smpl_transl for _ in range(num_dec_layers)
        ]
        all_gt_keypoints2d_list = [
            gt_keypoints2d for _ in range(num_dec_layers)
        ]
        all_gt_keypoints3d_list = [
            gt_keypoints3d for _ in range(num_dec_layers)
        ]
        all_has_smpl_list = [has_smpl for _ in range(num_dec_layers)]
        all_has_keypoints3d_list = [
            has_keypoints3d for _ in range(num_dec_layers)
        ]
        all_has_keypoints2d_list = [
            has_keypoints2d for _ in range(num_dec_layers)
        ]
        all_gt_ignore_list = [None for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        # all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        # all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        # all_gt_bboxes_ignore_list = [
        #     gt_bboxes_ignore for _ in range(num_dec_layers)
        # ]
        # computer loss for each layer
        (kp2d_loss, kp3d_loss, vert_loss, pose_loss, beta_loss) = multi_apply(
            self.compute_losses, pred_pose, pred_betas, pred_keypoints3d,
            pred_vertices, pred_cameras, all_gt_smpl_body_pose_list,
            all_gt_smpl_betas_list, all_gt_keypoints2d_list,
            all_gt_keypoints3d_list, all_has_keypoints2d_list,
            all_has_keypoints3d_list, all_has_smpl_list, img_metas_list,
            all_gt_ignore_list)

        losses = {}
        losses['keypoints2d_loss'] = kp2d_loss[-1]
        losses['keypoints3d_loss'] = kp3d_loss[-1]
        losses['vertex_loss'] = vert_loss[-1]
        losses['smpl_pose_loss'] = pose_loss[-1]
        losses['smpl_betas_loss'] = beta_loss[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for (kp2d_loss_i, kp3d_loss_i, vert_loss_i, pose_loss_i,
             beta_loss_i) in zip(kp2d_loss[:-1], kp3d_loss[:-1],
                                 vert_loss[:-1], pose_loss[:-1],
                                 beta_loss[:-1]):
            losses[f'd{num_dec_layer}.keypoints2d_loss'] = kp2d_loss_i
            losses[f'd{num_dec_layer}.keypoints3d_loss'] = kp3d_loss_i
            losses[f'd{num_dec_layer}.vertex_loss'] = vert_loss_i
            losses[f'd{num_dec_layer}.smpl_pose_loss'] = pose_loss_i
            losses[f'd{num_dec_layer}.smpl_betas_loss'] = beta_loss_i
            num_dec_layer += 1

        return losses

    def compute_losses(self,
                       outputs_poses,
                       outputs_shapes,
                       outputs_kp3ds,
                       outputs_verts,
                       outputs_cameras,
                       all_gt_smpl_body_pose_list,
                       all_gt_smpl_betas_list,
                       all_gt_kp2d_list,
                       all_gt_kp3d_list,
                       all_has_keypoints2d_list,
                       all_has_keypoints3d_list,
                       all_has_smpl_list,
                       img_metas_list,
                       all_gt_ignore_list=None):
        """_summary_
            loss_single
                get_targets
        Args:
            outputs_poses (_type_): with shape [B, N, 24, 3, 3]
            outputs_shapes (_type_): _description_
            all_gt_smpl_body_pose_list (_type_): _description_
            all_gt_smpl_betas_list (_type_): _description_
            all_gt_kp2d_list (Torch.tensor):
            all_gt_kp3d_list (list): with shape [B, N, K, D]
            img_metas_list (_type_): _description_
            all_gt_ignore_list (_type_): _description_
        """
        num_img = outputs_poses.size(0)  # batch_size
        all_pred_smpl_pose_list = [outputs_poses[i] for i in range(num_img)]
        all_pred_smpl_shape_list = [outputs_shapes[i] for i in range(num_img)]
        all_pred_kp3d_list = [outputs_kp3ds[i] for i in range(num_img)]
        all_pred_vert_list = [outputs_verts[i] for i in range(num_img)]
        all_pred_cam_list = [outputs_cameras[i] for i in range(num_img)]

        gt_bboxes_ignore_list = [all_gt_ignore_list for _ in range(num_img)]

        if all_has_keypoints2d_list is None:
            all_has_keypoints2d_list = [
                all_has_keypoints2d_list for _ in range(num_img)
            ]

        if all_has_keypoints3d_list is None:
            all_has_keypoints3d_list = [
                all_has_keypoints3d_list for _ in range(num_img)
            ]

        if all_has_smpl_list is None:
            all_has_smpl_list = [all_has_smpl_list for _ in range(num_img)]

        # for each batch data
        (kp2d_list, kp2d_weight_list, kp3d_list, kp3d_weight_list,
         smpl_pose_list, smpl_pose_weight_list, smpl_shape_list,
         smpl_shape_weight_list, vert_list, vert_weight_list, has_smpl_list,
         has_keypoints2d_list, has_keypoints3d_list, pos_inds_list,
         neg_inds_list) = multi_apply(
             self.prepare_targets,
             all_pred_smpl_pose_list,
             all_pred_smpl_shape_list,
             all_pred_kp3d_list,
             all_pred_vert_list,
             all_pred_cam_list,
             all_gt_smpl_body_pose_list,
             all_gt_smpl_betas_list,
             all_gt_kp2d_list,
             all_gt_kp3d_list,
             all_has_keypoints2d_list,
             all_has_keypoints3d_list,
             all_has_smpl_list,
             img_metas_list,
             gt_bboxes_ignore_list,
         )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        K = outputs_kp3ds.shape[-2]

        gt_kp2d = torch.cat(kp2d_list, 0)
        kp2d_weight = torch.cat(kp2d_weight_list, 0)
        pred_cam = outputs_cameras.reshape(-1, 3)
        # pred_kp2d = torch.cat()

        gt_kp3d = torch.cat(kp3d_list, 0)
        kp3d_weight = torch.cat(kp3d_weight_list, 0)
        pred_kp3d = outputs_kp3ds.reshape(-1, K, 3)

        gt_smpl_pose = torch.cat(smpl_pose_list, 0)
        smpl_pose_weight = torch.cat(smpl_pose_weight_list, 0)
        pred_smpl_pose = outputs_poses.reshape(-1, 24, 3, 3)

        gt_smpl_shape = torch.cat(smpl_shape_list, 0)
        smpl_shape_weight = torch.cat(smpl_shape_weight_list, 0)
        pred_smpl_shape = outputs_shapes.reshape(-1, 10)

        gt_vert = torch.cat(vert_list, 0)
        vert_weight = torch.cat(vert_weight_list, 0)
        pred_verts = outputs_verts.reshape(-1, 6890, 3)

        has_smpl = torch.cat(has_smpl_list, 0).squeeze()
        has_keypoints2d = torch.cat(has_keypoints2d_list, 0).squeeze()
        has_keypoints3d = torch.cat(has_keypoints3d_list, 0).squeeze()

        # losses = {}
        if self.loss_keypoints2d is not None:
            keypoints2d_loss = self.compute_keypoints2d_loss(
                pred_kp3d, pred_cam, gt_kp2d, has_keypoints2d=has_keypoints2d)
        else:
            keypoints2d_loss = 0.0

        if self.loss_keypoints3d is not None:
            keypoints3d_loss = self.compute_keypoints3d_loss(
                pred_kp3d,
                gt_kp3d,
                has_keypoints3d=has_keypoints3d,
            )
        else:
            keypoints3d_loss = 0.0

        if self.loss_vertex is not None:
            vertex_loss = self.compute_vertex_loss(pred_verts,
                                                   gt_vert,
                                                   has_smpl=has_smpl)
        else:
            vertex_loss = 0.0

        if self.loss_smpl_pose is not None:
            smpl_pose_loss = self.compute_smpl_pose_loss(pred_smpl_pose,
                                                         gt_smpl_pose,
                                                         has_smpl=has_smpl)
        else:
            smpl_pose_loss = 0.0

        if self.loss_smpl_betas is not None:
            smpl_betas_loss = self.compute_smpl_betas_loss(pred_smpl_shape,
                                                           gt_smpl_shape,
                                                           has_smpl=has_smpl)
        else:
            smpl_betas_loss = 0.0
        # if self.loss_iou is not None:
        #     losses['iou_loss'] = self.loss_iou()

        # if self.loss_bbox is not None:
        #     losses['bbox_loss'] = self.loss_bbox()

        # if self.loss_cls is not None:
        #     losses['cls_loss'] = self.loss_bbox()

        return (keypoints2d_loss, keypoints3d_loss, vertex_loss,
                smpl_pose_loss, smpl_betas_loss)

    def prepare_targets(self, pred_smpl_pose, pred_smpl_shape, pred_kp3d,
                        pred_vert, pred_cam, gt_smpl_pose, gt_smpl_shape,
                        gt_kp2d, gt_kp3d, has_keypoints2d, has_keypoints3d,
                        has_smpl, img_meta, gt_bboxes_ignore):
        """_summary_

        Args:
            all_pred_smpl_pose (_type_): _description_
            all_pred_smpl_shape (_type_): _description_
            all_pred_kp3d (_type_): _description_
            all_pred_vert (_type_): _description_
            all_gt_smpl_body_pose (_type_): _description_
            all_gt_smpl_betas (_type_): _description_
            all_gt_kp2d (_type_): _description_
            all_gt_kp3d (_type_): with shape [N, K, D]
            img_meta (_type_): _description_
            gt_bboxes_ignore (_type_): _description_
        """
        num_query = pred_smpl_pose.shape[0]
        assign_result = self.assigner.assign(pred_smpl_pose, pred_smpl_shape,
                                             pred_kp3d, pred_vert, pred_cam,
                                             gt_smpl_pose, gt_smpl_shape,
                                             gt_kp2d, gt_kp3d, has_keypoints2d,
                                             has_keypoints3d, has_smpl,
                                             img_meta, gt_bboxes_ignore)

        gt_smpl_pose = gt_smpl_pose.float()
        gt_smpl_shape = gt_smpl_shape.float()
        gt_kp2d = gt_kp2d.float()
        gt_kp3d = gt_kp3d.float()
        has_keypoints2d = has_keypoints2d.float()
        has_keypoints3d = has_keypoints3d.float()
        has_smpl = has_smpl.float()

        sampling_result = self.sampler.sample(assign_result, pred_smpl_pose,
                                              gt_smpl_pose)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # img_h, img_w, _ = img_meta['img_shape']

        # kp2d target
        kp2d_targets = torch.zeros_like(pred_kp3d[..., :2])
        kp2d_weights = torch.zeros_like(pred_kp3d[..., :2])
        kp2d_targets[pos_inds] = gt_kp2d[sampling_result.pos_assigned_gt_inds][
            ..., :2]
        kp2d_weights[pos_inds] = gt_kp2d[sampling_result.pos_assigned_gt_inds][
            ..., [2]].repeat(1, 1, 2)
        kp2d_targets = torch.cat(
            [kp2d_targets, kp2d_weights[..., 0].unsqueeze(-1)], dim=-1)
        # kp3d target
        kp3d_targets = torch.zeros_like(pred_kp3d)
        kp3d_weights = torch.zeros_like(pred_kp3d)
        kp3d_targets[pos_inds] = gt_kp3d[sampling_result.pos_assigned_gt_inds][
            ..., :3]
        kp3d_weights[pos_inds] = gt_kp3d[sampling_result.pos_assigned_gt_inds][
            ..., [3]].repeat(1, 1, 3)
        kp3d_targets = torch.cat(
            [kp3d_targets, kp3d_weights[..., 0].unsqueeze(-1)], dim=-1)

        # smpl_pose target
        smpl_pose_targets = torch.zeros_like(pred_smpl_pose)
        smpl_pose_weights = torch.zeros_like(pred_smpl_pose)
        gt_smpl_pose_rotmat = batch_rodrigues(gt_smpl_pose.view(-1, 3)).view(
            -1, 24, 3, 3)
        smpl_pose_targets[pos_inds] = gt_smpl_pose_rotmat[
            sampling_result.pos_assigned_gt_inds]
        smpl_pose_weights[pos_inds] = 1.0

        # smpl_beta target
        smpl_shape_targets = torch.zeros_like(pred_smpl_shape)
        smpl_shape_weights = torch.zeros_like(pred_smpl_shape)
        smpl_shape_targets[pos_inds] = gt_smpl_shape[
            sampling_result.pos_assigned_gt_inds]
        smpl_shape_weights[pos_inds] = 1.0

        # verts
        if self.body_model_train is not None:
            gt_output = self.body_model_train(
                betas=gt_smpl_shape,
                body_pose=gt_smpl_pose_rotmat[:, 1:],
                global_orient=gt_smpl_pose_rotmat[:, 0].unsqueeze(1),
                pose2rot=False)
            gt_vertices = gt_output['vertices']
            gt_model_joints = gt_output['joints']

            vert_targets = torch.zeros_like(pred_vert)
            vert_weights = torch.zeros_like(pred_vert)
            vert_targets[pos_inds] = gt_vertices[
                sampling_result.pos_assigned_gt_inds]
            vert_weights[pos_inds] = 1.0

        if has_keypoints2d is not None:
            has_keypoints2d_ = torch.zeros(
                (num_query, 1)).to(smpl_pose_targets.device)
            has_keypoints2d_[pos_inds] = has_keypoints2d[
                sampling_result.pos_assigned_gt_inds]
        else:
            has_keypoints2d_ = None

        if has_keypoints3d is not None:
            has_keypoints3d_ = torch.zeros(
                (num_query, 1)).to(smpl_pose_targets.device)
            has_keypoints3d_[pos_inds] = has_keypoints3d[
                sampling_result.pos_assigned_gt_inds]
        else:
            has_keypoints3d_ = None

        if has_smpl is not None:
            has_smpl_ = torch.zeros(
                (num_query, 1)).to(smpl_pose_targets.device)
            # if len(sampling_result.pos_assigned_gt_inds) == 1:
            #     has_smpl_[pos_inds] = has_smpl
            # else:
            has_smpl_[pos_inds] = has_smpl[
                sampling_result.pos_assigned_gt_inds]
        else:
            has_smpl_ = None
        return (kp2d_targets, kp2d_weights, kp3d_targets, kp3d_weights,
                smpl_pose_targets, smpl_pose_weights, smpl_shape_targets,
                smpl_shape_weights, vert_targets, vert_weights, has_smpl_,
                has_keypoints2d_, has_keypoints3d_, pos_inds, neg_inds)

    def forward_test(self, img, img_metas, **kwargs):
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        features = self.backbone(img)
        if self.neck is not None:
            features = self.neck(features)
        pred_pose, pred_betas, pred_cam, _, _  = \
            self.head(features, img_metas)

        # pred_pose = pred_pose[-1]
        # pred_betas = pred_betas[-1]
        # pred_cam = pred_cam[-1]

        L, B, N = pred_pose.shape[:3]
        if self.body_model_test is not None:
            pred_output = self.body_model_test(
                betas=pred_betas.reshape(L * B * N, 10),
                body_pose=pred_pose.reshape(L * B * N, 24, 3, 3)[:, 1:],
                global_orient=pred_pose.reshape(L * B * N, 24, 3,
                                                3)[:, 0].unsqueeze(1),
                pose2rot=False)
        else:
            raise ValueError('Please provide a builded body model.')

        pred_keypoints_3d = pred_output['joints'].reshape(L, B, N, -1, 3)
        pred_keypoints_3d = (pred_keypoints_3d -
                             pred_keypoints_3d[..., [0], :])
        pred_keypoints_3d = pred_keypoints_3d.detach().cpu().numpy()
        # pred_vertices = pred_output['vertices'].reshape(L, B, N, 6890, 3)
        pred_cam = pred_cam.detach().cpu().numpy()
        pred_pose = pred_pose.detach().cpu().numpy()
        pred_betas = pred_betas.detach().cpu().numpy()
        # batch, instance_num, kp_num, 4
        gt_keypoints3d = kwargs['keypoints3d'].repeat([1, N, 1, 1]).clone()
        # keypoints3d_mask = kwargs['keypoints3d_mask']
        gt_keypoints3d = gt_keypoints3d.detach().cpu().numpy()
        # gt_keypoints3d, _ = convert_kps(
        #                 gt_keypoints3d,
        #                 src='human_data',
        #                 dst='h36m')

        cost = np.sum((pred_keypoints_3d[-1] - gt_keypoints3d[..., :3]),
                      axis=(2, 3))
        index = np.argmin(abs(cost), -1)

        pred_keypoints_3d_ = []
        pred_pose_ = []
        pred_betas_ = []
        pred_cam_ = []

        for batch_i in range(B):
            ind = index[batch_i]
            pred_keypoints_3d_.append(pred_keypoints_3d[-1, batch_i, ind])
            pred_pose_.append(pred_pose[-1, batch_i, ind])
            pred_betas_.append(pred_betas[-1, batch_i, ind])
            pred_cam_.append(pred_cam[-1, batch_i, ind])

        # for img_id in range(len(img_metas)):
        #     pred_pose_ = pred_pose[:, img_id]
        #     pred_betas_ = pred_betas[:, img_id]
        #     pred_cam_ = pred_cam[:, img_id]
        #     pred_keypoints_3d_ = pred_keypoints_3d[:, img_id]
        #     pred_vertices_ = pred_vertices[:, img_id]
        #     img_shape_ = img_metas[img_id]['img_shape']

        #     result_list.append()

        all_preds = {}
        all_preds['keypoints_3d'] = np.array(pred_keypoints_3d_)
        all_preds['smpl_pose'] = np.array(pred_pose_)
        all_preds['smpl_beta'] = np.array(pred_betas_)
        all_preds['camera'] = np.array(pred_cam_)
        # all_preds['vertices'] = pred_vertices.detach().cpu().numpy()

        image_path = []
        for img_meta in img_metas:
            image_path.append(img_meta['image_path'])
        all_preds['image_path'] = image_path
        all_preds['image_idx'] = kwargs['sample_idx']
        return all_preds
        # loss

    def compute_keypoints3d_loss(
            self,
            pred_keypoints3d: torch.Tensor,
            gt_keypoints3d: torch.Tensor,
            has_keypoints3d: Optional[torch.Tensor] = None):
        """Compute loss for 3d keypoints."""
        keypoints3d_conf = gt_keypoints3d[:, :, 3].float().unsqueeze(-1)
        keypoints3d_conf = keypoints3d_conf.repeat(1, 1, 3)
        pred_keypoints3d = pred_keypoints3d.float()
        gt_keypoints3d = gt_keypoints3d[:, :, :3].float()

        # currently, only mpi_inf_3dhp and h36m have 3d keypoints
        # both datasets have right_hip_extra and left_hip_extra
        right_hip_idx = get_keypoint_idx('right_hip_extra', self.convention)
        left_hip_idx = get_keypoint_idx('left_hip_extra', self.convention)
        gt_pelvis = (gt_keypoints3d[:, right_hip_idx, :] +
                     gt_keypoints3d[:, left_hip_idx, :]) / 2
        pred_pelvis = (pred_keypoints3d[:, right_hip_idx, :] +
                       pred_keypoints3d[:, left_hip_idx, :]) / 2

        gt_keypoints3d = gt_keypoints3d - gt_pelvis[:, None, :]
        pred_keypoints3d = pred_keypoints3d - pred_pelvis[:, None, :]
        loss = self.loss_keypoints3d(pred_keypoints3d,
                                     gt_keypoints3d,
                                     reduction_override='none')

        # If has_keypoints3d is not None, then computes the losses on the
        # instances that have ground-truth keypoints3d.
        # But the zero confidence keypoints will be included in mean.
        # Otherwise, only compute the keypoints3d
        # which have positive confidence.

        # has_keypoints3d is None when the key has_keypoints3d
        # is not in the datasets
        if has_keypoints3d is None:

            valid_pos = keypoints3d_conf > 0
            if keypoints3d_conf[valid_pos].numel() == 0:
                return torch.Tensor([0]).type_as(gt_keypoints3d)
            loss = torch.sum(loss * keypoints3d_conf)
            loss /= keypoints3d_conf[valid_pos].numel()
        else:

            keypoints3d_conf = keypoints3d_conf[has_keypoints3d == 1]
            if keypoints3d_conf.shape[0] == 0:
                return torch.Tensor([0]).type_as(gt_keypoints3d)
            loss = loss[has_keypoints3d == 1]
            loss = (loss * keypoints3d_conf).mean()
        return loss

    def compute_keypoints2d_loss(
            self,
            pred_keypoints3d: torch.Tensor,
            pred_cam: torch.Tensor,
            gt_keypoints2d: torch.Tensor,
            img_res: Optional[int] = 512,
            focal_length: Optional[int] = 5000.,
            has_keypoints2d: Optional[torch.Tensor] = None):
        """Compute loss for 2d keypoints."""
        keypoints2d_conf = gt_keypoints2d[:, :, 2].float().unsqueeze(-1)
        keypoints2d_conf = keypoints2d_conf.repeat(1, 1, 2)
        gt_keypoints2d = gt_keypoints2d[:, :, :2].float()
        pred_keypoints2d = project_points(pred_keypoints3d,
                                          pred_cam,
                                          focal_length=focal_length,
                                          img_res=img_res)
        # Normalize keypoints to [-1,1]
        # The coordinate origin of pred_keypoints_2d is
        # the center of the input image.
        pred_keypoints2d = 2 * pred_keypoints2d / (img_res - 1)
        # The coordinate origin of gt_keypoints_2d is
        # the top left corner of the input image.
        gt_keypoints2d = 2 * gt_keypoints2d / (img_res - 1) - 1
        loss = self.loss_keypoints2d(pred_keypoints2d,
                                     gt_keypoints2d,
                                     reduction_override='none')

        # If has_keypoints2d is not None, then computes the losses on the
        # instances that have ground-truth keypoints2d.
        # But the zero confidence keypoints will be included in mean.
        # Otherwise, only compute the keypoints2d
        # which have positive confidence.
        # has_keypoints2d is None when the key has_keypoints2d
        # is not in the datasets

        if has_keypoints2d is None:
            valid_pos = keypoints2d_conf > 0
            if keypoints2d_conf[valid_pos].numel() == 0:
                return torch.Tensor([0]).type_as(gt_keypoints2d)
            loss = torch.sum(loss * keypoints2d_conf)
            loss /= keypoints2d_conf[valid_pos].numel()
        else:
            keypoints2d_conf = keypoints2d_conf[has_keypoints2d == 1]
            if keypoints2d_conf.shape[0] == 0:
                return torch.Tensor([0]).type_as(gt_keypoints2d)
            loss = loss[has_keypoints2d == 1]
            loss = (loss * keypoints2d_conf).mean()

        return loss

    def compute_vertex_loss(self, pred_vertices: torch.Tensor,
                            gt_vertices: torch.Tensor, has_smpl: torch.Tensor):
        """Compute loss for vertices."""
        gt_vertices = gt_vertices.float()
        conf = has_smpl.float().view(-1, 1, 1)
        conf = conf.repeat(1, gt_vertices.shape[1], gt_vertices.shape[2])
        loss = self.loss_vertex(pred_vertices,
                                gt_vertices,
                                reduction_override='none')
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_vertices)
        loss = torch.sum(loss * conf) / conf[valid_pos].numel()
        return loss

    def compute_smpl_pose_loss(self, pred_pose: torch.Tensor,
                               gt_pose: torch.Tensor, has_smpl: torch.Tensor):
        """Compute loss for smpl pose."""
        conf = has_smpl.float().view(-1)
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_pose)
        pred_pose = pred_pose[valid_pos]
        gt_pose = gt_pose[valid_pos]
        conf = conf[valid_pos]
        # gt_rotmat = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)
        loss = self.loss_smpl_pose(pred_pose,
                                   gt_pose,
                                   reduction_override='none')
        loss = loss.view(loss.shape[0], -1).mean(-1)
        loss = torch.mean(loss * conf)
        return loss

    def compute_smpl_betas_loss(self, pred_betas: torch.Tensor,
                                gt_betas: torch.Tensor,
                                has_smpl: torch.Tensor):
        """Compute loss for smpl betas."""
        conf = has_smpl.float().view(-1)
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_betas)
        pred_betas = pred_betas[valid_pos]
        gt_betas = gt_betas[valid_pos]
        conf = conf[valid_pos]
        loss = self.loss_smpl_betas(pred_betas,
                                    gt_betas,
                                    reduction_override='none')
        loss = loss.view(loss.shape[0], -1).mean(-1)
        loss = torch.mean(loss * conf)
        return loss

    def compute_camera_loss(self, cameras: torch.Tensor):
        """Compute loss for predicted camera parameters."""
        loss = self.loss_camera(cameras)
        return loss
