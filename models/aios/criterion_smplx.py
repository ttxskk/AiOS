import copy
import os
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms
from torch import Tensor
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, accuracy,
                       get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .utils import PoseProjector, sigmoid_focal_loss, MLP, OKSLoss
from typing import Optional, Union
from detrsmpl.core.conventions.keypoints_mapping import (get_keypoint_idx,
                                                         convert_kps)
from detrsmpl.utils.geometry import (batch_rodrigues, project_points_new)
from config.config import cfg
from util.human_models import smpl_x
from detrsmpl.utils.transforms import rotmat_to_aa

class SetCriterion(nn.Module):
    def __init__(self,
                 num_classes,
                 matcher,
                 weight_dict,
                 focal_alpha,
                 losses,
                 num_box_decoder_layers=2,
                 num_hand_face_decoder_layers=4,
                 num_body_points=17,
                 num_hand_points=6,
                 num_face_points=6,
                 smpl_loss_config=None,
                 convention='smplx_137'):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.vis = 0.1
        self.abs = 1
        self.num_body_points = num_body_points
        self.num_hand_points = num_hand_points
        self.num_face_points = num_face_points
        self.num_box_decoder_layers = num_box_decoder_layers
        self.num_hand_face_decoder_layers = num_hand_face_decoder_layers
        self.convention = convention
        self.body_oks = OKSLoss(linear=True,
                           num_keypoints=num_body_points,
                           eps=1e-6,
                           reduction='mean',
                           loss_weight=1.0)
        self.hand_oks = OKSLoss(linear=True,
                           num_keypoints=num_hand_points,
                           eps=1e-6,
                           reduction='mean',
                           loss_weight=1.0)
        self.face_oks = OKSLoss(linear=True,
                           num_keypoints=num_face_points,
                           eps=1e-6,
                           reduction='mean',
                           loss_weight=1.0)

    def loss_labels(self,
                    outputs,
                    targets,
                    indices,
                    idx,
                    num_boxes,
                    data_batch,
                    log=True):
        """Classification loss (Binary focal loss) targets dicts must contain
        the key "labels" containing a tensor of dim [nb_target_boxes]"""
        indices = indices[0]
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat(
            [t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2],
                                    self.num_classes,
                                    dtype=torch.int64,
                                    device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([
            src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1
        ],
                                            dtype=src_logits.dtype,
                                            layout=src_logits.layout,
                                            device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits,
                                     target_classes_onehot,
                                     num_boxes,
                                     alpha=self.focal_alpha,
                                     gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx],
                                                   target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes,
                         data_batch):
        """Compute the cardinality error, ie the absolute error in the number
        of predicted non-empty boxes This is not really a loss, it is intended
        for logging purposes only.

        It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['labels']) for v in targets],
                                      device=device)
        if tgt_lengths == 0:
            return  {'cardinality_error': pred_logits.sum()*0}
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) !=
                     pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_keypoints(self, outputs, targets, indices, 
                       idx, num_boxes, data_batch,
                       face_hand_kpt=False):
        """Compute the losses related to the keypoints."""
        indices = indices[0]
        device = outputs['pred_logits'].device
        losses = {}
        
        src_body_keypoints = outputs['pred_keypoints'][idx]  # xyxyvv
        if len(src_body_keypoints) == 0:
            losses.append({
                'loss_keypoints': src_body_keypoints.sum() * 0 + \
                    outputs['pred_smpl_cam'][idx].float().sum()*0,
                'loss_oks': src_body_keypoints.sum() * 0,
            })
        else:
            Z_pred = src_body_keypoints[:, 0:(self.num_body_points * 2)]  # [2, 2*14]
            V_pred = src_body_keypoints[:, (self.num_body_points * 2):]
            targets_body_keypoints = torch.cat(
                [t['keypoints'][i] for t, (_, i) in zip(targets, indices)],
                dim=0)
            targets_area = torch.cat(
                [t['area'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            target_body_boxes_conf = torch.cat(
                [t[i] for t, (_, i) in zip(data_batch['body_bbox_valid'], indices)], dim=0)
            Z_gt = targets_body_keypoints[:, 0:(self.num_body_points * 2)]
            V_gt: torch.Tensor = targets_body_keypoints[:, (self.num_body_points * 2):]
            body_kps_conf = V_gt.sum(-1)>0
            body_num_boxes = (body_kps_conf * target_body_boxes_conf).sum()
            oks_loss = self.body_oks(Z_pred,
                                Z_gt,
                                V_gt,
                                targets_area,
                                weight=None,
                                avg_factor=None,
                                reduction_override=None)
            oks_loss*= body_kps_conf * target_body_boxes_conf
            pose_loss = F.l1_loss(Z_pred, Z_gt, reduction='none')
            pose_loss = pose_loss * V_gt.repeat_interleave(2, dim=1)
            pose_loss = pose_loss.sum(-1) * target_body_boxes_conf
            if body_num_boxes>0:
                losses['loss_keypoints'] = pose_loss.sum() / body_num_boxes
                losses['loss_oks'] = oks_loss.sum() / body_num_boxes
            else:
                losses['loss_keypoints'] = src_body_keypoints.sum() * 0
                losses['loss_oks'] = src_body_keypoints.sum() * 0

        # lhand
        if 'pred_lhand_keypoints' in outputs and face_hand_kpt:
            src_lhand_keypoints = outputs['pred_lhand_keypoints'][idx]  # xyxyvv
            if len(src_lhand_keypoints) == 0:
                losses.update({
                    'loss_lhand_keypoints': src_lhand_keypoints.sum() * 0,
                    'loss_lhand_oks':src_lhand_keypoints.sum() * 0,
                })
            else:
                Z_pred = src_lhand_keypoints[:, 0:(self.num_hand_points * 2)]  # [2, 2*14]
                V_pred = src_lhand_keypoints[:, (self.num_hand_points * 2):]
                targets_lhand_keypoints = torch.cat(
                    [t['lhand_keypoints'][i] for t, (_, i) in zip(targets, indices)],
                    dim=0)  # i is batch_size
                targets_area = torch.cat(
                    [t['area'][i] for t, (_, i) in zip(targets, indices)], dim=0)
                target_lhand_boxes_conf = torch.cat(
                    [t[i] for t, (_, i) in zip(data_batch['lhand_bbox_valid'], indices)], dim=0)
                
                Z_gt = targets_lhand_keypoints[:, 0:(self.num_hand_points * 2)]
                V_gt: torch.Tensor = targets_lhand_keypoints[:, (self.num_hand_points * 2):]
                lhand_kps_conf = V_gt.sum(-1)>0
                lhand_num_boxes = (lhand_kps_conf*target_lhand_boxes_conf).sum()
                oks_loss = self.hand_oks(Z_pred,
                                    Z_gt,
                                    V_gt,
                                    targets_area,
                                    weight=None,
                                    avg_factor=None,
                                    reduction_override=None)
                oks_loss = oks_loss*lhand_kps_conf*target_lhand_boxes_conf
                pose_loss = F.l1_loss(Z_pred, Z_gt, reduction='none')
                pose_loss = pose_loss * V_gt.repeat_interleave(2, dim=1)
                pose_loss = pose_loss.sum(-1)*target_lhand_boxes_conf
                if lhand_num_boxes>0:
                    losses['loss_lhand_keypoints'] = pose_loss.sum() / lhand_num_boxes
                    losses['loss_lhand_oks'] = oks_loss.sum() / lhand_num_boxes
                else:
                    losses['loss_lhand_keypoints'] = src_lhand_keypoints.sum() * 0
                    losses['loss_lhand_oks'] = src_lhand_keypoints.sum() * 0
                    
        # rhand
        if 'pred_rhand_keypoints' in outputs and face_hand_kpt:
            src_rhand_keypoints = outputs['pred_rhand_keypoints'][idx]  # xyxyvv
            if len(src_rhand_keypoints) == 0:
                losses.update({
                    'loss_rhand_keypoints':
                    src_rhand_keypoints.sum() * 0,
                    'loss_rhand_oks':
                    src_rhand_keypoints.sum() * 0,
                })
            else:
                Z_pred = src_rhand_keypoints[:, 0:(self.num_hand_points * 2)]  # [2, 2*14]
                V_pred = src_rhand_keypoints[:, (self.num_hand_points * 2):]
                targets_rhand_keypoints = torch.cat(
                    [t['rhand_keypoints'][i] for t, (_, i) in zip(targets, indices)],
                    dim=0)
                targets_area = torch.cat(
                    [t['area'][i] for t, (_, i) in zip(targets, indices)], dim=0)
                target_rhand_boxes_conf = torch.cat(
                    [t[i] for t, (_, i) in zip(data_batch['rhand_bbox_valid'], indices)], dim=0)
                Z_gt = targets_rhand_keypoints[:, 0:(self.num_hand_points * 2)]
                V_gt: torch.Tensor = targets_rhand_keypoints[:, (self.num_hand_points * 2):]
                rhand_kps_conf = V_gt.sum(-1)>0
                rhand_num_boxes = (rhand_kps_conf*target_rhand_boxes_conf).sum()
                oks_loss = self.hand_oks(Z_pred,
                                    Z_gt,
                                    V_gt,
                                    targets_area,
                                    weight=None,
                                    avg_factor=None,
                                    reduction_override=None)
                oks_loss = oks_loss*rhand_kps_conf*target_rhand_boxes_conf
                pose_loss = F.l1_loss(Z_pred, Z_gt, reduction='none')
                pose_loss = pose_loss * V_gt.repeat_interleave(2, dim=1)
                pose_loss = pose_loss.sum(-1)*target_rhand_boxes_conf
                if rhand_num_boxes>0:
                    losses['loss_rhand_keypoints'] = pose_loss.sum() / rhand_num_boxes
                    losses['loss_rhand_oks'] = oks_loss.sum() / rhand_num_boxes
                else:
                    losses['loss_rhand_keypoints'] =  src_rhand_keypoints.sum() * 0
                    losses['loss_rhand_oks'] = src_rhand_keypoints.sum() * 0

        #   face
        if 'pred_face_keypoints' in outputs and face_hand_kpt:
            src_face_keypoints = outputs['pred_face_keypoints'][idx]  # xyxyvv
            if len(src_face_keypoints) == 0:
                losses.update({
                    'loss_face_keypoints': src_face_keypoints.sum() * 0,
                    'loss_face_oks':  src_face_keypoints.sum() * 0,
                })
            else:
                Z_pred = src_face_keypoints[:, 0:(self.num_face_points * 2)]  # [2, 2*14]
                V_pred = src_face_keypoints[:, (self.num_face_points * 2):]
                targets_face_keypoints = torch.cat(
                    [t['face_keypoints'][i] for t, (_, i) in zip(targets, indices)],
                    dim=0)
                targets_area = torch.cat(
                    [t['area'][i] for t, (_, i) in zip(targets, indices)], dim=0)
                target_face_boxes_conf = torch.cat(
                    [t[i] for t, (_, i) in zip(data_batch['face_bbox_valid'], indices)], dim=0)
                Z_gt = targets_face_keypoints[:, 0:(self.num_face_points * 2)]
                V_gt: torch.Tensor = targets_face_keypoints[:, (self.num_face_points * 2):]
                face_kps_conf = V_gt.sum(-1)>0
                face_num_boxes = (lhand_kps_conf*target_face_boxes_conf).sum()
                oks_loss = self.face_oks(Z_pred,
                                    Z_gt,
                                    V_gt,
                                    targets_area,
                                    weight=None,
                                    avg_factor=None,
                                    reduction_override=None)
                oks_loss = oks_loss*face_kps_conf*target_face_boxes_conf
                pose_loss = F.l1_loss(Z_pred, Z_gt, reduction='none')
                pose_loss = pose_loss * V_gt.repeat_interleave(2, dim=1)
                pose_loss = pose_loss.sum(-1)*target_face_boxes_conf    
                if face_num_boxes>0:
                    losses['loss_face_keypoints'] = pose_loss.sum() / face_num_boxes
                    losses['loss_face_oks'] = oks_loss.sum() / face_num_boxes
                else:
                    losses['loss_face_keypoints'] = src_face_keypoints.sum() * 0
                    losses['loss_face_oks'] = src_face_keypoints.sum() * 0
        
        return losses 

    def loss_smpl_pose(self, outputs, targets, indices, idx, num_boxes,
                       data_batch, face_hand_kpt=False):
        device = outputs['pred_logits'].device
        indices = indices[0]
        
        pred_smpl_body_pose = outputs['pred_smpl_pose'][idx] # 22
        pred_smpl_lhand_pose = outputs['pred_smpl_lhand_pose'][idx] # 15
        pred_smpl_rhand_pose = outputs['pred_smpl_rhand_pose'][idx] # 15
        pred_smpl_jaw_pose = outputs['pred_smpl_jaw_pose'][idx]

        pred_smplx_pose = torch.cat((pred_smpl_body_pose, pred_smpl_lhand_pose,
                                     pred_smpl_rhand_pose, pred_smpl_jaw_pose),
                                    dim=1)
        targets_smpl_pose = torch.cat(
            [t[i] for t, (_, i) in zip(data_batch['smplx_pose'], indices)],
            dim=0)
        targets_smpl_pose = batch_rodrigues(targets_smpl_pose.view(
            -1, 3)).view(-1, 53, 3, 3)
        conf = torch.cat([
            t[i] for t, (_, i) in zip(data_batch['smplx_pose_valid'], indices)
            ], dim=0)
        body_pose_valid = conf[:, :22].sum(-1) > 0
        lhand_pose_valid = conf[:, 22:37].sum(-1) > 0
        rhand_pose_valid = conf[:, 37:52].sum(-1) > 0
        face_pose_valid = conf[:, 52].sum(-1) > 0

        losses = {}
        loss_smpl_pose = \
            F.l1_loss(
                pred_smplx_pose,
                targets_smpl_pose,
                reduction='none'
            )
        loss_smpl_pose = loss_smpl_pose.sum([-1,-2]) * conf
        
        if face_hand_kpt:
            losses = {
                'loss_smpl_pose_root': loss_smpl_pose[:, 0].sum() / (body_pose_valid.sum() + 1e-6),
                'loss_smpl_pose_body': loss_smpl_pose[:, 1:22].sum() / (body_pose_valid.sum() + 1e-6),
                'loss_smpl_pose_lhand': loss_smpl_pose[:, 22:37].sum() / (lhand_pose_valid.sum() + 1e-6),
                'loss_smpl_pose_rhand': loss_smpl_pose[:, 37:52].sum() / (rhand_pose_valid.sum() + 1e-6),
                'loss_smpl_pose_jaw': loss_smpl_pose[:, 52].sum() / (face_pose_valid.sum() + 1e-6),
            }
        else:
            losses = {
                'loss_smpl_pose_root': loss_smpl_pose[:, 0].sum() / (body_pose_valid.sum() + 1e-6),
                'loss_smpl_pose_body': loss_smpl_pose[:, 1:22].sum() / (body_pose_valid.sum() + 1e-6),
                'loss_smpl_pose_lhand': 0 * loss_smpl_pose[:, 22:37].sum()/(lhand_pose_valid.sum() + 1e-6),
                'loss_smpl_pose_rhand': 0 * loss_smpl_pose[:, 37:52].sum() / (rhand_pose_valid.sum() + 1e-6),
                'loss_smpl_pose_jaw': 0*loss_smpl_pose[:, 52].sum() / (face_pose_valid.sum() + 1e-6),
            }
        return losses

    def loss_smpl_beta(self, outputs, targets, indices, idx, num_boxes,
                       data_batch, face_hand_kpt=False):
        indices = indices[0]
        device = outputs['pred_logits'].device
        pred_smpl_betas = outputs['pred_smpl_beta'][idx]
        
        targets_smpl_betas = torch.cat(
            [t[i] for t, (_, i) in zip(data_batch['smplx_shape'], indices)],
            dim=0)

        losses = {}
        conf = torch.cat([t[i] for t, (_, i) in zip(data_batch['smplx_shape_valid'], indices)], dim=0)
        if conf.sum() == 0:
            return {
                'loss_smpl_beta': pred_smpl_betas.sum() * 0
            }
        loss_smpl_betas = \
            F.l1_loss(
                pred_smpl_betas,
                targets_smpl_betas,
                reduction='none'
            )
        loss_smpl_betas = loss_smpl_betas.sum(-1) * conf
        losses = {'loss_smpl_beta': loss_smpl_betas.sum() / (conf.sum() + 1e-6)}
        return losses

    def loss_smpl_expr(self, outputs, targets, indices, idx, num_boxes,
                       data_batch, face_hand_kpt=False):
        indices = indices[0]
        device = outputs['pred_logits'].device
        pred_smpl_expr = outputs['pred_smpl_expr'][idx]
        targets_smpl_expr = torch.cat([t[i] for t, (_, i) in zip(data_batch['smplx_expr'], indices)], dim=0)

        conf = torch.cat([t[i] for t, (_, i) in zip(data_batch['smplx_expr_valid'], indices)], dim=0)
        if conf.sum() == 0:
            return {
                'loss_smpl_expr': pred_smpl_expr.sum() * 0
            }

        loss_smpl_expr = \
            F.l1_loss(
                pred_smpl_expr,
                targets_smpl_expr,
                reduction='none'
            )
        loss_smpl_expr = loss_smpl_expr.sum(-1) * conf
        
        losses = {} 
        if face_hand_kpt:
            losses = {'loss_smpl_expr': loss_smpl_expr.sum() / (conf.sum() + 1e-6)}
        else:
            losses = {'loss_smpl_expr': 0 * loss_smpl_expr.sum() / (conf.sum() + 1e-6) }
            
        return losses

    def loss_smpl_kp3d(self,
                       outputs,
                       targets,
                       indices,
                       idx,
                       num_boxes,
                       data_batch,
                       has_keypoints3d=None,
                       face_hand_kpt=False):

        # supervision for keypoints3d wo/ ra
        device = outputs['pred_logits'].device
        indices = indices[0]
        
        pred_smpl_kp3d = outputs['pred_smpl_kp3d'][idx].float()
        # meta_info['joint_valid'] * meta_info['is_3D'][:, None, None])
        targets_smpl_kp3d = torch.cat(
            [t[i] for t, (_, i) in zip(data_batch['joint_cam'], indices)],
            dim=0)
        losses = {}
        targets_kp3d_conf = targets_smpl_kp3d[:,:,3:].clone()
        targets_smpl_kp3d = targets_smpl_kp3d[:,:,:3]
        
        targets_is_3d = torch.cat([
            t[None, None].repeat(len(i), 1, 1)
            for t, (_, i) in zip(data_batch['is_3D'], indices)
        ], dim=0)

        targets_kp3d_conf = (targets_kp3d_conf * targets_is_3d)
        pelvis_idx = get_keypoint_idx('pelvis', self.convention)
        targets_pelvis = targets_smpl_kp3d[..., pelvis_idx, :]
        pred_pelvis = pred_smpl_kp3d[..., pelvis_idx, :]

        targets_smpl_kp3d = targets_smpl_kp3d - targets_pelvis[:, None, :]
        pred_smpl_kp3d = pred_smpl_kp3d - pred_pelvis[:, None, :]

        losses = {}
        body_idx = smpl_x.joint_part['body']
        face_idx = smpl_x.joint_part['face']
        lhand_idx = smpl_x.joint_part['lhand']
        rhand_idx = smpl_x.joint_part['rhand']
        
        loss_smpl_kp3d = F.l1_loss(pred_smpl_kp3d,
                                   targets_smpl_kp3d,
                                   reduction='none')

        body_kp3d_valid = targets_kp3d_conf[:, body_idx].sum([-1,-2]) > 0
        lhand_kp3d_valid = targets_kp3d_conf[:, lhand_idx].sum([-1,-2]) > 0
        rhand_kp3d_valid = targets_kp3d_conf[:, rhand_idx].sum([-1,-2]) > 0
        face_kp3d_valid = targets_kp3d_conf[:, face_idx].sum([-1,-2]) > 0
        
        loss_smpl_kp3d = loss_smpl_kp3d * targets_kp3d_conf
        
        if face_hand_kpt:
            losses['loss_smpl_body_kp3d'] = torch.sum(loss_smpl_kp3d[:, body_idx, :])  / (body_kp3d_valid.sum() + 1e-6)
            losses['loss_smpl_lhand_kp3d'] = torch.sum(loss_smpl_kp3d[:, lhand_idx, :]) / (lhand_kp3d_valid.sum() + 1e-6)
            losses['loss_smpl_rhand_kp3d'] = torch.sum(loss_smpl_kp3d[:, rhand_idx, :]) / (rhand_kp3d_valid.sum() + 1e-6)
            losses['loss_smpl_face_kp3d'] = torch.sum(loss_smpl_kp3d[:, face_idx, :]) / (face_kp3d_valid.sum() + 1e-6)
        else:
            losses['loss_smpl_body_kp3d'] = torch.sum(loss_smpl_kp3d[:, body_idx, :])  / (body_kp3d_valid.sum() + 1e-6)
            losses['loss_smpl_lhand_kp3d'] = 0*torch.sum(loss_smpl_kp3d[:, lhand_idx, :]) / (lhand_kp3d_valid.sum() + 1e-6)
            losses['loss_smpl_rhand_kp3d'] = 0*torch.sum(loss_smpl_kp3d[:, rhand_idx, :]) / (rhand_kp3d_valid.sum() + 1e-6)
            losses['loss_smpl_face_kp3d'] = 0*torch.sum(loss_smpl_kp3d[:, face_idx, :]) / (face_kp3d_valid.sum() + 1e-6)
        return losses

    def loss_smpl_kp3d_ra(self,
                          outputs,
                          targets,
                          indices,
                          idx,
                          num_boxes,
                          data_batch,
                          has_keypoints3d=None,
                          face_hand_kpt=False):
        # supervision for keypoints3d w/ ra
        device = outputs['pred_logits'].device
        indices = indices[0]

        pred_smpl_kp3d = outputs['pred_smpl_kp3d'][idx].float()
        targets_smpl_kp3d = torch.cat([
            t[i] for t, (_, i) in zip(data_batch['smplx_joint_cam'], indices)],
            dim=0)
        losses = {}
        targets_kp3d_conf = targets_smpl_kp3d[:,:,3:].clone()
        targets_smpl_kp3d = targets_smpl_kp3d[:,:,:3]
        
        targets_is_3d = torch.cat([
            t[None, None].repeat(len(i), 1, 1)
            for t, (_, i) in zip(data_batch['is_3D'], indices)],dim=0)
        
        targets_kp3d_conf = (targets_kp3d_conf * targets_is_3d).repeat(1, 1, 3)
        pelvis_idx = get_keypoint_idx('pelvis', self.convention)
        targets_pelvis = targets_smpl_kp3d[..., pelvis_idx, :]
        pred_pelvis = pred_smpl_kp3d[..., pelvis_idx, :]

        targets_smpl_kp3d = targets_smpl_kp3d - targets_pelvis[:, None, :]
        pred_smpl_kp3d = pred_smpl_kp3d - pred_pelvis[:, None, :]
        # calculate body, face and hand loss separately:

        losses = {}
        body_idx = smpl_x.joint_part['body']
        face_idx = smpl_x.joint_part['face']
        lhand_idx = smpl_x.joint_part['lhand']
        rhand_idx = smpl_x.joint_part['rhand']

        body_kp3d_valid = targets_kp3d_conf[:, body_idx].sum([-1,-2]) > 0
        lhand_kp3d_valid = targets_kp3d_conf[:, lhand_idx].sum([-1,-2]) > 0
        rhand_kp3d_valid = targets_kp3d_conf[:, rhand_idx].sum([-1,-2]) > 0
        face_kp3d_valid = targets_kp3d_conf[:, face_idx].sum([-1,-2]) > 0

        loss_smpl_body_kp3d = F.l1_loss(pred_smpl_kp3d[:, body_idx, :],
                                        targets_smpl_kp3d[:, body_idx, :],
                                        reduction='none')
        
        loss_smpl_body_kp3d = torch.sum(
            loss_smpl_body_kp3d * targets_kp3d_conf[:, body_idx, :])
        losses['loss_smpl_body_kp3d_ra'] = loss_smpl_body_kp3d / (body_kp3d_valid.sum() + 1e-6)
        
        face_cam = pred_smpl_kp3d[:, face_idx, :]
        neck_cam = pred_smpl_kp3d[:, smpl_x.neck_idx, None, :]
        face_cam = face_cam - neck_cam
        loss_smpl_face_kp3d = F.l1_loss(face_cam,
                                        targets_smpl_kp3d[:, face_idx, :],
                                        reduction='none')
        loss_smpl_face_kp3d = torch.sum(
            loss_smpl_face_kp3d * targets_kp3d_conf[:, face_idx, :])
        if face_hand_kpt:
            losses['loss_smpl_face_kp3d_ra'] = (loss_smpl_face_kp3d / (face_kp3d_valid.sum() + 1e-6))
        else:
            losses['loss_smpl_face_kp3d_ra'] = 0 * (loss_smpl_face_kp3d / (face_kp3d_valid.sum() + 1e-6))
        
        lhand_cam = pred_smpl_kp3d[:, lhand_idx, :]
        lwrist_cam = pred_smpl_kp3d[:, smpl_x.lwrist_idx, None, :]
        lhand_cam = lhand_cam - lwrist_cam
        loss_smpl_lhand_kp3d = F.l1_loss(lhand_cam,
                                            targets_smpl_kp3d[:, lhand_idx, :],
                                            reduction='none')
        loss_smpl_lhand_kp3d = torch.sum(
            loss_smpl_lhand_kp3d * targets_kp3d_conf[:, lhand_idx, :])
        
        if face_hand_kpt:
            losses['loss_smpl_lhand_kp3d_ra'] = (loss_smpl_lhand_kp3d / (lhand_kp3d_valid.sum() + 1e-6))
        else:
            losses['loss_smpl_lhand_kp3d_ra'] = 0*(loss_smpl_lhand_kp3d / (lhand_kp3d_valid.sum() + 1e-6))
            
        rhand_cam = pred_smpl_kp3d[:, rhand_idx, :]
        rwrist_cam = pred_smpl_kp3d[:, smpl_x.rwrist_idx, None, :]
        rhand_cam = rhand_cam - rwrist_cam

        loss_smpl_rhand_kp3d = F.l1_loss(rhand_cam,
                                            targets_smpl_kp3d[:, rhand_idx, :],
                                            reduction='none')
        loss_smpl_rhand_kp3d = torch.sum(
            loss_smpl_rhand_kp3d * targets_kp3d_conf[:, rhand_idx, :])
        
        if face_hand_kpt:
            losses['loss_smpl_rhand_kp3d_ra'] = (loss_smpl_rhand_kp3d / (rhand_kp3d_valid.sum() + 1e-6))
        else:
            losses['loss_smpl_rhand_kp3d_ra'] = 0*(loss_smpl_rhand_kp3d / (rhand_kp3d_valid.sum() + 1e-6))

        return losses

    def loss_smpl_kp2d(self,
                       outputs,
                       targets,
                       indices,
                       idx,
                       num_boxes,
                       data_batch,
                       focal_length=5000.,
                       has_keypoints2d=None,
                       face_hand_kpt=False):
        """Compute loss for 2d keypoints."""
        device = outputs['pred_logits'].device
        indices = indices[0]
        pred_smpl_kp3d = outputs['pred_smpl_kp3d'][idx].float()#.detach()
        pred_cam = outputs['pred_smpl_cam'][idx].float()
        targets_kp2d = torch.cat([t[i] for t, (_, i) in zip(data_batch['joint_img'], indices)], dim=0)

        keypoints2d_conf =  targets_kp2d[:,:,2:].clone()
        targets_kp2d = targets_kp2d[:, :, :2].float()
        targets_kp2d[:,:,0] = targets_kp2d[:,:,0]/cfg.output_hm_shape[2]
        targets_kp2d[:,:,1] = targets_kp2d[:,:,1]/cfg.output_hm_shape[1]
        img_wh =  torch.cat([data_batch['img_shape'][i][None] for i in idx[0]], dim=0).flip(-1)

        pred_smpl_kp2d = project_points_new(
            points_3d=pred_smpl_kp3d,
            pred_cam=pred_cam,
            focal_length=focal_length,
            camera_center=img_wh/2
        )
        pred_smpl_kp2d = pred_smpl_kp2d / img_wh[:, None]
        
        losses = {}
        body_idx = smpl_x.joint_part['body']
        face_idx = smpl_x.joint_part['face']
        lhand_idx = smpl_x.joint_part['lhand']
        rhand_idx = smpl_x.joint_part['rhand']
        
        body_kp2d_valid = keypoints2d_conf[:, body_idx].sum([-1,-2]) > 0
        lhand_kp2d_valid = keypoints2d_conf[:, lhand_idx].sum([-1,-2]) > 0
        rhand_kp2d_valid = keypoints2d_conf[:, rhand_idx].sum([-1,-2]) > 0
        face_kp2d_valid = keypoints2d_conf[:, face_idx].sum([-1,-2]) > 0
        
        loss_smpl_kp2d = F.l1_loss(pred_smpl_kp2d,
                                   targets_kp2d,
                                   reduction='none')
        loss_smpl_kp2d = loss_smpl_kp2d * keypoints2d_conf


        if face_hand_kpt:
            losses['loss_smpl_body_kp2d'] = torch.sum(loss_smpl_kp2d[:, body_idx, :])  / (body_kp2d_valid.sum() + 1e-6)
            losses['loss_smpl_lhand_kp2d'] = torch.sum(loss_smpl_kp2d[:, lhand_idx, :]) / (lhand_kp2d_valid.sum() + 1e-6)
            losses['loss_smpl_rhand_kp2d'] = torch.sum(loss_smpl_kp2d[:, rhand_idx, :]) / (rhand_kp2d_valid.sum() + 1e-6)
            losses['loss_smpl_face_kp2d'] = torch.sum(loss_smpl_kp2d[:, face_idx, :]) / (face_kp2d_valid.sum() + 1e-6)

        else:
            losses['loss_smpl_body_kp2d'] = torch.sum(loss_smpl_kp2d[:, body_idx, :])  / (body_kp2d_valid.sum() + 1e-6)
            losses['loss_smpl_lhand_kp2d'] = 0*torch.sum(loss_smpl_kp2d[:, lhand_idx, :]) / (lhand_kp2d_valid.sum() + 1e-6)
            losses['loss_smpl_rhand_kp2d'] = 0*torch.sum(loss_smpl_kp2d[:, rhand_idx, :]) / (rhand_kp2d_valid.sum() + 1e-6)
            losses['loss_smpl_face_kp2d'] = 0*torch.sum(loss_smpl_kp2d[:, face_idx, :]) / (face_kp2d_valid.sum() + 1e-6)
        return losses

    def loss_boxes(self, outputs, targets, indices, 
                   idx, num_boxes, data_batch,
                   face_hand_box=False):
        """Compute the losses related to the bounding boxes, the L1 regression
        loss and the GIoU loss targets dicts must contain the key "boxes"
        containing a tensor of dim [nb_target_boxes, 4] The target boxes are
        expected in format (center_x, center_y, w, h), normalized by the image
        size."""
        indices = indices[0]
        device = outputs['pred_logits'].device
        assert 'pred_boxes' in outputs
                
        src_body_boxes = outputs['pred_boxes'][idx]
        target_body_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_body_boxes_conf = torch.cat(
            [t[i] for t, (_, i) in zip(data_batch['body_bbox_valid'], indices)], dim=0)
        
        loss_body_bbox = F.l1_loss(src_body_boxes, target_body_boxes, reduction='none')
        loss_body_bbox = loss_body_bbox * target_body_boxes_conf[:,None]
        
        losses = {}
        losses['loss_body_bbox'] = loss_body_bbox.sum() / num_boxes
        loss_body_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_body_boxes),
                box_ops.box_cxcywh_to_xyxy(target_body_boxes)))
        
        loss_body_giou = loss_body_giou * target_body_boxes_conf
        losses['loss_body_giou'] = loss_body_giou.sum() / num_boxes
        
        if 'pred_lhand_boxes' in outputs and face_hand_box:
            src_lhand_boxes = outputs['pred_lhand_boxes'][idx]
            target_lhand_boxes = torch.cat(
                [t['lhand_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            target_lhand_boxes_conf = torch.cat(
                [t[i] for t, (_, i) in zip(data_batch['lhand_bbox_valid'], indices)], dim=0)
            # print(target_lhand_boxes_conf)
            loss_lhand_bbox = F.l1_loss(src_lhand_boxes, target_lhand_boxes, reduction='none')
            loss_lhand_bbox = loss_lhand_bbox * target_lhand_boxes_conf[:,None]
            num_lhand_boxes = (target_lhand_boxes_conf>0).sum()
            loss_lhand_giou = 1 - torch.diag(
                box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_lhand_boxes),
                    box_ops.box_cxcywh_to_xyxy(target_lhand_boxes)))
            loss_lhand_giou = loss_lhand_giou * target_lhand_boxes_conf
            if num_lhand_boxes > 0:
                losses['loss_lhand_bbox'] = loss_lhand_bbox.sum() / num_lhand_boxes
                losses['loss_lhand_giou'] = loss_lhand_giou.sum() / num_lhand_boxes
            else:
                losses['loss_lhand_bbox'] = loss_lhand_bbox.sum() * 0
                losses['loss_lhand_giou'] = loss_lhand_giou.sum() * 0

           
        
        if 'pred_rhand_boxes' in outputs and face_hand_box:
            src_rhand_boxes = outputs['pred_rhand_boxes'][idx]
            target_rhand_boxes = torch.cat(
                [t['rhand_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            target_rhand_boxes_conf = torch.cat(
                [t[i] for t, (_, i) in zip(data_batch['rhand_bbox_valid'], indices)], dim=0)
            loss_rhand_bbox = F.l1_loss(src_rhand_boxes, target_rhand_boxes, reduction='none')
            loss_rhand_bbox = loss_rhand_bbox * target_rhand_boxes_conf[:,None]
            num_rhand_boxes = (target_rhand_boxes_conf>0).sum()
            loss_rhand_giou = 1 - torch.diag(
                box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_rhand_boxes),
                    box_ops.box_cxcywh_to_xyxy(target_rhand_boxes)))
            loss_rhand_giou = loss_rhand_giou * target_rhand_boxes_conf
            if num_rhand_boxes > 0:
                losses['loss_rhand_bbox'] = loss_rhand_bbox.sum() / num_rhand_boxes
                losses['loss_rhand_giou'] = loss_rhand_giou.sum() / num_rhand_boxes
            else:
                losses['loss_rhand_bbox'] = loss_rhand_bbox.sum() * 0
                losses['loss_rhand_giou'] = loss_rhand_giou.sum() * 0
                
        if 'pred_face_boxes' in outputs and face_hand_box:
            src_face_boxes = outputs['pred_face_boxes'][idx]
            target_face_boxes = torch.cat(
                [t['face_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            target_face_boxes_conf = torch.cat(
                [t[i] for t, (_, i) in zip(data_batch['face_bbox_valid'], indices)], dim=0)
            loss_face_bbox = F.l1_loss(src_face_boxes, target_face_boxes, reduction='none')
            loss_face_bbox = loss_face_bbox * target_face_boxes_conf[:,None]
            num_face_boxes = (target_face_boxes_conf>0).sum()
            loss_face_giou = 1 - torch.diag(
                box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_face_boxes),
                    box_ops.box_cxcywh_to_xyxy(target_face_boxes)))       
            loss_face_giou = loss_face_giou * target_face_boxes_conf
            if num_face_boxes > 0:
                losses['loss_face_bbox'] = loss_face_bbox.sum() / num_face_boxes
                losses['loss_face_giou'] = loss_face_giou.sum() / num_face_boxes  
            else:
                losses['loss_face_bbox'] = loss_face_bbox.sum() * 0
                losses['loss_face_giou'] = loss_face_giou.sum() * 0

        return losses

    def loss_dn_boxes(self, outputs, targets, indices, idx, num_boxes,
                      data_batch):
        """
        Input:
            - src_boxes: bs, num_dn, 4
            - tgt_boxes: bs, num_dn, 4

        """
        indices = indices[0]
        num_tgt = outputs['num_tgt']
        src_boxes = outputs['dn_bbox_pred']
        tgt_boxes = outputs['dn_bbox_input']

        if 'num_tgt' not in outputs:
            device = outputs['pred_logits'].device
            losses = {
                'dn_loss_bbox': src_boxes.sum()*0,
                'dn_loss_giou': src_boxes.sum()*0,
            }
            return losses
        
        if 'num_tgt' not in outputs:
            device = outputs['pred_logits'].device
            losses = {
                'dn_loss_bbox': src_boxes.sum()*0,
                'dn_loss_giou': src_boxes.sum()*0,
            }
            return losses


        return self.tgt_loss_boxes(src_boxes, tgt_boxes, num_tgt)

    def loss_dn_labels(self, outputs, targets, indices, idx, num_boxes,
                       data_batch):
        """
        Input:
            - src_logits: bs, num_dn, num_classes
            - tgt_labels: bs, num_dn

        """
        indices = indices[0]
        if 'num_tgt' not in outputs:
            device = outputs['pred_logits'].device
            losses = {
                'dn_loss_ce': outputs['pred_logits'].sum()*0,
            }
            return losses
        num_tgt = outputs['num_tgt']
        src_logits = outputs['dn_class_pred']  # bs, num_dn, text_len
        tgt_labels = outputs['dn_class_input']

        return self.tgt_loss_labels(src_logits, tgt_labels, num_tgt)

    @torch.no_grad()
    def loss_matching_cost(self, outputs, targets, indices, idx, num_boxes,
                           data_batch):
        """
        Input:
            - src_logits: bs, num_dn, num_classes
            - tgt_labels: bs, num_dn

        """
        cost_mean_dict = indices[1]
        losses = {'set_{}'.format(k): v for k, v in cost_mean_dict.items()}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, data_batch, indices, num_boxes,
                 **kwargs):
        loss_map = {
            'smpl_pose': self.loss_smpl_pose,
            'smpl_beta': self.loss_smpl_beta,
            'smpl_expr': self.loss_smpl_expr,
            'smpl_kp2d': self.loss_smpl_kp2d,
            'smpl_kp2d_ba': self.loss_smpl_kp2d_ba,
            'smpl_kp3d_ra': self.loss_smpl_kp3d_ra,
            'smpl_kp3d': self.loss_smpl_kp3d,
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'keypoints': self.loss_keypoints,
            'boxes': self.loss_boxes,
            'dn_label': self.loss_dn_labels,
            'dn_bbox': self.loss_dn_boxes,
            'matching': self.loss_matching_cost,
        }
        
        idx = self._get_src_permutation_idx(indices[0])
        # pdb.set_trace()
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, idx, num_boxes,
                              data_batch, **kwargs)

    def prep_for_dn2(self, mask_dict):
        known_bboxs = mask_dict['known_bboxs']
        known_labels = mask_dict['known_labels']
        output_known_coord = mask_dict['output_known_coord']
        output_known_class = mask_dict['output_known_class']
        num_tgt = mask_dict['pad_size']

        return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt

    ## SMPL losses

    def forward(self, outputs, targets, data_batch, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        # import pdb; pdb.set_trace()
        outputs_without_aux = {
            k: v
            for k, v in outputs.items() if k != 'aux_outputs'
        }
        device = next(iter(outputs.values())).device

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t['boxes']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes],
                                    dtype=torch.float,
                                    device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # loss for final layer
        # pdb.set_trace()
        indices = self.matcher(outputs_without_aux, targets, data_batch)
        if return_indices:
            indices0_copy = indices
            indices_list = []
        losses = {}
        smpl_loss = ['smpl_pose', 'smpl_beta', 'smpl_expr', 'smpl_kp2d',
                     'smpl_kp2d_ba', 'smpl_kp3d', 'smpl_kp3d_ra']

        for loss in self.losses:
            kwargs = {}
            if loss == 'keypoints' or loss in smpl_loss:
                kwargs.update({'face_hand_kpt': True})
            if loss == 'boxes':
                kwargs.update({'face_hand_box': True})

            losses.update(
                self.get_loss(
                    loss, outputs, targets, 
                    data_batch, indices, 
                    num_boxes, **kwargs
                    ))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, data_batch)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'boxes':
                        kwargs.update({'face_hand_box': False})
                        if idx >= self.num_box_decoder_layers:
                            kwargs.update({'face_hand_box': True})
                            
                    if loss == 'masks':
                        continue
                    
                    if loss == 'keypoints':
                        if idx < self.num_box_decoder_layers:
                            continue
                        elif idx < self.num_hand_face_decoder_layers:
                            kwargs.update({'face_hand_kpt': False})
                        else:
                            kwargs.update({'face_hand_kpt': True})
                            
                    if loss in smpl_loss: 
                        if idx < self.num_box_decoder_layers:
                            continue
                        elif idx < self.num_hand_face_decoder_layers:
                            kwargs.update({'face_hand_kpt': False})
                        else:
                            kwargs.update({'face_hand_kpt': True})                    
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    
                    # if loss == 'smpl_expr' and idx < self.num_box_decoder_layers:
                    #     continue
                        
                    
                    # import pdb;pdb.set_trace()
                    l_dict = self.get_loss(loss, aux_outputs, targets,
                                           data_batch, indices, num_boxes,
                                           **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss in ['dn_bbox', 'dn_label', 'keypoints']:
                    continue
                if loss in [
                        'smpl_pose', 'smpl_beta', 'smpl_kp2d_ba', 'smpl_kp2d',
                        'smpl_kp3d_ra', 'smpl_kp3d', 'smpl_expr'
                ]:
                    continue
                kwargs = {}
                if loss == 'labels':
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, interm_outputs, targets,
                                       data_batch, indices, num_boxes,
                                       **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # aux_init loss
        if 'query_expand' in outputs:
            interm_outputs = outputs['query_expand']
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss in ['dn_bbox', 'dn_label']:
                    continue
                kwargs = {}

                if loss == 'labels':
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, interm_outputs, targets,
                                       data_batch, indices, num_boxes,
                                       **kwargs)
                l_dict = {k + f'_query_expand': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def tgt_loss_boxes(
        self,
        src_boxes,
        tgt_boxes,
        num_tgt,
    ):
        """
        Input:
            - src_boxes: bs, num_dn, 4
            - tgt_boxes: bs, num_dn, 4

        """
        
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')

        losses = {}
        losses['dn_loss_bbox'] = loss_bbox.sum() / num_tgt

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.flatten(0, 1)),
                box_ops.box_cxcywh_to_xyxy(tgt_boxes.flatten(0, 1))))
        losses['dn_loss_giou'] = loss_giou.sum() / num_tgt
        return losses

    def tgt_loss_labels(self,
                        src_logits: Tensor,
                        tgt_labels: Tensor,
                        num_tgt: int,
                        log: bool = True):
        """
        Input:
            - src_logits: bs, num_dn, num_classes
            - tgt_labels: bs, num_dn

        """
        target_classes_onehot = torch.zeros([
            src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1
        ],
                                            dtype=src_logits.dtype,
                                            layout=src_logits.layout,
                                            device=src_logits.device)
        target_classes_onehot.scatter_(2, tgt_labels.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits,
                                     target_classes_onehot,
                                     num_tgt,
                                     alpha=self.focal_alpha,
                                     gamma=2) * src_logits.shape[1]
        losses = {'dn_loss_ce': loss_ce}

        return losses


class SetCriterion_Box(nn.Module):
    def __init__(self,
                 num_classes,
                 matcher,
                 weight_dict,
                 focal_alpha,
                 losses,
                 num_box_decoder_layers=2,
                 num_hand_face_decoder_layers=4,
                 num_body_points=17,
                 num_hand_points=6,
                 num_face_points=6,
                 smpl_loss_config=None,
                 convention='smplx_137'):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.vis = 0.1
        self.abs = 1
        self.num_body_points = 0
        self.num_hand_points = 0
        self.num_face_points = 0
        self.num_box_decoder_layers = num_box_decoder_layers
        self.num_hand_face_decoder_layers = num_hand_face_decoder_layers
        self.convention = convention


    def loss_labels(self,
                    outputs,
                    targets,
                    indices,
                    idx,
                    num_boxes,
                    data_batch,
                    log=True):
        """Classification loss (Binary focal loss) targets dicts must contain
        the key "labels" containing a tensor of dim [nb_target_boxes]"""
        indices = indices[0]
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat(
            [t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2],
                                    self.num_classes,
                                    dtype=torch.int64,
                                    device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([
            src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1
        ],
                                            dtype=src_logits.dtype,
                                            layout=src_logits.layout,
                                            device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits,
                                     target_classes_onehot,
                                     num_boxes,
                                     alpha=self.focal_alpha,
                                     gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx],
                                                   target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes,
                         data_batch):
        """Compute the cardinality error, ie the absolute error in the number
        of predicted non-empty boxes This is not really a loss, it is intended
        for logging purposes only.

        It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['labels']) for v in targets],
                                      device=device)
        if tgt_lengths == 0:
            return  {'cardinality_error': pred_logits.sum()*0}
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) !=
                     pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_smpl_pose(self, outputs, targets, indices, idx, num_boxes,
                       data_batch, face_hand_kpt=False):
        indices = indices[0]
        device = outputs['pred_logits'].device

        pred_smpl_body_pose = outputs['pred_smpl_pose'][idx] # 22
        pred_smpl_lhand_pose = outputs['pred_smpl_lhand_pose'][idx] # 15
        pred_smpl_rhand_pose = outputs['pred_smpl_rhand_pose'][idx] # 15
        pred_smpl_jaw_pose = outputs['pred_smpl_jaw_pose'][idx]

        pred_smplx_pose = torch.cat((pred_smpl_body_pose, pred_smpl_lhand_pose,
                                     pred_smpl_rhand_pose, pred_smpl_jaw_pose),
                                    dim=1)

        targets_smpl_pose = torch.cat(
            [t[i] for t, (_, i) in zip(data_batch['smplx_pose'], indices)],
            dim=0)
        targets_smpl_pose = batch_rodrigues(targets_smpl_pose.view(
            -1, 3)).view(-1, 53, 3, 3)
        conf = torch.cat([
            t[i] for t, (_, i) in zip(data_batch['smplx_pose_valid'], indices)
            ], dim=0)
        body_pose_valid = conf[:, :22].sum(-1) > 0
        lhand_pose_valid = conf[:, 22:37].sum(-1) > 0
        rhand_pose_valid = conf[:, 37:52].sum(-1) > 0
        face_pose_valid = conf[:, 52].sum(-1) > 0
        
        losses = {}
        loss_smpl_pose = \
            F.l1_loss(
                pred_smplx_pose,
                targets_smpl_pose,
                reduction='none'
            )
        loss_smpl_pose = loss_smpl_pose.sum([-1,-2]) * conf

        if face_hand_kpt:
            losses = {
                'loss_smpl_pose_root': loss_smpl_pose[:, 0].sum() / (body_pose_valid.sum() + 1e-6),
                'loss_smpl_pose_body': loss_smpl_pose[:, 1:22].sum() / (body_pose_valid.sum() + 1e-6),
                'loss_smpl_pose_lhand': loss_smpl_pose[:, 22:37].sum() / (lhand_pose_valid.sum() + 1e-6),
                'loss_smpl_pose_rhand': loss_smpl_pose[:, 37:52].sum() / (rhand_pose_valid.sum() + 1e-6),
                'loss_smpl_pose_jaw': loss_smpl_pose[:, 52].sum() / (face_pose_valid.sum() + 1e-6),
            }
        else:
            losses = {
                'loss_smpl_pose_root': loss_smpl_pose[:, 0].sum() / (body_pose_valid.sum() + 1e-6),
                'loss_smpl_pose_body': loss_smpl_pose[:, 1:22].sum() / (body_pose_valid.sum() + 1e-6),
                'loss_smpl_pose_lhand': 0 * loss_smpl_pose[:, 22:37].sum()/(lhand_pose_valid.sum() + 1e-6),
                'loss_smpl_pose_rhand': 0 * loss_smpl_pose[:, 37:52].sum() / (rhand_pose_valid.sum() + 1e-6),
                'loss_smpl_pose_jaw': 0*loss_smpl_pose[:, 52].sum() / (face_pose_valid.sum() + 1e-6),
            }
        return losses

    def loss_smpl_beta(self, outputs, targets, indices, idx, num_boxes,
                       data_batch, face_hand_kpt=False):
        indices = indices[0]
        device = outputs['pred_logits'].device
        pred_smpl_betas = outputs['pred_smpl_beta'][idx]

        targets_smpl_betas = torch.cat(
            [t[i] for t, (_, i) in zip(data_batch['smplx_shape'], indices)],
            dim=0)

        losses = {}
        conf = torch.cat([t[i] for t, (_, i) in zip(data_batch['smplx_shape_valid'], indices)], dim=0)
        if conf.sum() == 0:
            return {
                'loss_smpl_beta': pred_smpl_betas.sum() * 0
            }
        loss_smpl_betas = \
            F.l1_loss(
                pred_smpl_betas,
                targets_smpl_betas,
                reduction='none'
            )
        loss_smpl_betas = loss_smpl_betas.sum(-1) * conf
        losses = {'loss_smpl_beta': loss_smpl_betas.sum() / (conf.sum() + 1e-6)}
        return losses

    def loss_smpl_expr(self, outputs, targets, indices, idx, num_boxes,
                       data_batch, face_hand_kpt=False):
        indices = indices[0]
        device = outputs['pred_logits'].device
        pred_smpl_expr = outputs['pred_smpl_expr'][idx]
        targets_smpl_expr = torch.cat([t[i] for t, (_, i) in zip(data_batch['smplx_expr'], indices)], dim=0)

        losses = {}
        conf = torch.cat([t[i] for t, (_, i) in zip(data_batch['smplx_expr_valid'], indices)], dim=0)
        if conf.sum() == 0:
            return {
                'loss_smpl_expr': pred_smpl_expr.sum() * 0
            }

        loss_smpl_expr = \
            F.l1_loss(
                pred_smpl_expr,
                targets_smpl_expr,
                reduction='none'
            )

        loss_smpl_expr = loss_smpl_expr.sum(-1) * conf
        if face_hand_kpt:
            losses = {'loss_smpl_expr': loss_smpl_expr.sum() / (conf.sum() + 1e-6)}
        else:
            losses = {'loss_smpl_expr': 0*loss_smpl_expr.sum() / (conf.sum() + 1e-6) }
            
        return losses

    def loss_smpl_kp3d(self,
                       outputs,
                       targets,
                       indices,
                       idx,
                       num_boxes,
                       data_batch,
                       has_keypoints3d=None,
                       face_hand_kpt=False):

        # supervision for keypoints3d wo/ ra
        device = outputs['pred_logits'].device
        indices = indices[0]
        pred_smpl_kp3d = outputs['pred_smpl_kp3d'][idx].float()

        targets_smpl_kp3d = torch.cat(
            [t[i] for t, (_, i) in zip(data_batch['joint_cam'], indices)],
            dim=0)
        losses = {}
        targets_kp3d_conf = targets_smpl_kp3d[:,:,3:].clone()
        targets_smpl_kp3d = targets_smpl_kp3d[:,:,:3]
        
        targets_is_3d = torch.cat([
            t[None, None].repeat(len(i), 1, 1)
            for t, (_, i) in zip(data_batch['is_3D'], indices)
        ], dim=0)

        targets_kp3d_conf = (targets_kp3d_conf * targets_is_3d)
        pelvis_idx = get_keypoint_idx('pelvis', self.convention)
        targets_pelvis = targets_smpl_kp3d[..., pelvis_idx, :]
        pred_pelvis = pred_smpl_kp3d[..., pelvis_idx, :]

        targets_smpl_kp3d = targets_smpl_kp3d - targets_pelvis[:, None, :]
        pred_smpl_kp3d = pred_smpl_kp3d - pred_pelvis[:, None, :]

        losses = {}
        body_idx = smpl_x.joint_part['body']
        face_idx = smpl_x.joint_part['face']
        lhand_idx = smpl_x.joint_part['lhand']
        rhand_idx = smpl_x.joint_part['rhand']
        
        loss_smpl_kp3d = F.l1_loss(pred_smpl_kp3d,
                                   targets_smpl_kp3d,
                                   reduction='none')
        
        body_kp3d_valid = targets_kp3d_conf[:, body_idx].sum([-1,-2]) > 0
        lhand_kp3d_valid = targets_kp3d_conf[:, lhand_idx].sum([-1,-2]) > 0
        rhand_kp3d_valid = targets_kp3d_conf[:, rhand_idx].sum([-1,-2]) > 0
        face_kp3d_valid = targets_kp3d_conf[:, face_idx].sum([-1,-2]) > 0
        
        loss_smpl_kp3d = loss_smpl_kp3d * targets_kp3d_conf
        
        if face_hand_kpt:
            losses['loss_smpl_body_kp3d'] = torch.sum(loss_smpl_kp3d[:, body_idx, :])  / (body_kp3d_valid.sum() + 1e-6)
            losses['loss_smpl_lhand_kp3d'] = torch.sum(loss_smpl_kp3d[:, lhand_idx, :]) / (lhand_kp3d_valid.sum() + 1e-6)
            losses['loss_smpl_rhand_kp3d'] = torch.sum(loss_smpl_kp3d[:, rhand_idx, :]) / (rhand_kp3d_valid.sum() + 1e-6)
            losses['loss_smpl_face_kp3d'] = torch.sum(loss_smpl_kp3d[:, face_idx, :]) / (face_kp3d_valid.sum() + 1e-6)
        else:
            losses['loss_smpl_body_kp3d'] = torch.sum(loss_smpl_kp3d[:, body_idx, :])  / (body_kp3d_valid.sum() + 1e-6)
            losses['loss_smpl_lhand_kp3d'] =  0*torch.sum(loss_smpl_kp3d[:, lhand_idx, :]) / (lhand_kp3d_valid.sum() + 1e-6)
            losses['loss_smpl_rhand_kp3d'] = 0*torch.sum(loss_smpl_kp3d[:, rhand_idx, :]) / (rhand_kp3d_valid.sum() + 1e-6)
            losses['loss_smpl_face_kp3d'] = 0*torch.sum(loss_smpl_kp3d[:, face_idx, :]) / (face_kp3d_valid.sum() + 1e-6)
        return losses


    def loss_smpl_kp3d_ra(self,
                          outputs,
                          targets,
                          indices,
                          idx,
                          num_boxes,
                          data_batch,
                          has_keypoints3d=None,
                          face_hand_kpt=False):
        # supervision for keypoints3d w/ ra
        device = outputs['pred_logits'].device
        indices = indices[0]

        pred_smpl_kp3d = outputs['pred_smpl_kp3d'][idx].float()
        targets_smpl_kp3d = torch.cat([
            t[i] for t, (_, i) in zip(data_batch['smplx_joint_cam'], indices)],
            dim=0)
        losses = {}
        targets_kp3d_conf = targets_smpl_kp3d[:,:,3:].clone()
        targets_smpl_kp3d = targets_smpl_kp3d[:,:,:3]
        
        targets_is_3d = torch.cat([
            t[None, None].repeat(len(i), 1, 1)
            for t, (_, i) in zip(data_batch['is_3D'], indices)],dim=0)
        
        targets_kp3d_conf = (targets_kp3d_conf * targets_is_3d).repeat(1, 1, 3)
        pelvis_idx = get_keypoint_idx('pelvis', self.convention)
        targets_pelvis = targets_smpl_kp3d[..., pelvis_idx, :]
        pred_pelvis = pred_smpl_kp3d[..., pelvis_idx, :]

        targets_smpl_kp3d = targets_smpl_kp3d - targets_pelvis[:, None, :]
        pred_smpl_kp3d = pred_smpl_kp3d - pred_pelvis[:, None, :]
        # calculate body, face and hand loss separately:

        losses = {}
        body_idx = smpl_x.joint_part['body']
        face_idx = smpl_x.joint_part['face']
        lhand_idx = smpl_x.joint_part['lhand']
        rhand_idx = smpl_x.joint_part['rhand']

        body_kp3d_valid = targets_kp3d_conf[:, body_idx].sum([-1,-2]) > 0
        lhand_kp3d_valid = targets_kp3d_conf[:, lhand_idx].sum([-1,-2]) > 0
        rhand_kp3d_valid = targets_kp3d_conf[:, rhand_idx].sum([-1,-2]) > 0
        face_kp3d_valid = targets_kp3d_conf[:, face_idx].sum([-1,-2]) > 0

        loss_smpl_body_kp3d = F.l1_loss(pred_smpl_kp3d[:, body_idx, :],
                                        targets_smpl_kp3d[:, body_idx, :],
                                        reduction='none')
        
        loss_smpl_body_kp3d = torch.sum(
            loss_smpl_body_kp3d * targets_kp3d_conf[:, body_idx, :])
        losses['loss_smpl_body_kp3d_ra'] = loss_smpl_body_kp3d / (body_kp3d_valid.sum() + 1e-6)
        
        face_cam = pred_smpl_kp3d[:, face_idx, :]
        neck_cam = pred_smpl_kp3d[:, smpl_x.neck_idx, None, :]
        face_cam = face_cam - neck_cam
        loss_smpl_face_kp3d = F.l1_loss(face_cam,
                                        targets_smpl_kp3d[:, face_idx, :],
                                        reduction='none')
        loss_smpl_face_kp3d = torch.sum(
            loss_smpl_face_kp3d * targets_kp3d_conf[:, face_idx, :])
        if face_hand_kpt:
            losses['loss_smpl_face_kp3d_ra'] = (loss_smpl_face_kp3d / (face_kp3d_valid.sum() + 1e-6))
        else:
            losses['loss_smpl_face_kp3d_ra'] = 0 * (loss_smpl_face_kp3d / (face_kp3d_valid.sum() + 1e-6))
        
        lhand_cam = pred_smpl_kp3d[:, lhand_idx, :]
        lwrist_cam = pred_smpl_kp3d[:, smpl_x.lwrist_idx, None, :]
        lhand_cam = lhand_cam - lwrist_cam
        loss_smpl_lhand_kp3d = F.l1_loss(lhand_cam,
                                            targets_smpl_kp3d[:, lhand_idx, :],
                                            reduction='none')
        loss_smpl_lhand_kp3d = torch.sum(
            loss_smpl_lhand_kp3d * targets_kp3d_conf[:, lhand_idx, :])
        
        if face_hand_kpt:
            losses['loss_smpl_lhand_kp3d_ra'] = (loss_smpl_lhand_kp3d / (lhand_kp3d_valid.sum() + 1e-6))
        else:
            losses['loss_smpl_lhand_kp3d_ra'] = 0*(loss_smpl_lhand_kp3d / (lhand_kp3d_valid.sum() + 1e-6))
            
        rhand_cam = pred_smpl_kp3d[:, rhand_idx, :]
        rwrist_cam = pred_smpl_kp3d[:, smpl_x.rwrist_idx, None, :]
        rhand_cam = rhand_cam - rwrist_cam

        loss_smpl_rhand_kp3d = F.l1_loss(rhand_cam,
                                            targets_smpl_kp3d[:, rhand_idx, :],
                                            reduction='none')
        loss_smpl_rhand_kp3d = torch.sum(
            loss_smpl_rhand_kp3d * targets_kp3d_conf[:, rhand_idx, :])
        
        if face_hand_kpt:
            losses['loss_smpl_rhand_kp3d_ra'] = (loss_smpl_rhand_kp3d / (rhand_kp3d_valid.sum() + 1e-6))
        else:
            losses['loss_smpl_rhand_kp3d_ra'] = 0*(loss_smpl_rhand_kp3d / (rhand_kp3d_valid.sum() + 1e-6))

        return losses

    def loss_smpl_kp2d(self,
                       outputs,
                       targets,
                       indices,
                       idx,
                       num_boxes,
                       data_batch,
                       focal_length=5000.,
                       has_keypoints2d=None,
                       face_hand_kpt=False):
        """Compute loss for 2d keypoints."""
        device = outputs['pred_logits'].device
        indices = indices[0]
        
        pred_smpl_kp3d = outputs['pred_smpl_kp3d'][idx].float()#.detach()
        pred_cam = outputs['pred_smpl_cam'][idx].float()

        targets_kp2d = torch.cat([t[i] for t, (_, i) in zip(data_batch['joint_img'], indices)], dim=0)
        keypoints2d_conf =  targets_kp2d[:,:,2:].clone()

        targets_kp2d = targets_kp2d[:, :, :2].float()
        targets_kp2d[:,:,0] = targets_kp2d[:,:,0]/cfg.output_hm_shape[2]
        targets_kp2d[:,:,1] = targets_kp2d[:,:,1]/cfg.output_hm_shape[1]
        img_wh =  torch.cat([data_batch['img_shape'][i][None] for i in idx[0]], dim=0).flip(-1)
        pred_smpl_kp2d = project_points_new(
            points_3d=pred_smpl_kp3d,
            pred_cam=pred_cam,
            focal_length=focal_length,
            camera_center=img_wh/2
        )
        pred_smpl_kp2d = pred_smpl_kp2d / img_wh[:, None]
        
        losses = {}
        body_idx = smpl_x.joint_part['body']
        face_idx = smpl_x.joint_part['face']
        lhand_idx = smpl_x.joint_part['lhand']
        rhand_idx = smpl_x.joint_part['rhand']
        
        body_kp2d_valid = keypoints2d_conf[:, body_idx].sum([-1,-2]) > 0
        lhand_kp2d_valid = keypoints2d_conf[:, lhand_idx].sum([-1,-2]) > 0
        rhand_kp2d_valid = keypoints2d_conf[:, rhand_idx].sum([-1,-2]) > 0
        face_kp2d_valid = keypoints2d_conf[:, face_idx].sum([-1,-2]) > 0
        
        loss_smpl_kp2d = F.l1_loss(pred_smpl_kp2d,
                                   targets_kp2d,
                                   reduction='none')
        loss_smpl_kp2d = loss_smpl_kp2d * keypoints2d_conf

        if face_hand_kpt:
            losses['loss_smpl_body_kp2d'] = torch.sum(loss_smpl_kp2d[:, body_idx, :])  / (body_kp2d_valid.sum() + 1e-6)
            losses['loss_smpl_lhand_kp2d'] = torch.sum(loss_smpl_kp2d[:, lhand_idx, :]) / (lhand_kp2d_valid.sum() + 1e-6)
            losses['loss_smpl_rhand_kp2d'] = torch.sum(loss_smpl_kp2d[:, rhand_idx, :]) / (rhand_kp2d_valid.sum() + 1e-6)
            losses['loss_smpl_face_kp2d'] = torch.sum(loss_smpl_kp2d[:, face_idx, :]) / (face_kp2d_valid.sum() + 1e-6)

        else:
            losses['loss_smpl_body_kp2d'] = torch.sum(loss_smpl_kp2d[:, body_idx, :])  / (body_kp2d_valid.sum() + 1e-6)
            losses['loss_smpl_lhand_kp2d'] = 0*torch.sum(loss_smpl_kp2d[:, lhand_idx, :]) / (lhand_kp2d_valid.sum() + 1e-6)
            losses['loss_smpl_rhand_kp2d'] = 0*torch.sum(loss_smpl_kp2d[:, rhand_idx, :]) / (rhand_kp2d_valid.sum() + 1e-6)
            losses['loss_smpl_face_kp2d'] = 0*torch.sum(loss_smpl_kp2d[:, face_idx, :]) / (face_kp2d_valid.sum() + 1e-6)
        return losses

    def loss_smpl_kp2d_ba(self,
                          outputs,
                          targets,
                          indices,
                          idx,
                          num_boxes,
                          data_batch,
                          focal_length=5000.,
                          has_keypoints2d=None,
                        face_hand_kpt=False):
        """Compute loss for 2d keypoints."""
        device = outputs['pred_logits'].device
        indices = indices[0]
        # pdb.set_trace()
        pred_smpl_kp3d = outputs['pred_smpl_kp3d'][idx].float()#.detach()
        pred_cam = outputs['pred_smpl_cam'][idx].float()

        # pdb.set_trace()

        # max_img_res = orig_img_res.max(-1)[0]
        # torch.cat([ torch.Tensor([orig_img_res[0]]*9), torch.Tensor([orig_img_res[1]]*9)], 0)
        # torch.cat([orig_img_res[i][None].repeat(num,1) for i, num in enumerate(instance_num)], 0)

        # orig_img_res = torch.Tensor([t['orig_size'] for t, (_, i) in zip(targets, indices)]).type_as(pred_smpl_kp3d)
        # orig_img_res = torch.Tensor([target['orig_size'] for target in targets]).type_as(pred_smpl_kp3d)
        # max_img_res = torch.cat([torch.full_like(src, i) for i, (src, _) in zip(max_img_res, indices)]).type_as(pred_smpl_kp3d)
        valid_num=0
        for indice in indices[0]:
            valid_num+=len(indice)
        targets_kp2d = torch.cat(
            [t[i] for t, (_, i) in zip(data_batch['joint_img'], indices)],
            dim=0)
        losses = {}

        
        
        keypoints2d_conf =  targets_kp2d[:,:,2:].clone()
        targets_kp2d = targets_kp2d[:,:,:2]
        
        keypoints2d_conf = keypoints2d_conf.repeat(1, 1, 2)
        targets_kp2d = targets_kp2d[:, :, :2].float()
        targets_kp2d[:, :, 0] = targets_kp2d[:, :, 0] / cfg.output_hm_shape[2]
        targets_kp2d[:, :, 1] = targets_kp2d[:, :, 1] / cfg.output_hm_shape[1]
        # targets_kp2d = targets_kp2d * 2 - 1
        img_wh =  torch.cat([data_batch['img_shape'][i][None] for i in idx[0]], dim=0).flip(-1)

        pred_smpl_kp2d = project_points_new(
            points_3d=pred_smpl_kp3d,
            pred_cam=pred_cam,
            focal_length=focal_length,
            camera_center=img_wh/2
        )

        pred_smpl_kp2d = pred_smpl_kp2d / img_wh[:, None]
        
        if valid_num == 0:
            losses['loss_smpl_body_kp2d_ba'] = 0 + pred_smpl_kp2d.sum()*0

            losses['loss_smpl_lhand_kp2d_ba'] = 0 + pred_smpl_kp2d.sum()*0
        
            losses['loss_smpl_rhand_kp2d_ba'] = 0 + pred_smpl_kp2d.sum()*0
        
            losses['loss_smpl_face_kp2d_ba'] = 0 + pred_smpl_kp2d.sum()*0
            return losses        
        # rhand bbox
        rhand_bbox_valid = torch.cat(
            [t[i] for t, (_, i) in zip(data_batch['rhand_bbox_valid'], indices) ], dim=0)
        rhand_bbox_gt = torch.cat(
            [t['rhand_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        rhand_bbox_gt = (box_ops.box_cxcywh_to_xyxy(rhand_bbox_gt).
                         reshape(-1,2,2)*img_wh[:, None]).reshape(-1, 4)
        num_rhand_bbox = rhand_bbox_valid.sum()
        # lhand bbox
        lhand_bbox_valid = torch.cat([
            t[i] for t, (_, i) in zip(data_batch['lhand_bbox_valid'], indices)], dim=0)
        lhand_bbox_gt = torch.cat(
            [t['lhand_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        lhand_bbox_gt = (box_ops.box_cxcywh_to_xyxy(lhand_bbox_gt).
                         reshape(-1,2,2)*img_wh[:, None]).reshape(-1, 4)
        num_lhand_bbox = lhand_bbox_valid.sum()
        # face bbox
        face_bbox_valid = torch.cat(
            [t[i] for t, (_, i) in zip(data_batch['face_bbox_valid'], indices)], dim=0)
        face_bbox_gt = torch.cat(
            [t['face_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        face_bbox_gt = (box_ops.box_cxcywh_to_xyxy(face_bbox_gt).
                        reshape(-1,2,2)*img_wh[:, None]).reshape(-1, 4)
        num_face_bbox = face_bbox_valid.sum()
        img_shape = torch.cat(
            [t[None].repeat(len(i), 1) for t, (_, i) in zip(data_batch['img_shape'], indices)], dim=0)
        
        # joint_proj = (joint_proj / 2 + 0.5)
        # joint_proj[:, :, 0] = joint_proj[:, :, 0] * img_shape[:, 1:]
        # joint_proj[:, :, 1] = joint_proj[:, :, 1] * img_shape[:, :1]

        if not (lhand_bbox_valid + rhand_bbox_valid + face_bbox_valid == 0).all():
            for part_name, bbox in (
                    ('lhand', lhand_bbox_gt), 
                    ('rhand', rhand_bbox_gt), 
                    ('face', face_bbox_gt)):
                
                x = targets_kp2d[:, smpl_x.joint_part[part_name], 0]
                y = targets_kp2d[:, smpl_x.joint_part[part_name], 1]
                # trunc = joint_trunc[:, smpl_x.joint_part[part_name], 0]
                trunc = keypoints2d_conf[:, smpl_x.joint_part[part_name], 0].clone()
                # x in [0, 1]? bbox in [0, 1]. 
                x -= (bbox[:, None, 0] / img_shape[:, 1:])
                # x 
                x *= (img_shape[:, 1:] / (bbox[:, None, 2] - bbox[:, None, 0] + 1e-6))
                
                y -= (bbox[:, None, 1] / img_shape[:, :1])
                y *= (img_shape[:, :1] / (bbox[:, None, 3] - bbox[:, None, 1] + 1e-6))
                # transformed to 0-1 bbox space

                trunc *= ((x >= 0) * (x <= 1) *
                          (y >= 0) * (y <= 1))

                
                coord = torch.stack((x, y), 2)
                

                targets_kp2d = torch.cat(
                    (targets_kp2d[:, :smpl_x.joint_part[part_name][0], :], coord,
                     targets_kp2d[:, smpl_x.joint_part[part_name][-1] + 1:, :]),
                    1)
                
                x_pred = pred_smpl_kp2d[:, smpl_x.joint_part[part_name], 0]
                y_pred = pred_smpl_kp2d[:, smpl_x.joint_part[part_name], 1]
                # bbox: xyxy img_shape: hw
                x_pred -= (bbox[:, None, 0] / img_shape[:, 1:])
                x_pred *= (img_shape[:, 1:] / (bbox[:, None, 2] - bbox[:, None, 0] + 1e-6))
                
                y_pred -= (bbox[:, None, 1] / img_shape[:, :1])
                y_pred *= (img_shape[:, :1] / (bbox[:, None, 3] - bbox[:, None, 1] + 1e-6))

                coord_pred = torch.stack((x_pred, y_pred), 2)
                trans = []

                for bid in range(coord_pred.shape[0]):
                    mask = trunc[bid] == 1
                    
                    if torch.sum(mask) == 0:
                        trans.append(torch.zeros((2)).float().cuda())
                    else:
                        trans.append(
                            (-coord_pred[bid, mask, :2] + targets_kp2d[:, smpl_x.joint_part[part_name], :][bid, mask, :2]).mean(0))
                trans = torch.stack(trans)[:, None, :]
                
                coord_pred = coord_pred + trans  # global translation alignment
                pred_smpl_kp2d = torch.cat(
                    (pred_smpl_kp2d[:, :smpl_x.joint_part[part_name][0], :], coord_pred,
                     pred_smpl_kp2d[:, smpl_x.joint_part[part_name][-1] + 1:, :]),
                    1)
            
        loss_smpl_kp2d_ba = F.l1_loss(pred_smpl_kp2d,
                                   targets_kp2d[:, :, :2],
                                   reduction='none')
        valid_pos = keypoints2d_conf > 0
        
        losses = {}
        if keypoints2d_conf[valid_pos].numel() == 0:
            return {
                'loss_smpl_body_kp2d_ba': loss_smpl_kp2d_ba.sum()*0,
                'loss_smpl_lhand_kp2d_ba': loss_smpl_kp2d_ba.sum()*0,
                'loss_smpl_rhand_kp2d_ba': loss_smpl_kp2d_ba.sum()*0,
                'loss_smpl_face_kp2d_ba': loss_smpl_kp2d_ba.sum()*0,             
            }
        # loss /= targets_kp3d_conf[valid_pos].numel()
        
        loss_smpl_kp2d_ba = loss_smpl_kp2d_ba * keypoints2d_conf
        losses['loss_smpl_body_kp2d_ba'] = torch.sum(loss_smpl_kp2d_ba[:, 
                                                smpl_x.joint_part['body'], :]) / num_boxes
        if face_hand_kpt:
            if num_lhand_bbox>0:
                losses['loss_smpl_lhand_kp2d_ba'] = torch.sum(loss_smpl_kp2d_ba[:, 
                                                        smpl_x.joint_part['lhand'], :]) / num_lhand_bbox
            else:
                losses['loss_smpl_lhand_kp2d_ba'] = loss_smpl_kp2d_ba.sum()*0
            if num_rhand_bbox>0:
                losses['loss_smpl_rhand_kp2d_ba'] = torch.sum(loss_smpl_kp2d_ba[:, 
                                                        smpl_x.joint_part['rhand'], :]) / num_rhand_bbox
            else:
                losses['loss_smpl_rhand_kp2d_ba'] = loss_smpl_kp2d_ba.sum()*0
            if num_face_bbox>0:
                losses['loss_smpl_face_kp2d_ba'] = torch.sum(loss_smpl_kp2d_ba[:, 
                                                        smpl_x.joint_part['face'], :]) / num_face_bbox
            else:
                losses['loss_smpl_face_kp2d_ba'] = loss_smpl_kp2d_ba.sum()*0
        else:
            losses['loss_smpl_lhand_kp2d_ba'] = 0*torch.sum(loss_smpl_kp2d_ba[:, 
                                                    smpl_x.joint_part['lhand'], :]) / num_lhand_bbox

            losses['loss_smpl_rhand_kp2d_ba'] = 0*torch.sum(loss_smpl_kp2d_ba[:, 
                                                    smpl_x.joint_part['rhand'], :]) / num_rhand_bbox

            losses['loss_smpl_face_kp2d_ba'] = 0*torch.sum(loss_smpl_kp2d_ba[:, 
                                                        smpl_x.joint_part['face'], :]) / num_face_bbox
        return losses


    def loss_boxes(self, outputs, targets, indices, 
                   idx, num_boxes, data_batch,
                   face_hand_box=False):
        """Compute the losses related to the bounding boxes, the L1 regression
        loss and the GIoU loss targets dicts must contain the key "boxes"
        containing a tensor of dim [nb_target_boxes, 4] The target boxes are
        expected in format (center_x, center_y, w, h), normalized by the image
        size."""
        indices = indices[0]
        device = outputs['pred_logits'].device
        assert 'pred_boxes' in outputs
                
        src_body_boxes = outputs['pred_boxes'][idx]
        target_body_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_body_boxes_conf = torch.cat(
            [t[i] for t, (_, i) in zip(data_batch['body_bbox_valid'], indices)], dim=0)
        
        loss_body_bbox = F.l1_loss(src_body_boxes, target_body_boxes, reduction='none')
        loss_body_bbox = loss_body_bbox * target_body_boxes_conf[:,None]
        
        losses = {}
        losses['loss_body_bbox'] = loss_body_bbox.sum() / num_boxes
        loss_body_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_body_boxes),
                box_ops.box_cxcywh_to_xyxy(target_body_boxes)))
        
        loss_body_giou = loss_body_giou * target_body_boxes_conf
        losses['loss_body_giou'] = loss_body_giou.sum() / num_boxes
        
        if 'pred_lhand_boxes' in outputs and face_hand_box:
            src_lhand_boxes = outputs['pred_lhand_boxes'][idx]
            target_lhand_boxes = torch.cat(
                [t['lhand_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            target_lhand_boxes_conf = torch.cat(
                [t[i] for t, (_, i) in zip(data_batch['lhand_bbox_valid'], indices)], dim=0)
            # print(target_lhand_boxes_conf)
            loss_lhand_bbox = F.l1_loss(src_lhand_boxes, target_lhand_boxes, reduction='none')
            loss_lhand_bbox = loss_lhand_bbox * target_lhand_boxes_conf[:,None]
            num_lhand_boxes = (target_lhand_boxes_conf>0).sum()
            loss_lhand_giou = 1 - torch.diag(
                box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_lhand_boxes),
                    box_ops.box_cxcywh_to_xyxy(target_lhand_boxes)))
            loss_lhand_giou = loss_lhand_giou * target_lhand_boxes_conf
            if num_lhand_boxes > 0:
                losses['loss_lhand_bbox'] = loss_lhand_bbox.sum() / num_lhand_boxes
                losses['loss_lhand_giou'] = loss_lhand_giou.sum() / num_lhand_boxes
            else:
                losses['loss_lhand_bbox'] = loss_lhand_bbox.sum() * 0
                losses['loss_lhand_giou'] = loss_lhand_giou.sum() * 0
           
        
        if 'pred_rhand_boxes' in outputs and face_hand_box:
            src_rhand_boxes = outputs['pred_rhand_boxes'][idx]
            target_rhand_boxes = torch.cat(
                [t['rhand_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            target_rhand_boxes_conf = torch.cat(
                [t[i] for t, (_, i) in zip(data_batch['rhand_bbox_valid'], indices)], dim=0)
            loss_rhand_bbox = F.l1_loss(src_rhand_boxes, target_rhand_boxes, reduction='none')
            loss_rhand_bbox = loss_rhand_bbox * target_rhand_boxes_conf[:,None]
            num_rhand_boxes = (target_rhand_boxes_conf>0).sum()
            loss_rhand_giou = 1 - torch.diag(
                box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_rhand_boxes),
                    box_ops.box_cxcywh_to_xyxy(target_rhand_boxes)))
            loss_rhand_giou = loss_rhand_giou * target_rhand_boxes_conf
            if num_rhand_boxes > 0:
                losses['loss_rhand_bbox'] = loss_rhand_bbox.sum() / num_rhand_boxes
                losses['loss_rhand_giou'] = loss_rhand_giou.sum() / num_rhand_boxes
            else:
                losses['loss_rhand_bbox'] = loss_rhand_bbox.sum() * 0
                losses['loss_rhand_giou'] = loss_rhand_giou.sum() * 0
                
        if 'pred_face_boxes' in outputs and face_hand_box:
            src_face_boxes = outputs['pred_face_boxes'][idx]
            target_face_boxes = torch.cat(
                [t['face_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            target_face_boxes_conf = torch.cat(
                [t[i] for t, (_, i) in zip(data_batch['face_bbox_valid'], indices)], dim=0)
            loss_face_bbox = F.l1_loss(src_face_boxes, target_face_boxes, reduction='none')
            loss_face_bbox = loss_face_bbox * target_face_boxes_conf[:,None]
            num_face_boxes = (target_face_boxes_conf>0).sum()
            loss_face_giou = 1 - torch.diag(
                box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_face_boxes),
                    box_ops.box_cxcywh_to_xyxy(target_face_boxes)))       
            loss_face_giou = loss_face_giou * target_face_boxes_conf
            if num_face_boxes > 0:
                losses['loss_face_bbox'] = loss_face_bbox.sum() / num_face_boxes
                losses['loss_face_giou'] = loss_face_giou.sum() / num_face_boxes  
            else:
                losses['loss_face_bbox'] = loss_face_bbox.sum() * 0
                losses['loss_face_giou'] = loss_face_giou.sum() * 0

        return losses

    def loss_dn_boxes(self, outputs, targets, indices, idx, num_boxes,
                      data_batch):
        """
        Input:
            - src_boxes: bs, num_dn, 4
            - tgt_boxes: bs, num_dn, 4

        """
        indices = indices[0]
        num_tgt = outputs['num_tgt']
        src_boxes = outputs['dn_bbox_pred']
        tgt_boxes = outputs['dn_bbox_input']
        valid_num=0
        for indice in indices[0]:
            valid_num+=len(indice)
        if valid_num == 0:
            device = outputs['pred_logits'].device
            losses = {
                'dn_loss_bbox': src_boxes.sum()*0,
                'dn_loss_giou': src_boxes.sum()*0,
            }
            return losses   
        if 'num_tgt' not in outputs:
            device = outputs['pred_logits'].device
            losses = {
                'dn_loss_bbox': src_boxes.sum()*0,
                'dn_loss_giou': src_boxes.sum()*0,
            }
            return losses
        
        if 'num_tgt' not in outputs:
            device = outputs['pred_logits'].device
            losses = {
                'dn_loss_bbox': src_boxes.sum()*0,
                'dn_loss_giou': src_boxes.sum()*0,
            }
            return losses


        return self.tgt_loss_boxes(src_boxes, tgt_boxes, num_tgt)

    def loss_dn_labels(self, outputs, targets, indices, idx, num_boxes,
                       data_batch):
        """
        Input:
            - src_logits: bs, num_dn, num_classes
            - tgt_labels: bs, num_dn

        """
        indices = indices[0]
        if 'num_tgt' not in outputs:
            device = outputs['pred_logits'].device
            losses = {
                'dn_loss_ce': outputs['pred_logits'].sum()*0,
            }
            return losses
        valid_num = 0
        for indice in indices[0]:
            valid_num+=len(indice)
        if valid_num == 0:
            device = outputs['pred_logits'].device
            losses = {
                'dn_loss_ce': outputs['pred_logits'].sum()*0,
            }
            return losses 
        num_tgt = outputs['num_tgt']
        src_logits = outputs['dn_class_pred']  # bs, num_dn, text_len
        tgt_labels = outputs['dn_class_input']

        return self.tgt_loss_labels(src_logits, tgt_labels, num_tgt)

    @torch.no_grad()
    def loss_matching_cost(self, outputs, targets, indices, idx, num_boxes,
                           data_batch):
        """
        Input:
            - src_logits: bs, num_dn, num_classes
            - tgt_labels: bs, num_dn

        """
        cost_mean_dict = indices[1]
        losses = {'set_{}'.format(k): v for k, v in cost_mean_dict.items()}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, data_batch, indices, num_boxes,
                 **kwargs):
        loss_map = {
            'smpl_pose': self.loss_smpl_pose,
            'smpl_beta': self.loss_smpl_beta,
            'smpl_expr': self.loss_smpl_expr,
            'smpl_kp2d': self.loss_smpl_kp2d,
            'smpl_kp2d_ba': self.loss_smpl_kp2d_ba,
            'smpl_kp3d_ra': self.loss_smpl_kp3d_ra,
            'smpl_kp3d': self.loss_smpl_kp3d,
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'dn_label': self.loss_dn_labels,
            'dn_bbox': self.loss_dn_boxes,
            'matching': self.loss_matching_cost,
        }
        
        idx = self._get_src_permutation_idx(indices[0])
        # pdb.set_trace()
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, idx, num_boxes,
                              data_batch, **kwargs)

    def prep_for_dn2(self, mask_dict):
        known_bboxs = mask_dict['known_bboxs']
        known_labels = mask_dict['known_labels']
        output_known_coord = mask_dict['output_known_coord']
        output_known_class = mask_dict['output_known_class']
        num_tgt = mask_dict['pad_size']

        return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt


    def forward(self, outputs, targets, data_batch, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        # import pdb; pdb.set_trace()
        outputs_without_aux = {
            k: v
            for k, v in outputs.items() if k != 'aux_outputs'
        }
        device = next(iter(outputs.values())).device

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t['boxes']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes],
                                    dtype=torch.float,
                                    device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # loss for final layer
        # pdb.set_trace()
        indices = self.matcher(outputs_without_aux, targets)
        if return_indices:
            indices0_copy = indices
            indices_list = []
        losses = {}
        smpl_loss = ['smpl_pose', 'smpl_beta', 'smpl_expr', 'smpl_kp2d',
                     'smpl_kp2d_ba', 'smpl_kp3d', 'smpl_kp3d_ra']
        # import pdb; pdb.set_trace()
        for loss in self.losses:
            # print(loss)
            # print(self.get_loss(loss, outputs, targets, indices, num_boxes))
            kwargs = {}

            if loss == 'keypoints' or loss in smpl_loss:
                kwargs.update({'face_hand_kpt': True})
            if loss == 'boxes':
                kwargs.update({'face_hand_box': True})

            losses.update(
                self.get_loss(
                    loss, outputs, targets, 
                    data_batch, indices, 
                    num_boxes, **kwargs
                    ))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'boxes':
                        kwargs.update({'face_hand_box': False})
                        if idx >= self.num_box_decoder_layers:
                            kwargs.update({'face_hand_box': True})
                            
                    if loss == 'masks':
                        continue
                    
                    if loss == 'keypoints':
                        if idx < self.num_box_decoder_layers:
                            continue
                        elif idx < self.num_hand_face_decoder_layers:
                            kwargs.update({'face_hand_kpt': False})
                        else:
                            kwargs.update({'face_hand_kpt': True})
                            
                    if loss in smpl_loss: 
                        if idx < self.num_box_decoder_layers:
                            continue
                        elif idx < self.num_hand_face_decoder_layers:
                            kwargs.update({'face_hand_kpt': False})
                        else:
                            kwargs.update({'face_hand_kpt': True})                    
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    
                    # if loss == 'smpl_expr' and idx < self.num_box_decoder_layers:
                    #     continue
                        
                    
                    # import pdb;pdb.set_trace()
                    l_dict = self.get_loss(loss, aux_outputs, targets,
                                           data_batch, indices, num_boxes,
                                           **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss in ['dn_bbox', 'dn_label', 'keypoints']:
                    continue
                if loss in [
                        'smpl_pose', 'smpl_beta', 'smpl_kp2d_ba', 'smpl_kp2d',
                        'smpl_kp3d_ra', 'smpl_kp3d', 'smpl_expr'
                ]:
                    continue
                kwargs = {}
                if loss == 'labels':
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, interm_outputs, targets,
                                       data_batch, indices, num_boxes,
                                       **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # aux_init loss
        if 'query_expand' in outputs:
            interm_outputs = outputs['query_expand']
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss in ['dn_bbox', 'dn_label']:
                    continue
                kwargs = {}

                if loss == 'labels':
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, interm_outputs, targets,
                                       data_batch, indices, num_boxes,
                                       **kwargs)
                l_dict = {k + f'_query_expand': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def tgt_loss_boxes(
        self,
        src_boxes,
        tgt_boxes,
        num_tgt,
    ):
        """
        Input:
            - src_boxes: bs, num_dn, 4
            - tgt_boxes: bs, num_dn, 4

        """
        
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')

        losses = {}
        losses['dn_loss_bbox'] = loss_bbox.sum() / num_tgt

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.flatten(0, 1)),
                box_ops.box_cxcywh_to_xyxy(tgt_boxes.flatten(0, 1))))
        losses['dn_loss_giou'] = loss_giou.sum() / num_tgt
        return losses

    def tgt_loss_labels(self,
                        src_logits: Tensor,
                        tgt_labels: Tensor,
                        num_tgt: int,
                        log: bool = True):
        """
        Input:
            - src_logits: bs, num_dn, num_classes
            - tgt_labels: bs, num_dn

        """
        target_classes_onehot = torch.zeros([
            src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1
        ],
                                            dtype=src_logits.dtype,
                                            layout=src_logits.layout,
                                            device=src_logits.device)
        target_classes_onehot.scatter_(2, tgt_labels.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits,
                                     target_classes_onehot,
                                     num_tgt,
                                     alpha=self.focal_alpha,
                                     gamma=2) * src_logits.shape[1]
        losses = {'dn_loss_ce': loss_ce}

        return losses

