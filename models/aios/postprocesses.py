import copy
import os
import math
from scipy.optimize import linear_sum_assignment
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms
from torch import Tensor
from pycocotools.coco import COCO
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, accuracy,
                       get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from detrsmpl.utils.demo_utils import convert_verts_to_cam_coord, xywh2xyxy, xyxy2xywh
import numpy as np
from detrsmpl.core.conventions.keypoints_mapping import convert_kps
from detrsmpl.models.body_models.builder import build_body_model
from detrsmpl.utils.geometry import batch_rodrigues, project_points, weak_perspective_projection,project_points_new
from util.human_models import smpl_x
from detrsmpl.core.conventions.keypoints_mapping import get_keypoint_idx
class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the
    coco api."""
    def __init__(self,
                 num_select=100,
                 nms_iou_threshold=-1,
                 num_body_points=17,
                 body_model=None) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold
        self.num_body_points = num_body_points
        self.body_model = build_body_model(
            dict(type='GenderedSMPL',
                 keypoint_src='h36m',
                 keypoint_dst='h36m',
                 model_path='data/body_models/smpl',
                 keypoint_approximate=True,
                 joints_regressor=
                 'data/body_models/J_regressor_h36m.npy'))

    @torch.no_grad()
    def forward(self,
                outputs,
                target_sizes,
                targets,
                data_batch_nc,
                device,
                not_to_xyxy=False,
                test=False):
        # import pdb; pdb.set_trace()
        num_select = self.num_select
        self.body_model.to(device)

        out_logits, out_bbox, out_keypoints= \
            outputs['pred_logits'], outputs['pred_boxes'], \
            outputs['pred_keypoints']

        out_smpl_pose, out_smpl_beta, out_smpl_cam, out_smpl_kp3d = \
            outputs['pred_smpl_pose'], outputs['pred_smpl_beta'], \
            outputs['pred_smpl_cam'], outputs['pred_smpl_kp3d']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = \
            torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        # bbox
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:, :, 2:] = boxes[:, :, 2:] - boxes[:, :, :2]
        boxes_norm = torch.gather(boxes, 1,
                                  topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        target_sizes = target_sizes.type_as(boxes)
        # from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes_norm * scale_fct[:, None, :]

        # keypoints
        topk_keypoints = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        keypoints = torch.gather(
            out_keypoints, 1,
            topk_keypoints.unsqueeze(-1).repeat(1, 1,
                                                self.num_body_points * 3))

        Z_pred = keypoints[:, :, :(self.num_body_points * 2)]
        V_pred = keypoints[:, :, (self.num_body_points * 2):]
        img_h, img_w = target_sizes.unbind(1)
        Z_pred = Z_pred * torch.stack([img_w, img_h], dim=1).repeat(
            1, self.num_body_points)[:, None, :]
        keypoints_res = torch.zeros_like(keypoints)
        keypoints_res[..., 0::3] = Z_pred[..., 0::2]
        keypoints_res[..., 1::3] = Z_pred[..., 1::2]
        keypoints_res[..., 2::3] = V_pred[..., 0::1]

        # smpl out_smpl_pose, out_smpl_beta, out_smpl_cam, out_smpl_kp3d
        topk_smpl = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        smpl_pose = torch.gather(
            out_smpl_pose, 1, topk_smpl[:, :, None, None,
                                        None].repeat(1, 1, 24, 3, 3))
        smpl_beta = torch.gather(out_smpl_beta, 1,
                                 topk_smpl[:, :, None].repeat(1, 1, 10))
        smpl_cam = torch.gather(out_smpl_cam, 1,
                                topk_smpl[:, :, None].repeat(1, 1, 3))
        smpl_kp3d = torch.gather(
            out_smpl_kp3d, 1,
            topk_smpl[:, :, None, None].repeat(1, 1, out_smpl_kp3d.shape[-2],
                                               3))
        tgt_smpl_kp3d = data_batch_nc['keypoints3d_smpl']
        tgt_smpl_pose = [
            torch.concat([
                data_batch_nc['smpl_global_orient'][i][:, None],
                data_batch_nc['smpl_body_pose'][i]
            ],
                         dim=-2)
            for i in range(len(data_batch_nc['smpl_body_pose']))
        ]
        tgt_smpl_beta = data_batch_nc['smpl_betas']
        tgt_keypoints = data_batch_nc['keypoints2d_ori']
        tgt_bbox = data_batch_nc['bbox_xywh']

        indices = []
        # pred
        pred_smpl_kp3d = []
        pred_smpl_pose = []
        pred_smpl_beta = []
        pred_scores = []
        pred_labels = []
        pred_boxes = []
        pred_keypoints = []
        pred_smpl_cam = []

        # gt
        gt_smpl_kp3d = []
        gt_smpl_pose = []
        gt_smpl_beta = []
        gt_boxes = []
        gt_keypoints = []
        image_idx = []

        results = []
        for i, kp3d in enumerate(tgt_smpl_kp3d):
            # kp3d
            conf = tgt_smpl_kp3d[i][..., [3]]
            gt_kp3d = tgt_smpl_kp3d[i][..., :3]
            pred_kp3d = smpl_kp3d[i]

            gt_output = self.body_model(
                betas=tgt_smpl_beta[i].float(),
                body_pose=tgt_smpl_pose[i][:, 1:].float().reshape(-1, 69),
                global_orient=tgt_smpl_pose[i][:, [0]].float().reshape(-1, 3),
                gender=torch.zeros(tgt_smpl_beta[i].shape[0]),
                pose2rot=True)
            gt_kp3d = gt_output['joints']
            # gt_kp3d,_ = convert_kps(
            #     gt_kp3d,
            #     src='smpl_54',
            #     dst='h36m',
            # )
            assert gt_kp3d.shape[-2] == 17

            H36M_TO_J17 = [
                6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9
            ]
            H36M_TO_J14 = H36M_TO_J17[:14]
            joint_mapper = H36M_TO_J14
            pred_pelvis = pred_kp3d[:, 0]
            gt_pelvis = gt_kp3d[:, 0]
            gt_keypoints3d = gt_kp3d[:, joint_mapper, :]
            pred_keypoints3d = pred_kp3d[:, joint_mapper, :]

            pred_keypoints3d = (pred_keypoints3d -
                                pred_pelvis[:, None, :]) * 1000
            gt_keypoints3d = (gt_keypoints3d - gt_pelvis[:, None, :]) * 1000
            
            cost_kp3d = torch.abs((pred_keypoints3d[:, None] -
                                   gt_keypoints3d[None])).sum([-2, -1])
            
            
            tgt_bbox[i][..., 2] = tgt_bbox[i][..., 0] + tgt_bbox[i][..., 2]
            tgt_bbox[i][..., 3] = tgt_bbox[i][..., 1] + tgt_bbox[i][..., 3]
            gt_bbox = tgt_bbox[i][..., :4].float()
            pred_bbox = boxes[i]
            # box_iou = box_ops.box_iou(pred_bbox,gt_bbox)[0]
            cost_giou = -box_ops.generalized_box_iou(pred_bbox, gt_bbox)
            
            indice = linear_sum_assignment(cost_giou.cpu())
            pred_ind, gt_ind = indice

            indices.append(indice)
            # bbox
            
            # cost_bbox = torch.cdist(pred_bbox, gt_bbox, p=1)
            # indice = linear_sum_assignment(cost_giou.cpu())
            # pred_ind, gt_ind = indice
            # indices.append(indice)

            # pred
            pred_scores.append(scores[i][pred_ind].detach().cpu().numpy())
            pred_labels.append(labels[i][pred_ind].detach().cpu().numpy())
            pred_boxes.append(boxes[i][pred_ind].detach().cpu().numpy())
            pred_keypoints.append(
                keypoints_res[i][pred_ind].detach().cpu().numpy())

            pred_smpl_kp3d.append(
                smpl_kp3d[i][pred_ind].detach().cpu().numpy())
            pred_smpl_pose.append(
                smpl_pose[i][pred_ind].detach().cpu().numpy())
            pred_smpl_beta.append(
                smpl_beta[i][pred_ind].detach().cpu().numpy())
            pred_smpl_cam.append(smpl_cam[i][pred_ind].detach().cpu().numpy())

            # gt
            gt_smpl_kp3d.append(
                tgt_smpl_kp3d[i][gt_ind].detach().cpu().numpy())
            gt_smpl_pose.append(
                tgt_smpl_pose[i][gt_ind].detach().cpu().numpy())
            gt_smpl_beta.append(
                tgt_smpl_beta[i][gt_ind].detach().cpu().numpy())
            gt_boxes.append(tgt_bbox[i][gt_ind].detach().cpu().numpy())
            gt_keypoints.append(
                tgt_keypoints[i][gt_ind].detach().cpu().numpy())
            image_idx.append(targets[i]['image_id'].detach().cpu().numpy())
            # gt_output = self.body_model(
            #     betas=tgt_smpl_beta[i].float(),
            #     body_pose=tgt_smpl_pose[i][:,1:].float().reshape(-1, 69),
            #     global_orient=tgt_smpl_pose[i][:,[0]].float().reshape(-1, 3),
            #     pose2rot=True
            #     )

        results.append({
            'scores': pred_scores,
            'labels': pred_labels,
            'boxes': pred_boxes,
            'keypoints': pred_keypoints,
            'pred_smpl_pose': pred_smpl_pose,
            'pred_smpl_beta': pred_smpl_beta,
            'pred_smpl_cam': pred_smpl_cam,
            'pred_smpl_kp3d': pred_smpl_kp3d,
            'gt_smpl_pose': gt_smpl_pose,
            'gt_smpl_beta': gt_smpl_beta,
            'gt_smpl_kp3d': gt_smpl_kp3d,
            'gt_boxes': gt_bbox,
            'gt_keypoints': gt_keypoints,
            'image_idx': image_idx,
        })
        # results.append({
        #     'scores': scores[i][pred_ind],
        #     'labels': labels[i][pred_ind],
        #     'boxes': boxes[i][pred_ind],
        #     'keypoints': keypoints_res[i][pred_ind],
        #     'pred_smpl_pose': smpl_pose[i][pred_ind],
        #     'pred_smpl_beta': tgt_smpl_beta[i][gt_ind],
        #     'pred_smpl_cam': smpl_cam[i][pred_ind],
        #     'pred_smpl_kp3d': smpl_kp3d[i][pred_ind],
        #     'gt_smpl_pose': tgt_smpl_pose[i][gt_ind],
        #     'gt_smpl_beta': tgt_smpl_beta[i][gt_ind],
        #     'gt_smpl_kp3d': tgt_smpl_kp3d[i][gt_ind],
        #     'gt_boxes': tgt_bbox[i][gt_ind],
        #     'gt_keypoints': tgt_keypoints[i][gt_ind],
        #     'image_idx': targets[i]['image_id'],
        #     }
        # )

        if self.nms_iou_threshold > 0:
            raise NotImplementedError
            item_indices = [
                nms(b, s, iou_threshold=self.nms_iou_threshold)
                for b, s in zip(boxes, scores)
            ]
            # import pdb; pdb.set_trace()
            results = [{
                'scores': s[i],
                'labels': l[i],
                'boxes': b[i]
            } for s, l, b, i in zip(scores, labels, boxes, item_indices)]
        else:
            results = results

        return results


class PostProcess_aios(nn.Module):
    """This module converts the model's output into the format expected by the
    coco api."""
    def __init__(self,
                 num_select=100,
                 nms_iou_threshold=-1,
                 num_body_points=17) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold
        self.num_body_points = num_body_points

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        num_select = self.num_select
        out_logits, out_bbox, out_keypoints = outputs['pred_logits'], outputs[
            'pred_boxes'], outputs['pred_keypoints']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(
            out_logits.shape[0], -1),
                                               num_select,
                                               dim=1)
        scores = topk_values

        # bbox
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:, :, 2:] = boxes[:, :, 2:] - boxes[:, :, :2]
        boxes = torch.gather(boxes, 1,
                             topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # keypoints
        topk_keypoints = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        keypoints = torch.gather(
            out_keypoints, 1,
            topk_keypoints.unsqueeze(-1).repeat(1, 1,
                                                self.num_body_points * 3))

        Z_pred = keypoints[:, :, :(self.num_body_points * 2)]
        V_pred = keypoints[:, :, (self.num_body_points * 2):]
        img_h, img_w = target_sizes.unbind(1)
        Z_pred = Z_pred * torch.stack([img_w, img_h], dim=1).repeat(
            1, self.num_body_points)[:, None, :]
        keypoints_res = torch.zeros_like(keypoints)
        keypoints_res[..., 0::3] = Z_pred[..., 0::2]
        keypoints_res[..., 1::3] = Z_pred[..., 1::2]
        keypoints_res[..., 2::3] = V_pred[..., 0::1]

        if self.nms_iou_threshold > 0:
            raise NotImplementedError
            item_indices = [
                nms(b, s, iou_threshold=self.nms_iou_threshold)
                for b, s in zip(boxes, scores)
            ]
            # import ipdb; ipdb.set_trace()
            results = [{
                'scores': s[i],
                'labels': l[i],
                'boxes': b[i]
            } for s, l, b, i in zip(scores, labels, boxes, item_indices)]
        else:
            results = [{
                'scores': s,
                'labels': l,
                'boxes': b,
                'keypoints': k
            } for s, l, b, k in zip(scores, labels, boxes, keypoints_res)]

        return results


class PostProcess_SMPLX(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(
        self, 
        num_select=100, 
        nms_iou_threshold=-1,
        num_body_points=17,
        body_model= dict(
            type='smplx',
            keypoint_src='smplx',
            num_expression_coeffs=10,
            keypoint_dst='smplx_137',
            model_path='data/body_models/smplx',
            use_pca=False,
            use_face_contour=True)
        ) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold
        self.num_body_points=num_body_points
        self.body_model = build_body_model(body_model)
        
    @torch.no_grad()
    def forward(self, outputs, target_sizes, targets, data_batch_nc, not_to_xyxy=False, test=False):
        # import pdb; pdb.set_trace()
        num_select = self.num_select
        
        out_logits, out_bbox, out_keypoints= \
            outputs['pred_logits'], outputs['pred_boxes'], \
            outputs['pred_keypoints']
            
        out_smpl_pose, out_smpl_beta, out_smpl_expr, out_smpl_cam, out_smpl_kp3d, out_smpl_verts = \
            outputs['pred_smpl_fullpose'], outputs['pred_smpl_beta'], outputs['pred_smpl_expr'], \
            outputs['pred_smpl_cam'], outputs['pred_smpl_kp3d'], outputs['pred_smpl_verts']
            
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = \
            torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        # bbox
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
        boxes_norm = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        target_sizes = target_sizes.type_as(boxes)
        # from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes_norm * scale_fct[:, None, :]


        # keypoints
        topk_keypoints = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        keypoints = torch.gather(out_keypoints, 1, topk_keypoints.unsqueeze(-1).repeat(1, 1, self.num_body_points*3))

        Z_pred = keypoints[:, :, :(self.num_body_points*2)]
        V_pred = keypoints[:, :, (self.num_body_points*2):]
        img_h, img_w = target_sizes.unbind(1)
        Z_pred = Z_pred * torch.stack([img_w, img_h], dim=1).repeat(1, self.num_body_points)[:, None, :]
        keypoints_res = torch.zeros_like(keypoints)
        keypoints_res[..., 0::3] = Z_pred[..., 0::2]
        keypoints_res[..., 1::3] = Z_pred[..., 1::2]
        keypoints_res[..., 2::3] = V_pred[..., 0::1]

        # smpl out_smpl_pose, out_smpl_beta, out_smpl_cam, out_smpl_kp3d
        topk_smpl = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        
        smpl_pose = torch.gather(out_smpl_pose, 1, topk_smpl[:,:,None].repeat(1, 1, 159))        
        smpl_beta = torch.gather(out_smpl_beta, 1, topk_smpl[:,:,None].repeat(1, 1, 10))   
        smpl_expr = torch.gather(out_smpl_expr, 1, topk_smpl[:,:,None].repeat(1, 1, 10))   
        smpl_cam = torch.gather(out_smpl_cam, 1, topk_smpl[:,:,None].repeat(1, 1, 3))   
        smpl_kp3d = torch.gather(out_smpl_kp3d, 1, topk_smpl[:,:,None, None].repeat(1, 1, out_smpl_kp3d.shape[-2],3))
        smpl_verts = torch.gather(out_smpl_verts, 1, topk_smpl[:,:,None, None].repeat(1, 1, out_smpl_verts.shape[-2],3))

        tgt_smpl_kp3d = data_batch_nc['joint_cam']
        tgt_smpl_kp3d_conf = data_batch_nc['joint_valid']
        tgt_smpl_pose = data_batch_nc['smplx_pose']
        tgt_smpl_beta = data_batch_nc['smplx_shape']
        tgt_smpl_expr = data_batch_nc['smplx_expr']
        tgt_keypoints = data_batch_nc['joint_img']
        tgt_img_shape = data_batch_nc['img_shape']
        tgt_ann_idx = data_batch_nc['ann_idx']
        # tgt_img_path = data_batch_nc['img_shape']
                
        
        tgt_bbox_center = torch.stack(data_batch_nc['body_bbox_center'])
        tgt_bbox_size = torch.stack(data_batch_nc['body_bbox_size'])
        tgt_bbox = torch.cat([tgt_bbox_center-tgt_bbox_size/2,tgt_bbox_size],dim=-1)
        tgt_bbox = tgt_bbox * scale_fct
        tgt_verts = data_batch_nc['smplx_mesh_cam']
        tgt_bb2img_trans = data_batch_nc['bb2img_trans']
        indices = []
        # pred
        pred_smpl_kp3d = []
        pred_smpl_pose = []
        pred_smpl_beta = []
        pred_smpl_verts = []
        pred_smpl_expr = []
        pred_scores = []
        pred_labels = []
        pred_boxes = []
        pred_keypoints = []
        pred_smpl_cam = []
        
        # gt
        gt_smpl_kp3d = []
        gt_smpl_pose = []
        gt_smpl_beta = []
        gt_smpl_expr = []
        gt_smpl_verts = []
        gt_boxes = []
        gt_keypoints = []
        gt_bb2img_trans = []
        image_idx = []       
        
        results = []
        for i, kp3d in enumerate(tgt_smpl_kp3d):
            # kp3d
            conf =  tgt_smpl_kp3d_conf[i][...,]
            gt_kp3d = tgt_smpl_kp3d[i][...,:3]
            pred_kp3d = smpl_kp3d[i]
            pred_kp3d_match,_ = convert_kps(pred_kp3d,'smplx','smplx_137')
            # pred_kp3d_match = pred_kp3d
            cost_kp3d = torch.abs((pred_kp3d_match[:,None] - 
              gt_kp3d[None])* conf[None]).sum([-2,-1])

            # bbox
            tgt_bbox[i][...,2] = tgt_bbox[i][...,0] + tgt_bbox[i][...,2]
            tgt_bbox[i][...,3] = tgt_bbox[i][...,1] + tgt_bbox[i][...,3]
            gt_bbox = tgt_bbox[i][..., :4][None].float()
            pred_bbox = boxes[i]
            # box_iou = box_ops.box_iou(pred_bbox,gt_bbox)[0]
            cost_giou = -box_ops.generalized_box_iou(pred_bbox,gt_bbox)
            # cost_bbox = torch.cdist(pred_bbox, gt_bbox, p=1)
            indice = linear_sum_assignment(cost_kp3d.cpu())
            pred_ind, gt_ind = indice
            indices=(indice)
            # pred
            pred_scores=(scores[i][pred_ind].detach().cpu().numpy())
            pred_labels=(labels[i][pred_ind].detach().cpu().numpy())
            pred_boxes=(boxes[i][pred_ind].detach().cpu().numpy())
            pred_keypoints=(keypoints_res[i][pred_ind].detach().cpu().numpy())
            
            pred_smpl_kp3d=(smpl_kp3d[i][pred_ind].detach().cpu().numpy())
            pred_smpl_pose=(smpl_pose[i][pred_ind].detach().cpu().numpy())
            pred_smpl_beta=(smpl_beta[i][pred_ind].detach().cpu().numpy())
            pred_smpl_cam=(smpl_cam[i][pred_ind].detach().cpu().numpy())
            pred_smpl_expr=(smpl_expr[i][pred_ind].detach().cpu().numpy())
            pred_smpl_verts=(smpl_verts[i][pred_ind].detach().cpu().numpy())
            # gt 
            # gt_smpl_kp3d=(tgt_smpl_kp3d[i][gt_ind].detach().cpu().numpy())
            # gt_smpl_pose=(tgt_smpl_pose[i][gt_ind].detach().cpu().numpy())
            # gt_smpl_beta=(tgt_smpl_beta[i][gt_ind].detach().cpu().numpy())
            # gt_boxes=(tgt_bbox[i][gt_ind].detach().cpu().numpy())
            # gt_smpl_expr=(tgt_smpl_expr[i][gt_ind].detach().cpu().numpy())
            
            # gt_smpl_verts=(tgt_verts[i][gt_ind].detach().cpu().numpy())
            # gt_keypoints=(tgt_keypoints[i][gt_ind].detach().cpu().numpy())
            # gt_bb2img_trans=(tgt_bb2img_trans[i][gt_ind].detach().cpu().numpy())

            gt_smpl_kp3d=(tgt_smpl_kp3d[i].detach().cpu().numpy())
            gt_smpl_pose=(tgt_smpl_pose[i].detach().cpu().numpy())
            gt_smpl_beta=(tgt_smpl_beta[i].detach().cpu().numpy())
            gt_boxes=(tgt_bbox[i].detach().cpu().numpy())
            gt_smpl_expr=(tgt_smpl_expr[i].detach().cpu().numpy())
            
            gt_smpl_verts=(tgt_verts[i].detach().cpu().numpy())
            gt_ann_idx=(tgt_ann_idx[i].detach().cpu().numpy())
            gt_keypoints=(tgt_keypoints[i].detach().cpu().numpy())
            gt_img_shape=(tgt_img_shape[i].detach().cpu().numpy())
            gt_bb2img_trans=(tgt_bb2img_trans[i].detach().cpu().numpy())
            if 'image_id' in targets[i]:
                image_idx=(targets[i]['image_id'].detach().cpu().numpy())
            
            # pred_smpl_pose = np.concatenate(pred_smpl_pose,axis = 0)
            # gt_bb2img_trans = np.concatenate(gt_bb2img_trans,axis = 0)
            # gt_smpl_verts = np.concatenate(gt_smpl_verts,axis = 0)
            # pred_smpl_verts = np.concatenate(pred_smpl_verts, axis = 0)
            # pred_smpl_cam = np.concatenate(pred_smpl_cam, axis = 0)
            #  import ipdb;ipdb.set_trace()
            smplx_root_pose = pred_smpl_pose[:,:3]
            smplx_body_pose = pred_smpl_pose[:,3:66]
            smplx_lhand_pose = pred_smpl_pose[:,66:111]
            smplx_rhand_pose = pred_smpl_pose[:,111:156]
            smplx_jaw_pose = pred_smpl_pose[:,156:]
            
            # pred_smpl_kp3d = np.concatenate(pred_smpl_kp3d,axis = 0)

            pred_smpl_cam = torch.Tensor(pred_smpl_cam)
            pred_smpl_kp3d = torch.Tensor(pred_smpl_kp3d)
            # pred_smpl_kp2d = weak_perspective_projection(pred_smpl_kp3d, scale=pred_smpl_cam[:, :1], translation=pred_smpl_cam[:, 1:3])
            # pred_smpl_verts2d = weak_perspective_projection(pred_smpl_kp3d, scale=pred_smpl_cam[:, :1], translation=pred_smpl_cam[:, 1:3])
            img_wh = tgt_img_shape[i].flip(-1)[None]
            pred_smpl_kp2d = project_points_new(
                points_3d=pred_smpl_kp3d,
                pred_cam=pred_smpl_cam,
                focal_length=5000,
                camera_center=img_wh/2
            )
            
            pred_smpl_kp2d = pred_smpl_kp2d.numpy()
            pred_smpl_cam = pred_smpl_cam.numpy()
            # cam_trans = get_camera_trans(pred_smpl_cam)
            
            # pred_smpl_kp2d = (pred_smpl_kp2d+1)/2
            # pred_smpl_kp2d[:, :,0] = pred_smpl_kp2d[:, :, 0] * gt_img_shape[1]
            # pred_smpl_kp2d[:, :, 1] = pred_smpl_kp2d[:, :, 1] * gt_img_shape[0]
            # # joint_proj = np.dot(out['bb2img_trans'], joint_proj.transpose(1, 0)).transpose(1, 0)
            # # joint_proj[:, 0] = joint_proj[:, 0] / self.resolution[1] * 3840  # restore to original resolution
            # # joint_proj[:, 1] = joint_proj[:, 1] / self.resolution[0] * 2160  # restore to original resolution
            
            
            results.append({
                    'scores': pred_scores, 
                    'labels': pred_labels, 
                    'boxes': pred_boxes[0], 
                    'keypoints': pred_keypoints[0],
                    'smplx_root_pose': smplx_root_pose[0], 
                    'smplx_body_pose': smplx_body_pose[0],
                    'smplx_lhand_pose': smplx_lhand_pose[0],
                    'smplx_rhand_pose': smplx_rhand_pose[0],
                    'smplx_jaw_pose': smplx_jaw_pose[0],
                    'smplx_shape': pred_smpl_beta[0], 
                    'smplx_expr': pred_smpl_expr[0], 
                    'cam_trans': pred_smpl_cam[0], 
                    'smplx_mesh_cam': pred_smpl_verts[0],
                    'smplx_mesh_cam_target': gt_smpl_verts,
                    'gt_ann_idx':gt_ann_idx,
                    'gt_smpl_kp3d':gt_smpl_kp3d,
                    'smplx_joint_proj': pred_smpl_kp2d[0],
                    'image_idx': image_idx,
                    'bb2img_trans': gt_bb2img_trans,
                    'img_shape': gt_img_shape
                })
                # results.append({
                #     'scores': scores[i][pred_ind], 
                #     'labels': labels[i][pred_ind], 
                #     'boxes': boxes[i][pred_ind], 
                #     'keypoints': keypoints_res[i][pred_ind],
                #     'pred_smpl_pose': smpl_pose[i][pred_ind], 
                #     'pred_smpl_beta': tgt_smpl_beta[i][gt_ind], 
                #     'pred_smpl_cam': smpl_cam[i][pred_ind], 
                #     'pred_smpl_kp3d': smpl_kp3d[i][pred_ind], 
                #     'gt_smpl_pose': tgt_smpl_pose[i][gt_ind],
                #     'gt_smpl_beta': tgt_smpl_beta[i][gt_ind],
                #     'gt_smpl_kp3d': tgt_smpl_kp3d[i][gt_ind],
                #     'gt_boxes': tgt_bbox[i][gt_ind],
                #     'gt_keypoints': tgt_keypoints[i][gt_ind],
                #     'image_idx': targets[i]['image_id'],
                #     }
                # )   

        if self.nms_iou_threshold > 0:
            raise NotImplementedError
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]
            # import pdb; pdb.set_trace()
            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]
        else:
            results = results

        return results
    
    
class PostProcess_SMPLX_Multi(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(
        self, 
        num_select=100, 
        nms_iou_threshold=-1,
        num_body_points=17,
        body_model= dict(
            type='smplx',
            keypoint_src='smplx',
            num_expression_coeffs=10,
            num_betas=10,
            gender='neutral',
            keypoint_dst='smplx_137',
            model_path='data/body_models/smplx',
            use_pca=False,
            use_face_contour=True,
            ),
        ) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold
        self.num_body_points=num_body_points
        
        # -1 for neutral; 0 for male; 1 for femal
        gender_body_model = {}
        gender_body_model[-1] = build_body_model(body_model)
        
        body_model['gender']='male'
        gender_body_model[0] = build_body_model(body_model)
        
        body_model['gender']='female'
        gender_body_model[1] = build_body_model(body_model)
        
        self.body_model = gender_body_model
    @torch.no_grad()
    def forward(self, outputs, target_sizes, targets, data_batch_nc, not_to_xyxy=False, test=False,  dataset = None):
        # import pdb; pdb.set_trace()
        batch_size = outputs['pred_keypoints'].shape[0]
        results = []
        device = outputs['pred_keypoints'].device
        for body_model in self.body_model.values():
            body_model.to(device)

        num_select = 1
        out_logits, out_bbox, out_keypoints= \
            outputs['pred_logits'], outputs['pred_boxes'], \
            outputs['pred_keypoints']
        
        out_smpl_pose, out_smpl_beta, out_smpl_expr, out_smpl_cam, out_smpl_kp3d, out_smpl_verts = \
            outputs['pred_smpl_fullpose'], outputs['pred_smpl_beta'], outputs['pred_smpl_expr'], \
            outputs['pred_smpl_cam'], outputs['pred_smpl_kp3d'], outputs['pred_smpl_verts']

        out_smpl_kp2d = []
        for bs in range(batch_size):
            out_kp3d_i = out_smpl_kp3d[bs]
            out_cam_i = out_smpl_cam[bs]
            out_img_shape = data_batch_nc['img_shape'][bs].flip(-1)[None]
            # out_kp3d_i = out_kp3d_i - out_kp3d_i[:, [0]]
            out_kp2d_i = project_points_new(
                points_3d=out_kp3d_i,
                pred_cam=out_cam_i,
                focal_length=5000,
                camera_center=out_img_shape/2
            )   
            out_smpl_kp2d.append(out_kp2d_i.detach().cpu().numpy())
        out_smpl_kp2d = torch.tensor(out_smpl_kp2d).to(device)

            
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = \
            torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        
        # bbox
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
        
        # gather gt bbox
        boxes_norm = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        target_sizes = target_sizes.type_as(boxes)
        # from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes_norm * scale_fct[:, None, :]

        # smplx kp2d
        topk_smpl_kp2d = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        pred_smpl_kp2d = torch.gather(
            out_smpl_kp2d, 1, 
            topk_smpl_kp2d.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 137, 2)) 
               
       
        # keypoints
        topk_keypoints = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        keypoints = torch.gather(
            out_keypoints, 1, 
            topk_keypoints.unsqueeze(-1).repeat(1, 1, self.num_body_points*3))

        Z_pred = keypoints[:, :, :(self.num_body_points * 2)]
        V_pred = keypoints[:, :, (self.num_body_points * 2):]
        img_h, img_w = target_sizes.unbind(1)
        Z_pred = Z_pred * torch.stack([img_w, img_h], dim=1).repeat(1, self.num_body_points)[:, None, :]
        keypoints_res = torch.zeros_like(keypoints)
        keypoints_res[..., 0::3] = Z_pred[..., 0::2]
        keypoints_res[..., 1::3] = Z_pred[..., 1::2]
        keypoints_res[..., 2::3] = V_pred[..., 0::1]

        # smpl out_smpl_pose, out_smpl_beta, out_smpl_cam, out_smpl_kp3d
        topk_smpl = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        
        smpl_pose = torch.gather(out_smpl_pose, 1, topk_smpl[:,:,None].repeat(1, 1, 159))        
        smpl_beta = torch.gather(out_smpl_beta, 1, topk_smpl[:,:,None].repeat(1, 1, 10))   
        smpl_expr = torch.gather(out_smpl_expr, 1, topk_smpl[:,:,None].repeat(1, 1, 10))   
        smpl_cam = torch.gather(out_smpl_cam, 1, topk_smpl[:,:,None].repeat(1, 1, 3))   
        smpl_kp3d = torch.gather(out_smpl_kp3d, 1, topk_smpl[:,:,None, None].repeat(1, 1, out_smpl_kp3d.shape[-2],3))
        smpl_verts = torch.gather(out_smpl_verts, 1, topk_smpl[:,:,None, None].repeat(1, 1, out_smpl_verts.shape[-2],3))
      
        tgt_smpl_kp3d = data_batch_nc['joint_cam']
        # tgt_smpl_kp3d_conf = data_batch_nc['joint_valid']
        tgt_smpl_pose = data_batch_nc['smplx_pose']
        tgt_smpl_beta = data_batch_nc['smplx_shape']
        tgt_smpl_expr = data_batch_nc['smplx_expr']
        tgt_keypoints = data_batch_nc['joint_img']
        tgt_img_shape = data_batch_nc['img_shape']
        # tgt_bbox_center = data_batch_nc['body_bbox_center']
        # tgt_bbox_size = data_batch_nc['body_bbox_size']
        tgt_bb2img_trans = data_batch_nc['bb2img_trans']
        tgt_ann_idx = data_batch_nc['ann_idx']
        
        pred_indice_list = []
        gt_indice_list = []
        tgt_verts = []
        tgt_kp3d = []
        tgt_bbox = []
        for bbox_center, bbox_size, pose, \
            beta, expr, gender, gt_kp2d, _, pred_kp2d, pred_kp3d, boxe, scale \
                in zip(
                    data_batch_nc['body_bbox_center'], 
                    data_batch_nc['body_bbox_size'], 
                    # data_batch_nc['bb2img_trans'],
                    data_batch_nc['smplx_pose'],
                    data_batch_nc['smplx_shape'],
                    data_batch_nc['smplx_expr'],
                    data_batch_nc['gender'],
                    data_batch_nc['joint_img'],
                    data_batch_nc['joint_cam'],
                    # keypoints_res, smpl_kp3d, boxes, scale_fct,
                    pred_smpl_kp2d, smpl_kp3d, boxes, scale_fct,
                ):
            # build smplx verts
            gt_verts = []
            gt_kp3d = []
            gt_bbox = []
            gender_ = gender.cpu().numpy()
        
            for i, g in enumerate(gender_):
                gt_out = self.body_model[g](
                    betas=beta[i].reshape(-1, 10),
                    global_orient=pose[i, :3].reshape(-1, 3).unsqueeze(1),
                    body_pose=pose[i, 3:66].reshape(-1, 21 * 3),
                    left_hand_pose=pose[i, 66:111].reshape(-1, 15 * 3),
                    right_hand_pose=pose[i, 111:156].reshape(-1, 15 * 3),
                    jaw_pose=pose[i, 156:159].reshape(-1, 3),
                    leye_pose=torch.zeros_like(pose[i, 156:159]),
                    reye_pose=torch.zeros_like(pose[i, 156:159]),
                    expression=expr[i].reshape(-1, 10),                
                )
                gt_verts.append(gt_out['vertices'][0].detach().cpu().numpy())
                gt_kp3d.append(gt_out['joints'][0].detach().cpu().numpy())
            
            tgt_verts.append(gt_verts)
            tgt_kp3d.append(gt_kp3d)
                    
            # bbox
            gt_bbox = torch.cat(
                [bbox_center - bbox_size / 2, bbox_size ], dim=-1)
            gt_bbox = gt_bbox * scale
            # xywh2xyxy
            gt_bbox[..., 2] = gt_bbox[..., 0] + gt_bbox[..., 2]
            gt_bbox[..., 3] = gt_bbox[..., 1] + gt_bbox[..., 3]
            tgt_bbox.append(gt_bbox[..., :4].float())
            
            pred_bbox = boxe.clone()
            # box_iou = box_ops.box_iou(pred_bbox,gt_bbox)[0]
            cost_giou = -box_ops.generalized_box_iou(pred_bbox, gt_bbox)
            
            
            cost_bbox = torch.cdist(
                box_ops.box_xyxy_to_cxcywh(pred_bbox)/scale, 
                box_ops.box_xyxy_to_cxcywh(gt_bbox)/scale, p=1)
            
            # smpl kp2d
            gt_kp2d_conf = gt_kp2d[:,:,2:3]
            gt_kp2d_ = (gt_kp2d[:, :, :2] * scale[:2]) /torch.tensor([12, 16]).to(device)
            
            gt_kp2d_body = gt_kp2d_[:, smpl_x.joint_part['body']]
            gt_kp2d_body_conf  = gt_kp2d_conf[:, smpl_x.joint_part['body']]
            pred_kp2d_body = pred_kp2d[:, smpl_x.joint_part['body']] # smplx kps head

            if dataset.__class__.__name__ == 'UBody_MM':
                cost_keypoints = torch.abs(
                    (pred_kp2d_body[:, None]/scale[:2] - gt_kp2d_body[None]/scale[:2])*gt_kp2d_body_conf[None]
                                        ).sum([-2,-1])/gt_kp2d_body_conf[None].sum()            
            else:
                cost_keypoints = torch.abs(
                (pred_kp2d_body[:, None]/scale[:2] - gt_kp2d_body[None]/scale[:2])
                                       ).sum([-2,-1])
            # smpl kp3d
            gt_kp3d_ = torch.tensor(np.array(gt_kp3d) - np.array(gt_kp3d)[:, [0]]).to(device)
            pred_kp3d_ = (pred_kp3d - pred_kp3d[:, [0]])
            cost_kp3d = torch.abs((pred_kp3d_[:, None] - gt_kp3d_[None])).sum([-2,-1])
            
            # 1. kps
            indice = linear_sum_assignment(cost_keypoints.cpu())
            
            pred_ind, gt_ind = indice
            pred_indice_list.append(pred_ind)
            gt_indice_list.append(gt_ind)
            
        pred_scores = torch.cat(
            [t[i] for t, i in zip(scores, pred_indice_list)]
            ).detach().cpu().numpy()
        pred_labels = torch.cat(
            [t[i] for t, i in zip(labels, pred_indice_list)]
            ).detach().cpu().numpy()
        pred_boxes = torch.cat(
            [t[i] for t, i in zip(boxes, pred_indice_list)]
            ).detach().cpu().numpy()
        pred_keypoints = torch.cat(
            [t[i] for t, i in zip(keypoints_res, pred_indice_list)]
            ).detach().cpu().numpy()
        


        
        pred_smpl_kp2d = []
        pred_smpl_kp3d = []
        pred_smpl_cam = []
        img_wh_list = []
        for i, img_wh in enumerate(tgt_img_shape):
            
            kp3d = smpl_kp3d[i][pred_indice_list[i]]
            cam = smpl_cam[i][pred_indice_list[i]]
            img_wh = img_wh.flip(-1)[None]
            
            kp2d = project_points_new(
                points_3d=kp3d,
                pred_cam=cam,
                focal_length=5000,
                camera_center=img_wh/2
            )     
            num_instance = kp2d.shape[0]
            img_wh_list.append(img_wh.repeat(num_instance,1).cpu().numpy())
            pred_smpl_kp2d.append(kp2d.detach().cpu().numpy())
            pred_smpl_kp3d.append(kp3d.detach().cpu().numpy())
            pred_smpl_cam.append(cam.detach().cpu().numpy())
        pred_smpl_pose = torch.cat(
            [t[i] for t, i in zip(smpl_pose, pred_indice_list)]
            ).detach().cpu().numpy()
        pred_smpl_beta = torch.cat(
            [t[i] for t, i in zip(smpl_beta, pred_indice_list)]
            ).detach().cpu().numpy()
        pred_smpl_expr = torch.cat(
            [t[i] for t, i in zip(smpl_expr, pred_indice_list)]
            ).detach().cpu().numpy()
        pred_smpl_verts = torch.cat(
            [t[i] for t, i in zip(smpl_verts, pred_indice_list)]
            ).detach().cpu().numpy()

        pred_smpl_kp2d = np.concatenate(pred_smpl_kp2d, 0)
        pred_smpl_kp3d = np.concatenate(pred_smpl_kp3d, 0)
        pred_smpl_cam = np.concatenate(pred_smpl_cam, 0)       
        img_wh_list = np.concatenate(img_wh_list, 0)   

        gt_smpl_kp3d = torch.cat(tgt_smpl_kp3d).detach().cpu().numpy()
        gt_smpl_pose = torch.cat(tgt_smpl_pose).detach().cpu().numpy()
        gt_smpl_beta = torch.cat(tgt_smpl_beta).detach().cpu().numpy()
        gt_boxes = torch.cat(tgt_bbox).detach().cpu().numpy()
        gt_smpl_expr = torch.cat(tgt_smpl_expr).detach().cpu().numpy()
        # gt_img_shape = torch.cat(tgt_img_shape).detach().cpu().numpy()
        gt_smpl_verts = np.concatenate(
            [np.array(t)[i] for t, i in zip(tgt_verts, gt_indice_list)], 0)
        gt_ann_idx = torch.cat([t.repeat(len(i)) for t, i in zip(tgt_ann_idx, gt_indice_list)],dim=0).cpu().numpy()
        
        gt_keypoints = torch.cat(tgt_keypoints).detach().cpu().numpy()
        # gt_img_shape = tgt_img_shape.detach().cpu().numpy()
        gt_bb2img_trans = torch.stack(tgt_bb2img_trans).detach().cpu().numpy()

        if 'image_id' in targets[i]:
            image_idx=(targets[i]['image_id'].detach().cpu().numpy())

        smplx_root_pose = pred_smpl_pose[:,:3]
        smplx_body_pose = pred_smpl_pose[:,3:66]
        smplx_lhand_pose = pred_smpl_pose[:,66:111]
        smplx_rhand_pose = pred_smpl_pose[:,111:156]
        smplx_jaw_pose = pred_smpl_pose[:,156:]

        results.append({
                    'scores': pred_scores, 
                    'labels': pred_labels, 
                    'boxes': pred_boxes, 
                    'keypoints': pred_keypoints,
                    'smplx_root_pose': smplx_root_pose, 
                    'smplx_body_pose': smplx_body_pose,
                    'smplx_lhand_pose': smplx_lhand_pose,
                    'smplx_rhand_pose': smplx_rhand_pose,
                    'smplx_jaw_pose': smplx_jaw_pose,
                    'smplx_shape': pred_smpl_beta, 
                    'smplx_expr': pred_smpl_expr, 
                    'cam_trans': pred_smpl_cam, 
                    'smplx_mesh_cam': pred_smpl_verts,
                    'smplx_mesh_cam_target': gt_smpl_verts,
                    'gt_smpl_kp3d':gt_smpl_kp3d,
                    'smplx_joint_proj': pred_smpl_kp2d,
                    # 'image_idx': image_idx,
                    "img": data_batch_nc['img'].cpu().numpy(),
                    'bb2img_trans': gt_bb2img_trans,
                    'img_shape': img_wh_list,
                    'gt_ann_idx': gt_ann_idx
                })

        if self.nms_iou_threshold > 0:
            raise NotImplementedError
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]
            # import pdb; pdb.set_trace()
            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]
        else:
            results = results

        return results


class PostProcess_SMPLX_Multi_Infer(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(
        self, 
        num_select=100, 
        nms_iou_threshold=-1,
        num_body_points=17,
        body_model= dict(
            type='smplx',
            keypoint_src='smplx',
            num_expression_coeffs=10,
            num_betas=10,
            gender='neutral',
            keypoint_dst='smplx_137',
            model_path='data/body_models/smplx',
            use_pca=False,
            use_face_contour=True)
        ) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold
        self.num_body_points=num_body_points
        
        # -1 for neutral; 0 for male; 1 for femal
        gender_body_model = {}
        gender_body_model[-1] = build_body_model(body_model)
        
        body_model['gender']='male'
        gender_body_model[0] = build_body_model(body_model)
        
        body_model['gender']='female'
        gender_body_model[1] = build_body_model(body_model)
        
        self.body_model = gender_body_model
        
    @torch.no_grad()
    def forward(self, outputs, target_sizes, targets, data_batch_nc, image_shape= None, not_to_xyxy=False, test=False):
        """
        image_shape(target_sizes): input image shape
        
        """
        # import pdb; pdb.set_trace()
        batch_size = outputs['pred_keypoints'].shape[0]
        results = []
        device = outputs['pred_keypoints'].device
        # for body_model in self.body_model.values():
        #     body_model.to(device)
        
        pred_kp_coco = outputs['pred_keypoints']
        num_select = self.num_select
        out_logits, out_bbox= outputs['pred_logits'], outputs['pred_boxes']
        
        out_body_bbox, out_lhand_bbox, out_rhand_bbox, out_face_bbox = \
            outputs['pred_boxes'], outputs['pred_lhand_boxes'], \
                outputs['pred_rhand_boxes'], outputs['pred_face_boxes']
            
        out_smpl_pose, out_smpl_beta, out_smpl_expr, out_smpl_cam, out_smpl_kp3d, out_smpl_verts = \
            outputs['pred_smpl_fullpose'], outputs['pred_smpl_beta'], outputs['pred_smpl_expr'], \
            outputs['pred_smpl_cam'], outputs['pred_smpl_kp3d'], outputs['pred_smpl_verts']

        out_smpl_kp2d = []
        for bs in range(batch_size):
            out_kp3d_i = out_smpl_kp3d[bs]
            out_cam_i = out_smpl_cam[bs]
            out_img_shape = data_batch_nc['img_shape'][bs].flip(-1)[None]

            out_kp2d_i = project_points_new(
                points_3d=out_kp3d_i,
                pred_cam=out_cam_i,
                focal_length=5000,
                camera_center=out_img_shape/2
            )   
            out_smpl_kp2d.append(out_kp2d_i.detach().cpu().numpy())
        out_smpl_kp2d = torch.tensor(out_smpl_kp2d).to(device)

            
        # assert len(out_logits) == len(target_sizes)
        # assert target_sizes.shape[1] == 2
        
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = \
            torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        
        # bbox
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            out_body_bbox = box_ops.box_cxcywh_to_xyxy(out_body_bbox)
            out_lhand_bbox = box_ops.box_cxcywh_to_xyxy(out_lhand_bbox)
            out_rhand_bbox = box_ops.box_cxcywh_to_xyxy(out_rhand_bbox)
            out_face_bbox = box_ops.box_cxcywh_to_xyxy(out_face_bbox)
            
        # gather body bbox
        target_sizes = target_sizes.type_as(boxes)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        
        boxes_norm = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        boxes = boxes_norm * scale_fct[:, None, :]
        
        body_bbox_norm = torch.gather(out_body_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        body_boxes = body_bbox_norm * scale_fct[:, None, :]
        
        lhand_bbox_norm = torch.gather(out_lhand_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        lhand_boxes = lhand_bbox_norm * scale_fct[:, None, :]
        
        rhand_bbox_norm = torch.gather(out_rhand_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        rhand_boxes = rhand_bbox_norm * scale_fct[:, None, :]
        
        face_bbox_norm = torch.gather(out_face_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        face_boxes = face_bbox_norm * scale_fct[:, None, :]
        
        # from relative [0, 1] to absolute [0, height] coordinates

        # smplx kp2d
        topk_smpl_kp2d = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        pred_smpl_kp2d = torch.gather(
            out_smpl_kp2d, 1, 
            topk_smpl_kp2d.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 144, 2))        
        # pred_smpl_kp2d = np.concatenate(pred_smpl_kp2d, 0)
        pred_kp_coco = pred_kp_coco[..., 0:17*2].reshape(pred_kp_coco.shape[0], pred_kp_coco.shape[1], 17, 2)
        # pred_kp_coco_norm = torch.gather(
        #     pred_kp_coco, 1, 
        #     topk_smpl_kp2d.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 17, 2))  
        # pred_kp_coco = pred_kp_coco_norm * scale_fct[:, None, :2]
        # smpl param
        topk_smpl = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        smpl_pose = torch.gather(out_smpl_pose, 1, topk_smpl[:,:,None].repeat(1, 1, 159))        
        smpl_beta = torch.gather(out_smpl_beta, 1, topk_smpl[:,:,None].repeat(1, 1, 10))   
        smpl_expr = torch.gather(out_smpl_expr, 1, topk_smpl[:,:,None].repeat(1, 1, 10))   
        smpl_cam = torch.gather(out_smpl_cam, 1, topk_smpl[:,:,None].repeat(1, 1, 3))   
        smpl_kp3d = torch.gather(out_smpl_kp3d, 1, topk_smpl[:,:,None, None].repeat(1, 1, out_smpl_kp3d.shape[-2],3))
        smpl_verts = torch.gather(out_smpl_verts, 1, topk_smpl[:,:,None, None].repeat(1, 1, out_smpl_verts.shape[-2],3))
        # smpl_verts = smpl_verts - smpl_kp3d[:,:, [0]]
        (s, tx, ty) = (smpl_cam[..., 0] + 1e-9), smpl_cam[..., 1], smpl_cam[..., 2]
        depth, dx, dy = 1./s, tx/s, ty/s
        transl = torch.stack([dx, dy, depth], -1) 
        
        smplx_root_pose = smpl_pose[:, :, :3]
        smplx_body_pose = smpl_pose[:, :, 3:66]
        smplx_lhand_pose = smpl_pose[:, :, 66:111]
        smplx_rhand_pose = smpl_pose[:, :, 111:156]
        smplx_jaw_pose = smpl_pose[:, :, 156:]
        
        if 'ann_idx' in data_batch_nc:
            image_idx=[target.cpu().numpy()[0] for target in data_batch_nc['ann_idx']]

        for bs in range(batch_size):
            results.append({
                        'scores': scores[bs], 
                        'labels': labels[bs], 
                        'keypoints_coco': pred_kp_coco[bs],
                        'smpl_kp3d': smpl_kp3d[bs],
                        'smplx_root_pose': smplx_root_pose[bs], 
                        'smplx_body_pose': smplx_body_pose[bs],
                        'smplx_lhand_pose': smplx_lhand_pose[bs],
                        'smplx_rhand_pose': smplx_rhand_pose[bs],
                        'smplx_jaw_pose': smplx_jaw_pose[bs],
                        'smplx_shape': smpl_beta[bs], 
                        'smplx_expr': smpl_expr[bs], 
                        'smplx_joint_proj': pred_smpl_kp2d[bs],
                        'smpl_verts': smpl_verts[bs],
                        'image_idx': image_idx[bs],
                        'cam_trans': transl[bs],
                        'body_bbox': body_boxes[bs],
                        'lhand_bbox': lhand_boxes[bs],
                        'rhand_bbox': rhand_boxes[bs],
                        'face_bbox': face_boxes[bs],
                        'bb2img_trans': data_batch_nc['bb2img_trans'][bs],
                        'img2bb_trans': data_batch_nc['img2bb_trans'][bs],
                        'img': data_batch_nc['img'][bs],
                        'img_shape': data_batch_nc['img_shape'][bs]
                    })

        if self.nms_iou_threshold > 0:
            raise NotImplementedError
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]
            # import pdb; pdb.set_trace()
            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]
        else:
            results = results

        return results


class PostProcess_SMPLX_Multi_Box(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(
        self, 
        num_select=100, 
        nms_iou_threshold=-1,
        num_body_points=17,
        body_model= dict(
            type='smplx',
            keypoint_src='smplx',
            num_expression_coeffs=10,
            num_betas=10,
            gender='neutral',
            keypoint_dst='smplx_137',
            model_path='data/body_models/smplx',
            use_pca=False,
            use_face_contour=True)
        ) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold
        self.num_body_points=num_body_points
        
        # -1 for neutral; 0 for male; 1 for femal
        gender_body_model = {}
        gender_body_model[-1] = build_body_model(body_model)
        
        body_model['gender']='male'
        gender_body_model[0] = build_body_model(body_model)
        
        body_model['gender']='female'
        gender_body_model[1] = build_body_model(body_model)
        
        self.body_model = gender_body_model
        
        
    @torch.no_grad()
    def forward(self, outputs, target_sizes, targets, data_batch_nc, not_to_xyxy=False, test=False):
        # import pdb; pdb.set_trace()
        batch_size = outputs['pred_smpl_beta'].shape[0]
        results = []
        device = outputs['pred_smpl_beta'].device
        for body_model in self.body_model.values():
            body_model.to(device)
        # test with instance num
        # num_select=data_batch_nc['joint_img'][0].shape[0]
        num_select = self.num_select
        out_logits, out_bbox= outputs['pred_logits'], outputs['pred_boxes']

        out_smpl_pose, out_smpl_beta, out_smpl_expr, out_smpl_cam, out_smpl_kp3d, out_smpl_verts = \
            outputs['pred_smpl_fullpose'], outputs['pred_smpl_beta'], outputs['pred_smpl_expr'], \
            outputs['pred_smpl_cam'], outputs['pred_smpl_kp3d'], outputs['pred_smpl_verts']

        out_smpl_kp2d = []
        
        for bs in range(batch_size):
            out_kp3d_i = out_smpl_kp3d[bs]
            out_cam_i = out_smpl_cam[bs]
            out_img_shape = data_batch_nc['img_shape'][bs].flip(-1)[None]
            # out_kp3d_i = out_kp3d_i - out_kp3d_i[:, [0]]
            out_kp2d_i = project_points_new(
                points_3d=out_kp3d_i,
                pred_cam=out_cam_i,
                focal_length=5000,
                camera_center=out_img_shape/2
            )   
            out_smpl_kp2d.append(out_kp2d_i.detach().cpu().numpy())
        out_smpl_kp2d = torch.tensor(out_smpl_kp2d).to(device)

            
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = \
            torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        
        # bbox
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
        
        # gather gt bbox
        boxes_norm = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        target_sizes = target_sizes.type_as(boxes)
        # from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes_norm * scale_fct[:, None, :]

        # smplx kp2d
        topk_smpl_kp2d = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        pred_smpl_kp2d = torch.gather(
            out_smpl_kp2d, 1, 
            topk_smpl_kp2d.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 137, 2)) 

        # smpl out_smpl_pose, out_smpl_beta, out_smpl_cam, out_smpl_kp3d
        topk_smpl = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        smpl_pose = torch.gather(out_smpl_pose, 1, topk_smpl[:,:,None].repeat(1, 1, 159))        
        smpl_beta = torch.gather(out_smpl_beta, 1, topk_smpl[:,:,None].repeat(1, 1, 10))   
        smpl_expr = torch.gather(out_smpl_expr, 1, topk_smpl[:,:,None].repeat(1, 1, 10))   
        smpl_cam = torch.gather(out_smpl_cam, 1, topk_smpl[:,:,None].repeat(1, 1, 3))   
        smpl_kp3d = torch.gather(out_smpl_kp3d, 1, topk_smpl[:,:,None, None].repeat(1, 1, out_smpl_kp3d.shape[-2],3))
        smpl_verts = torch.gather(out_smpl_verts, 1, topk_smpl[:,:,None, None].repeat(1, 1, out_smpl_verts.shape[-2],3))

        tgt_smpl_kp3d = data_batch_nc['joint_cam']
        tgt_smpl_pose = data_batch_nc['smplx_pose']
        tgt_smpl_beta = data_batch_nc['smplx_shape']
        tgt_smpl_expr = data_batch_nc['smplx_expr']
        tgt_keypoints = data_batch_nc['joint_img']
        tgt_img_shape = data_batch_nc['img_shape']
        tgt_bb2img_trans = data_batch_nc['bb2img_trans']
        tgt_ann_idx = data_batch_nc['ann_idx']

        pred_indice_list = []
        gt_indice_list = []
        tgt_verts = []
        tgt_kp3d = []
        tgt_bbox = []
        for bbox_center, bbox_size, pose, \
            beta, expr, gender, gt_kp2d, _, pred_kp2d, pred_kp3d, boxe, scale \
                in zip(
                    data_batch_nc['body_bbox_center'], 
                    data_batch_nc['body_bbox_size'], 
                    data_batch_nc['smplx_pose'],
                    data_batch_nc['smplx_shape'],
                    data_batch_nc['smplx_expr'],
                    data_batch_nc['gender'],
                    data_batch_nc['joint_img'],
                    data_batch_nc['joint_cam'],
                    pred_smpl_kp2d, smpl_kp3d, boxes, scale_fct,
                ):
            # build smplx verts
            gt_verts = []
            gt_kp3d = []
            gt_bbox = []
            gender_ = gender.cpu().numpy()
        
            for i, g in enumerate(gender_):
                gt_out = self.body_model[g](
                    betas=beta[i].reshape(-1, 10),
                    global_orient=pose[i, :3].reshape(-1, 3).unsqueeze(1),
                    body_pose=pose[i, 3:66].reshape(-1, 21 * 3),
                    left_hand_pose=pose[i, 66:111].reshape(-1, 15 * 3),
                    right_hand_pose=pose[i, 111:156].reshape(-1, 15 * 3),
                    jaw_pose=pose[i, 156:159].reshape(-1, 3),
                    leye_pose=torch.zeros_like(pose[i, 156:159]),
                    reye_pose=torch.zeros_like(pose[i, 156:159]),
                    expression=expr[i].reshape(-1, 10),                
                )
                gt_verts.append(gt_out['vertices'][0].detach().cpu().numpy())
                gt_kp3d.append(gt_out['joints'][0].detach().cpu().numpy())
            
            tgt_verts.append(gt_verts)
            tgt_kp3d.append(gt_kp3d)
                    
            # bbox
            gt_bbox = torch.cat(
                [bbox_center - bbox_size / 2, bbox_size ], dim=-1)
            gt_bbox = gt_bbox * scale
            # xywh2xyxy
            gt_bbox[..., 2] = gt_bbox[..., 0] + gt_bbox[..., 2]
            gt_bbox[..., 3] = gt_bbox[..., 1] + gt_bbox[..., 3]
            tgt_bbox.append(gt_bbox[..., :4].float())
            
            pred_bbox = boxe.clone()
            # box_iou = box_ops.box_iou(pred_bbox,gt_bbox)[0]
            cost_giou = -box_ops.generalized_box_iou(pred_bbox, gt_bbox)
            
            
            cost_bbox = torch.cdist(
                box_ops.box_xyxy_to_cxcywh(pred_bbox)/scale, 
                box_ops.box_xyxy_to_cxcywh(gt_bbox)/scale, p=1)
            
            # smpl kp2d
            gt_kp2d_conf = gt_kp2d[:,:,2:3]
            gt_kp2d_ = (gt_kp2d[:, :, :2] * scale[:2]) /torch.tensor([12, 16]).to(device)
             
            gt_kp2d_body, _ = convert_kps(gt_kp2d_,'smplx_137', 'coco', approximate=True)
            pred_kp2d_body, _ = convert_kps(pred_kp2d,'smplx_137', 'coco', approximate=True)
            cost_keypoints = torch.abs(
                (pred_kp2d_body[:, None]/scale[:2] - gt_kp2d_body[None]/scale[:2])
                                       ).sum([-2,-1])
          
                    
            # smpl kp3d
            gt_kp3d_ = torch.tensor(np.array(gt_kp3d) - np.array(gt_kp3d)[:, [0]]).to(device)
            pred_kp3d_ = (pred_kp3d - pred_kp3d[:, [0]])
            cost_kp3d = torch.abs((pred_kp3d_[:, None] - gt_kp3d_[None])).sum([-2,-1])
            
            # 1. kps
            indice = linear_sum_assignment(cost_keypoints.cpu())

            pred_ind, gt_ind = indice
            pred_indice_list.append(pred_ind)
            gt_indice_list.append(gt_ind)
            
        pred_scores = torch.cat(
            [t[i] for t, i in zip(scores, pred_indice_list)]
            ).detach().cpu().numpy()
        pred_labels = torch.cat(
            [t[i] for t, i in zip(labels, pred_indice_list)]
            ).detach().cpu().numpy()
        pred_boxes = torch.cat(
            [t[i] for t, i in zip(boxes, pred_indice_list)]
            ).detach().cpu().numpy()

        pred_smpl_kp2d = []
        pred_smpl_kp3d = []
        pred_smpl_cam = []
        img_wh_list = []
        for i, img_wh in enumerate(tgt_img_shape):
            
            kp3d = smpl_kp3d[i][pred_indice_list[i]]
            cam = smpl_cam[i][pred_indice_list[i]]
            img_wh = img_wh.flip(-1)[None]
            
            kp2d = project_points_new(
                points_3d=kp3d,
                pred_cam=cam,
                focal_length=5000,
                camera_center=img_wh/2
            )     
            num_instance = kp2d.shape[0]
            img_wh_list.append(img_wh.repeat(num_instance,1).cpu().numpy())
            pred_smpl_kp2d.append(kp2d.detach().cpu().numpy())
            pred_smpl_kp3d.append(kp3d.detach().cpu().numpy())
            pred_smpl_cam.append(cam.detach().cpu().numpy())
        
        pred_smpl_pose = torch.cat(
            [t[i] for t, i in zip(smpl_pose, pred_indice_list)]
            ).detach().cpu().numpy()
        pred_smpl_beta = torch.cat(
            [t[i] for t, i in zip(smpl_beta, pred_indice_list)]
            ).detach().cpu().numpy()
        pred_smpl_expr = torch.cat(
            [t[i] for t, i in zip(smpl_expr, pred_indice_list)]
            ).detach().cpu().numpy()
        pred_smpl_verts = torch.cat(
            [t[i] for t, i in zip(smpl_verts, pred_indice_list)]
            ).detach().cpu().numpy()
        
        pred_smpl_kp2d = np.concatenate(pred_smpl_kp2d, 0)
        pred_smpl_kp3d = np.concatenate(pred_smpl_kp3d, 0)
        pred_smpl_cam = np.concatenate(pred_smpl_cam, 0)       
        img_wh_list = np.concatenate(img_wh_list, 0)   

        gt_smpl_kp3d = torch.cat(tgt_smpl_kp3d).detach().cpu().numpy()
        gt_smpl_pose = torch.cat(tgt_smpl_pose).detach().cpu().numpy()
        gt_smpl_beta = torch.cat(tgt_smpl_beta).detach().cpu().numpy()
        gt_boxes = torch.cat(tgt_bbox).detach().cpu().numpy()
        gt_smpl_expr = torch.cat(tgt_smpl_expr).detach().cpu().numpy()
        # gt_img_shape = torch.cat(tgt_img_shape).detach().cpu().numpy()
        gt_smpl_verts = np.concatenate(
            [np.array(t)[i] for t, i in zip(tgt_verts, gt_indice_list)], 0)
        gt_ann_idx = torch.cat([t.repeat(len(i)) for t, i in zip(tgt_ann_idx, gt_indice_list)],dim=0).cpu().numpy()
        
        gt_keypoints = torch.cat(tgt_keypoints).detach().cpu().numpy()
        # gt_img_shape = tgt_img_shape.detach().cpu().numpy()
        gt_bb2img_trans = torch.stack(tgt_bb2img_trans).detach().cpu().numpy()

        if 'image_id' in targets[i]:
            image_idx=(targets[i]['image_id'].detach().cpu().numpy())

        smplx_root_pose = pred_smpl_pose[:,:3]
        smplx_body_pose = pred_smpl_pose[:,3:66]
        smplx_lhand_pose = pred_smpl_pose[:,66:111]
        smplx_rhand_pose = pred_smpl_pose[:,111:156]
        smplx_jaw_pose = pred_smpl_pose[:,156:]

        results.append({
                    'scores': pred_scores, 
                    'labels': pred_labels, 
                    'boxes': pred_boxes, 
                    # 'keypoints': pred_keypoints,
                    'smplx_root_pose': smplx_root_pose, 
                    'smplx_body_pose': smplx_body_pose,
                    'smplx_lhand_pose': smplx_lhand_pose,
                    'smplx_rhand_pose': smplx_rhand_pose,
                    'smplx_jaw_pose': smplx_jaw_pose,
                    'smplx_shape': pred_smpl_beta, 
                    'smplx_expr': pred_smpl_expr, 
                    'cam_trans': pred_smpl_cam, 
                    'smplx_mesh_cam': pred_smpl_verts,
                    'smplx_mesh_cam_target': gt_smpl_verts,
                    'gt_smpl_kp3d':gt_smpl_kp3d,
                    'smplx_joint_proj': pred_smpl_kp2d,
                    # 'image_idx': image_idx,
                    "img": data_batch_nc['img'].cpu().numpy(),
                    'bb2img_trans': gt_bb2img_trans,
                    'img_shape': img_wh_list,
                    'gt_ann_idx': gt_ann_idx
                })

        if self.nms_iou_threshold > 0:
            raise NotImplementedError
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]
            # import pdb; pdb.set_trace()
            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]
        else:
            results = results

        return results


class PostProcess_SMPLX_Multi_Infer_Box(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(
        self, 
        num_select=100, 
        nms_iou_threshold=-1,
        num_body_points=17,
        body_model= dict(
            type='smplx',
            keypoint_src='smplx',
            num_expression_coeffs=10,
            num_betas=10,
            gender='neutral',
            keypoint_dst='smplx_137',
            model_path='data/body_models/smplx',
            use_pca=False,
            use_face_contour=True)
        ) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold
        self.num_body_points=num_body_points
        
        # -1 for neutral; 0 for male; 1 for femal
        gender_body_model = {}
        gender_body_model[-1] = build_body_model(body_model)
        
        body_model['gender']='male'
        gender_body_model[0] = build_body_model(body_model)
        
        body_model['gender']='female'
        gender_body_model[1] = build_body_model(body_model)
        
        self.body_model = gender_body_model
        
    @torch.no_grad()
    def forward(self, outputs, target_sizes, targets, data_batch_nc, image_shape= None, not_to_xyxy=False, test=False):
        """
        image_shape(target_sizes): input image shape
        
        """

        batch_size = outputs['pred_smpl_beta'].shape[0]
        results = []
        device = outputs['pred_smpl_beta'].device

        num_select = self.num_select
        out_logits, out_bbox= outputs['pred_logits'], outputs['pred_boxes']
        
        out_body_bbox, out_lhand_bbox, out_rhand_bbox, out_face_bbox = \
            outputs['pred_boxes'], outputs['pred_lhand_boxes'], \
                outputs['pred_rhand_boxes'], outputs['pred_face_boxes']
            
        out_smpl_pose, out_smpl_beta, out_smpl_expr, out_smpl_cam, out_smpl_kp3d, out_smpl_verts = \
            outputs['pred_smpl_fullpose'], outputs['pred_smpl_beta'], outputs['pred_smpl_expr'], \
            outputs['pred_smpl_cam'], outputs['pred_smpl_kp3d'], outputs['pred_smpl_verts']

        out_smpl_kp2d = []
        for bs in range(batch_size):
            out_kp3d_i = out_smpl_kp3d[bs]
            out_cam_i = out_smpl_cam[bs]
            out_img_shape = data_batch_nc['img_shape'][bs].flip(-1)[None]

            out_kp2d_i = project_points_new(
                points_3d=out_kp3d_i,
                pred_cam=out_cam_i,
                focal_length=5000,
                camera_center=out_img_shape/2
            )   
            out_smpl_kp2d.append(out_kp2d_i.detach().cpu().numpy())
        out_smpl_kp2d = torch.tensor(out_smpl_kp2d).to(device)

            
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = \
            torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        
        # bbox
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            out_body_bbox = box_ops.box_cxcywh_to_xyxy(out_body_bbox)
            out_lhand_bbox = box_ops.box_cxcywh_to_xyxy(out_lhand_bbox)
            out_rhand_bbox = box_ops.box_cxcywh_to_xyxy(out_rhand_bbox)
            out_face_bbox = box_ops.box_cxcywh_to_xyxy(out_face_bbox)
            
        # gather body bbox
        target_sizes = target_sizes.type_as(boxes)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        
        boxes_norm = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        boxes = boxes_norm * scale_fct[:, None, :]
        
        body_bbox_norm = torch.gather(out_body_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        body_boxes = body_bbox_norm * scale_fct[:, None, :]
        
        lhand_bbox_norm = torch.gather(out_lhand_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        lhand_boxes = lhand_bbox_norm * scale_fct[:, None, :]
        
        rhand_bbox_norm = torch.gather(out_rhand_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        rhand_boxes = rhand_bbox_norm * scale_fct[:, None, :]
        
        face_bbox_norm = torch.gather(out_face_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        face_boxes = face_bbox_norm * scale_fct[:, None, :]
        
        # from relative [0, 1] to absolute [0, height] coordinates

        # smplx kp2d
        topk_smpl_kp2d = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        pred_smpl_kp2d = torch.gather(
            out_smpl_kp2d, 1, 
            topk_smpl_kp2d.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 144, 2))        

        # smpl param
        topk_smpl = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        smpl_pose = torch.gather(out_smpl_pose, 1, topk_smpl[:,:,None].repeat(1, 1, 159))        
        smpl_beta = torch.gather(out_smpl_beta, 1, topk_smpl[:,:,None].repeat(1, 1, 10))   
        smpl_expr = torch.gather(out_smpl_expr, 1, topk_smpl[:,:,None].repeat(1, 1, 10))   
        smpl_cam = torch.gather(out_smpl_cam, 1, topk_smpl[:,:,None].repeat(1, 1, 3))   
        smpl_kp3d = torch.gather(out_smpl_kp3d, 1, topk_smpl[:,:,None, None].repeat(1, 1, out_smpl_kp3d.shape[-2],3))
        smpl_verts = torch.gather(out_smpl_verts, 1, topk_smpl[:,:,None, None].repeat(1, 1, out_smpl_verts.shape[-2],3))
        # smpl_verts = smpl_verts - smpl_kp3d[:,:, [0]]
        (s, tx, ty) = (smpl_cam[..., 0] + 1e-9), smpl_cam[..., 1], smpl_cam[..., 2]
        depth, dx, dy = 1./s, tx/s, ty/s
        transl = torch.stack([dx, dy, depth], -1) 
        
        smplx_root_pose = smpl_pose[:, :, :3]
        smplx_body_pose = smpl_pose[:, :, 3:66]
        smplx_lhand_pose = smpl_pose[:, :, 66:111]
        smplx_rhand_pose = smpl_pose[:, :, 111:156]
        smplx_jaw_pose = smpl_pose[:, :, 156:]
        
        if 'ann_idx' in data_batch_nc:
            image_idx=[target.cpu().numpy()[0] for target in data_batch_nc['ann_idx']]

        for bs in range(batch_size):
            results.append({
                        'scores': scores[bs], 
                        'labels': labels[bs], 
                        'smpl_kp3d': smpl_kp3d[bs],
                        'smplx_root_pose': smplx_root_pose[bs], 
                        'smplx_body_pose': smplx_body_pose[bs],
                        'smplx_lhand_pose': smplx_lhand_pose[bs],
                        'smplx_rhand_pose': smplx_rhand_pose[bs],
                        'smplx_jaw_pose': smplx_jaw_pose[bs],
                        'smplx_shape': smpl_beta[bs], 
                        'smplx_expr': smpl_expr[bs], 
                        'smplx_joint_proj': pred_smpl_kp2d[bs],
                        'smpl_verts': smpl_verts[bs],
                        'image_idx': image_idx[bs],
                        'cam_trans': transl[bs],
                        'body_bbox': body_boxes[bs],
                        'lhand_bbox': lhand_boxes[bs],
                        'rhand_bbox': rhand_boxes[bs],
                        'face_bbox': face_boxes[bs],
                        'bb2img_trans': data_batch_nc['bb2img_trans'][bs],
                        'img2bb_trans': data_batch_nc['img2bb_trans'][bs],
                        'img': data_batch_nc['img'][bs],
                        'img_shape': data_batch_nc['img_shape'][bs]
                    })

        if self.nms_iou_threshold > 0:
            raise NotImplementedError
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]
            # import pdb; pdb.set_trace()
            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]
        else:
            results = results

        return results
