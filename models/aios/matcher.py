import torch, os
from scipy.optimize import linear_sum_assignment
from torch import nn
from .utils import OKSLoss
import numpy as np
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 focal_alpha=0.25,
                 cost_keypoints=1.0,
                 cost_kpvis=0.1,
                 cost_oks=0.01,
                 num_body_points=17):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'
        self.cost_keypoints = cost_keypoints
        self.cost_kpvis = cost_kpvis
        self.cost_oks = cost_oks
        self.focal_alpha = focal_alpha
        self.num_body_points = num_body_points
        if num_body_points == 17:
            self.sigmas = np.array([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
                1.07, .87, .87, .89, .89
            ],
                                   dtype=np.float32) / 10.0

        elif num_body_points == 14:
            self.sigmas = np.array([
                .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89,
                .79, .79
            ]) / 10.0
        else:
            raise ValueError(f'Unsupported keypoints number {num_body_points}')

    @torch.no_grad()
    def forward(self, outputs, targets, data_batch=None):
        bs, num_queries = outputs['pred_logits'].shape[:2]
        out_prob = outputs['pred_logits'].flatten(0, 1).sigmoid()
        out_bbox = outputs['pred_boxes'].flatten(0, 1)
        
        out_keypoints = outputs['pred_keypoints'].flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets])
        tgt_keypoints = torch.cat([v['keypoints'] for v in targets])
        tgt_area = torch.cat([v['area'] for v in targets])
        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**
                                        gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * (
            (1 - out_prob)**gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                         box_cxcywh_to_xyxy(tgt_bbox), data_batch)

        # compute the keypoint costs
        Z_pred = out_keypoints[:, 0:(self.num_body_points * 2)]
        V_pred = out_keypoints[:, (self.num_body_points * 2):]
        Z_gt = tgt_keypoints[:, 0:(self.num_body_points * 2)]
        V_gt: torch.Tensor = tgt_keypoints[:, (self.num_body_points * 2):]
        if Z_pred.sum() > 0:
            sigmas = Z_pred.new_tensor(self.sigmas)
            variances = (sigmas * 2)**2
            kpt_preds = Z_pred.reshape(-1, Z_pred.size(-1) // 2, 2)
            kpt_gts = Z_gt.reshape(-1, Z_gt.size(-1) // 2, 2)
            squared_distance = (kpt_preds[:, None, :, 0] - kpt_gts[None, :, :, 0]) ** 2 + \
                               (kpt_preds[:, None, :, 1] - kpt_gts[None, :, :, 1]) ** 2
            squared_distance0 = squared_distance / (tgt_area[:, None] *
                                                    variances[None, :] * 2)
            squared_distance1 = torch.exp(-squared_distance0)
            squared_distance1 = squared_distance1 * V_gt
            oks = squared_distance1.sum(dim=-1) / (V_gt.sum(dim=-1) + 1e-6)
            oks = oks.clamp(min=1e-6)
            cost_oks = 1 - oks
            # import pdb; pdb.set_trace()
            cost_keypoints = torch.abs(Z_pred[:, None, :] - Z_gt[None])
            cost_keypoints = cost_keypoints * V_gt.repeat_interleave(
                2, dim=1)[None]
            cost_keypoints = cost_keypoints.sum(-1)
            cost_bbox = torch.zeros_like(cost_keypoints)
            cost_giou = torch.zeros_like(
                cost_keypoints)  # [bs*query, instance_num]
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + self.cost_keypoints * cost_keypoints + self.cost_oks * cost_oks
            C = C.view(bs, num_queries, -1).cpu()

        else:
            cost_oks = torch.zeros_like(cost_bbox)
            cost_keypoints = torch.zeros_like(cost_bbox)
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + self.cost_keypoints * cost_keypoints + self.cost_oks * cost_oks
            C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v['boxes']) for v in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]
        if tgt_ids.shape[0] > 0:
            cost_mean_dict = {
                'class': cost_class.mean(),
                'bbox': cost_bbox.mean(),
                'giou': cost_giou.mean(),
                'keypoints': cost_keypoints.mean()
            }
        else:
            cost_mean_dict = {
                'class': torch.zeros_like(cost_class.mean()),
                'bbox': torch.zeros_like(cost_bbox.mean()),
                'giou': torch.zeros_like(cost_giou.mean()),
                'keypoints': torch.zeros_like(cost_keypoints.mean()),
            }

        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices], cost_mean_dict


def build_matcher(args):
    if args.matcher_type == 'HungarianMatcher':
        return HungarianMatcher(cost_class=args.set_cost_class,
                                cost_bbox=args.set_cost_bbox,
                                cost_giou=args.set_cost_giou,
                                focal_alpha=args.focal_alpha,
                                cost_keypoints=args.set_cost_keypoints,
                                cost_kpvis=args.set_cost_kpvis,
                                cost_oks=args.set_cost_oks,
                                num_body_points=args.num_body_points)
    elif args.matcher_type == 'HungarianMatcherBox':
        return HungarianMatcherBox(cost_class=args.set_cost_class,
                                cost_bbox=args.set_cost_bbox,
                                cost_giou=args.set_cost_giou,
                                focal_alpha=args.focal_alpha)
    else:
        raise NotImplementedError('Unknown args.matcher_type: {}'.format(
            args.matcher_type))




class HungarianMatcherBox(nn.Module):
    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 focal_alpha=0.25,
                 cost_keypoints=1.0,
                 cost_kpvis=0.1,
                 cost_oks=0.01,
                 num_body_points=17):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'
        self.cost_keypoints = cost_keypoints
        self.cost_kpvis = cost_kpvis
        self.cost_oks = cost_oks
        self.focal_alpha = focal_alpha
        self.num_body_points = num_body_points
        if num_body_points == 17:
            self.sigmas = np.array([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
                1.07, .87, .87, .89, .89
            ],
                                   dtype=np.float32) / 10.0

        elif num_body_points == 14:
            self.sigmas = np.array([
                .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89,
                .79, .79
            ]) / 10.0
        else:
            raise ValueError(f'Unsupported keypoints number {num_body_points}')

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_logits'].shape[:2]
        out_prob = outputs['pred_logits'].flatten(0, 1).sigmoid()
        out_bbox = outputs['pred_boxes'].flatten(0, 1)


        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets])

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**
                                        gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * (
            (1 - out_prob)**gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                         box_cxcywh_to_xyxy(tgt_bbox))


        cost_oks = torch.zeros_like(cost_bbox)
        cost_keypoints = torch.zeros_like(cost_bbox)
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v['boxes']) for v in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]

        if tgt_ids.shape[0] > 0:
            cost_mean_dict = {
                'class': cost_class.mean(),
                'bbox': cost_bbox.mean(),
                'giou': cost_giou.mean(),
            }
        else:
            cost_mean_dict = {
                'class': torch.zeros_like(cost_class.mean()),
                'bbox': torch.zeros_like(cost_bbox.mean()),
                'giou': torch.zeros_like(cost_giou.mean()),
            }

        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices], cost_mean_dict
