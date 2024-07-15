import copy
import pdb
import os
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from util import box_ops
from util.keypoint_ops import keypoint_xyzxyz_to_xyxyzz
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid)
from .backbones import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .utils import PoseProjector, sigmoid_focal_loss, MLP
from .postprocesses import PostProcess_SMPLX, PostProcess_aios
from .postprocesses import PostProcess_SMPLX_Multi as PostProcess_SMPLX
from .postprocesses import PostProcess_SMPLX_Multi_Box
from .postprocesses import  PostProcess_SMPLX_Multi_Infer, PostProcess_SMPLX_Multi_Infer_Box
from .criterion_smplx import SetCriterion, SetCriterion_Box
from ..registry import MODULE_BUILD_FUNCS
from detrsmpl.core.conventions.keypoints_mapping import convert_kps
from detrsmpl.models.body_models.builder import build_body_model
from util.human_models import smpl_x
from detrsmpl.core.conventions.keypoints_mapping import get_keypoint_idxs_by_part
import numpy as np

from detrsmpl.utils.geometry import (rot6d_to_rotmat)
from detrsmpl.utils.transforms import rotmat_to_aa
import cv2
from config.config import cfg


class AiOSSMPLX(nn.Module):
    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        aux_loss=False,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=False,
        fix_refpoints_hw=-1,
        num_feature_levels=1,
        nheads=8,
        two_stage_type='no',
        dec_pred_class_embed_share=False,
        dec_pred_bbox_embed_share=False,
        dec_pred_pose_embed_share=False,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_batch_gt_fuse=False,
        dn_labelbook_size=100,
        dn_attn_mask_type_list=['group2group'],
        cls_no_bias=False,
        num_group=100,
        num_body_points=17,
        num_hand_points=10,
        num_face_points=10,
        num_box_decoder_layers=2,
        num_hand_face_decoder_layers=4,
        body_model=dict(
            type='smplx',
            keypoint_src='smplx',
            num_expression_coeffs=10,
            keypoint_dst='smplx_137',
            model_path='data/body_models/smplx',
            use_pca=False,
            use_face_contour=True),
        train=True,
        inference=False,
        focal_length=[5000., 5000.],
        camera_3d_size=2.5
    ):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)
        self.num_body_points = num_body_points
        self.num_hand_points = num_hand_points
        self.num_face_points = num_face_points
        self.num_whole_body_points = num_body_points + 2*num_hand_points + num_face_points
        self.num_box_decoder_layers = num_box_decoder_layers
        self.num_hand_face_decoder_layers = num_hand_face_decoder_layers
        self.focal_length = focal_length
        self.camera_3d_size=camera_3d_size
        self.inference = inference
        if train:
            self.smpl_convention = 'smplx'
        else:
            self.smpl_convention = 'h36m'
        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy  # False
        self.fix_refpoints_hw = fix_refpoints_hw  # -1

        # for dn training
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_batch_gt_fuse = dn_batch_gt_fuse
        self.dn_labelbook_size = dn_labelbook_size
        self.dn_attn_mask_type_list = dn_attn_mask_type_list
        assert all([
            i in ['match2dn', 'dn2dn', 'group2group']
            for i in dn_attn_mask_type_list
        ])
        assert not dn_batch_gt_fuse

        if inference:
            body_model=dict(
                type='smplx',
                keypoint_src='smplx',
                num_expression_coeffs=10,
                num_betas=10,
                keypoint_dst='smplx',
                model_path='data/body_models/smplx',
                use_pca=False,
                use_face_contour=True)
        self.body_model = build_body_model(body_model)
        for param in self.body_model.parameters():
            param.requires_grad = False       
        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)  # 3
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels,
                                  hidden_dim,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == 'no', 'two_stage_type should be no if num_feature_levels=1 !!!'
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1],
                              hidden_dim,
                              kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            ])

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, 'Why not iter_update?'

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share  # false
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share  # false

        # 1.1 prepare class & box embed
        _class_embed = nn.Linear(hidden_dim,
                                 num_classes,
                                 bias=(not cls_no_bias))
        if not cls_no_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _class_embed.bias.data = torch.ones(self.num_classes) * bias_value

        # 1.2 box embed layer list
        if dec_pred_class_embed_share:
            class_embed_layerlist = [
                _class_embed for i in range(transformer.num_decoder_layers)
            ]
        else:
            class_embed_layerlist = [
                copy.deepcopy(_class_embed)
                for i in range(transformer.num_decoder_layers)
            ]

        ###########################################################################
        #                    body bbox + l/r hand box + face box
        ###########################################################################
        # 1.1 body bbox embed
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        
        # 1.2 body bbox embed layer list
        self.num_group = num_group
        if dec_pred_bbox_embed_share:
            box_body_embed_layerlist = [
                _bbox_embed for i in range(transformer.num_decoder_layers)
            ]
        else:
            box_body_embed_layerlist = [
                copy.deepcopy(_bbox_embed)
                for i in range(transformer.num_decoder_layers)
            ]

        # 2.1 lhand bbox embed
        _bbox_hand_embed = MLP(hidden_dim, hidden_dim, 2, 3) # TODO: the out shape should be 2 not 4
        nn.init.constant_(_bbox_hand_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_hand_embed.layers[-1].bias.data, 0)

        _bbox_hand_hw_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_bbox_hand_hw_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_hand_hw_embed.layers[-1].bias.data, 0)
        # 2.2 lhand bbox embed layer list
        if dec_pred_pose_embed_share:
            box_hand_embed_layerlist = \
                [_bbox_hand_embed for i in range(transformer.num_decoder_layers - num_box_decoder_layers+1)]
        else:
            box_hand_embed_layerlist = [
                copy.deepcopy(_bbox_hand_embed)
                for i in range(transformer.num_decoder_layers -
                            num_box_decoder_layers + 1)
            ]

        if dec_pred_pose_embed_share:
            box_hand_hw_embed_layerlist = [
                _bbox_hand_hw_embed for i in range(
                    transformer.num_decoder_layers - num_box_decoder_layers)
                ]
        else:
            box_hand_hw_embed_layerlist = [
                copy.deepcopy(_bbox_hand_hw_embed)
                for i in range(transformer.num_decoder_layers -
                            num_box_decoder_layers)
            ]
                        
        # 4.1 face bbox embed
        _bbox_face_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_bbox_face_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_face_embed.layers[-1].bias.data, 0)

        _bbox_face_hw_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_bbox_face_hw_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_face_hw_embed.layers[-1].bias.data, 0)
        
        # 4.2 face bbox embed layer list
        if dec_pred_pose_embed_share:
            box_face_embed_layerlist = [
                _bbox_face_embed for i in range(
                    transformer.num_decoder_layers - num_box_decoder_layers + 1)
                ]
        else:
            box_face_embed_layerlist = [
                copy.deepcopy(_bbox_face_embed)
                for i in range(transformer.num_decoder_layers -
                            num_box_decoder_layers + 1)
            ]

        if dec_pred_pose_embed_share:
            box_face_hw_embed_layerlist = [
                _bbox_face_hw_embed for i in range(
                    transformer.num_decoder_layers - num_box_decoder_layers)]
        else:
            box_face_hw_embed_layerlist = [
                copy.deepcopy(_bbox_face_hw_embed)
                for i in range(transformer.num_decoder_layers -
                            num_box_decoder_layers)
            ]            
        ###########################################################################
        #                    body kp2d + l/r hand kp2d + face kp2d
        ###########################################################################
            
        ######## body #######
        # 1.1 body kp2d embed
        _pose_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_pose_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_pose_embed.layers[-1].bias.data, 0)

        # 1.2 body kp2d embed layer list
        if num_body_points == 17:
            if dec_pred_pose_embed_share:
                pose_embed_layerlist = \
                    [_pose_embed for i in range(transformer.num_decoder_layers - num_box_decoder_layers+1)]
            else:
                pose_embed_layerlist = [
                    copy.deepcopy(_pose_embed)
                    for i in range(transformer.num_decoder_layers -
                                num_box_decoder_layers + 1)
                ]
        else:
            if dec_pred_pose_embed_share:
                pose_embed_layerlist = [
                    _pose_embed for i in range(transformer.num_decoder_layers -
                                            num_box_decoder_layers)
                ]
            else:
                pose_embed_layerlist = [
                    copy.deepcopy(_pose_embed)
                    for i in range(transformer.num_decoder_layers -
                                num_box_decoder_layers)
                ]

        # 1.3 body kp bbox embed 
        _pose_hw_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        
        # 1.4 body kp bbox embed layer list
        pose_hw_embed_layerlist = [
            _pose_hw_embed for i in range(transformer.num_decoder_layers -
                                        num_box_decoder_layers)
        ]
            
        ######## lhand #######
        # 2.1 lhand kp2d embed
        _pose_hand_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_pose_hand_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_pose_hand_embed.layers[-1].bias.data, 0)

        # 2.2 lhand kp2d embed layer list
        if dec_pred_pose_embed_share:
            pose_hand_embed_layerlist = \
                [_pose_hand_embed for i in range(transformer.num_decoder_layers - num_hand_face_decoder_layers+1)]
        else:
            pose_hand_embed_layerlist = [
                copy.deepcopy(_pose_hand_embed)
                for i in range(transformer.num_decoder_layers -
                            num_hand_face_decoder_layers + 1)
            ]

        # 2.3 lhand kp bbox embed 
        _pose_hand_hw_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        
        # 2.4 lhand kp bbox embed layer list
        pose_hand_hw_embed_layerlist = [
            _pose_hand_hw_embed for i in range(transformer.num_decoder_layers -
                                        num_hand_face_decoder_layers)
        ]
            

        ######## face #######
        # 4.1 face kp2d embed
        _pose_face_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_pose_face_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_pose_face_embed.layers[-1].bias.data, 0)

        # 4.2 face kp2d embed layer list
        if dec_pred_pose_embed_share:
            pose_face_embed_layerlist = \
                [_pose_face_embed for i in range(transformer.num_decoder_layers - num_hand_face_decoder_layers+1)]
        else:
            pose_face_embed_layerlist = [
                copy.deepcopy(_pose_face_embed)
                for i in range(transformer.num_decoder_layers -
                            num_hand_face_decoder_layers + 1)
            ]

        # 4.3 face kp bbox embed 
        _pose_face_hw_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        
        # 4.4 face kp bbox embed layer list
        pose_face_hw_embed_layerlist = [
            _pose_face_hw_embed for i in range(transformer.num_decoder_layers -
                                        num_hand_face_decoder_layers)
        ]

        ###########################################################################
        #                    smpl pose + betas + kp2d + kp3d + cam
        ###########################################################################
        
        # 1. smpl pose embed
        if body_model['type'].upper()=='SMPL':
            self.body_model_joint_num = 24
        elif body_model['type'].upper()=='SMPLX':
            self.body_model_joint_num = 22
        else:
            raise ValueError(
            f'Only supports SMPL or SMPLX, but get {body_model.type}')      
        #TODO: 

        _smpl_pose_embed = MLP(hidden_dim * (self.num_body_points + 4),
                            hidden_dim, self.body_model_joint_num * 6, 3)
        nn.init.constant_(_smpl_pose_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_smpl_pose_embed.layers[-1].bias.data, 0)  

        if dec_pred_bbox_embed_share:
            smpl_pose_embed_layerlist = [
                _smpl_pose_embed
                for i in range(transformer.num_decoder_layers -
                               num_box_decoder_layers)
            ]
        else:
            smpl_pose_embed_layerlist = [
                copy.deepcopy(_smpl_pose_embed)
                for i in range(transformer.num_decoder_layers -
                               num_box_decoder_layers)
            ]

        # 2. smpl betas embed
        _smpl_beta_embed = MLP(hidden_dim * (self.num_body_points + 4),
                               hidden_dim, 10, 3)
        nn.init.constant_(_smpl_beta_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_smpl_beta_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            smpl_beta_embed_layerlist = [
                _smpl_beta_embed
                for i in range(transformer.num_decoder_layers -
                               num_box_decoder_layers)
            ]
        else:
            smpl_beta_embed_layerlist = [
                copy.deepcopy(_smpl_beta_embed)
                for i in range(transformer.num_decoder_layers -
                               num_box_decoder_layers)
            ]

        # 3. smpl cam embed
        _cam_embed = MLP(hidden_dim * (self.num_body_points + 4), hidden_dim,
                         3, 3)
        nn.init.constant_(_cam_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_cam_embed.layers[-1].bias.data, 0)
        
        if dec_pred_bbox_embed_share:
            cam_embed_layerlist = [
                _cam_embed for i in range(transformer.num_decoder_layers -
                                          num_box_decoder_layers)
            ]
        else:
            cam_embed_layerlist = [
                copy.deepcopy(_cam_embed)
                for i in range(transformer.num_decoder_layers -
                               num_box_decoder_layers)
            ]

        # 2. smplx hand pose embed
        _smplx_hand_pose_embed_layer_2_3 = \
            MLP(hidden_dim, hidden_dim, 15 * 6, 3)
        nn.init.constant_(_smplx_hand_pose_embed_layer_2_3.layers[-1].weight.data, 0)
        nn.init.constant_(_smplx_hand_pose_embed_layer_2_3.layers[-1].bias.data, 0)
        
        _smplx_hand_pose_embed_layer_4_5 = \
            MLP(hidden_dim * (self.num_hand_points + 3), hidden_dim, 15 * 6, 3)
        nn.init.constant_(_smplx_hand_pose_embed_layer_4_5.layers[-1].weight.data, 0)
        nn.init.constant_(_smplx_hand_pose_embed_layer_4_5.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            smplx_hand_pose_embed_layerlist = [
                _smplx_hand_pose_embed_layer_2_3
                if i<2 else _smplx_hand_pose_embed_layer_4_5
                for i in range(transformer.num_decoder_layers -
                               num_box_decoder_layers)
            ]
        else:
            smplx_hand_pose_embed_layerlist = [
                copy.deepcopy(_smplx_hand_pose_embed_layer_2_3)
                if i<2 else copy.deepcopy(_smplx_hand_pose_embed_layer_4_5)
                for i in range(transformer.num_decoder_layers -
                               num_box_decoder_layers)
            ]

        # 3. smplx face expression 

        _smplx_expression_embed_layer_2_3 = \
            MLP(hidden_dim, hidden_dim, 10, 3)
        nn.init.constant_(_smplx_expression_embed_layer_2_3.layers[-1].weight.data, 0)
        nn.init.constant_(_smplx_expression_embed_layer_2_3.layers[-1].bias.data, 0)
        
        _smplx_expression_embed_layer_4_5 = \
            MLP(hidden_dim * (self.num_hand_points + 2), hidden_dim, 10, 3)
        nn.init.constant_(_smplx_expression_embed_layer_4_5.layers[-1].weight.data, 0)
        nn.init.constant_(_smplx_expression_embed_layer_4_5.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            smplx_expression_embed_layerlist = [
                _smplx_expression_embed_layer_2_3
                if i<2 else _smplx_expression_embed_layer_4_5
                for i in range(transformer.num_decoder_layers -
                               num_box_decoder_layers)
            ]
        else:
            smplx_expression_embed_layerlist = [
                copy.deepcopy(_smplx_expression_embed_layer_2_3)
                if i<2 else copy.deepcopy(_smplx_expression_embed_layer_4_5)
                for i in range(transformer.num_decoder_layers -
                               num_box_decoder_layers)
            ]
        

        # 4. smplx jaw pose embed
        _smplx_jaw_embed_2_3 = MLP(hidden_dim * 1,
                               hidden_dim, 6, 3)
        nn.init.constant_(_smplx_jaw_embed_2_3.layers[-1].weight.data, 0)
        nn.init.constant_(_smplx_jaw_embed_2_3.layers[-1].bias.data, 0)
        
        _smplx_jaw_embed_4_5 = MLP(hidden_dim * (self.num_face_points + 2),
                               hidden_dim, 6, 3)
        nn.init.constant_(_smplx_jaw_embed_4_5.layers[-1].weight.data, 0)
        nn.init.constant_(_smplx_jaw_embed_4_5.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            smplx_jaw_embed_layerlist = [
                _smplx_jaw_embed_2_3 if i<2 else _smplx_jaw_embed_4_5
                for i in range(
                    transformer.num_decoder_layers - num_box_decoder_layers)
            ]
        else:
            smplx_jaw_embed_layerlist = [
                copy.deepcopy(_smplx_jaw_embed_2_3) 
                if i<2 else copy.deepcopy(_smplx_jaw_embed_4_5) 
                for i in range(
                    transformer.num_decoder_layers -  num_box_decoder_layers)
            ]

        self.bbox_embed = nn.ModuleList(box_body_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.pose_embed = nn.ModuleList(pose_embed_layerlist)
        self.pose_hw_embed = nn.ModuleList(pose_hw_embed_layerlist)

        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.pose_embed = self.pose_embed
        self.transformer.decoder.pose_hw_embed = self.pose_hw_embed
        self.transformer.decoder.class_embed = self.class_embed
        
        # smpl
        self.smpl_pose_embed = nn.ModuleList(smpl_pose_embed_layerlist)
        self.smpl_beta_embed = nn.ModuleList(smpl_beta_embed_layerlist)
        self.smpl_cam_embed = nn.ModuleList(cam_embed_layerlist)

        # smplx hand kp
        self.bbox_hand_embed = nn.ModuleList(box_hand_embed_layerlist)
        self.bbox_hand_hw_embed = nn.ModuleList(box_hand_hw_embed_layerlist)
        self.pose_hand_embed = nn.ModuleList(pose_hand_embed_layerlist)
        self.pose_hand_hw_embed = nn.ModuleList(pose_hand_hw_embed_layerlist)

        self.transformer.decoder.bbox_hand_embed = self.bbox_hand_embed
        self.transformer.decoder.bbox_hand_hw_embed = self.bbox_hand_hw_embed
        self.transformer.decoder.pose_hand_embed = self.pose_hand_embed
        self.transformer.decoder.pose_hand_hw_embed = self.pose_hand_hw_embed

        # smplx face kp
        self.bbox_face_embed = nn.ModuleList(box_face_embed_layerlist)
        self.bbox_face_hw_embed = nn.ModuleList(box_face_hw_embed_layerlist)
        self.pose_face_embed = nn.ModuleList(pose_face_embed_layerlist)
        self.pose_face_hw_embed = nn.ModuleList(pose_face_hw_embed_layerlist)               
    
        self.transformer.decoder.bbox_face_embed = self.bbox_face_embed
        self.transformer.decoder.bbox_face_hw_embed = self.bbox_face_hw_embed
        self.transformer.decoder.pose_face_embed = self.pose_face_embed
        self.transformer.decoder.pose_face_hw_embed = self.pose_face_hw_embed
            
        # smplx 
        self.smpl_hand_pose_embed = nn.ModuleList(smplx_hand_pose_embed_layerlist)
        self.smpl_expr_embed = nn.ModuleList(smplx_expression_embed_layerlist)
        self.smpl_jaw_embed = nn.ModuleList(smplx_jaw_embed_layerlist)

        #
        self.transformer.decoder.num_hand_face_decoder_layers = num_hand_face_decoder_layers
        self.transformer.decoder.num_box_decoder_layers = num_box_decoder_layers
        self.transformer.decoder.num_body_points = num_body_points
        self.transformer.decoder.num_hand_points = num_hand_points
        self.transformer.decoder.num_face_points = num_face_points
        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in [
            'no', 'standard'
        ], 'unknown param {} of two_stage_type'.format(two_stage_type)
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(
                    _bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed

            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(
                    _class_embed)
            self.refpoint_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def prepare_for_dn2(self, targets):
        if not self.training:
            device = targets[0]['boxes'].device
            bs = len(targets)
            
            num_points = self.num_body_points + 4
            attn_mask2 = torch.zeros(
                bs,
                self.nheads,
                self.num_group * num_points,
                self.num_group * num_points,
                device=device,
                dtype=torch.bool)

            group_bbox_kpt = num_points
            group_nobbox_kpt = self.num_body_points
            kpt_index = [
                x for x in range(self.num_group * num_points) 
                if x % num_points in [
                    0, 
                    self.num_body_points+1, 
                    self.num_body_points+2, 
                    self.num_body_points+3
                    ]
                ]
            for matchj in range(self.num_group * num_points):
                sj = (matchj // group_bbox_kpt) * group_bbox_kpt
                ej = (matchj // group_bbox_kpt + 1)*group_bbox_kpt
                if sj > 0:
                    attn_mask2[:, :, matchj, :sj] = True
                if ej < self.num_group * num_points:
                    attn_mask2[:, :, matchj, ej:] = True

            for match_x in range(self.num_group * num_points):
                if match_x % group_bbox_kpt in [0, 
                                                self.num_body_points+1, 
                                                self.num_body_points+2, 
                                                self.num_body_points+3]:
                    attn_mask2[:,:,match_x,kpt_index]=False


            num_points = self.num_whole_body_points + 4
            attn_mask3 = torch.zeros(
                bs,
                self.nheads,
                self.num_group * (num_points), 
                self.num_group * (num_points),
                device=device, 
                dtype=torch.bool)

            group_bbox_kpt = (num_points)
            # group_nobbox_kpt = self.num_body_points
            kpt_index = [
                x for x in range(self.num_group * (num_points)) if x % (num_points) in 
                [0, 
                 1+self.num_body_points, 
                 2+self.num_body_points+self.num_hand_points, 
                 3+self.num_body_points+self.num_hand_points*2
                 ]
                ]
            for matchj in range(self.num_group * num_points):
                sj = (matchj // group_bbox_kpt) * group_bbox_kpt
                ej = (matchj // group_bbox_kpt + 1)*group_bbox_kpt
                if sj > 0:
                    attn_mask3[:, :, matchj, :sj] = True
                if ej < self.num_group * num_points:
                    attn_mask3[:, :, matchj, ej:] = True

            for match_x in range(self.num_group * num_points):
                if match_x % group_bbox_kpt in [
                    0, 
                    1 + self.num_body_points, 
                    2 + self.num_body_points + self.num_hand_points, 
                    3 + self.num_body_points + self.num_hand_points * 2]:
                    
                    attn_mask3[:, :, match_x, kpt_index] = False
            attn_mask2 = attn_mask2.flatten(0, 1)
            attn_mask3 = attn_mask3.flatten(0, 1)
            return None, None, None, attn_mask2, attn_mask3, None

        # targets, dn_scalar, noise_scale = dn_args
        device = targets[0]['boxes'].device
        bs = len(targets)
        dn_number = self.dn_number  # 100
        dn_box_noise_scale = self.dn_box_noise_scale  # 0.4
        dn_label_noise_ratio = self.dn_label_noise_ratio  # 0.5

        # gather gt boxes and labels
        gt_boxes = [t['boxes'] for t in targets]
        gt_labels = [t['labels'] for t in targets]
        gt_keypoints = [t['keypoints'] for t in targets]

        # repeat them
        def get_indices_for_repeat(now_num, target_num, device='cuda'):
            """
            Input:
                - now_num: int
                - target_num: int
            Output:
                - indices: tensor[target_num]
            """
            out_indice = []
            base_indice = torch.arange(now_num).to(device)
            multiplier = target_num // now_num
            out_indice.append(base_indice.repeat(multiplier))
            residue = target_num % now_num
            out_indice.append(base_indice[torch.randint(0,
                                                        now_num, (residue, ),
                                                        device=device)])
            return torch.cat(out_indice)

        if self.dn_batch_gt_fuse:
            raise NotImplementedError
            gt_boxes_bsall = torch.cat(gt_boxes)  # num_boxes, 4
            gt_labels_bsall = torch.cat(gt_labels)
            num_gt_bsall = gt_boxes_bsall.shape[0]
            if num_gt_bsall > 0:
                indices = get_indices_for_repeat(num_gt_bsall, dn_number,
                                                 device)
                gt_boxes_expand = gt_boxes_bsall[indices][None].repeat(
                    bs, 1, 1)  # bs, num_dn, 4
                gt_labels_expand = gt_labels_bsall[indices][None].repeat(
                    bs, 1)  # bs, num_dn
            else:
                # all negative samples when no gt boxes
                gt_boxes_expand = torch.rand(bs, dn_number, 4, device=device)
                gt_labels_expand = torch.ones(
                    bs, dn_number, dtype=torch.int64, device=device) * int(
                        self.num_classes)
        else:
            gt_boxes_expand = []
            gt_labels_expand = []
            gt_keypoints_expand = []  # here
            for idx, (gt_boxes_i, gt_labels_i, gt_keypoint_i) in enumerate(
                    zip(gt_boxes, gt_labels, gt_keypoints)):  # idx -> batch id
                num_gt_i = gt_boxes_i.shape[0]  # instance num
                if num_gt_i > 0:
                    indices = get_indices_for_repeat(num_gt_i, dn_number,
                                                     device)
                    gt_boxes_expand_i = gt_boxes_i[indices]  # num_dn, 4
                    gt_labels_expand_i = gt_labels_i[indices]  # add smpl
                    gt_keypoints_expand_i = gt_keypoint_i[indices]
                else:
                    # all negative samples when no gt boxes
                    gt_boxes_expand_i = torch.rand(dn_number, 4, device=device)
                    gt_labels_expand_i = torch.ones(
                        dn_number, dtype=torch.int64, device=device) * int(
                            self.num_classes)
                    gt_keypoints_expand_i = torch.rand(dn_number,
                                                       self.num_body_points *
                                                       3,
                                                       device=device)
                gt_boxes_expand.append(gt_boxes_expand_i)  # add smpl
                gt_labels_expand.append(gt_labels_expand_i)
                gt_keypoints_expand.append(gt_keypoints_expand_i)
            gt_boxes_expand = torch.stack(gt_boxes_expand)
            gt_labels_expand = torch.stack(gt_labels_expand)
            gt_keypoints_expand = torch.stack(gt_keypoints_expand)
        knwon_boxes_expand = gt_boxes_expand.clone()
        knwon_labels_expand = gt_labels_expand.clone()

        # add noise
        if dn_label_noise_ratio > 0:
            prob = torch.rand_like(knwon_labels_expand.float())
            chosen_indice = prob < dn_label_noise_ratio
            new_label = torch.randint_like(
                knwon_labels_expand[chosen_indice], 0,
                self.dn_labelbook_size)  # randomly put a new one here
            knwon_labels_expand[chosen_indice] = new_label

        if dn_box_noise_scale > 0:
            diff = torch.zeros_like(knwon_boxes_expand)
            diff[..., :2] = knwon_boxes_expand[..., 2:] / 2
            diff[..., 2:] = knwon_boxes_expand[..., 2:]
            knwon_boxes_expand += torch.mul(
                (torch.rand_like(knwon_boxes_expand) * 2 - 1.0),
                diff) * dn_box_noise_scale
            knwon_boxes_expand = knwon_boxes_expand.clamp(min=0.0, max=1.0)

        input_query_label = self.label_enc(knwon_labels_expand)
        input_query_bbox = inverse_sigmoid(knwon_boxes_expand)

        # prepare mask
        if 'group2group' in self.dn_attn_mask_type_list:
            attn_mask = torch.zeros(bs,
                                    self.nheads,
                                    dn_number + self.num_queries,
                                    dn_number + self.num_queries,
                                    device=device,
                                    dtype=torch.bool)
            attn_mask[:, :, dn_number:, :dn_number] = True
            for idx, (gt_boxes_i,
                      gt_labels_i) in enumerate(zip(gt_boxes,
                                                    gt_labels)):  # for batch
                num_gt_i = gt_boxes_i.shape[0]
                if num_gt_i == 0:
                    continue
                for matchi in range(dn_number):
                    si = (matchi // num_gt_i) * num_gt_i
                    ei = (matchi // num_gt_i + 1) * num_gt_i
                    if si > 0:
                        attn_mask[idx, :, matchi, :si] = True
                    if ei < dn_number:
                        attn_mask[idx, :, matchi, ei:dn_number] = True
            attn_mask = attn_mask.flatten(0, 1)

        if 'group2group' in self.dn_attn_mask_type_list:
            # self.num_body_points = self.num_body_points +3
            num_points = self.num_body_points + 4
            attn_mask2 = torch.zeros(
                bs,
                self.nheads,
                dn_number + self.num_group * num_points,
                dn_number + self.num_group * num_points,
                device=device,
                dtype=torch.bool)
            attn_mask2[:, :, dn_number:, :dn_number] = True
            group_bbox_kpt = num_points
            # group_nobbox_kpt = self.num_body_points
            kpt_index = [x for x in range(self.num_group * num_points) 
                         if x % num_points in [
                             0, self.num_body_points+1, self.num_body_points+2, self.num_body_points+3]]
            for matchj in range(self.num_group * num_points):
                sj = (matchj // group_bbox_kpt) * group_bbox_kpt
                ej = (matchj // group_bbox_kpt + 1)*group_bbox_kpt
                if sj > 0:
                    attn_mask2[:, :, dn_number:, dn_number:][:, :, matchj, :sj] = True
                if ej < self.num_group * num_points:
                    attn_mask2[:, :, dn_number:, dn_number:][:, :, matchj, ej:] = True

            for match_x in range(self.num_group * num_points):
                if match_x % group_bbox_kpt in [0, 
                                                self.num_body_points+1, 
                                                self.num_body_points+2, 
                                                self.num_body_points+3]:
                    attn_mask2[:, :, dn_number:, dn_number:][:,:,match_x,kpt_index]=False

            for idx, (gt_boxes_i, gt_labels_i) in enumerate(zip(gt_boxes, gt_labels)):
                num_gt_i = gt_boxes_i.shape[0]
                if num_gt_i == 0:
                    continue
                for matchi in range(dn_number):
                    si = (matchi // num_gt_i) * num_gt_i
                    ei = (matchi // num_gt_i + 1) * num_gt_i
                    if si > 0:
                        attn_mask2[idx, :, matchi, :si] = True
                    if ei < dn_number:
                        attn_mask2[idx, :, matchi, ei:dn_number] = True
            attn_mask2 = attn_mask2.flatten(0, 1)


        if 'group2group' in self.dn_attn_mask_type_list:
            
            # self.num_body_points = self.num_body_points +3
            num_points = self.num_whole_body_points + 4
            attn_mask3 = torch.zeros(
                bs,
                self.nheads,
                dn_number + self.num_group * (num_points), dn_number + self.num_group * (num_points),
                                    device=device, dtype=torch.bool)
            attn_mask3[:, :, dn_number:, :dn_number] = True
            group_bbox_kpt = (num_points)
            # group_nobbox_kpt = self.num_body_points
            kpt_index = [
                x for x in range(self.num_group * (num_points)) if x % (num_points) in 
                [0, 
                 1+self.num_body_points, 
                 2+self.num_body_points+self.num_hand_points, 
                 3+self.num_body_points+self.num_hand_points*2
                 ]
                ]
            for matchj in range(self.num_group * num_points):
                sj = (matchj // group_bbox_kpt) * group_bbox_kpt
                ej = (matchj // group_bbox_kpt + 1)*group_bbox_kpt
                if sj > 0:
                    attn_mask3[:, :, dn_number:, dn_number:][:, :, matchj, :sj] = True
                if ej < self.num_group * num_points:
                    attn_mask3[:, :, dn_number:, dn_number:][:, :, matchj, ej:] = True

            for match_x in range(self.num_group * num_points):
                if match_x % group_bbox_kpt in [0, 
                                                1 + self.num_body_points, 
                                                2 + self.num_body_points + self.num_hand_points, 
                                                3 + self.num_body_points + self.num_hand_points * 2]:
                    
                    attn_mask3[:, :, dn_number:, dn_number:][:,:,match_x,kpt_index]=False

            for idx, (gt_boxes_i, gt_labels_i) in enumerate(zip(gt_boxes, gt_labels)):
                num_gt_i = gt_boxes_i.shape[0]
                if num_gt_i == 0:
                    continue
                for matchi in range(dn_number):
                    si = (matchi // num_gt_i) * num_gt_i
                    ei = (matchi // num_gt_i + 1) * num_gt_i
                    if si > 0:
                        attn_mask3[idx, :, matchi, :si] = True
                    if ei < dn_number:
                        attn_mask3[idx, :, matchi, ei:dn_number] = True
            attn_mask3 = attn_mask3.flatten(0, 1)




        mask_dict = {
            'pad_size': dn_number,
            'known_bboxs': gt_boxes_expand,
            'known_labels': gt_labels_expand,
            'known_keypoints': gt_keypoints_expand
        }

        return input_query_label, input_query_bbox, attn_mask, attn_mask2, attn_mask3, mask_dict

    def dn_post_process2(self, outputs_class, outputs_coord,
                         outputs_body_keypoints_list, mask_dict):
        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = [
                outputs_class_i[:, :mask_dict['pad_size'], :]
                for outputs_class_i in outputs_class
            ]
            output_known_coord = [
                outputs_coord_i[:, :mask_dict['pad_size'], :]
                for outputs_coord_i in outputs_coord
            ]

            outputs_class = [
                outputs_class_i[:, mask_dict['pad_size']:, :]
                for outputs_class_i in outputs_class
            ]
            outputs_coord = [
                outputs_coord_i[:, mask_dict['pad_size']:, :]
                for outputs_coord_i in outputs_coord
            ]
            outputs_keypoint = outputs_body_keypoints_list

            mask_dict.update({
                'output_known_coord': output_known_coord,
                'output_known_class': output_known_class
            })
        return outputs_class, outputs_coord, outputs_keypoint

    def forward(self, data_batch: NestedTensor, targets: List = None):
        """The forward expects a NestedTensor, which consists of:

           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """

        if isinstance(data_batch, dict):
            samples, targets = self.prepare_targets(data_batch)
            # import pdb; pdb.set_trace()
        elif isinstance(data_batch, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(data_batch)
        else:
            samples = data_batch
        # print(samples.data['img'].shape)
        # exit()
        features, poss = self.backbone(samples)
        srcs = []
        masks = []
        for l, feat in enumerate(features):  # len(features=3)
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(),
                                     size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        if self.dn_number > 0 or targets is not None:
            input_query_label, input_query_bbox, attn_mask,attn_mask2, attn_mask3, mask_dict =\
                self.prepare_for_dn2(targets)
        else:
            assert targets is None
            input_query_bbox = input_query_label = attn_mask = attn_mask2 = attn_mask3 = mask_dict = None


        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask,
            attn_mask2, attn_mask3)

        # update human boxes
        effective_dn_number = self.dn_number if self.training else 0
        outputs_body_bbox_list = []
        outputs_class = []
        
        for dec_lid, (layer_ref_sig, layer_body_bbox_embed, layer_cls_embed,
                      layer_hs) in enumerate(
                          zip(reference[:-1], self.bbox_embed,
                              self.class_embed, hs)):
            if dec_lid < self.num_box_decoder_layers:
                # human det
                layer_delta_unsig = layer_body_bbox_embed(layer_hs)
                layer_body_box_outputs_unsig = \
                    layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
                layer_body_box_outputs_unsig = layer_body_box_outputs_unsig.sigmoid()
                layer_cls = layer_cls_embed(layer_hs)
                outputs_body_bbox_list.append(layer_body_box_outputs_unsig)
                outputs_class.append(layer_cls)
                
            elif dec_lid < self.num_box_decoder_layers + 2:
                bs = layer_ref_sig.shape[0]                
                # dn body bbox
                layer_hs_body_bbox_dn = layer_hs[:, :effective_dn_number, :]  # dn content query
                reference_before_sigmoid_body_bbox_dn = layer_ref_sig[:, :effective_dn_number, :]  # dn position query
                layer_body_box_delta_unsig_dn = layer_body_bbox_embed(layer_hs_body_bbox_dn)
                layer_body_box_outputs_unsig_dn = layer_body_box_delta_unsig_dn + inverse_sigmoid(
                    reference_before_sigmoid_body_bbox_dn)
                layer_body_box_outputs_unsig_dn = layer_body_box_outputs_unsig_dn.sigmoid()
                
                # norm body bbox
                layer_hs_body_bbox_norm = layer_hs[:, effective_dn_number:, :][
                    :, 0::(self.num_body_points + 4), :]  # norm content query
                reference_before_sigmoid_body_bbox_norm = layer_ref_sig[:, effective_dn_number:, :][
                    :, 0::(self.num_body_points+ 4), :]  # norm position query
                layer_body_box_delta_unsig_norm = layer_body_bbox_embed(layer_hs_body_bbox_norm)
                layer_body_box_outputs_unsig_norm = layer_body_box_delta_unsig_norm + inverse_sigmoid(
                    reference_before_sigmoid_body_bbox_norm)
                layer_body_box_outputs_unsig_norm = layer_body_box_outputs_unsig_norm.sigmoid()

                layer_body_box_outputs_unsig = torch.cat(
                    (layer_body_box_outputs_unsig_dn, layer_body_box_outputs_unsig_norm), dim=1)

                # classfication
                layer_cls_dn = layer_cls_embed(layer_hs_body_bbox_dn)
                layer_cls_norm = layer_cls_embed(layer_hs_body_bbox_norm)
                layer_cls = torch.cat((layer_cls_dn, layer_cls_norm), dim=1)

                outputs_class.append(layer_cls)
                outputs_body_bbox_list.append(layer_body_box_outputs_unsig)                
            else:
                bs = layer_ref_sig.shape[0]                
                # dn body bbox
                layer_hs_body_bbox_dn = layer_hs[:, :effective_dn_number, :]  # dn content query
                reference_before_sigmoid_body_bbox_dn = layer_ref_sig[:, :effective_dn_number, :]  # dn position query
                layer_body_box_delta_unsig_dn = layer_body_bbox_embed(layer_hs_body_bbox_dn)
                layer_body_box_outputs_unsig_dn = layer_body_box_delta_unsig_dn + inverse_sigmoid(
                    reference_before_sigmoid_body_bbox_dn)
                layer_body_box_outputs_unsig_dn = layer_body_box_outputs_unsig_dn.sigmoid()
                
                # norm body bbox
                layer_hs_body_bbox_norm = layer_hs[:, effective_dn_number:, :][
                    :, 0::(self.num_whole_body_points + 4), :]  # norm content query
                reference_before_sigmoid_body_bbox_norm = layer_ref_sig[:,effective_dn_number:, :][
                    :, 0::(self.num_whole_body_points + 4), :]  # norm position query
                layer_body_box_delta_unsig_norm = layer_body_bbox_embed(layer_hs_body_bbox_norm)
                layer_body_box_outputs_unsig_norm = layer_body_box_delta_unsig_norm + inverse_sigmoid(
                    reference_before_sigmoid_body_bbox_norm)
                layer_body_box_outputs_unsig_norm = layer_body_box_outputs_unsig_norm.sigmoid()

                layer_body_box_outputs_unsig = torch.cat(
                    (layer_body_box_outputs_unsig_dn, layer_body_box_outputs_unsig_norm), dim=1)

                # classfication
                layer_cls_dn = layer_cls_embed(layer_hs_body_bbox_dn)
                layer_cls_norm = layer_cls_embed(layer_hs_body_bbox_norm)
                layer_cls = torch.cat((layer_cls_dn, layer_cls_norm), dim=1)

                outputs_class.append(layer_cls)
                outputs_body_bbox_list.append(layer_body_box_outputs_unsig)       
        
        # 找query
        q_index = torch.topk(layer_cls_norm.max(-1)[0], 100, dim=1)[1]
        q_value = torch.topk(layer_cls_norm.max(-1)[0], 100, dim=1)[0]
        # update hand and face boxes
        outputs_lhand_bbox_list = []
        outputs_rhand_bbox_list = []
        outputs_face_bbox_list = []
        # update keypoints boxes
        outputs_body_keypoints_list = []
        outputs_body_keypoints_hw = []
        outputs_lhand_keypoints_list = []
        outputs_lhand_keypoints_hw = []        
        outputs_rhand_keypoints_list = []
        outputs_rhand_keypoints_hw = []
        outputs_face_keypoints_list = []
        outputs_face_keypoints_hw = []             
        
        outputs_smpl_pose_list = []
        outputs_smpl_lhand_pose_list = []
        outputs_smpl_rhand_pose_list = []
        outputs_smpl_expr_list = []
        outputs_smpl_jaw_pose_list = []
        outputs_smpl_beta_list = []
        outputs_smpl_cam_list = []
        # outputs_smpl_cam_f_list = []
        outputs_smpl_kp2d_list = []
        outputs_smpl_kp3d_list = []
        outputs_smpl_verts_list = []
        body_kpt_index = [
            x for x in range(self.num_group * (self.num_body_points + 4))
            if x % (self.num_body_points + 4) in range(1,self.num_body_points+1)
        ]
        body_kpt_index_2 = [
            x for x in range(self.num_group * (self.num_whole_body_points + 4))
            if (x % (self.num_whole_body_points + 4) in range(1,self.num_body_points+1))
        ]
        lhand_kpt_index = [
            x for x in range(self.num_group * (self.num_whole_body_points + 4))
            if (x % (self.num_whole_body_points + 4) in range(
                self.num_body_points+2, 
                self.num_body_points+self.num_hand_points+2))]
        
        rhand_kpt_index = [
            x for x in range(self.num_group * (self.num_whole_body_points + 4))
            if (x % (self.num_whole_body_points + 4) in range(
                self.num_body_points+self.num_hand_points+3, 
                self.num_body_points+self.num_hand_points*2+3))
        ]
        
        face_kpt_index = [
            x for x in range(self.num_group * (self.num_whole_body_points + 4))
            if (x % (self.num_whole_body_points + 4) in range(
                self.num_body_points+self.num_hand_points*2+4, 
                self.num_body_points+self.num_hand_points*2+self.num_face_points+4))
        ]
        
        # body box, kps, lhand box
        body_index = list(range(0,self.num_body_points+2))
        # rhand box and face box        
        body_index.extend(
            [self.num_body_points + self.num_hand_points + 2, self.num_body_points + 2 * self.num_hand_points + 3]
        )
        smpl_pose_index = [
            x for x in range(self.num_group * (self.num_whole_body_points + 4))
            if (x % (self.num_whole_body_points + 4) in body_index) 
        ]
        
        # smpl lhand
        lhand_index = list(range(self.num_body_points+1, self.num_body_points+self.num_hand_points+3))
        # body box
        lhand_index.insert(0, 0)
        
        smpl_lhand_pose_index = [
            x for x in range(self.num_group * (self.num_whole_body_points + 4))
            if (x % (self.num_whole_body_points + 4) in lhand_index)]
        
        # smpl rhand
        rhand_index = list(range(self.num_body_points + self.num_hand_points + 2, self.num_body_points + self.num_hand_points * 2 +3))
        rhand_index.insert(0,self.num_body_points+1)
        rhand_index.insert(0,0)
        smpl_rhand_pose_index = [
            x for x in range(self.num_group * (self.num_whole_body_points + 4))
            if (x % (self.num_whole_body_points + 4) in rhand_index)]
        
        # smpl face
        face_index = list(range(self.num_body_points + self.num_hand_points * 2 + 3, self.num_body_points + self.num_hand_points * 2 + self.num_face_points + 4))
        face_index.insert(0,0)
        
        smpl_face_pose_index = [
            x for x in range(self.num_group * (self.num_whole_body_points + 4))
            if (x % (self.num_whole_body_points + 4) in face_index)]
        
        for dec_lid, (layer_ref_sig, layer_hs) in enumerate(zip(reference[:-1], hs)):
            if dec_lid < self.num_box_decoder_layers:
                assert isinstance(layer_hs, torch.Tensor)
                bs = layer_hs.shape[0]
                layer_body_kps_res = layer_hs.new_zeros(
                    (bs, self.num_queries,
                     self.num_body_points * 3))  # [-, 900, 42]
                outputs_body_keypoints_list.append(layer_body_kps_res)
                
                # lhand
                layer_lhand_bbox_res = layer_hs.new_zeros(
                    (bs, self.num_queries, 4))  # [-, 900, 42]
                outputs_lhand_bbox_list.append(layer_lhand_bbox_res)
                layer_lhand_kps_res = layer_hs.new_zeros(
                    (bs, self.num_queries,
                     self.num_hand_points * 3))  # [-, 900, 42]
                outputs_lhand_keypoints_list.append(layer_lhand_kps_res)                

                # rhand
                layer_rhand_bbox_res = layer_hs.new_zeros(
                    (bs, self.num_queries, 4))  # [-, 900, 42]
                outputs_rhand_bbox_list.append(layer_rhand_bbox_res)                
                layer_rhand_kps_res = layer_hs.new_zeros(
                    (bs, self.num_queries,
                     self.num_hand_points * 3))  # [-, 900, 42]
                outputs_rhand_keypoints_list.append(layer_rhand_kps_res)
                
                # face
                layer_face_bbox_res = layer_hs.new_zeros(
                    (bs, self.num_queries, 4))  # [-, 900, 42]
                outputs_face_bbox_list.append(layer_face_bbox_res)
                layer_face_kps_res = layer_hs.new_zeros(
                    (bs, self.num_queries,
                     self.num_face_points * 3))  # [-, 900, 42]
                outputs_face_keypoints_list.append(layer_face_kps_res)
                
                
                # smpl or smplx
                smpl_pose = layer_hs.new_zeros((bs, self.num_queries, self.body_model_joint_num * 3))
                smpl_rhand_pose = layer_hs.new_zeros(
                    (bs, self.num_queries, 15 * 3))
                smpl_lhand_pose = layer_hs.new_zeros(
                    (bs, self.num_queries, 15 * 3))
                smpl_expr = layer_hs.new_zeros((bs, self.num_queries, 10))
                smpl_jaw_pose = layer_hs.new_zeros((bs, self.num_queries, 6))
                smpl_beta = layer_hs.new_zeros((bs, self.num_queries, 10))
                smpl_cam = layer_hs.new_zeros((bs, self.num_queries, 3))
                # smpl_cam_f = layer_hs.new_zeros((bs, self.num_queries, 1))
                # smpl_kp2d = layer_hs.new_zeros((bs, self.num_queries, self.num_body_points,3))
                smpl_kp3d = layer_hs.new_zeros(
                    (bs, self.num_queries, self.num_body_points, 4))
                outputs_smpl_pose_list.append(smpl_pose)
                outputs_smpl_rhand_pose_list.append(smpl_rhand_pose)
                outputs_smpl_lhand_pose_list.append(smpl_lhand_pose)
                outputs_smpl_expr_list.append(smpl_expr)
                outputs_smpl_jaw_pose_list.append(smpl_jaw_pose)
                outputs_smpl_beta_list.append(smpl_beta)
                outputs_smpl_cam_list.append(smpl_cam)
                # outputs_smpl_cam_f_list.append(smpl_cam_f)
                # outputs_smpl_kp2d_list.append(smpl_kp2d)
                outputs_smpl_kp3d_list.append(smpl_kp3d)
            elif dec_lid < self.num_box_decoder_layers +2:
                bs = layer_ref_sig.shape[0]
                layer_hs_body_kpt = \
                    layer_hs[:, effective_dn_number:, :].index_select(
                        1, torch.tensor(body_kpt_index, device=layer_hs.device))
                # body kp2d
                delta_body_kp_xy_unsig = \
                    self.pose_embed[dec_lid - self.num_box_decoder_layers](layer_hs_body_kpt)
                layer_ref_sig_body_kpt = \
                    layer_ref_sig[:,effective_dn_number:, :].index_select(1,torch.tensor(body_kpt_index,device=layer_hs.device))
                layer_outputs_unsig_body_keypoints = delta_body_kp_xy_unsig + inverse_sigmoid(
                    layer_ref_sig_body_kpt[..., :2])
                vis_xy_unsig = torch.ones_like(
                    layer_outputs_unsig_body_keypoints,
                    device=layer_outputs_unsig_body_keypoints.device)
                xyv = torch.cat((layer_outputs_unsig_body_keypoints,
                                 vis_xy_unsig[:, :, 0].unsqueeze(-1)),
                                dim=-1)
                xyv = xyv.sigmoid()
                layer_res = xyv.reshape(
                    (bs, self.num_group, self.num_body_points,
                     3)).flatten(2, 3)
                layer_hw = layer_ref_sig_body_kpt[..., 2:].reshape(
                    bs, self.num_group, self.num_body_points, 2).flatten(2, 3)
                layer_res = keypoint_xyzxyz_to_xyxyzz(layer_res)
                outputs_body_keypoints_list.append(layer_res)
                outputs_body_keypoints_hw.append(layer_hw)
                
                # lhand bbox
                layer_hs_lhand_bbox = \
                    layer_hs[:, effective_dn_number:, :][:, (self.num_body_points + 1)::(self.num_body_points + 4), :]
                    
                delta_lhand_bbox_xy_unsig = self.bbox_hand_embed[dec_lid - self.num_box_decoder_layers](layer_hs_lhand_bbox)             
                layer_ref_sig_lhand_bbox = \
                    layer_ref_sig[:,effective_dn_number:, :][
                        :, (self.num_body_points + 1)::(self.num_body_points + 4), :].clone() 
                layer_ref_unsig_lhand_bbox = inverse_sigmoid(layer_ref_sig_lhand_bbox)
                delta_lhand_bbox_hw_unsig = self.bbox_hand_hw_embed[
                    dec_lid-self.num_box_decoder_layers](layer_hs_lhand_bbox)
                layer_ref_unsig_lhand_bbox[..., :2] +=delta_lhand_bbox_xy_unsig[..., :2]
                layer_ref_unsig_lhand_bbox[..., 2:] +=delta_lhand_bbox_hw_unsig
                layer_ref_sig_lhand_bbox = layer_ref_unsig_lhand_bbox.sigmoid()
                outputs_lhand_bbox_list.append(layer_ref_sig_lhand_bbox)
                
                layer_lhand_kps_res = layer_hs.new_zeros(
                    (bs, self.num_queries,
                     self.num_hand_points * 3))  # [-, 900, 42]
                outputs_lhand_keypoints_list.append(layer_lhand_kps_res)
                                
                # rhand bbox
                layer_hs_rhand_bbox = \
                    layer_hs[:, effective_dn_number:, :][
                        :, (self.num_body_points + 2)::(self.num_body_points + 4), :]
                delta_rhand_bbox_xy_unsig = self.bbox_hand_embed[
                    dec_lid - self.num_box_decoder_layers](layer_hs_rhand_bbox)             
                layer_ref_sig_rhand_bbox = \
                    layer_ref_sig[:,effective_dn_number:, :][
                        :, (self.num_body_points + 2)::(self.num_body_points + 4), :].clone()
                layer_ref_unsig_rhand_bbox = inverse_sigmoid(layer_ref_sig_rhand_bbox)
                delta_rhand_bbox_hw_unsig = self.bbox_hand_hw_embed[
                    dec_lid-self.num_box_decoder_layers](layer_hs_rhand_bbox)
                layer_ref_unsig_rhand_bbox[..., :2] +=delta_rhand_bbox_xy_unsig[..., :2]
                layer_ref_unsig_rhand_bbox[..., 2:] +=delta_rhand_bbox_hw_unsig
                layer_ref_sig_rhand_bbox = layer_ref_unsig_rhand_bbox.sigmoid()
                outputs_rhand_bbox_list.append(layer_ref_sig_rhand_bbox)
                
                # rhand kps
                layer_rhand_kps_res = layer_hs.new_zeros(
                    (bs, self.num_queries,
                     self.num_hand_points * 3))  # [-, 900, 42]
                outputs_rhand_keypoints_list.append(layer_rhand_kps_res)
                
                # face bbox
                layer_hs_face_bbox = \
                    layer_hs[:, effective_dn_number:, :][
                        :, (self.num_body_points + 3)::(self.num_body_points + 4), :]
                delta_face_bbox_xy_unsig = self.bbox_face_embed[
                    dec_lid - self.num_box_decoder_layers](layer_hs_face_bbox)             
                layer_ref_sig_face_bbox = \
                    layer_ref_sig[:,effective_dn_number:, :][
                        :, (self.num_body_points + 3)::(self.num_body_points + 4), :].clone()
                layer_ref_unsig_face_bbox = inverse_sigmoid(layer_ref_sig_face_bbox)
                delta_face_bbox_hw_unsig = self.bbox_face_hw_embed[
                    dec_lid-self.num_box_decoder_layers](layer_hs_face_bbox)
                layer_ref_unsig_face_bbox[..., :2] +=delta_face_bbox_xy_unsig[..., :2]
                layer_ref_unsig_face_bbox[..., 2:] +=delta_face_bbox_hw_unsig                
                layer_ref_sig_face_bbox = layer_ref_unsig_face_bbox.sigmoid()
                
                outputs_face_bbox_list.append(layer_ref_sig_face_bbox)
                
                # face kps
                layer_face_kps_res = layer_hs.new_zeros(
                    (bs, self.num_queries,
                     self.num_face_points * 3))  # [-, 900, 42]
                outputs_face_keypoints_list.append(layer_face_kps_res)
                
                # smpl or smplx
                bs, _, feat_dim = layer_hs.shape
                smpl_feats = layer_hs[:, effective_dn_number:, :].reshape(
                    bs, -1, feat_dim * (self.num_body_points + 4))
                smpl_lhand_pose_feats = layer_hs[:, effective_dn_number:, :][
                    :, (self.num_body_points + 1):: (self.num_body_points + 4), :].reshape(
                        bs, -1, feat_dim)
                smpl_rhand_pose_feats = layer_hs[:, effective_dn_number:, :][
                    :, (self.num_body_points + 2):: (self.num_body_points + 4), :].reshape(
                        bs, -1, feat_dim)
                smpl_face_pose_feats = layer_hs[:, effective_dn_number:, :][
                    :, (self.num_body_points + 3):: (self.num_body_points + 4), :].reshape(
                        bs, -1, feat_dim)
                                  
                smpl_pose = self.smpl_pose_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_feats)
                smpl_pose = rot6d_to_rotmat(smpl_pose.reshape(-1, 6)).reshape(
                    bs, self.num_group, self.body_model_joint_num, 3, 3)
                
                smpl_lhand_pose = self.smpl_hand_pose_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_lhand_pose_feats)
                smpl_lhand_pose = rot6d_to_rotmat(smpl_lhand_pose.reshape(
                    -1, 6)).reshape(bs, self.num_group, 15, 3, 3)
                
                smpl_rhand_pose = self.smpl_hand_pose_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_rhand_pose_feats)
                smpl_rhand_pose = rot6d_to_rotmat(smpl_rhand_pose.reshape(
                    -1, 6)).reshape(bs, self.num_group, 15, 3, 3)
                
                smpl_jaw_pose = self.smpl_jaw_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_face_pose_feats)
                smpl_jaw_pose = rot6d_to_rotmat(smpl_jaw_pose.reshape(-1, 6)).reshape(
                    bs, self.num_group, 1, 3, 3)
                                 
                smpl_beta = self.smpl_beta_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_feats)
                smpl_cam = self.smpl_cam_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_feats)
                smpl_expr = self.smpl_expr_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_face_pose_feats)
                # smpl_jaw_pose = layer_hs.new_zeros(bs, self.num_group, 3)
                leye_pose = torch.zeros_like(smpl_jaw_pose)
                reye_pose = torch.zeros_like(smpl_jaw_pose)



                if self.body_model is not None:
                    smpl_pose_ = rotmat_to_aa(smpl_pose)
                    smpl_lhand_pose_ = layer_hs.new_zeros(bs, self.num_group, 15, 3)
                    smpl_rhand_pose_ = layer_hs.new_zeros(bs, self.num_group, 15, 3)
                    smpl_jaw_pose_ = rotmat_to_aa(smpl_jaw_pose)
                    leye_pose_ = rotmat_to_aa(leye_pose)
                    reye_pose_ = rotmat_to_aa(reye_pose)
                    
                    pred_output = self.body_model(
                        betas=smpl_beta.reshape(-1, 10),
                        body_pose=smpl_pose_[:, :,  1:].reshape(-1, 21 * 3),
                        global_orient=smpl_pose_[:, :, 0].reshape(
                            -1, 3).unsqueeze(1),
                        left_hand_pose=smpl_lhand_pose_.reshape(-1, 15 * 3),
                        right_hand_pose=smpl_rhand_pose_.reshape(-1, 15 * 3),
                        leye_pose=leye_pose_,
                        reye_pose=reye_pose_,
                        jaw_pose=smpl_jaw_pose_.reshape(-1, 3),
                        # expression=smpl_expr.reshape(-1, 10),
                        expression=layer_hs.new_zeros(bs, self.num_group, 10).reshape(-1, 10)
                    )
                    smpl_kp3d = pred_output['joints'].reshape(
                        bs, self.num_group, -1, 3)
                    smpl_verts = pred_output['vertices'].reshape(
                        bs, self.num_group, -1, 3)
                outputs_smpl_pose_list.append(smpl_pose)
                outputs_smpl_rhand_pose_list.append(smpl_rhand_pose)
                outputs_smpl_lhand_pose_list.append(smpl_lhand_pose)
                outputs_smpl_expr_list.append(smpl_expr)
                outputs_smpl_jaw_pose_list.append(smpl_jaw_pose)
                outputs_smpl_beta_list.append(smpl_beta)
                outputs_smpl_cam_list.append(smpl_cam)
                outputs_smpl_kp3d_list.append(smpl_kp3d)
            else:
                bs = layer_ref_sig.shape[0]
                layer_hs_body_kpt = \
                    layer_hs[:, effective_dn_number:, :].index_select(
                        1, torch.tensor(body_kpt_index_2, device=layer_hs.device))

                # body kp2d
                delta_body_kp_xy_unsig = \
                    self.pose_embed[
                        dec_lid - self.num_box_decoder_layers](layer_hs_body_kpt)
                layer_ref_sig_body_kpt = \
                    layer_ref_sig[:,effective_dn_number:, :].index_select(
                        1,torch.tensor(body_kpt_index_2,device=layer_hs.device))
                layer_outputs_unsig_body_keypoints = \
                    delta_body_kp_xy_unsig + inverse_sigmoid(
                    layer_ref_sig_body_kpt[..., :2])
                vis_xy_unsig = torch.ones_like(
                    layer_outputs_unsig_body_keypoints,
                    device=layer_outputs_unsig_body_keypoints.device)
                xyv = torch.cat((layer_outputs_unsig_body_keypoints,
                                 vis_xy_unsig[:, :, 0].unsqueeze(-1)),
                                dim=-1)
                xyv = xyv.sigmoid()
                layer_res = xyv.reshape(
                    (bs, self.num_group, self.num_body_points,
                     3)).flatten(2, 3)
                layer_hw = layer_ref_sig_body_kpt[..., 2:].reshape(
                    bs, self.num_group, self.num_body_points, 2).flatten(2, 3)
                layer_res = keypoint_xyzxyz_to_xyxyzz(layer_res)
                outputs_body_keypoints_list.append(layer_res)
                outputs_body_keypoints_hw.append(layer_hw)
                
                # lhand bbox
                layer_hs_lhand_bbox = \
                    layer_hs[:, effective_dn_number:, :][
                        :, (self.num_body_points + 1)::(self.num_whole_body_points + 4), :]
                    
                delta_lhand_bbox_xy_unsig = self.bbox_hand_embed[
                    dec_lid - self.num_box_decoder_layers](layer_hs_lhand_bbox)             
                layer_ref_sig_lhand_bbox = \
                    layer_ref_sig[:,effective_dn_number:, :][
                        :, (self.num_body_points + 1)::(self.num_whole_body_points + 4), :].clone()
                layer_ref_unsig_lhand_bbox = inverse_sigmoid(layer_ref_sig_lhand_bbox)
                delta_lhand_bbox_hw_unsig = self.bbox_hand_hw_embed[
                    dec_lid-self.num_box_decoder_layers](layer_hs_lhand_bbox)
                layer_ref_unsig_lhand_bbox[..., :2] +=delta_lhand_bbox_xy_unsig[..., :2]
                layer_ref_unsig_lhand_bbox[..., 2:] +=delta_lhand_bbox_hw_unsig
                layer_ref_sig_lhand_bbox = layer_ref_unsig_lhand_bbox.sigmoid()
                outputs_lhand_bbox_list.append(layer_ref_sig_lhand_bbox)
                
                # lhand kps
                layer_hs_lhand_kps_res = \
                    layer_hs[:, effective_dn_number:, :].index_select(
                        1, torch.tensor(lhand_kpt_index, device=layer_hs.device))
                delta_lhand_kp_xy_unsig = \
                    self.pose_hand_embed[
                        dec_lid - self.num_hand_face_decoder_layers](layer_hs_lhand_kps_res)                
                layer_ref_sig_lhand_kpt = \
                    layer_ref_sig[:,effective_dn_number:, :].index_select(
                        1,torch.tensor(lhand_kpt_index,device=layer_hs.device)) 
                layer_outputs_unsig_lhand_keypoints = delta_lhand_kp_xy_unsig + inverse_sigmoid(
                    layer_ref_sig_lhand_kpt[..., :2])                    
                lhand_vis_xy_unsig = torch.ones_like(
                    layer_outputs_unsig_lhand_keypoints,
                    device=layer_outputs_unsig_lhand_keypoints.device)
                lhand_xyv = torch.cat((layer_outputs_unsig_lhand_keypoints,
                                 lhand_vis_xy_unsig[:, :, 0].unsqueeze(-1)),
                                dim=-1)
                lhand_xyv = lhand_xyv.sigmoid()
                layer_lhand_kps_res = lhand_xyv.reshape(
                    (bs, self.num_group, self.num_hand_points,
                     3)).flatten(2, 3)
                layer_lhand_hw = layer_ref_sig_lhand_kpt[..., 2:].reshape(
                    bs, self.num_group, self.num_hand_points, 2).flatten(2, 3)
                layer_lhand_kps_res = keypoint_xyzxyz_to_xyxyzz(layer_lhand_kps_res)
                outputs_lhand_keypoints_list.append(layer_lhand_kps_res)
                outputs_lhand_keypoints_hw.append(layer_lhand_hw)

                # rhand bbox
                layer_hs_rhand_bbox = \
                    layer_hs[:, effective_dn_number:, :][
                        :, (self.num_body_points + self.num_hand_points + 2)::(self.num_whole_body_points + 4), :]
                delta_rhand_bbox_xy_unsig = self.bbox_hand_embed[
                    dec_lid - self.num_box_decoder_layers](layer_hs_rhand_bbox)             
                layer_ref_sig_rhand_bbox = \
                    layer_ref_sig[:,effective_dn_number:, :][
                        :, (self.num_body_points + self.num_hand_points + 2)::(self.num_whole_body_points + 4), :].clone()                  
                layer_ref_unsig_rhand_bbox = inverse_sigmoid(layer_ref_sig_rhand_bbox)
                delta_rhand_bbox_hw_unsig = self.bbox_hand_hw_embed[
                    dec_lid-self.num_box_decoder_layers](layer_hs_rhand_bbox)
                layer_ref_unsig_rhand_bbox[..., :2] +=delta_rhand_bbox_xy_unsig[..., :2]
                layer_ref_unsig_rhand_bbox[..., 2:] +=delta_rhand_bbox_hw_unsig
                layer_ref_sig_rhand_bbox = layer_ref_unsig_rhand_bbox.sigmoid()
                outputs_rhand_bbox_list.append(layer_ref_sig_rhand_bbox)
                
                # rhand kps
                layer_hs_rhand_kps_res = \
                    layer_hs[:, effective_dn_number:, :].index_select(
                        1, torch.tensor(rhand_kpt_index, device=layer_hs.device))
                delta_rhand_kp_xy_unsig = \
                    self.pose_hand_embed[
                        dec_lid - self.num_hand_face_decoder_layers](layer_hs_rhand_kps_res)                
                layer_ref_sig_rhand_kpt = \
                    layer_ref_sig[:,effective_dn_number:, :].index_select(
                        1,torch.tensor(rhand_kpt_index,device=layer_hs.device)) 
                layer_outputs_unsig_rhand_keypoints = delta_rhand_kp_xy_unsig + inverse_sigmoid(
                    layer_ref_sig_rhand_kpt[..., :2])                    
                rhand_vis_xy_unsig = torch.ones_like(
                    layer_outputs_unsig_rhand_keypoints,
                    device=layer_outputs_unsig_rhand_keypoints.device)
                rhand_xyv = torch.cat((layer_outputs_unsig_rhand_keypoints,
                                 rhand_vis_xy_unsig[:, :, 0].unsqueeze(-1)),
                                dim=-1)
                rhand_xyv = rhand_xyv.sigmoid()
                layer_rhand_kps_res = rhand_xyv.reshape(
                    (bs, self.num_group, self.num_hand_points,
                     3)).flatten(2, 3)
                layer_rhand_hw = layer_ref_sig_rhand_kpt[..., 2:].reshape(
                    bs, self.num_group, self.num_hand_points, 2).flatten(2, 3)
                layer_rhand_kps_res = keypoint_xyzxyz_to_xyxyzz(layer_rhand_kps_res)
                outputs_rhand_keypoints_list.append(layer_rhand_kps_res)
                outputs_rhand_keypoints_hw.append(layer_rhand_hw)
                
                # face bbox
                layer_hs_face_bbox = \
                    layer_hs[:, effective_dn_number:, :][
                        :, (self.num_body_points + 2 * self.num_hand_points + 3)::(self.num_whole_body_points + 4), :]
                delta_face_bbox_xy_unsig = self.bbox_face_embed[dec_lid - self.num_box_decoder_layers](layer_hs_face_bbox)             
                layer_ref_sig_face_bbox = \
                    layer_ref_sig[:,effective_dn_number:, :][
                        :, (self.num_body_points + 2 * self.num_hand_points + 3)::(self.num_whole_body_points + 4), :].clone()               
                layer_ref_unsig_face_bbox = inverse_sigmoid(layer_ref_sig_face_bbox)
                delta_face_bbox_hw_unsig = self.bbox_face_hw_embed[
                    dec_lid-self.num_box_decoder_layers](layer_hs_face_bbox)
                layer_ref_unsig_face_bbox[..., :2] +=delta_face_bbox_xy_unsig[..., :2]
                layer_ref_unsig_face_bbox[..., 2:] +=delta_face_bbox_hw_unsig
                layer_ref_sig_face_bbox = layer_ref_unsig_face_bbox.sigmoid()   
                outputs_face_bbox_list.append(layer_ref_sig_face_bbox)
                
                # face kps
                layer_hs_face_kps_res = \
                    layer_hs[:, effective_dn_number:, :].index_select(
                        1, torch.tensor(face_kpt_index, device=layer_hs.device))
                delta_face_kp_xy_unsig = \
                    self.pose_face_embed[
                        dec_lid - self.num_hand_face_decoder_layers](layer_hs_face_kps_res)                
                layer_ref_sig_face_kpt = \
                    layer_ref_sig[:,effective_dn_number:, :].index_select(
                        1,torch.tensor(face_kpt_index,device=layer_hs.device)) 
                layer_outputs_unsig_face_keypoints = delta_face_kp_xy_unsig + inverse_sigmoid(
                    layer_ref_sig_face_kpt[..., :2])                    
                face_vis_xy_unsig = torch.ones_like(
                    layer_outputs_unsig_face_keypoints,
                    device=layer_outputs_unsig_face_keypoints.device)
                face_xyv = torch.cat((layer_outputs_unsig_face_keypoints,
                                 face_vis_xy_unsig[:, :, 0].unsqueeze(-1)),
                                dim=-1)
                face_xyv = face_xyv.sigmoid()
                layer_face_kps_res = face_xyv.reshape(
                    (bs, self.num_group, self.num_face_points,
                     3)).flatten(2, 3)
                layer_face_hw = layer_ref_sig_face_kpt[..., 2:].reshape(
                    bs, self.num_group, self.num_face_points, 2).flatten(2, 3)
                layer_face_kps_res = keypoint_xyzxyz_to_xyxyzz(layer_face_kps_res)
                outputs_face_keypoints_list.append(layer_face_kps_res)
                outputs_face_keypoints_hw.append(layer_face_hw)
                                
                # pdb.set_trace()
                bs, _, feat_dim = layer_hs.shape
                smpl_body_pose_feats = layer_hs[:, effective_dn_number:, :].index_select(
                    1, torch.tensor(smpl_pose_index, device=layer_hs.device)
                    ).reshape(bs, -1, feat_dim * (self.num_body_points + 4))
                smpl_lhand_pose_feats = layer_hs[:, effective_dn_number:, :].index_select(
                    1, torch.tensor(smpl_lhand_pose_index, device=layer_hs.device)
                    ).reshape(bs, -1, feat_dim * (self.num_hand_points + 3))
                smpl_rhand_pose_feats = layer_hs[:, effective_dn_number:, :].index_select(
                    1, torch.tensor(smpl_rhand_pose_index, device=layer_hs.device)
                    ).reshape(bs, -1, feat_dim * (self.num_hand_points + 3))
                smpl_face_pose_feats = layer_hs[:, effective_dn_number:, :].index_select(
                    1, torch.tensor(smpl_face_pose_index, device=layer_hs.device)
                    ).reshape(bs, -1, feat_dim * (self.num_face_points + 2))
                                                
                smpl_pose = self.smpl_pose_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_body_pose_feats)
                
                smpl_pose = rot6d_to_rotmat(smpl_pose.reshape(-1, 6)).reshape(
                    bs, self.num_group, self.body_model_joint_num, 3, 3)
                smpl_lhand_pose = self.smpl_hand_pose_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_lhand_pose_feats)
                smpl_lhand_pose = rot6d_to_rotmat(smpl_lhand_pose.reshape(
                    -1, 6)).reshape(bs, self.num_group, 15, 3, 3)
                smpl_rhand_pose = self.smpl_hand_pose_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_rhand_pose_feats)
                smpl_rhand_pose = rot6d_to_rotmat(smpl_rhand_pose.reshape(
                    -1, 6)).reshape(bs, self.num_group, 15, 3, 3)

                smpl_expr = self.smpl_expr_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_face_pose_feats)
                smpl_jaw_pose = self.smpl_jaw_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_face_pose_feats)
                smpl_jaw_pose = rot6d_to_rotmat(smpl_jaw_pose.reshape(-1, 6)).reshape(
                    bs, self.num_group, 1, 3, 3)
                smpl_beta = self.smpl_beta_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_body_pose_feats)
                smpl_cam = self.smpl_cam_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_body_pose_feats)
                # smpl_cam_f = self.smpl_cam_f_embed[
                #     dec_lid - self.num_box_decoder_layers](smpl_body_pose_feats)
                
                num_samples = smpl_beta.reshape(-1, 10).shape[0]
                device = smpl_beta.device
                leye_pose = torch.zeros_like(smpl_jaw_pose)
                reye_pose = torch.zeros_like(smpl_jaw_pose)

                if self.body_model is not None:
                    # print(smpl_pose)
                    # exit()
                    smpl_pose_ = rotmat_to_aa(smpl_pose)
                    smpl_lhand_pose_ = rotmat_to_aa(smpl_lhand_pose)
                    smpl_rhand_pose_ = rotmat_to_aa(smpl_rhand_pose)
                    smpl_jaw_pose_ = rotmat_to_aa(smpl_jaw_pose)
                    leye_pose_ = rotmat_to_aa(leye_pose)
                    reye_pose_ = rotmat_to_aa(reye_pose)
                    
                    pred_output = self.body_model(
                        betas=smpl_beta.reshape(-1, 10),
                        body_pose=smpl_pose_[:, :,  1:].reshape(-1, 21 * 3),
                        global_orient=smpl_pose_[:, :, 0].reshape(
                            -1, 3).unsqueeze(1),
                        left_hand_pose=smpl_lhand_pose_.reshape(-1, 15 * 3),
                        right_hand_pose=smpl_rhand_pose_.reshape(-1, 15 * 3),
                        leye_pose=leye_pose_,
                        reye_pose=reye_pose_,
                        jaw_pose=smpl_jaw_pose_.reshape(-1, 3),
                        expression=smpl_expr.reshape(-1, 10),
                    )
                    smpl_kp3d = pred_output['joints'].reshape(
                        bs, self.num_group, -1, 3)
                    smpl_verts = pred_output['vertices'].reshape(
                        bs, self.num_group, -1, 3)
                outputs_smpl_pose_list.append(smpl_pose)
                outputs_smpl_rhand_pose_list.append(smpl_rhand_pose)
                outputs_smpl_lhand_pose_list.append(smpl_lhand_pose)
                outputs_smpl_expr_list.append(smpl_expr)
                outputs_smpl_jaw_pose_list.append(smpl_jaw_pose)
                outputs_smpl_beta_list.append(smpl_beta)
                outputs_smpl_cam_list.append(smpl_cam)
                # outputs_smpl_cam_f_list.append(smpl_cam_f)
                outputs_smpl_kp3d_list.append(smpl_kp3d)
                if not self.training:
                    outputs_smpl_verts_list.append(smpl_verts)
        dn_mask_dict = mask_dict
        if self.dn_number > 0 and dn_mask_dict is not None:
            outputs_class, outputs_body_bbox_list, outputs_body_keypoints_list = self.dn_post_process2(
                outputs_class, outputs_body_bbox_list, outputs_body_keypoints_list,
                dn_mask_dict)
            dn_class_input = dn_mask_dict['known_labels']
            dn_bbox_input = dn_mask_dict['known_bboxs']
            dn_class_pred = dn_mask_dict['output_known_class']
            dn_bbox_pred = dn_mask_dict['output_known_coord']

        for idx, (_out_class, _out_bbox, _out_keypoint) in enumerate(
                zip(outputs_class, outputs_body_bbox_list,
                    outputs_body_keypoints_list)):
            assert _out_class.shape[1] == _out_bbox.shape[
                1] == _out_keypoint.shape[1]
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_body_bbox_list[-1],
            'pred_lhand_boxes': outputs_lhand_bbox_list[-1],
            'pred_rhand_boxes': outputs_rhand_bbox_list[-1],
            'pred_face_boxes': outputs_face_bbox_list[-1],
            'pred_keypoints': outputs_body_keypoints_list[-1],
            'pred_lhand_keypoints': outputs_lhand_keypoints_list[-1],
            'pred_rhand_keypoints': outputs_rhand_keypoints_list[-1],
            'pred_face_keypoints': outputs_face_keypoints_list[-1],
            'pred_smpl_pose': outputs_smpl_pose_list[-1],
            'pred_smpl_rhand_pose': outputs_smpl_rhand_pose_list[-1],
            'pred_smpl_lhand_pose': outputs_smpl_lhand_pose_list[-1],
            'pred_smpl_jaw_pose': outputs_smpl_jaw_pose_list[-1],
            'pred_smpl_expr': outputs_smpl_expr_list[-1],
            'pred_smpl_beta': outputs_smpl_beta_list[-1],  # [B, 100, 10]
            'pred_smpl_cam': outputs_smpl_cam_list[-1],
            # 'pred_smpl_cam_f': outputs_smpl_cam_f_list[-1],
            'pred_smpl_kp3d': outputs_smpl_kp3d_list[-1]
        }
        if not self.training:
            full_pose = torch.cat((outputs_smpl_pose_list[-1],
                               outputs_smpl_lhand_pose_list[-1],
                               outputs_smpl_rhand_pose_list[-1],
                               outputs_smpl_jaw_pose_list[-1]),dim=2)
            bs,num_q,_,_,_ = full_pose.shape
            full_pose = rotmat_to_aa(full_pose).reshape(bs,num_q,53*3)
            out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_body_bbox_list[-1],
            'pred_lhand_boxes': outputs_lhand_bbox_list[-1],
            'pred_rhand_boxes': outputs_rhand_bbox_list[-1],
            'pred_face_boxes': outputs_face_bbox_list[-1],
            'pred_keypoints': outputs_body_keypoints_list[-1],
            'pred_lhand_keypoints': outputs_lhand_keypoints_list[-1],
            'pred_rhand_keypoints': outputs_rhand_keypoints_list[-1],
            'pred_face_keypoints': outputs_face_keypoints_list[-1],
            'pred_smpl_pose': outputs_smpl_pose_list[-1],
            'pred_smpl_rhand_pose': outputs_smpl_rhand_pose_list[-1],
            'pred_smpl_lhand_pose': outputs_smpl_lhand_pose_list[-1],
            'pred_smpl_jaw_pose': outputs_smpl_jaw_pose_list[-1],
            'pred_smpl_expr': outputs_smpl_expr_list[-1],
            'pred_smpl_beta': outputs_smpl_beta_list[-1],  # [B, 100, 10]
            'pred_smpl_cam': outputs_smpl_cam_list[-1],
            'pred_smpl_kp3d': outputs_smpl_kp3d_list[-1],
            'pred_smpl_verts': outputs_smpl_verts_list[-1],
            'pred_smpl_fullpose': full_pose
        }

        if self.dn_number > 0 and dn_mask_dict is not None:
            out.update({
                'dn_class_input': dn_class_input,
                'dn_bbox_input': dn_bbox_input,
                'dn_class_pred': dn_class_pred[-1],
                'dn_bbox_pred': dn_bbox_pred[-1],
                'num_tgt': dn_mask_dict['pad_size']
            })

        if self.aux_loss:
            out['aux_outputs'] = \
                self._set_aux_loss(
                    outputs_class,
                    outputs_body_bbox_list,
                    outputs_lhand_bbox_list,
                    outputs_rhand_bbox_list,
                    outputs_face_bbox_list,
                    outputs_body_keypoints_list,
                    outputs_lhand_keypoints_list,
                    outputs_rhand_keypoints_list,
                    outputs_face_keypoints_list,
                    outputs_smpl_pose_list,
                    outputs_smpl_rhand_pose_list,
                    outputs_smpl_lhand_pose_list,
                    outputs_smpl_jaw_pose_list,
                    outputs_smpl_expr_list,
                    outputs_smpl_beta_list,
                    outputs_smpl_cam_list,
                    # outputs_smpl_cam_f_list,
                    outputs_smpl_kp3d_list
                ) # with key pred_logits, pred_bbox, pred_keypoints
            if self.dn_number > 0 and dn_mask_dict is not None:
                assert len(dn_class_pred[:-1]) == len(
                    dn_bbox_pred[:-1]) == len(out['aux_outputs'])
                for aux_out, dn_class_pred_i, dn_bbox_pred_i in zip(
                        out['aux_outputs'], dn_class_pred, dn_bbox_pred):
                    aux_out.update({
                        'dn_class_input': dn_class_input,
                        'dn_bbox_input': dn_bbox_input,
                        'dn_class_pred': dn_class_pred_i,
                        'dn_bbox_pred': dn_bbox_pred_i,
                        'num_tgt': dn_mask_dict['pad_size']
                    })
        # for encoder output
        if hs_enc is not None:
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            interm_pose = torch.zeros_like(outputs_body_keypoints_list[0])
            out['interm_outputs'] = {
                'pred_logits': interm_class,
                'pred_boxes': interm_coord,
                'pred_keypoints': interm_pose
            }

        return out, targets, data_batch

    @torch.jit.unused
    def _set_aux_loss(self, 
                      outputs_class, 
                      outputs_body_coord, 
                      outputs_lhand_coord,
                      outputs_rhand_coord,
                      outputs_face_coord,
                      outputs_body_keypoints,
                      outputs_lhand_keypoints,
                      outputs_rhand_keypoints,
                      outputs_face_keypoints,
                      outputs_smpl_pose, 
                      outputs_smpl_rhand_pose,
                      outputs_smpl_lhand_pose, 
                      outputs_smpl_jaw_pose,
                      outputs_smpl_expr, 
                      outputs_smpl_beta, 
                      outputs_smpl_cam,
                    #   outputs_smpl_cam_f,
                      outputs_smpl_kp3d):

        return [{
            'pred_logits': a,
            'pred_boxes': b,
            'pred_lhand_boxes': c,
            'pred_rhand_boxes': d,
            'pred_face_boxes': e,
            'pred_keypoints': f,
            'pred_lhand_keypoints': g,
            'pred_rhand_keypoints': h,
            'pred_face_keypoints': i,
            'pred_smpl_pose': j,
            'pred_smpl_rhand_pose': k,
            'pred_smpl_lhand_pose': l,
            'pred_smpl_jaw_pose': m,
            'pred_smpl_expr': n,
            'pred_smpl_beta': o,
            'pred_smpl_cam': p,
            'pred_smpl_kp3d': q
        } for a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q in zip(
            outputs_class[:-1], 
            outputs_body_coord[:-1],
            outputs_lhand_coord[:-1],
            outputs_rhand_coord[:-1],
            outputs_face_coord[:-1],
            outputs_body_keypoints[:-1],
            outputs_lhand_keypoints[:-1],
            outputs_rhand_keypoints[:-1],
            outputs_face_keypoints[:-1],
            outputs_smpl_pose[:-1], 
            outputs_smpl_rhand_pose[:-1],
            outputs_smpl_lhand_pose[:-1], 
            outputs_smpl_jaw_pose[:-1],
            outputs_smpl_expr[:-1], 
            outputs_smpl_beta[:-1],
            outputs_smpl_cam[:-1], 
            outputs_smpl_kp3d[:-1])]

    def prepare_targets(self, data_batch):

        data_batch_coco = []
        instance_dict = {}
        img_list = data_batch['img'].float()
        batch_size, _, input_img_h, input_img_w = img_list.shape
        device = img_list.device
        masks = torch.ones((batch_size, input_img_h, input_img_w),
                           dtype=torch.bool,
                           device=device)
        

        # cv2.imread(data_batch['img_metas'][img_id]['image_path']).shape
        for img_id in range(batch_size):
            img_h, img_w = data_batch['img_shape'][img_id]
            masks[img_id, :img_h, :img_w] = 0
            
            if not self.inference:
                instance_body_bbox = torch.cat([data_batch['body_bbox_center'][img_id],\
                                                data_batch['body_bbox_size'][img_id]],dim=-1)
                instance_face_bbox = torch.cat([data_batch['face_bbox_center'][img_id],\
                                                data_batch['face_bbox_size'][img_id]],dim=-1)
                instance_lhand_bbox = torch.cat([data_batch['lhand_bbox_center'][img_id],\
                                                data_batch['lhand_bbox_size'][img_id]],dim=-1)
                instance_rhand_bbox = torch.cat([data_batch['rhand_bbox_center'][img_id],\
                                                data_batch['rhand_bbox_size'][img_id]],dim=-1)

                instance_kp2d = data_batch['joint_img'][img_id].clone().float()
                instance_kp2d_mask = data_batch['joint_trunc'][img_id].clone().float()
                instance_kp2d[:,:,2:] = instance_kp2d_mask
                body_kp2d, _  = convert_kps(instance_kp2d, 'smplx_137', 'coco', approximate=True)
                lhand_kp2d, _  = convert_kps(instance_kp2d, 'smplx_137', 'smplx_lhand', approximate=True)
                rhand_kp2d, _  = convert_kps(instance_kp2d, 'smplx_137', 'smplx_rhand', approximate=True)
                face_kp2d, _  = convert_kps(instance_kp2d, 'smplx_137', 'smplx_face', approximate=True)
                body_kp2d[:,:,0] = body_kp2d[:,:,0]/cfg.output_hm_shape[2]
                body_kp2d[:,:,1] = body_kp2d[:,:,1]/cfg.output_hm_shape[1]
                body_kp2d = torch.cat([body_kp2d[:,:,:2].flatten(1),body_kp2d[:,:,2]],dim=-1)

                lhand_kp2d[:,:,0] = lhand_kp2d[:,:,0]/cfg.output_hm_shape[2]
                lhand_kp2d[:,:,1] = lhand_kp2d[:,:,1]/cfg.output_hm_shape[1]
                lhand_kp2d = torch.cat([lhand_kp2d[:,:,:2].flatten(1),lhand_kp2d[:,:,2]],dim=-1)
                
                rhand_kp2d[:,:,0] = rhand_kp2d[:,:,0]/cfg.output_hm_shape[2]
                rhand_kp2d[:,:,1] = rhand_kp2d[:,:,1]/cfg.output_hm_shape[1]
                rhand_kp2d = torch.cat([rhand_kp2d[:,:,:2].flatten(1),rhand_kp2d[:,:,2]],dim=-1)

                face_kp2d[:,:,0] = face_kp2d[:,:,0]/cfg.output_hm_shape[2]
                face_kp2d[:,:,1] = face_kp2d[:,:,1]/cfg.output_hm_shape[1]
                face_kp2d = torch.cat([face_kp2d[:,:,:2].flatten(1),face_kp2d[:,:,2]],dim=-1)
                
                instance_dict = {}
                instance_dict['boxes'] = instance_body_bbox.float()
                instance_dict['face_boxes'] = instance_face_bbox.float()
                instance_dict['lhand_boxes'] = instance_lhand_bbox.float()
                instance_dict['rhand_boxes'] = instance_rhand_bbox.float()
                instance_dict['keypoints'] = body_kp2d.float()
                instance_dict['lhand_keypoints'] = lhand_kp2d.float()
                instance_dict['rhand_keypoints'] = rhand_kp2d.float()
                instance_dict['face_keypoints'] = face_kp2d.float()
            
                # instance_dict['orig_size'] = data_batch['ori_shape'][img_id]
                instance_dict['size'] = data_batch['img_shape'][img_id]  # after augmentation 
                
                instance_dict['area'] = instance_body_bbox[:, 2] * instance_body_bbox[:, 3]
                instance_dict['lhand_area'] = instance_lhand_bbox[:, 2] * instance_lhand_bbox[:, 3]
                instance_dict['rhand_area'] = instance_rhand_bbox[:, 2] * instance_rhand_bbox[:, 3]
                instance_dict['face_area'] = instance_face_bbox[:, 2] * instance_face_bbox[:, 3]

                instance_dict['labels'] = torch.ones(instance_body_bbox.shape[0],
                                                    dtype=torch.long,
                                                    device=device)
                data_batch_coco.append(instance_dict)               
            else:
                instance_body_bbox = torch.cat([data_batch['body_bbox_center'][img_id],\
                                                data_batch['body_bbox_size'][img_id]],dim=-1)
                instance_dict = {}
                # instance_dict['orig_size'] = data_batch['ori_shape'][img_id]
                instance_dict['size'] = data_batch['img_shape'][img_id]  # after augmentation 
                instance_dict['boxes'] = instance_body_bbox.float()    
                     
                data_batch_coco.append(instance_dict)  

        input_img = NestedTensor(img_list, masks)
        return input_img, data_batch_coco



@MODULE_BUILD_FUNCS.registe_with_name(module_name='aios_smplx')
def build_aios_smplx(args, cfg):
    # pdb.set_trace()
    num_classes = args.num_classes  # 2
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_class_embed_share = args.dec_pred_class_embed_share
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share

    if args.eval:
        body_model = args.body_model_test
        train = False
    else:
        body_model = args.body_model_train
        train = True
        
    model = AiOSSMPLX(
        backbone,
        transformer,
        num_classes=num_classes,  # 2
        num_queries=args.num_queries,  # 900
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,  # False
        fix_refpoints_hw=args.fix_refpoints_hw,  # -1
        num_feature_levels=args.num_feature_levels,  # 4
        nheads=args.nheads,  # 8
        dec_pred_class_embed_share=dec_pred_class_embed_share,  # false
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,  # False
        # two stage
        two_stage_type=args.two_stage_type,

        # box_share
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,  # False
        two_stage_class_embed_share=args.two_stage_class_embed_share,  # False
        dn_number=args.dn_number if args.use_dn else 0,  # 100
        dn_box_noise_scale=args.dn_box_noise_scale,  # 0.4
        dn_label_noise_ratio=args.dn_label_noise_ratio,  # 0.5
        dn_batch_gt_fuse=args.dn_batch_gt_fuse,  # false
        dn_attn_mask_type_list=args.dn_attn_mask_type_list,
        dn_labelbook_size=dn_labelbook_size,  # 100
        cls_no_bias=args.cls_no_bias,  # False
        num_group=args.num_group,  # 100
        num_body_points=args.num_body_points,  # 17
        num_hand_points=args.num_hand_points,  # 17
        num_face_points=args.num_face_points,  # 17
        num_box_decoder_layers=args.num_box_decoder_layers,  # 2
        num_hand_face_decoder_layers=args.num_hand_face_decoder_layers,
        # smpl_convention=convention
        body_model=body_model,
        train=train,
        inference=args.inference)
    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {
        'loss_ce': args.cls_loss_coef,  # 2
        # bbox
        'loss_body_bbox': args.body_bbox_loss_coef,  # 5
        'loss_rhand_bbox': args.rhand_bbox_loss_coef,  # 5
        'loss_lhand_bbox': args.lhand_bbox_loss_coef,  # 5
        'loss_face_bbox': args.face_bbox_loss_coef,  # 5
        # bbox giou
        'loss_body_giou': args.body_giou_loss_coef,  # 2
        'loss_rhand_giou': args.rhand_giou_loss_coef,  # 2
        'loss_lhand_giou': args.lhand_giou_loss_coef,  # 2
        'loss_face_giou': args.face_giou_loss_coef,  # 2
        # 2d kp
        'loss_keypoints': args.keypoints_loss_coef,  # 10
        'loss_rhand_keypoints': args.rhand_keypoints_loss_coef,  # 10
        'loss_lhand_keypoints': args.lhand_keypoints_loss_coef,  # 10
        'loss_face_keypoints': args.face_keypoints_loss_coef,  # 10
        # 2d kp oks
        'loss_oks': args.oks_loss_coef,  # 4
        'loss_rhand_oks': args.rhand_oks_loss_coef,  # 4
        'loss_lhand_oks': args.lhand_oks_loss_coef,  # 4
        'loss_face_oks': args.face_oks_loss_coef,  # 4
        # smpl param
        'loss_smpl_pose_root': args.smpl_pose_loss_root_coef,  # 0
        'loss_smpl_pose_body': args.smpl_pose_loss_body_coef,  # 0
        'loss_smpl_pose_lhand': args.smpl_pose_loss_lhand_coef,  # 0
        'loss_smpl_pose_rhand': args.smpl_pose_loss_rhand_coef,  # 0
        'loss_smpl_pose_jaw': args.smpl_pose_loss_jaw_coef,  # 0
        'loss_smpl_beta': args.smpl_beta_loss_coef,  # 0
        'loss_smpl_expr': args.smpl_expr_loss_coef, 
        # smpl kp3d ra
        'loss_smpl_body_kp3d_ra': args.smpl_body_kp3d_ra_loss_coef,  # 0
        'loss_smpl_lhand_kp3d_ra': args.smpl_lhand_kp3d_ra_loss_coef,  # 0
        'loss_smpl_rhand_kp3d_ra': args.smpl_rhand_kp3d_ra_loss_coef,  # 0
        'loss_smpl_face_kp3d_ra': args.smpl_face_kp3d_ra_loss_coef,  # 0
        # smpl kp3d
        'loss_smpl_body_kp3d': args.smpl_body_kp3d_loss_coef,  # 0
        'loss_smpl_face_kp3d': args.smpl_face_kp3d_loss_coef,  # 0
        'loss_smpl_lhand_kp3d': args.smpl_lhand_kp3d_loss_coef,  # 0
        'loss_smpl_rhand_kp3d': args.smpl_rhand_kp3d_loss_coef,  # 0
        # smpl kp2d
        'loss_smpl_body_kp2d': args.smpl_body_kp2d_loss_coef,  # 0
        'loss_smpl_lhand_kp2d': args.smpl_lhand_kp2d_loss_coef,  # 0
        'loss_smpl_rhand_kp2d': args.smpl_rhand_kp2d_loss_coef,  # 0
        'loss_smpl_face_kp2d': args.smpl_face_kp2d_loss_coef,  # 0
        
        # smpl kp2d ba
        'loss_smpl_body_kp2d_ba': args.smpl_body_kp2d_ba_loss_coef,
        'loss_smpl_face_kp2d_ba': args.smpl_face_kp2d_ba_loss_coef,
        'loss_smpl_lhand_kp2d_ba': args.smpl_lhand_kp2d_ba_loss_coef,
        'loss_smpl_rhand_kp2d_ba': args.smpl_rhand_kp2d_ba_loss_coef,
        
    }

    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    if args.use_dn:
        weight_dict.update({
            'dn_loss_ce':
            args.dn_label_coef,  # 0.3
            'dn_loss_bbox':
            args.bbox_loss_coef * args.dn_bbox_coef,  # 5 * 0.5
            'dn_loss_giou':
            args.giou_loss_coef * args.dn_bbox_coef,  # 2 * 0.5
        })

    clean_weight_dict = copy.deepcopy(weight_dict)

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):  # from 0 t 4 # ???
            for k, v in clean_weight_dict.items():
                if i < args.num_box_decoder_layers and ('keypoints' in k or 'oks' in k):
                    continue
                if i < args.num_box_decoder_layers and k in [
                    'loss_rhand_bbox', 'loss_lhand_bbox', 'loss_face_bbox',
                    'loss_rhand_giou', 'loss_lhand_giou', 'loss_face_giou']:
                    continue
                if i < args.num_hand_face_decoder_layers and k in [
                    'loss_rhand_keypoints', 'loss_lhand_keypoints', 
                    'loss_face_keypoints', 'loss_rhand_oks',
                    'loss_lhand_oks', 'loss_face_oks']:
                    continue
                if i < args.num_box_decoder_layers and 'smpl' in k:
                    continue
                aux_weight_dict.update({k + f'_{i}': v})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            # bbox
            'loss_body_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_rhand_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_lhand_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_face_bbox': 1.0 if not no_interm_box_loss else 0.0,
            # bbox giou
            'loss_body_giou': 1.0 if not no_interm_box_loss else 0.0,
            'loss_rhand_giou': 1.0 if not no_interm_box_loss else 0.0,
            'loss_lhand_giou': 1.0 if not no_interm_box_loss else 0.0,
            'loss_face_giou': 1.0 if not no_interm_box_loss else 0.0,
            # 2d kp
            'loss_keypoints': 1.0 if not no_interm_box_loss else 0.0,
            'loss_rhand_keypoints': 1.0 if not no_interm_box_loss else 0.0,
            'loss_lhand_keypoints': 1.0 if not no_interm_box_loss else 0.0,
            'loss_face_keypoints': 1.0 if not no_interm_box_loss else 0.0,
            # 2d oks
            'loss_oks': 1.0 if not no_interm_box_loss else 0.0,
            'loss_rhand_oks': 1.0 if not no_interm_box_loss else 0.0,
            'loss_lhand_oks': 1.0 if not no_interm_box_loss else 0.0,
            'loss_face_oks': 1.0 if not no_interm_box_loss else 0.0,
            # smpl param
            'loss_smpl_pose_root': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_pose_body': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_pose_lhand': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_pose_rhand': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_pose_jaw': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_beta': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_expr': 1.0 if not no_interm_box_loss else 0.0,
            # smpl kp3d ra
            'loss_smpl_body_kp3d_ra': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_lhand_kp3d_ra': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_rhand_kp3d_ra': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_face_kp3d_ra': 1.0 if not no_interm_box_loss else 0.0,
            # smpl kp3d
            'loss_smpl_body_kp3d': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_face_kp3d': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_lhand_kp3d': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_rhand_kp3d': 1.0 if not no_interm_box_loss else 0.0,
            # smpl kp2d
            'loss_smpl_body_kp2d': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_lhand_kp2d': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_rhand_kp2d': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_face_kp2d': 1.0 if not no_interm_box_loss else 0.0,
            # smpl kp2d ba
            'loss_smpl_body_kp2d_ba': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_lhand_kp2d_ba': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_rhand_kp2d_ba': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_face_kp2d_ba': 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef  # 1
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update({
            k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k]
            for k, v in clean_weight_dict_wo_dn.items() if 'keypoints' not in k
        })
        weight_dict.update(interm_weight_dict)

        interm_weight_dict.update({
            k + f'_query_expand': v * interm_loss_coef * _coeff_weight_dict[k]
            for k, v in clean_weight_dict_wo_dn.items()
        })  # ???
        weight_dict.update(interm_weight_dict)

    losses = cfg.losses
    
    if args.dn_number > 0:
        losses += ['dn_label', 'dn_bbox']
    losses += ['matching']

    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=args.focal_alpha,
        losses=losses,
        num_box_decoder_layers=args.num_box_decoder_layers,
        num_hand_face_decoder_layers=args.num_hand_face_decoder_layers,
        num_body_points=args.num_body_points,
        num_hand_points=args.num_hand_points,
        num_face_points=args.num_face_points,
        )

    criterion.to(device)
    if args.inference:
        postprocessors = {
            'bbox': 
                PostProcess_SMPLX_Multi_Infer(
                    num_select=args.num_select, 
                    nms_iou_threshold=args.nms_iou_threshold,
                    num_body_points=args.num_body_points),
        }
    else:
        postprocessors = {
            'bbox': 
                PostProcess_SMPLX(
                    num_select=args.num_select, 
                    nms_iou_threshold=args.nms_iou_threshold,
                    num_body_points=args.num_body_points),
        }
    postprocessors_aios = {
        'bbox':
        PostProcess_aios(num_select=args.num_select,
                           nms_iou_threshold=args.nms_iou_threshold,
                           num_body_points=args.num_body_points),
    }
    # criterion_smpl=build_architecture(cfg['smpl_loss'])
    return model, criterion, postprocessors, postprocessors_aios



class AiOSSMPLX_Box(nn.Module):
    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        aux_loss=False,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=False,
        fix_refpoints_hw=-1,
        num_feature_levels=1,
        nheads=8,
        two_stage_type='no',
        dec_pred_class_embed_share=False,
        dec_pred_bbox_embed_share=False,
        dec_pred_pose_embed_share=False,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_batch_gt_fuse=False,
        dn_labelbook_size=100,
        dn_attn_mask_type_list=['group2group'],
        cls_no_bias=False,
        num_group=100,
        num_body_points=0,
        num_hand_points=0,
        num_face_points=0,
        num_box_decoder_layers=2,
        num_hand_face_decoder_layers=4,
        body_model=dict(
            type='smplx',
            keypoint_src='smplx',
            num_expression_coeffs=10,
            keypoint_dst='smplx_137',
            model_path='data/body_models/smplx',
            use_pca=False,
            use_face_contour=True),
        train=True,
        inference=False,
        focal_length=[5000., 5000.],
        camera_3d_size=2.5
    ):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)
        self.num_body_points = num_body_points
        self.num_hand_points = num_hand_points
        self.num_face_points = num_face_points
        self.num_whole_body_points = num_body_points + 2*num_hand_points + num_face_points
        self.num_box_decoder_layers = num_box_decoder_layers
        self.num_hand_face_decoder_layers = num_hand_face_decoder_layers
        self.focal_length = focal_length
        self.camera_3d_size=camera_3d_size
        self.inference = inference
        if train:
            self.smpl_convention = 'smplx'
        else:
            self.smpl_convention = 'h36m'
        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy  # False
        self.fix_refpoints_hw = fix_refpoints_hw  # -1

        # for dn training
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_batch_gt_fuse = dn_batch_gt_fuse
        self.dn_labelbook_size = dn_labelbook_size
        self.dn_attn_mask_type_list = dn_attn_mask_type_list
        assert all([
            i in ['match2dn', 'dn2dn', 'group2group']
            for i in dn_attn_mask_type_list
        ])
        assert not dn_batch_gt_fuse

        # build human body
        # if train:
        #     self.body_model = build_body_model(body_model)
        if inference:
            body_model=dict(
                type='smplx',
                keypoint_src='smplx',
                num_expression_coeffs=10,
                num_betas=10,
                keypoint_dst='smplx',
                model_path='data/body_models/smplx',
                use_pca=False,
                use_face_contour=True)
        self.body_model = build_body_model(body_model)
        for param in self.body_model.parameters():
            param.requires_grad = False       
        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)  # 3
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels,
                                  hidden_dim,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == 'no', 'two_stage_type should be no if num_feature_levels=1 !!!'
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1],
                              hidden_dim,
                              kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            ])

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, 'Why not iter_update?'

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share  # false
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share  # false

        # 1.1 prepare class & box embed
        _class_embed = nn.Linear(hidden_dim,
                                 num_classes,
                                 bias=(not cls_no_bias))
        if not cls_no_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _class_embed.bias.data = torch.ones(self.num_classes) * bias_value

        # 1.2 box embed layer list
        if dec_pred_class_embed_share:
            class_embed_layerlist = [
                _class_embed for i in range(transformer.num_decoder_layers)
            ]
        else:
            class_embed_layerlist = [
                copy.deepcopy(_class_embed)
                for i in range(transformer.num_decoder_layers)
            ]


        ###########################################################################
        #                    body bbox + l/r hand box + face box
        ###########################################################################
        # 1.1 body bbox embed
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        
        # 1.2 body bbox embed layer list
        self.num_group = num_group
        if dec_pred_bbox_embed_share:
            box_body_embed_layerlist = [
                _bbox_embed for i in range(transformer.num_decoder_layers)
            ]
        else:
            box_body_embed_layerlist = [
                copy.deepcopy(_bbox_embed)
                for i in range(transformer.num_decoder_layers)
            ]

        # 2.1 lhand bbox embed
        _bbox_hand_embed = MLP(hidden_dim, hidden_dim, 2, 3) # TODO: the out shape should be 2 not 4
        nn.init.constant_(_bbox_hand_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_hand_embed.layers[-1].bias.data, 0)

        _bbox_hand_hw_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_bbox_hand_hw_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_hand_hw_embed.layers[-1].bias.data, 0)
        # 2.2 lhand bbox embed layer list
        if dec_pred_pose_embed_share:
            box_hand_embed_layerlist = \
                [_bbox_hand_embed for i in range(transformer.num_decoder_layers - num_box_decoder_layers+1)]
        else:
            box_hand_embed_layerlist = [
                copy.deepcopy(_bbox_hand_embed)
                for i in range(transformer.num_decoder_layers -
                            num_box_decoder_layers + 1)
            ]

        if dec_pred_pose_embed_share:
            box_hand_hw_embed_layerlist = [
                _bbox_hand_hw_embed for i in range(
                    transformer.num_decoder_layers - num_box_decoder_layers)
                ]
        else:
            box_hand_hw_embed_layerlist = [
                copy.deepcopy(_bbox_hand_hw_embed)
                for i in range(transformer.num_decoder_layers -
                            num_box_decoder_layers)
            ]
                        
        # 4.1 face bbox embed
        _bbox_face_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_bbox_face_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_face_embed.layers[-1].bias.data, 0)

        _bbox_face_hw_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_bbox_face_hw_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_face_hw_embed.layers[-1].bias.data, 0)
        
        # 4.2 face bbox embed layer list
        if dec_pred_pose_embed_share:
            box_face_embed_layerlist = [
                _bbox_face_embed for i in range(
                    transformer.num_decoder_layers - num_box_decoder_layers + 1)
                ]
        else:
            box_face_embed_layerlist = [
                copy.deepcopy(_bbox_face_embed)
                for i in range(transformer.num_decoder_layers -
                            num_box_decoder_layers + 1)
            ]

        if dec_pred_pose_embed_share:
            box_face_hw_embed_layerlist = [
                _bbox_face_hw_embed for i in range(
                    transformer.num_decoder_layers - num_box_decoder_layers)]
        else:
            box_face_hw_embed_layerlist = [
                copy.deepcopy(_bbox_face_hw_embed)
                for i in range(transformer.num_decoder_layers -
                            num_box_decoder_layers)
            ]            
        
        # 1. smpl pose embed
        if body_model['type'].upper()=='SMPL':
            self.body_model_joint_num = 24
        elif body_model['type'].upper()=='SMPLX':
            self.body_model_joint_num = 22
        else:
            raise ValueError(
            f'Only supports SMPL or SMPLX, but get {body_model.type}')      
        #TODO: 

        _smpl_pose_embed = MLP(hidden_dim *  4, hidden_dim, self.body_model_joint_num * 6, 3)
        nn.init.constant_(_smpl_pose_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_smpl_pose_embed.layers[-1].bias.data, 0)  

        if dec_pred_bbox_embed_share:
            smpl_pose_embed_layerlist = [
                _smpl_pose_embed
                for i in range(transformer.num_decoder_layers -
                               num_box_decoder_layers)
            ]
        else:
            smpl_pose_embed_layerlist = [
                copy.deepcopy(_smpl_pose_embed)
                for i in range(transformer.num_decoder_layers -
                               num_box_decoder_layers)
            ]

        # 2. smpl betas embed
        _smpl_beta_embed = MLP(hidden_dim * 4, hidden_dim, 10, 3)
        nn.init.constant_(_smpl_beta_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_smpl_beta_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            smpl_beta_embed_layerlist = [
                _smpl_beta_embed
                for i in range(transformer.num_decoder_layers -
                               num_box_decoder_layers)
            ]
        else:
            smpl_beta_embed_layerlist = [
                copy.deepcopy(_smpl_beta_embed)
                for i in range(transformer.num_decoder_layers -
                               num_box_decoder_layers)
            ]

        # 3. smpl cam embed
        _cam_embed = MLP(hidden_dim * 4, hidden_dim, 3, 3)
        nn.init.constant_(_cam_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_cam_embed.layers[-1].bias.data, 0)
        
        if dec_pred_bbox_embed_share:
            cam_embed_layerlist = [
                _cam_embed for i in range(transformer.num_decoder_layers -
                                          num_box_decoder_layers)
            ]
        else:
            cam_embed_layerlist = [
                copy.deepcopy(_cam_embed)
                for i in range(transformer.num_decoder_layers -
                               num_box_decoder_layers)
            ]
        
        ###########################################################################
        #  smplx body pose + hand pose + expression + betas + kp2d + kp3d + cam
        ###########################################################################

        # 2. smplx hand pose embed
        _smplx_hand_pose_embed_layer_2_3 = \
            MLP(hidden_dim * 2, hidden_dim, 15 * 6, 3)
        nn.init.constant_(_smplx_hand_pose_embed_layer_2_3.layers[-1].weight.data, 0)
        nn.init.constant_(_smplx_hand_pose_embed_layer_2_3.layers[-1].bias.data, 0)
        
        _smplx_hand_pose_embed_layer_4_5 = \
            MLP(hidden_dim * 2, hidden_dim, 15 * 6, 3)
        nn.init.constant_(_smplx_hand_pose_embed_layer_4_5.layers[-1].weight.data, 0)
        nn.init.constant_(_smplx_hand_pose_embed_layer_4_5.layers[-1].bias.data, 0)


        
        if dec_pred_bbox_embed_share:
            smplx_hand_pose_embed_layerlist = [
                _smplx_hand_pose_embed_layer_2_3
                if i<2 else _smplx_hand_pose_embed_layer_4_5
                for i in range(transformer.num_decoder_layers -
                               num_box_decoder_layers)
            ]
        else:
            smplx_hand_pose_embed_layerlist = [
                copy.deepcopy(_smplx_hand_pose_embed_layer_2_3)
                if i<2 else copy.deepcopy(_smplx_hand_pose_embed_layer_4_5)
                for i in range(transformer.num_decoder_layers -
                               num_box_decoder_layers)
            ]


        # 3. smplx face expression 

        _smplx_expression_embed_layer_2_3 = \
            MLP(hidden_dim*2, hidden_dim, 10, 3)
        nn.init.constant_(_smplx_expression_embed_layer_2_3.layers[-1].weight.data, 0)
        nn.init.constant_(_smplx_expression_embed_layer_2_3.layers[-1].bias.data, 0)
        
        _smplx_expression_embed_layer_4_5 = \
            MLP(hidden_dim * 2, hidden_dim, 10, 3)
        nn.init.constant_(_smplx_expression_embed_layer_4_5.layers[-1].weight.data, 0)
        nn.init.constant_(_smplx_expression_embed_layer_4_5.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            smplx_expression_embed_layerlist = [
                _smplx_expression_embed_layer_2_3
                if i<2 else _smplx_expression_embed_layer_4_5
                for i in range(transformer.num_decoder_layers -
                               num_box_decoder_layers)
            ]
        else:
            smplx_expression_embed_layerlist = [
                copy.deepcopy(_smplx_expression_embed_layer_2_3)
                if i<2 else copy.deepcopy(_smplx_expression_embed_layer_4_5)
                for i in range(transformer.num_decoder_layers -
                               num_box_decoder_layers)
            ]

        # 4. smplx jaw pose embed
        _smplx_jaw_embed_2_3 = MLP(hidden_dim * 2, hidden_dim, 6, 3)
        nn.init.constant_(_smplx_jaw_embed_2_3.layers[-1].weight.data, 0)
        nn.init.constant_(_smplx_jaw_embed_2_3.layers[-1].bias.data, 0)
        
        _smplx_jaw_embed_4_5 = MLP(hidden_dim * 2, hidden_dim, 6, 3)
        nn.init.constant_(_smplx_jaw_embed_4_5.layers[-1].weight.data, 0)
        nn.init.constant_(_smplx_jaw_embed_4_5.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            smplx_jaw_embed_layerlist = [
                _smplx_jaw_embed_2_3 if i<2 else _smplx_jaw_embed_4_5
                for i in range(
                    transformer.num_decoder_layers - num_box_decoder_layers)
            ]
        else:
            smplx_jaw_embed_layerlist = [
                copy.deepcopy(_smplx_jaw_embed_2_3) 
                if i<2 else copy.deepcopy(_smplx_jaw_embed_4_5) 
                for i in range(
                    transformer.num_decoder_layers -  num_box_decoder_layers)
            ]
            
        self.bbox_embed = nn.ModuleList(box_body_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)

        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed
        
        # smpl
        self.smpl_pose_embed = nn.ModuleList(smpl_pose_embed_layerlist)
        self.smpl_beta_embed = nn.ModuleList(smpl_beta_embed_layerlist)
        self.smpl_cam_embed = nn.ModuleList(cam_embed_layerlist)

        # smplx lhand kp
        self.bbox_hand_embed = nn.ModuleList(box_hand_embed_layerlist)
        self.bbox_hand_hw_embed = nn.ModuleList(box_hand_hw_embed_layerlist)

        self.transformer.decoder.bbox_hand_embed = self.bbox_hand_embed
        self.transformer.decoder.bbox_hand_hw_embed = self.bbox_hand_hw_embed

        # smplx face kp
        self.bbox_face_embed = nn.ModuleList(box_face_embed_layerlist)
        self.bbox_face_hw_embed = nn.ModuleList(box_face_hw_embed_layerlist)

        self.transformer.decoder.bbox_face_embed = self.bbox_face_embed
        self.transformer.decoder.bbox_face_hw_embed = self.bbox_face_hw_embed

        # smplx 
        self.smpl_hand_pose_embed = nn.ModuleList(smplx_hand_pose_embed_layerlist)

        self.smpl_expr_embed = nn.ModuleList(smplx_expression_embed_layerlist)
        self.smpl_jaw_embed = nn.ModuleList(smplx_jaw_embed_layerlist)

        self.transformer.decoder.num_hand_face_decoder_layers = num_hand_face_decoder_layers
        self.transformer.decoder.num_box_decoder_layers = num_box_decoder_layers
        self.transformer.decoder.num_body_points = num_body_points
        self.transformer.decoder.num_hand_points = num_hand_points
        self.transformer.decoder.num_face_points = num_face_points
        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in [
            'no', 'standard'
        ], 'unknown param {} of two_stage_type'.format(two_stage_type)
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(
                    _bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed

            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(
                    _class_embed)
            self.refpoint_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def prepare_for_dn2(self, targets):
        if not self.training:
            device = targets[0]['boxes'].device
            bs = len(targets)
            
            num_points = 4
            attn_mask2 = torch.zeros(
                bs,
                self.nheads,
                self.num_group * 4,
                self.num_group * 4,
                device=device,
                dtype=torch.bool)

            group_bbox_kpt = 4
            # body bbox index
            kpt_index = [x for x in range(self.num_group * 4) if x % 4 in [0]]
            
            for matchj in range(self.num_group * 4):
                sj = (matchj // group_bbox_kpt) * group_bbox_kpt
                ej = (matchj // group_bbox_kpt + 1)*group_bbox_kpt
                
                # for each instance, they should associate with their query (body hand face)
                if sj > 0:
                    attn_mask2[:, :, matchj, :sj] = True
                if ej < self.num_group * 4:
                    attn_mask2[:, :, matchj, ej:] = True

            for match_x in range(self.num_group * 4):
                if match_x % group_bbox_kpt in [0, 1, 2, 3]:
                    # each query (hand face body) should associate with all body query
                    attn_mask2[:,:,match_x, kpt_index]=False

            num_points = 4
            attn_mask3 = torch.zeros(
                bs,
                self.nheads,
                self.num_group * 4, 
                self.num_group * 4,
                device=device, 
                dtype=torch.bool)

            group_bbox_kpt = 4
            kpt_index = [x for x in range(self.num_group * 4) if x % 4 in [0]]
            for matchj in range(self.num_group * 4):
                sj = (matchj // group_bbox_kpt) * group_bbox_kpt
                ej = (matchj // group_bbox_kpt + 1)*group_bbox_kpt
                # for each instance, they should associate with their query (body hand face)
                if sj > 0:
                    attn_mask3[:, :, matchj, :sj] = True
                if ej < self.num_group * 4:
                    attn_mask3[:, :, matchj, ej:] = True

            for match_x in range(self.num_group * 4):
                if match_x % group_bbox_kpt in [0, 1,  2, 3]:
                    # each query (hand face body) should associate with all body query
                    attn_mask3[:, :, match_x, kpt_index] = False

            attn_mask2 = attn_mask2.flatten(0, 1)
            attn_mask3 = attn_mask3.flatten(0, 1)
            return None, None, None, attn_mask2, attn_mask3, None

        # targets, dn_scalar, noise_scale = dn_args
        device = targets[0]['boxes'].device
        bs = len(targets)
        dn_number = self.dn_number  # 100
        dn_box_noise_scale = self.dn_box_noise_scale  # 0.4
        dn_label_noise_ratio = self.dn_label_noise_ratio  # 0.5

        # gather gt boxes and labels
        gt_boxes = [t['boxes'] for t in targets]
        gt_labels = [t['labels'] for t in targets]
        gt_keypoints = [t['keypoints'] for t in targets]

        # repeat them
        def get_indices_for_repeat(now_num, target_num, device='cuda'):
            """
            Input:
                - now_num: int
                - target_num: int
            Output:
                - indices: tensor[target_num]
            """
            out_indice = []
            base_indice = torch.arange(now_num).to(device)
            multiplier = target_num // now_num
            out_indice.append(base_indice.repeat(multiplier))
            residue = target_num % now_num
            out_indice.append(base_indice[torch.randint(0,
                                                        now_num, (residue, ),
                                                        device=device)])
            return torch.cat(out_indice)

        if self.dn_batch_gt_fuse:
            raise NotImplementedError
            gt_boxes_bsall = torch.cat(gt_boxes)  # num_boxes, 4
            gt_labels_bsall = torch.cat(gt_labels)
            num_gt_bsall = gt_boxes_bsall.shape[0]
            if num_gt_bsall > 0:
                indices = get_indices_for_repeat(num_gt_bsall, dn_number,
                                                 device)
                gt_boxes_expand = gt_boxes_bsall[indices][None].repeat(
                    bs, 1, 1)  # bs, num_dn, 4
                gt_labels_expand = gt_labels_bsall[indices][None].repeat(
                    bs, 1)  # bs, num_dn
            else:
                # all negative samples when no gt boxes
                gt_boxes_expand = torch.rand(bs, dn_number, 4, device=device)
                gt_labels_expand = torch.ones(
                    bs, dn_number, dtype=torch.int64, device=device) * int(
                        self.num_classes)
        else:
            gt_boxes_expand = []
            gt_labels_expand = []
            gt_keypoints_expand = []  # here
            for idx, (gt_boxes_i, gt_labels_i, gt_keypoint_i) in enumerate(
                    zip(gt_boxes, gt_labels, gt_keypoints)):  # idx -> batch id
                num_gt_i = gt_boxes_i.shape[0]  # instance num
                if num_gt_i > 0:
                    indices = get_indices_for_repeat(num_gt_i, dn_number,
                                                     device)
                    gt_boxes_expand_i = gt_boxes_i[indices]  # num_dn, 4
                    gt_labels_expand_i = gt_labels_i[indices]  # add smpl
                    gt_keypoints_expand_i = gt_keypoint_i[indices]
                else:
                    # all negative samples when no gt boxes
                    gt_boxes_expand_i = torch.rand(dn_number, 4, device=device)
                    gt_labels_expand_i = torch.ones(
                        dn_number, dtype=torch.int64, device=device) * int(
                            self.num_classes)
                    gt_keypoints_expand_i = torch.rand(dn_number,
                                                       self.num_body_points *
                                                       3,
                                                       device=device)
                gt_boxes_expand.append(gt_boxes_expand_i)  # add smpl
                gt_labels_expand.append(gt_labels_expand_i)
                gt_keypoints_expand.append(gt_keypoints_expand_i)
            gt_boxes_expand = torch.stack(gt_boxes_expand)
            gt_labels_expand = torch.stack(gt_labels_expand)
            gt_keypoints_expand = torch.stack(gt_keypoints_expand)
        knwon_boxes_expand = gt_boxes_expand.clone()
        knwon_labels_expand = gt_labels_expand.clone()

        # add noise
        if dn_label_noise_ratio > 0:
            prob = torch.rand_like(knwon_labels_expand.float())
            chosen_indice = prob < dn_label_noise_ratio
            new_label = torch.randint_like(
                knwon_labels_expand[chosen_indice], 0,
                self.dn_labelbook_size)  # randomly put a new one here
            knwon_labels_expand[chosen_indice] = new_label

        if dn_box_noise_scale > 0:
            diff = torch.zeros_like(knwon_boxes_expand)
            diff[..., :2] = knwon_boxes_expand[..., 2:] / 2
            diff[..., 2:] = knwon_boxes_expand[..., 2:]
            knwon_boxes_expand += torch.mul(
                (torch.rand_like(knwon_boxes_expand) * 2 - 1.0),
                diff) * dn_box_noise_scale
            knwon_boxes_expand = knwon_boxes_expand.clamp(min=0.0, max=1.0)

        input_query_label = self.label_enc(knwon_labels_expand)
        input_query_bbox = inverse_sigmoid(knwon_boxes_expand)

        # prepare mask

        if 'group2group' in self.dn_attn_mask_type_list:
            attn_mask = torch.zeros(bs,
                                    self.nheads,
                                    dn_number + self.num_queries,
                                    dn_number + self.num_queries,
                                    device=device,
                                    dtype=torch.bool)
            attn_mask[:, :, dn_number:, :dn_number] = True
            for idx, (gt_boxes_i, gt_labels_i) in enumerate(
                    zip(gt_boxes, gt_labels)):  # for batch
                num_gt_i = gt_boxes_i.shape[0]
                if num_gt_i == 0:
                    continue
                for matchi in range(dn_number):
                    si = (matchi // num_gt_i) * num_gt_i
                    ei = (matchi // num_gt_i + 1) * num_gt_i
                    if si > 0:
                        attn_mask[idx, :, matchi, :si] = True
                    if ei < dn_number:
                        attn_mask[idx, :, matchi, ei:dn_number] = True
            attn_mask = attn_mask.flatten(0, 1)

        if 'group2group' in self.dn_attn_mask_type_list:
            # self.num_body_points = self.num_body_points +3
            num_points = 4
            attn_mask2 = torch.zeros(
                bs,
                self.nheads,
                dn_number + self.num_group * 4,
                dn_number + self.num_group * 4,
                device=device,
                dtype=torch.bool)
            attn_mask2[:, :, dn_number:, :dn_number] = True
            group_bbox_kpt = 4

            for matchj in range(self.num_group * 4):
                sj = (matchj // group_bbox_kpt) * group_bbox_kpt
                ej = (matchj // group_bbox_kpt + 1)*group_bbox_kpt 
                # for each instance, they should associate their body, hand, and face bbox
                if sj > 0:
                    attn_mask2[:, :, dn_number:, dn_number:][:, :, matchj, :sj] = True
                if ej < self.num_group * 4:
                    attn_mask2[:, :, dn_number:, dn_number:][:, :, matchj, ej:] = True
            # body bbox index
            kpt_index = [x for x in range(self.num_group * 4) if x % 4 in [0]]
            for match_x in range(self.num_group * 4):
                if match_x % group_bbox_kpt in [0, 1,  2, 3]:
                    # for each instance, they should associate their each query with 
                    # other instances' body query
                    attn_mask2[:, :, dn_number:, dn_number:][:, :, match_x, kpt_index]=False

            for idx, (gt_boxes_i, gt_labels_i) in enumerate(zip(gt_boxes, gt_labels)):
                num_gt_i = gt_boxes_i.shape[0]
                if num_gt_i == 0:
                    continue
                for matchi in range(dn_number):
                    si = (matchi // num_gt_i) * num_gt_i
                    ei = (matchi // num_gt_i + 1) * num_gt_i
                    if si > 0:
                        attn_mask2[idx, :, matchi, :si] = True
                    if ei < dn_number:
                        attn_mask2[idx, :, matchi, ei:dn_number] = True
            attn_mask2 = attn_mask2.flatten(0, 1)


        if 'group2group' in self.dn_attn_mask_type_list:
            num_points = 4
            attn_mask3 = torch.zeros(
                bs,
                self.nheads,
                dn_number + self.num_group * 4, dn_number + self.num_group * 4,
                device=device, dtype=torch.bool)
            attn_mask3[:, :, dn_number:, :dn_number] = True
            group_bbox_kpt = 4
            
            for matchj in range(self.num_group * 4):
                sj = (matchj // group_bbox_kpt) * group_bbox_kpt
                ej = (matchj // group_bbox_kpt + 1)*group_bbox_kpt
                # for each instance, they should associate their body, hand, and face bbox
                if sj > 0:
                    attn_mask3[:, :, dn_number:, dn_number:][:, :, matchj, :sj] = True
                if ej < self.num_group * 4:
                    attn_mask3[:, :, dn_number:, dn_number:][:, :, matchj, ej:] = True
            
            kpt_index = [x for x in range(self.num_group * 4) if x % 4 in [0]]
            for match_x in range(self.num_group * 4):
                if match_x % group_bbox_kpt in [0, 1,  2, 3]:
                    # for each instance, they should associate their each query with 
                    # other instances' body query
                    attn_mask3[:, :, dn_number:, dn_number:][:, :, match_x, kpt_index]=False

            for idx, (gt_boxes_i, gt_labels_i) in enumerate(zip(gt_boxes, gt_labels)):
                num_gt_i = gt_boxes_i.shape[0]
                if num_gt_i == 0:
                    continue
                for matchi in range(dn_number):
                    si = (matchi // num_gt_i) * num_gt_i
                    ei = (matchi // num_gt_i + 1) * num_gt_i
                    if si > 0:
                        attn_mask3[idx, :, matchi, :si] = True
                    if ei < dn_number:
                        attn_mask3[idx, :, matchi, ei:dn_number] = True
            attn_mask3 = attn_mask3.flatten(0, 1)

        mask_dict = {
            'pad_size': dn_number,
            'known_bboxs': gt_boxes_expand,
            'known_labels': gt_labels_expand,
            'known_keypoints': gt_keypoints_expand
        }

        return input_query_label, input_query_bbox, attn_mask, attn_mask2, attn_mask3, mask_dict

    def dn_post_process2(self, outputs_class, outputs_coord, mask_dict):
        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = [
                outputs_class_i[:, :mask_dict['pad_size'], :]
                for outputs_class_i in outputs_class
            ]
            output_known_coord = [
                outputs_coord_i[:, :mask_dict['pad_size'], :]
                for outputs_coord_i in outputs_coord
            ]

            outputs_class = [
                outputs_class_i[:, mask_dict['pad_size']:, :]
                for outputs_class_i in outputs_class
            ]
            outputs_coord = [
                outputs_coord_i[:, mask_dict['pad_size']:, :]
                for outputs_coord_i in outputs_coord
            ]

            mask_dict.update({
                'output_known_coord': output_known_coord,
                'output_known_class': output_known_class
            })
        return outputs_class, outputs_coord

    def forward(self, data_batch: NestedTensor, targets: List = None):
        """The forward expects a NestedTensor, which consists of:

           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """

        if isinstance(data_batch, dict):
            samples, targets = self.prepare_targets(data_batch)
            # import pdb; pdb.set_trace()
        elif isinstance(data_batch, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(data_batch)
        else:
            samples = data_batch
        features, poss = self.backbone(samples)
        srcs = []
        masks = []
        for l, feat in enumerate(features):  # len(features=3)
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(),
                                     size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        if self.dn_number > 0 or targets is not None:
            input_query_label, input_query_bbox, attn_mask,attn_mask2, attn_mask3, mask_dict =\
                self.prepare_for_dn2(targets)
        else:
            assert targets is None
            input_query_bbox = input_query_label = attn_mask = attn_mask2 = attn_mask3 = mask_dict = None


        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask,
            attn_mask2, attn_mask3)

        # update human boxes
        effective_dn_number = self.dn_number if self.training else 0
        outputs_body_bbox_list = []
        outputs_class = []
        
        for dec_lid, (layer_ref_sig, layer_body_bbox_embed, layer_cls_embed,
                      layer_hs) in enumerate(
                          zip(reference[:-1], self.bbox_embed,
                              self.class_embed, hs)):
            if dec_lid < self.num_box_decoder_layers:
                # human det
                layer_delta_unsig = layer_body_bbox_embed(layer_hs)
                layer_body_box_outputs_unsig = \
                    layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
                layer_body_box_outputs_unsig = layer_body_box_outputs_unsig.sigmoid()
                layer_cls = layer_cls_embed(layer_hs)
                outputs_body_bbox_list.append(layer_body_box_outputs_unsig)
                outputs_class.append(layer_cls)
                
            elif dec_lid < self.num_box_decoder_layers + 2:
                bs = layer_ref_sig.shape[0]                
                # dn body bbox
                layer_hs_body_bbox_dn = layer_hs[:, :effective_dn_number, :]  # dn content query
                reference_before_sigmoid_body_bbox_dn = layer_ref_sig[:, :effective_dn_number, :]  # dn position query
                layer_body_box_delta_unsig_dn = layer_body_bbox_embed(layer_hs_body_bbox_dn)
                layer_body_box_outputs_unsig_dn = layer_body_box_delta_unsig_dn + inverse_sigmoid(
                    reference_before_sigmoid_body_bbox_dn)
                layer_body_box_outputs_unsig_dn = layer_body_box_outputs_unsig_dn.sigmoid()
                
                # norm body bbox
                layer_hs_body_bbox_norm = layer_hs[:, effective_dn_number:, :][
                    :, 0::(self.num_body_points + 4), :]  # norm content query
                reference_before_sigmoid_body_bbox_norm = layer_ref_sig[:, effective_dn_number:, :][
                    :, 0::(self.num_body_points+ 4), :]  # norm position query
                layer_body_box_delta_unsig_norm = layer_body_bbox_embed(layer_hs_body_bbox_norm)
                layer_body_box_outputs_unsig_norm = layer_body_box_delta_unsig_norm + inverse_sigmoid(
                    reference_before_sigmoid_body_bbox_norm)
                layer_body_box_outputs_unsig_norm = layer_body_box_outputs_unsig_norm.sigmoid()

                layer_body_box_outputs_unsig = torch.cat(
                    (layer_body_box_outputs_unsig_dn, layer_body_box_outputs_unsig_norm), dim=1)

                # classfication
                layer_cls_dn = layer_cls_embed(layer_hs_body_bbox_dn)
                layer_cls_norm = layer_cls_embed(layer_hs_body_bbox_norm)
                layer_cls = torch.cat((layer_cls_dn, layer_cls_norm), dim=1)

                outputs_class.append(layer_cls)
                outputs_body_bbox_list.append(layer_body_box_outputs_unsig)                
            else:
                bs = layer_ref_sig.shape[0]                
                # dn body bbox
                layer_hs_body_bbox_dn = layer_hs[:, :effective_dn_number, :]  # dn content query
                reference_before_sigmoid_body_bbox_dn = layer_ref_sig[:, :effective_dn_number, :]  # dn position query
                layer_body_box_delta_unsig_dn = layer_body_bbox_embed(layer_hs_body_bbox_dn)
                layer_body_box_outputs_unsig_dn = layer_body_box_delta_unsig_dn + inverse_sigmoid(
                    reference_before_sigmoid_body_bbox_dn)
                layer_body_box_outputs_unsig_dn = layer_body_box_outputs_unsig_dn.sigmoid()
                
                # norm body bbox
                layer_hs_body_bbox_norm = layer_hs[:, effective_dn_number:, :][
                    :, 0::(self.num_whole_body_points + 4), :]  # norm content query
                reference_before_sigmoid_body_bbox_norm = layer_ref_sig[:,effective_dn_number:, :][
                    :, 0::(self.num_whole_body_points + 4), :]  # norm position query
                layer_body_box_delta_unsig_norm = layer_body_bbox_embed(layer_hs_body_bbox_norm)
                layer_body_box_outputs_unsig_norm = layer_body_box_delta_unsig_norm + inverse_sigmoid(
                    reference_before_sigmoid_body_bbox_norm)
                layer_body_box_outputs_unsig_norm = layer_body_box_outputs_unsig_norm.sigmoid()

                layer_body_box_outputs_unsig = torch.cat(
                    (layer_body_box_outputs_unsig_dn, layer_body_box_outputs_unsig_norm), dim=1)

                # classfication
                layer_cls_dn = layer_cls_embed(layer_hs_body_bbox_dn)
                layer_cls_norm = layer_cls_embed(layer_hs_body_bbox_norm)
                layer_cls = torch.cat((layer_cls_dn, layer_cls_norm), dim=1)

                outputs_class.append(layer_cls)
                outputs_body_bbox_list.append(layer_body_box_outputs_unsig)       
                
        # update hand and face boxes
        outputs_lhand_bbox_list = []
        outputs_rhand_bbox_list = []
        outputs_face_bbox_list = []
        # update keypoints boxes
        outputs_body_keypoints_list = []
        outputs_body_keypoints_hw = []
        outputs_lhand_keypoints_list = []
        outputs_lhand_keypoints_hw = []        
        outputs_rhand_keypoints_list = []
        outputs_rhand_keypoints_hw = []
        outputs_face_keypoints_list = []
        outputs_face_keypoints_hw = []             
        
        outputs_smpl_pose_list = []
        outputs_smpl_lhand_pose_list = []
        outputs_smpl_rhand_pose_list = []
        outputs_smpl_expr_list = []
        outputs_smpl_jaw_pose_list = []
        outputs_smpl_beta_list = []
        outputs_smpl_cam_list = []
        outputs_smpl_kp2d_list = []
        outputs_smpl_kp3d_list = []
        outputs_smpl_verts_list = []
        
        # smpl pose
        # body box, kps, lhand box
        body_index = [0, 1, 2, 3]
        smpl_pose_index = [
            x for x in range(self.num_group * 4) if (x % 4 in body_index)]
        
        # smpl lhand
        lhand_index = [0, 1]
        smpl_lhand_pose_index = [
            x for x in range(self.num_group * 4) if (x % 4 in lhand_index)]
        
        # smpl rhand
        rhand_index = [0, 2]
        smpl_rhand_pose_index = [
            x for x in range(self.num_group * 4) if (x % 4 in rhand_index)]
        
        # smpl face
        face_index = [0, 3]
        smpl_face_pose_index = [
            x for x in range(self.num_group * 4) if (x % 4 in face_index)]
        
        for dec_lid, (layer_ref_sig, layer_hs) in enumerate(zip(reference[:-1], hs)):
            if dec_lid < self.num_box_decoder_layers:
                assert isinstance(layer_hs, torch.Tensor)
                bs = layer_hs.shape[0]
                layer_body_kps_res = layer_hs.new_zeros(
                    (bs, self.num_queries,
                     self.num_body_points * 3))  # [-, 900, 42]
                outputs_body_keypoints_list.append(layer_body_kps_res)
                
                # lhand
                layer_lhand_bbox_res = layer_hs.new_zeros(
                    (bs, self.num_queries, 4))  # [-, 900, 42]
                outputs_lhand_bbox_list.append(layer_lhand_bbox_res)
                layer_lhand_kps_res = layer_hs.new_zeros(
                    (bs, self.num_queries,
                     self.num_hand_points * 3))  # [-, 900, 42]
                outputs_lhand_keypoints_list.append(layer_lhand_kps_res)                

                # rhand
                layer_rhand_bbox_res = layer_hs.new_zeros(
                    (bs, self.num_queries, 4))  # [-, 900, 42]
                outputs_rhand_bbox_list.append(layer_rhand_bbox_res)                
                layer_rhand_kps_res = layer_hs.new_zeros(
                    (bs, self.num_queries,
                     self.num_hand_points * 3))  # [-, 900, 42]
                outputs_rhand_keypoints_list.append(layer_rhand_kps_res)
                
                # face
                layer_face_bbox_res = layer_hs.new_zeros(
                    (bs, self.num_queries, 4))  # [-, 900, 42]
                outputs_face_bbox_list.append(layer_face_bbox_res)
                layer_face_kps_res = layer_hs.new_zeros(
                    (bs, self.num_queries,
                     self.num_face_points * 3))  # [-, 900, 42]
                outputs_face_keypoints_list.append(layer_face_kps_res)
                
                # smpl or smplx
                smpl_pose = layer_hs.new_zeros((bs, self.num_queries, self.body_model_joint_num * 3))
                smpl_rhand_pose = layer_hs.new_zeros(
                    (bs, self.num_queries, 15 * 3))
                smpl_lhand_pose = layer_hs.new_zeros(
                    (bs, self.num_queries, 15 * 3))
                smpl_expr = layer_hs.new_zeros((bs, self.num_queries, 10))
                smpl_jaw_pose = layer_hs.new_zeros((bs, self.num_queries, 6))
                smpl_beta = layer_hs.new_zeros((bs, self.num_queries, 10))
                smpl_cam = layer_hs.new_zeros((bs, self.num_queries, 3))
                # smpl_kp2d = layer_hs.new_zeros((bs, self.num_queries, self.num_body_points,3))
                smpl_kp3d = layer_hs.new_zeros(
                    (bs, self.num_queries, self.num_body_points, 4))
                outputs_smpl_pose_list.append(smpl_pose)
                outputs_smpl_rhand_pose_list.append(smpl_rhand_pose)
                outputs_smpl_lhand_pose_list.append(smpl_lhand_pose)
                outputs_smpl_expr_list.append(smpl_expr)
                outputs_smpl_jaw_pose_list.append(smpl_jaw_pose)
                outputs_smpl_beta_list.append(smpl_beta)
                outputs_smpl_cam_list.append(smpl_cam)
                # outputs_smpl_kp2d_list.append(smpl_kp2d)
                outputs_smpl_kp3d_list.append(smpl_kp3d)
            elif dec_lid < self.num_box_decoder_layers +2:
                bs = layer_ref_sig.shape[0]
                # lhand bbox
                layer_hs_lhand_bbox = \
                    layer_hs[:, effective_dn_number:, :][:, 1::4, :]
                    
                delta_lhand_bbox_xy_unsig = self.bbox_hand_embed[dec_lid - self.num_box_decoder_layers](layer_hs_lhand_bbox)             
                layer_ref_sig_lhand_bbox = \
                    layer_ref_sig[:,effective_dn_number:, :][:, 1::4, :].clone() 
                layer_ref_unsig_lhand_bbox = inverse_sigmoid(layer_ref_sig_lhand_bbox)
                delta_lhand_bbox_hw_unsig = self.bbox_hand_hw_embed[
                    dec_lid-self.num_box_decoder_layers](layer_hs_lhand_bbox)
                layer_ref_unsig_lhand_bbox[..., :2] +=delta_lhand_bbox_xy_unsig[..., :2]
                layer_ref_unsig_lhand_bbox[..., 2:] +=delta_lhand_bbox_hw_unsig
                layer_ref_sig_lhand_bbox = layer_ref_unsig_lhand_bbox.sigmoid()
                outputs_lhand_bbox_list.append(layer_ref_sig_lhand_bbox)
                
                # rhand bbox
                layer_hs_rhand_bbox = \
                    layer_hs[:, effective_dn_number:, :][:, 2::4, :]
                delta_rhand_bbox_xy_unsig = self.bbox_hand_embed[
                    dec_lid - self.num_box_decoder_layers](layer_hs_rhand_bbox)             
                layer_ref_sig_rhand_bbox = \
                    layer_ref_sig[:,effective_dn_number:, :][:, 2::4, :].clone()
                layer_ref_unsig_rhand_bbox = inverse_sigmoid(layer_ref_sig_rhand_bbox)
                delta_rhand_bbox_hw_unsig = self.bbox_hand_hw_embed[
                    dec_lid-self.num_box_decoder_layers](layer_hs_rhand_bbox)
                layer_ref_unsig_rhand_bbox[..., :2] +=delta_rhand_bbox_xy_unsig[..., :2]
                layer_ref_unsig_rhand_bbox[..., 2:] +=delta_rhand_bbox_hw_unsig
                layer_ref_sig_rhand_bbox = layer_ref_unsig_rhand_bbox.sigmoid()
                outputs_rhand_bbox_list.append(layer_ref_sig_rhand_bbox)
                
                # face bbox
                layer_hs_face_bbox = \
                    layer_hs[:, effective_dn_number:, :][:, 3::4, :]
                delta_face_bbox_xy_unsig = self.bbox_face_embed[
                    dec_lid - self.num_box_decoder_layers](layer_hs_face_bbox)             
                layer_ref_sig_face_bbox = \
                    layer_ref_sig[:,effective_dn_number:, :][:, 3::4, :].clone()
                layer_ref_unsig_face_bbox = inverse_sigmoid(layer_ref_sig_face_bbox)
                delta_face_bbox_hw_unsig = self.bbox_face_hw_embed[
                    dec_lid-self.num_box_decoder_layers](layer_hs_face_bbox)
                layer_ref_unsig_face_bbox[..., :2] +=delta_face_bbox_xy_unsig[..., :2]
                layer_ref_unsig_face_bbox[..., 2:] +=delta_face_bbox_hw_unsig                
                layer_ref_sig_face_bbox = layer_ref_unsig_face_bbox.sigmoid()
                
                outputs_face_bbox_list.append(layer_ref_sig_face_bbox)
                
                # smpl or smplx
                bs, _, feat_dim = layer_hs.shape
                smpl_feats = layer_hs[:, effective_dn_number:, :].index_select(
                    1, torch.tensor(smpl_pose_index, device=layer_hs.device)
                    ).reshape(bs, -1, feat_dim * 4)
                smpl_lhand_pose_feats = \
                    layer_hs[:, effective_dn_number:, :].index_select(
                    1, torch.tensor(smpl_lhand_pose_index, device=layer_hs.device)
                    ).reshape(bs, -1, feat_dim * 2)
                smpl_rhand_pose_feats = layer_hs[:, effective_dn_number:, :].index_select(
                    1, torch.tensor(smpl_rhand_pose_index, device=layer_hs.device)
                    ).reshape(bs, -1, feat_dim * 2)
                smpl_face_pose_feats = layer_hs[:, effective_dn_number:, :].index_select(
                    1, torch.tensor(smpl_face_pose_index, device=layer_hs.device)
                    ).reshape(bs, -1, feat_dim * 2)
                                  
                smpl_pose = self.smpl_pose_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_feats)
                smpl_pose = rot6d_to_rotmat(smpl_pose.reshape(-1, 6)).reshape(
                    bs, self.num_group, self.body_model_joint_num, 3, 3)
                
                smpl_lhand_pose = self.smpl_hand_pose_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_lhand_pose_feats)
                smpl_lhand_pose = rot6d_to_rotmat(smpl_lhand_pose.reshape(
                    -1, 6)).reshape(bs, self.num_group, 15, 3, 3)
                
                smpl_rhand_pose = self.smpl_hand_pose_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_rhand_pose_feats)
                smpl_rhand_pose = rot6d_to_rotmat(smpl_rhand_pose.reshape(
                    -1, 6)).reshape(bs, self.num_group, 15, 3, 3)
                
                smpl_jaw_pose = self.smpl_jaw_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_face_pose_feats)
                smpl_jaw_pose = rot6d_to_rotmat(smpl_jaw_pose.reshape(-1, 6)).reshape(
                    bs, self.num_group, 1, 3, 3)
                                 
                smpl_beta = self.smpl_beta_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_feats)
                smpl_cam = self.smpl_cam_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_feats)

                smpl_expr = self.smpl_expr_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_face_pose_feats)
                # smpl_jaw_pose = layer_hs.new_zeros(bs, self.num_group, 3)
                leye_pose = torch.zeros_like(smpl_jaw_pose)
                reye_pose = torch.zeros_like(smpl_jaw_pose)



                if self.body_model is not None:
                    smpl_pose_ = rotmat_to_aa(smpl_pose)
                    # smpl_lhand_pose_ = rotmat_to_aa(smpl_lhand_pose)
                    # smpl_rhand_pose_ = rotmat_to_aa(smpl_rhand_pose)
                    smpl_lhand_pose_ = layer_hs.new_zeros(bs, self.num_group, 15, 3)
                    smpl_rhand_pose_ = layer_hs.new_zeros(bs, self.num_group, 15, 3)
                    smpl_jaw_pose_ = rotmat_to_aa(smpl_jaw_pose)
                    leye_pose_ = rotmat_to_aa(leye_pose)
                    reye_pose_ = rotmat_to_aa(reye_pose)
                    
                    pred_output = self.body_model(
                        betas=smpl_beta.reshape(-1, 10),
                        body_pose=smpl_pose_[:, :,  1:].reshape(-1, 21 * 3),
                        global_orient=smpl_pose_[:, :, 0].reshape(
                            -1, 3).unsqueeze(1),
                        left_hand_pose=smpl_lhand_pose_.reshape(-1, 15 * 3),
                        right_hand_pose=smpl_rhand_pose_.reshape(-1, 15 * 3),
                        leye_pose=leye_pose_,
                        reye_pose=reye_pose_,
                        jaw_pose=smpl_jaw_pose_.reshape(-1, 3),
                        # expression=smpl_expr.reshape(-1, 10),
                        expression=layer_hs.new_zeros(bs, self.num_group, 10).reshape(-1, 10)
                    )
                    smpl_kp3d = pred_output['joints'].reshape(
                        bs, self.num_group, -1, 3)
                    smpl_verts = pred_output['vertices'].reshape(
                        bs, self.num_group, -1, 3)
                    # pred_vertices = pred_output['vertices'].reshape(bs, -1, 6890, 3)

                outputs_smpl_pose_list.append(smpl_pose)
                outputs_smpl_rhand_pose_list.append(smpl_rhand_pose)
                outputs_smpl_lhand_pose_list.append(smpl_lhand_pose)
                outputs_smpl_expr_list.append(smpl_expr)
                outputs_smpl_jaw_pose_list.append(smpl_jaw_pose)
                outputs_smpl_beta_list.append(smpl_beta)
                outputs_smpl_cam_list.append(smpl_cam)
                outputs_smpl_kp3d_list.append(smpl_kp3d)
                

            else:
                bs = layer_ref_sig.shape[0]
                # lhand bbox
                layer_hs_lhand_bbox = \
                    layer_hs[:, effective_dn_number:, :][:, 1::4, :]
                delta_lhand_bbox_xy_unsig = self.bbox_hand_embed[
                    dec_lid - self.num_box_decoder_layers](layer_hs_lhand_bbox)             
                layer_ref_sig_lhand_bbox = \
                    layer_ref_sig[:,effective_dn_number:, :][:, 1::4, :].clone()
                layer_ref_unsig_lhand_bbox = inverse_sigmoid(layer_ref_sig_lhand_bbox)
                delta_lhand_bbox_hw_unsig = self.bbox_hand_hw_embed[
                    dec_lid-self.num_box_decoder_layers](layer_hs_lhand_bbox)
                layer_ref_unsig_lhand_bbox[..., :2] +=delta_lhand_bbox_xy_unsig[..., :2]
                layer_ref_unsig_lhand_bbox[..., 2:] +=delta_lhand_bbox_hw_unsig
                layer_ref_sig_lhand_bbox = layer_ref_unsig_lhand_bbox.sigmoid()
                outputs_lhand_bbox_list.append(layer_ref_sig_lhand_bbox)
                
                # rhand bbox
                layer_hs_rhand_bbox = \
                    layer_hs[:, effective_dn_number:, :][:, 2::4, :]
                delta_rhand_bbox_xy_unsig = self.bbox_hand_embed[
                    dec_lid - self.num_box_decoder_layers](layer_hs_rhand_bbox)             
                layer_ref_sig_rhand_bbox = \
                    layer_ref_sig[:,effective_dn_number:, :][:, 2::4, :].clone()                  
                layer_ref_unsig_rhand_bbox = inverse_sigmoid(layer_ref_sig_rhand_bbox)
                delta_rhand_bbox_hw_unsig = self.bbox_hand_hw_embed[
                    dec_lid-self.num_box_decoder_layers](layer_hs_rhand_bbox)
                layer_ref_unsig_rhand_bbox[..., :2] +=delta_rhand_bbox_xy_unsig[..., :2]
                layer_ref_unsig_rhand_bbox[..., 2:] +=delta_rhand_bbox_hw_unsig
                layer_ref_sig_rhand_bbox = layer_ref_unsig_rhand_bbox.sigmoid()
                outputs_rhand_bbox_list.append(layer_ref_sig_rhand_bbox)

                # face bbox
                layer_hs_face_bbox = \
                    layer_hs[:, effective_dn_number:, :][:, 3::4, :]
                delta_face_bbox_xy_unsig = \
                    self.bbox_face_embed[dec_lid - self.num_box_decoder_layers](layer_hs_face_bbox)             
                layer_ref_sig_face_bbox = \
                    layer_ref_sig[:,effective_dn_number:, :][:, 3::4, :].clone()               
                layer_ref_unsig_face_bbox = inverse_sigmoid(layer_ref_sig_face_bbox)
                delta_face_bbox_hw_unsig = self.bbox_face_hw_embed[
                    dec_lid-self.num_box_decoder_layers](layer_hs_face_bbox)
                layer_ref_unsig_face_bbox[..., :2] +=delta_face_bbox_xy_unsig[..., :2]
                layer_ref_unsig_face_bbox[..., 2:] +=delta_face_bbox_hw_unsig
                layer_ref_sig_face_bbox = layer_ref_unsig_face_bbox.sigmoid()   
                outputs_face_bbox_list.append(layer_ref_sig_face_bbox)
                
                bs, _, feat_dim = layer_hs.shape
                smpl_body_pose_feats = layer_hs[:, effective_dn_number:, :].index_select(
                    1, torch.tensor(smpl_pose_index, device=layer_hs.device)
                    ).reshape(bs, -1, feat_dim * 4)
                smpl_lhand_pose_feats = layer_hs[:, effective_dn_number:, :].index_select(
                    1, torch.tensor(smpl_lhand_pose_index, device=layer_hs.device)
                    ).reshape(bs, -1, feat_dim * 2)
                smpl_rhand_pose_feats = layer_hs[:, effective_dn_number:, :].index_select(
                    1, torch.tensor(smpl_rhand_pose_index, device=layer_hs.device)
                    ).reshape(bs, -1, feat_dim * 2)
                smpl_face_pose_feats = layer_hs[:, effective_dn_number:, :].index_select(
                    1, torch.tensor(smpl_face_pose_index, device=layer_hs.device)
                    ).reshape(bs, -1, feat_dim * 2)
                                                
                smpl_pose = self.smpl_pose_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_body_pose_feats)
                
                smpl_pose = rot6d_to_rotmat(smpl_pose.reshape(-1, 6)).reshape(
                    bs, self.num_group, self.body_model_joint_num, 3, 3)
                smpl_lhand_pose = self.smpl_hand_pose_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_lhand_pose_feats)
                smpl_lhand_pose = rot6d_to_rotmat(smpl_lhand_pose.reshape(
                    -1, 6)).reshape(bs, self.num_group, 15, 3, 3)
                smpl_rhand_pose = self.smpl_hand_pose_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_rhand_pose_feats)
                smpl_rhand_pose = rot6d_to_rotmat(smpl_rhand_pose.reshape(
                    -1, 6)).reshape(bs, self.num_group, 15, 3, 3)

                smpl_expr = self.smpl_expr_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_face_pose_feats)
                smpl_jaw_pose = self.smpl_jaw_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_face_pose_feats)
                smpl_jaw_pose = rot6d_to_rotmat(smpl_jaw_pose.reshape(-1, 6)).reshape(
                    bs, self.num_group, 1, 3, 3)
                smpl_beta = self.smpl_beta_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_body_pose_feats)
                smpl_cam = self.smpl_cam_embed[
                    dec_lid - self.num_box_decoder_layers](smpl_body_pose_feats)
                
                num_samples = smpl_beta.reshape(-1, 10).shape[0]
                device = smpl_beta.device
                leye_pose = torch.zeros_like(smpl_jaw_pose)
                reye_pose = torch.zeros_like(smpl_jaw_pose)

                if self.body_model is not None:
                    smpl_pose_ = rotmat_to_aa(smpl_pose)
                    smpl_lhand_pose_ = rotmat_to_aa(smpl_lhand_pose)
                    smpl_rhand_pose_ = rotmat_to_aa(smpl_rhand_pose)
                    smpl_jaw_pose_ = rotmat_to_aa(smpl_jaw_pose)
                    leye_pose_ = rotmat_to_aa(leye_pose)
                    reye_pose_ = rotmat_to_aa(reye_pose)
                    
                    pred_output = self.body_model(
                        betas=smpl_beta.reshape(-1, 10),
                        body_pose=smpl_pose_[:, :,  1:].reshape(-1, 21 * 3),
                        global_orient=smpl_pose_[:, :, 0].reshape(
                            -1, 3).unsqueeze(1),
                        left_hand_pose=smpl_lhand_pose_.reshape(-1, 15 * 3),
                        right_hand_pose=smpl_rhand_pose_.reshape(-1, 15 * 3),
                        leye_pose=leye_pose_,
                        reye_pose=reye_pose_,
                        jaw_pose=smpl_jaw_pose_.reshape(-1, 3),
                        expression=smpl_expr.reshape(-1, 10),
                        # expression=layer_hs.new_zeros(bs, self.num_group, 10).reshape(-1, 10),
                    )
                    smpl_kp3d = pred_output['joints'].reshape(
                        bs, self.num_group, -1, 3)
                    smpl_verts = pred_output['vertices'].reshape(
                        bs, self.num_group, -1, 3)

                outputs_smpl_pose_list.append(smpl_pose)
                outputs_smpl_rhand_pose_list.append(smpl_rhand_pose)
                outputs_smpl_lhand_pose_list.append(smpl_lhand_pose)
                outputs_smpl_expr_list.append(smpl_expr)
                outputs_smpl_jaw_pose_list.append(smpl_jaw_pose)
                outputs_smpl_beta_list.append(smpl_beta)
                outputs_smpl_cam_list.append(smpl_cam)
                outputs_smpl_kp3d_list.append(smpl_kp3d)
                if not self.training:
                    outputs_smpl_verts_list.append(smpl_verts)
        dn_mask_dict = mask_dict
        if self.dn_number > 0 and dn_mask_dict is not None:
            outputs_class, outputs_body_bbox_list = self.dn_post_process2(
                outputs_class, outputs_body_bbox_list, dn_mask_dict)
            dn_class_input = dn_mask_dict['known_labels']
            dn_bbox_input = dn_mask_dict['known_bboxs']
            dn_class_pred = dn_mask_dict['output_known_class']
            dn_bbox_pred = dn_mask_dict['output_known_coord']

        for idx, (_out_class, _out_bbox) in enumerate(zip(outputs_class, outputs_body_bbox_list)):
            assert _out_class.shape[1] == _out_bbox.shape[1]

        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_body_bbox_list[-1],
            'pred_lhand_boxes': outputs_lhand_bbox_list[-1],
            'pred_rhand_boxes': outputs_rhand_bbox_list[-1],
            'pred_face_boxes': outputs_face_bbox_list[-1],
            'pred_smpl_pose': outputs_smpl_pose_list[-1],
            'pred_smpl_rhand_pose': outputs_smpl_rhand_pose_list[-1],
            'pred_smpl_lhand_pose': outputs_smpl_lhand_pose_list[-1],
            'pred_smpl_jaw_pose': outputs_smpl_jaw_pose_list[-1],
            'pred_smpl_expr': outputs_smpl_expr_list[-1],
            'pred_smpl_beta': outputs_smpl_beta_list[-1],  # [B, 100, 10]
            'pred_smpl_cam': outputs_smpl_cam_list[-1],
            'pred_smpl_kp3d': outputs_smpl_kp3d_list[-1]
        }
        if not self.training:
            full_pose = torch.cat((outputs_smpl_pose_list[-1],
                               outputs_smpl_lhand_pose_list[-1],
                               outputs_smpl_rhand_pose_list[-1],
                               outputs_smpl_jaw_pose_list[-1]),dim=2)
            bs,num_q,_,_,_ = full_pose.shape
            full_pose = rotmat_to_aa(full_pose).reshape(bs,num_q,53*3)
            out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_body_bbox_list[-1],
            'pred_lhand_boxes': outputs_lhand_bbox_list[-1],
            'pred_rhand_boxes': outputs_rhand_bbox_list[-1],
            'pred_face_boxes': outputs_face_bbox_list[-1],
            'pred_smpl_pose': outputs_smpl_pose_list[-1],
            'pred_smpl_rhand_pose': outputs_smpl_rhand_pose_list[-1],
            'pred_smpl_lhand_pose': outputs_smpl_lhand_pose_list[-1],
            'pred_smpl_jaw_pose': outputs_smpl_jaw_pose_list[-1],
            'pred_smpl_expr': outputs_smpl_expr_list[-1],
            'pred_smpl_beta': outputs_smpl_beta_list[-1],  # [B, 100, 10]
            'pred_smpl_cam': outputs_smpl_cam_list[-1],
            'pred_smpl_kp3d': outputs_smpl_kp3d_list[-1],
            'pred_smpl_verts': outputs_smpl_verts_list[-1],
            'pred_smpl_fullpose': full_pose
        }

        if self.dn_number > 0 and dn_mask_dict is not None:
            out.update({
                'dn_class_input': dn_class_input,
                'dn_bbox_input': dn_bbox_input,
                'dn_class_pred': dn_class_pred[-1],
                'dn_bbox_pred': dn_bbox_pred[-1],
                'num_tgt': dn_mask_dict['pad_size']
            })

        if self.aux_loss:
            out['aux_outputs'] = \
                self._set_aux_loss(
                    outputs_class,
                    outputs_body_bbox_list,
                    outputs_lhand_bbox_list,
                    outputs_rhand_bbox_list,
                    outputs_face_bbox_list,
                    outputs_smpl_pose_list,
                    outputs_smpl_rhand_pose_list,
                    outputs_smpl_lhand_pose_list,
                    outputs_smpl_jaw_pose_list,
                    outputs_smpl_expr_list,
                    outputs_smpl_beta_list,
                    outputs_smpl_cam_list,
                    outputs_smpl_kp3d_list
                ) # with key pred_logits, pred_bbox, pred_keypoints
            if self.dn_number > 0 and dn_mask_dict is not None:
                assert len(dn_class_pred[:-1]) == len(
                    dn_bbox_pred[:-1]) == len(out['aux_outputs'])
                for aux_out, dn_class_pred_i, dn_bbox_pred_i in zip(
                        out['aux_outputs'], dn_class_pred, dn_bbox_pred):
                    aux_out.update({
                        'dn_class_input': dn_class_input,
                        'dn_bbox_input': dn_bbox_input,
                        'dn_class_pred': dn_class_pred_i,
                        'dn_bbox_pred': dn_bbox_pred_i,
                        'num_tgt': dn_mask_dict['pad_size']
                    })
        # for encoder output
        if hs_enc is not None:
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            interm_pose = torch.zeros_like(outputs_body_keypoints_list[0])
            out['interm_outputs'] = {
                'pred_logits': interm_class,
                'pred_boxes': interm_coord,
                'pred_keypoints': interm_pose
            }

        return out, targets, data_batch

    @torch.jit.unused
    def _set_aux_loss(self, 
                      outputs_class, 
                      outputs_body_coord, 
                      outputs_lhand_coord,
                      outputs_rhand_coord,
                      outputs_face_coord,
                      outputs_smpl_pose, 
                      outputs_smpl_rhand_pose,
                      outputs_smpl_lhand_pose, 
                      outputs_smpl_jaw_pose,
                      outputs_smpl_expr, 
                      outputs_smpl_beta, 
                      outputs_smpl_cam,
                      outputs_smpl_kp3d):

        return [{
            'pred_logits': a,
            'pred_boxes': b,
            'pred_lhand_boxes': c,
            'pred_rhand_boxes': d,
            'pred_face_boxes': e,
            'pred_smpl_pose': j,
            'pred_smpl_rhand_pose': k,
            'pred_smpl_lhand_pose': l,
            'pred_smpl_jaw_pose': m,
            'pred_smpl_expr': n,
            'pred_smpl_beta': o,
            'pred_smpl_cam': p,
            'pred_smpl_kp3d': q
        } for a, b, c, d, e, j, k, l, m, n, o, p, q in zip(
            outputs_class[:-1], 
            outputs_body_coord[:-1],
            outputs_lhand_coord[:-1],
            outputs_rhand_coord[:-1],
            outputs_face_coord[:-1],
            outputs_smpl_pose[:-1], 
            outputs_smpl_rhand_pose[:-1],
            outputs_smpl_lhand_pose[:-1], 
            outputs_smpl_jaw_pose[:-1],
            outputs_smpl_expr[:-1], 
            outputs_smpl_beta[:-1],
            outputs_smpl_cam[:-1], 
            outputs_smpl_kp3d[:-1])]

    def prepare_targets(self, data_batch):

        data_batch_coco = []
        instance_dict = {}
        img_list = data_batch['img'].float()
        # input_img_h, input_img_w = data_batch['image_metas'][0]['batch_input_shape']
        batch_size, _, input_img_h, input_img_w = img_list.shape
        device = img_list.device
        masks = torch.ones((batch_size, input_img_h, input_img_w),
                           dtype=torch.bool,
                           device=device)
        
        if self.num_body_points == 17:
            ed_convention = 'coco'
        elif self.num_body_points == 14:
            ed_convention = 'crowdpose'

        # cv2.imread(data_batch['img_metas'][img_id]['image_path']).shape
        for img_id in range(batch_size):
            img_h, img_w = data_batch['img_shape'][img_id]
            masks[img_id, :img_h, :img_w] = 0
            
            if not self.inference:
                instance_body_bbox = torch.cat([data_batch['body_bbox_center'][img_id],\
                                                data_batch['body_bbox_size'][img_id]],dim=-1)
                instance_face_bbox = torch.cat([data_batch['face_bbox_center'][img_id],\
                                                data_batch['face_bbox_size'][img_id]],dim=-1)
                instance_lhand_bbox = torch.cat([data_batch['lhand_bbox_center'][img_id],\
                                                data_batch['lhand_bbox_size'][img_id]],dim=-1)
                instance_rhand_bbox = torch.cat([data_batch['rhand_bbox_center'][img_id],\
                                                data_batch['rhand_bbox_size'][img_id]],dim=-1)

                instance_kp2d = data_batch['joint_img'][img_id].clone().float()
                instance_kp2d_mask = data_batch['joint_trunc'][img_id].clone().float()
                instance_kp2d[:,:,2:] = instance_kp2d_mask
                body_kp2d, _  = convert_kps(instance_kp2d, 'smplx_137', 'coco', approximate=True)
                lhand_kp2d, _  = convert_kps(instance_kp2d, 'smplx_137', 'smplx_lhand', approximate=True)
                rhand_kp2d, _  = convert_kps(instance_kp2d, 'smplx_137', 'smplx_rhand', approximate=True)
                face_kp2d, _  = convert_kps(instance_kp2d, 'smplx_137', 'smplx_face', approximate=True)
                # from util.vis_utils import show_bbox
                # show_bbox(img_list[img_id],instance_kp2d.cpu().numpy(),data_batch['bbox_xywh'][img_id].cpu().numpy)
                body_kp2d[:,:,0] = body_kp2d[:,:,0]/cfg.output_hm_shape[2]
                body_kp2d[:,:,1] = body_kp2d[:,:,1]/cfg.output_hm_shape[1]
                body_kp2d = torch.cat([body_kp2d[:,:,:2].flatten(1),body_kp2d[:,:,2]],dim=-1)

                lhand_kp2d[:,:,0] = lhand_kp2d[:,:,0]/cfg.output_hm_shape[2]
                lhand_kp2d[:,:,1] = lhand_kp2d[:,:,1]/cfg.output_hm_shape[1]
                lhand_kp2d = torch.cat([lhand_kp2d[:,:,:2].flatten(1),lhand_kp2d[:,:,2]],dim=-1)
                
                rhand_kp2d[:,:,0] = rhand_kp2d[:,:,0]/cfg.output_hm_shape[2]
                rhand_kp2d[:,:,1] = rhand_kp2d[:,:,1]/cfg.output_hm_shape[1]
                rhand_kp2d = torch.cat([rhand_kp2d[:,:,:2].flatten(1),rhand_kp2d[:,:,2]],dim=-1)

                face_kp2d[:,:,0] = face_kp2d[:,:,0]/cfg.output_hm_shape[2]
                face_kp2d[:,:,1] = face_kp2d[:,:,1]/cfg.output_hm_shape[1]
                face_kp2d = torch.cat([face_kp2d[:,:,:2].flatten(1),face_kp2d[:,:,2]],dim=-1)
                
                instance_dict = {}
                instance_dict['boxes'] = instance_body_bbox.float()
                instance_dict['face_boxes'] = instance_face_bbox.float()
                instance_dict['lhand_boxes'] = instance_lhand_bbox.float()
                instance_dict['rhand_boxes'] = instance_rhand_bbox.float()
                instance_dict['keypoints'] = body_kp2d.float()
                instance_dict['lhand_keypoints'] = lhand_kp2d.float()
                instance_dict['rhand_keypoints'] = rhand_kp2d.float()
                instance_dict['face_keypoints'] = face_kp2d.float()
            
                # instance_dict['orig_size'] = data_batch['ori_shape'][img_id]
                instance_dict['size'] = data_batch['img_shape'][img_id]  # after augmentation 
                
                instance_dict['area'] = instance_body_bbox[:, 2] * instance_body_bbox[:, 3]
                instance_dict['lhand_area'] = instance_lhand_bbox[:, 2] * instance_lhand_bbox[:, 3]
                instance_dict['rhand_area'] = instance_rhand_bbox[:, 2] * instance_rhand_bbox[:, 3]
                instance_dict['face_area'] = instance_face_bbox[:, 2] * instance_face_bbox[:, 3]

                instance_dict['labels'] = torch.ones(instance_body_bbox.shape[0],
                                                    dtype=torch.long,
                                                    device=device)
                data_batch_coco.append(instance_dict)               
            else:
                instance_body_bbox = torch.cat([data_batch['body_bbox_center'][img_id],\
                                                data_batch['body_bbox_size'][img_id]],dim=-1)
                instance_dict = {}
                # instance_dict['orig_size'] = data_batch['ori_shape'][img_id]
                instance_dict['size'] = data_batch['img_shape'][img_id]  # after augmentation 
                instance_dict['boxes'] = instance_body_bbox.float()    
                     
                data_batch_coco.append(instance_dict)  

        input_img = NestedTensor(img_list, masks)
        return input_img, data_batch_coco

@MODULE_BUILD_FUNCS.registe_with_name(module_name='aios_smplx_box')
def build_aios_smplx_box(args, cfg):
    # pdb.set_trace()
    num_classes = args.num_classes  # 2
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_class_embed_share = args.dec_pred_class_embed_share
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share

    if args.eval:
        body_model = args.body_model_test
        train = False
    else:
        body_model = args.body_model_train
        train = True
        
    model = AiOSSMPLX_Box(
        backbone,
        transformer,
        num_classes=num_classes,  # 2
        num_queries=args.num_queries,  # 900
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,  # False
        fix_refpoints_hw=args.fix_refpoints_hw,  # -1
        num_feature_levels=args.num_feature_levels,  # 4
        nheads=args.nheads,  # 8
        dec_pred_class_embed_share=dec_pred_class_embed_share,  # false
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,  # False
        # two stage
        two_stage_type=args.two_stage_type,

        # box_share
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,  # False
        two_stage_class_embed_share=args.two_stage_class_embed_share,  # False
        dn_number=args.dn_number if args.use_dn else 0,  # 100
        dn_box_noise_scale=args.dn_box_noise_scale,  # 0.4
        dn_label_noise_ratio=args.dn_label_noise_ratio,  # 0.5
        dn_batch_gt_fuse=args.dn_batch_gt_fuse,  # false
        dn_attn_mask_type_list=args.dn_attn_mask_type_list,
        dn_labelbook_size=dn_labelbook_size,  # 100
        cls_no_bias=args.cls_no_bias,  # False
        num_group=args.num_group,  # 100
        num_body_points=0,  # 17
        num_hand_points=0,  # 17
        num_face_points=0,  # 17
        num_box_decoder_layers=args.num_box_decoder_layers,  # 2
        num_hand_face_decoder_layers=args.num_hand_face_decoder_layers,
        # smpl_convention=convention
        body_model=body_model,
        train=train,
        inference=args.inference)
    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {
        'loss_ce': args.cls_loss_coef,  # 2
        # bbox
        'loss_body_bbox': args.body_bbox_loss_coef,  # 5
        'loss_rhand_bbox': args.rhand_bbox_loss_coef,  # 5
        'loss_lhand_bbox': args.lhand_bbox_loss_coef,  # 5
        'loss_face_bbox': args.face_bbox_loss_coef,  # 5
        # bbox giou
        'loss_body_giou': args.body_giou_loss_coef,  # 2
        'loss_rhand_giou': args.rhand_giou_loss_coef,  # 2
        'loss_lhand_giou': args.lhand_giou_loss_coef,  # 2
        'loss_face_giou': args.face_giou_loss_coef,  # 2
        # smpl param
        'loss_smpl_pose_root': args.smpl_pose_loss_root_coef,  # 0
        'loss_smpl_pose_body': args.smpl_pose_loss_body_coef,  # 0
        'loss_smpl_pose_lhand': args.smpl_pose_loss_lhand_coef,  # 0
        'loss_smpl_pose_rhand': args.smpl_pose_loss_rhand_coef,  # 0
        'loss_smpl_pose_jaw': args.smpl_pose_loss_jaw_coef,  # 0
        'loss_smpl_beta': args.smpl_beta_loss_coef,  # 0
        'loss_smpl_expr': args.smpl_expr_loss_coef, 
        # smpl kp3d ra
        'loss_smpl_body_kp3d_ra': args.smpl_body_kp3d_ra_loss_coef,  # 0
        'loss_smpl_lhand_kp3d_ra': args.smpl_lhand_kp3d_ra_loss_coef,  # 0
        'loss_smpl_rhand_kp3d_ra': args.smpl_rhand_kp3d_ra_loss_coef,  # 0
        'loss_smpl_face_kp3d_ra': args.smpl_face_kp3d_ra_loss_coef,  # 0
        # smpl kp3d
        'loss_smpl_body_kp3d': args.smpl_body_kp3d_loss_coef,  # 0
        'loss_smpl_face_kp3d': args.smpl_face_kp3d_loss_coef,  # 0
        'loss_smpl_lhand_kp3d': args.smpl_lhand_kp3d_loss_coef,  # 0
        'loss_smpl_rhand_kp3d': args.smpl_rhand_kp3d_loss_coef,  # 0
        # smpl kp2d
        'loss_smpl_body_kp2d': args.smpl_body_kp2d_loss_coef,  # 0
        'loss_smpl_lhand_kp2d': args.smpl_lhand_kp2d_loss_coef,  # 0
        'loss_smpl_rhand_kp2d': args.smpl_rhand_kp2d_loss_coef,  # 0
        'loss_smpl_face_kp2d': args.smpl_face_kp2d_loss_coef,  # 0
    }

    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    if args.use_dn:
        weight_dict.update({
            'dn_loss_ce':
            args.dn_label_coef,  # 0.3
            'dn_loss_bbox':
            args.bbox_loss_coef * args.dn_bbox_coef,  # 5 * 0.5
            'dn_loss_giou':
            args.giou_loss_coef * args.dn_bbox_coef,  # 2 * 0.5
        })

    clean_weight_dict = copy.deepcopy(weight_dict)

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):  # from 0 t 4 # ???
            for k, v in clean_weight_dict.items():
                if i < args.num_box_decoder_layers and ('keypoints' in k or 'oks' in k):
                    continue
                if i < args.num_box_decoder_layers and k in [
                    'loss_rhand_bbox', 'loss_lhand_bbox', 'loss_face_bbox',
                    'loss_rhand_giou', 'loss_lhand_giou', 'loss_face_giou']:
                    continue
                if i < args.num_hand_face_decoder_layers and k in [
                    'loss_rhand_keypoints', 'loss_lhand_keypoints', 
                    'loss_face_keypoints', 'loss_rhand_oks',
                    'loss_lhand_oks', 'loss_face_oks']:
                    continue
                if i < args.num_box_decoder_layers and 'smpl' in k:
                    continue
                aux_weight_dict.update({k + f'_{i}': v})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            # bbox
            'loss_body_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_rhand_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_lhand_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_face_bbox': 1.0 if not no_interm_box_loss else 0.0,
            # bbox giou
            'loss_body_giou': 1.0 if not no_interm_box_loss else 0.0,
            'loss_rhand_giou': 1.0 if not no_interm_box_loss else 0.0,
            'loss_lhand_giou': 1.0 if not no_interm_box_loss else 0.0,
            'loss_face_giou': 1.0 if not no_interm_box_loss else 0.0,
            # smpl param
            'loss_smpl_pose_root': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_pose_body': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_pose_lhand': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_pose_rhand': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_pose_jaw': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_beta': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_expr': 1.0 if not no_interm_box_loss else 0.0,
            # smpl kp3d ra
            'loss_smpl_body_kp3d_ra': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_lhand_kp3d_ra': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_rhand_kp3d_ra': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_face_kp3d_ra': 1.0 if not no_interm_box_loss else 0.0,
            # smpl kp3d
            'loss_smpl_body_kp3d': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_face_kp3d': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_lhand_kp3d': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_rhand_kp3d': 1.0 if not no_interm_box_loss else 0.0,
            # smpl kp2d
            'loss_smpl_body_kp2d': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_lhand_kp2d': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_rhand_kp2d': 1.0 if not no_interm_box_loss else 0.0,
            'loss_smpl_face_kp2d': 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef  # 1
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update({
            k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k]
            for k, v in clean_weight_dict_wo_dn.items() if 'keypoints' not in k
        })
        weight_dict.update(interm_weight_dict)

        interm_weight_dict.update({
            k + f'_query_expand': v * interm_loss_coef * _coeff_weight_dict[k]
            for k, v in clean_weight_dict_wo_dn.items()
        })  # ???
        weight_dict.update(interm_weight_dict)

    losses = cfg.losses
    
    if args.dn_number > 0:
        losses += ['dn_label', 'dn_bbox']
    losses += ['matching']

    criterion = SetCriterion_Box(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=args.focal_alpha,
        losses=losses,
        num_box_decoder_layers=args.num_box_decoder_layers,
        num_hand_face_decoder_layers=args.num_hand_face_decoder_layers,
        num_body_points=0,
        num_hand_points=0,
        num_face_points=0,
        )

    criterion.to(device)
    if args.inference:
        postprocessors = {
            'bbox': 
                PostProcess_SMPLX_Multi_Infer_Box(
                    num_select=args.num_select, 
                    nms_iou_threshold=args.nms_iou_threshold,
                    num_body_points=0),
        }
    else:
        postprocessors = {
            'bbox': 
                PostProcess_SMPLX_Multi_Box(
                    num_select=args.num_select, 
                    nms_iou_threshold=args.nms_iou_threshold,
                    num_body_points=0),
        }
    postprocessors_aios = {
        'bbox':
        PostProcess_aios(num_select=args.num_select,
                           nms_iou_threshold=args.nms_iou_threshold,
                           num_body_points=0),
    }

    return model, criterion, postprocessors, postprocessors_aios

