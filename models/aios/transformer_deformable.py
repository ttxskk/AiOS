import copy
import os
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from util.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn

from .utils import sigmoid_focal_loss, MLP, _get_activation_fn, gen_sineembed_for_position
import pdb


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation='relu',
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()
        # pdb.set_trace()
        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads,
                                      n_points)  # 256 4 8 4
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self,
                src,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                key_padding_mask=None):
        # pdb.set_trace()
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points,
                              src, spatial_shapes, level_start_index,
                              key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)

        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation='relu',
        n_levels=4,
        n_heads=8,
        n_points=4,
        decoder_sa_type='ca',
        module_seq=['sa', 'ca', 'ffn'],
    ):
        super().__init__()
        # pdb.set_trace()
        self.module_seq = module_seq
        assert sorted(module_seq) == ['ca', 'ffn', 'sa']

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model,
                                               n_heads,
                                               dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation,
                                             d_model=d_ffn,
                                             batch_dim=1)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_proj = None
        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa']

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
            self,
            # for tgt
            tgt: Optional[Tensor],  # nq, bs, d_model
            tgt_query_pos: Optional[
                Tensor] = None,  # pos for query. MLP(Sine(pos))
            tgt_query_sine_embed: Optional[
                Tensor] = None,  # pos for query. Sine(pos)
            tgt_key_padding_mask: Optional[Tensor] = None,
            tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4

            # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
            memory_key_padding_mask: Optional[Tensor] = None,
            memory_level_start_index: Optional[Tensor] = None,  # num_levels
            memory_spatial_shapes: Optional[
                Tensor] = None,  # bs, num_levels, 2
            memory_pos: Optional[Tensor] = None,  # pos for memory

            # sa
        self_attn_mask: Optional[
            Tensor] = None,  # mask used for self-attention
            cross_attn_mask: Optional[
                Tensor] = None,  # mask used for cross-attention
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        # pdb.set_trace()
        assert cross_attn_mask is None

        if self.self_attn is not None:
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            tgt_reference_points.transpose(0, 1).contiguous(),
            memory.transpose(0, 1), memory_spatial_shapes,
            memory_level_start_index, memory_key_padding_mask).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
