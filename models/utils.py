import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from util import box_ops
from util.misc import NestedTensor
from util.utils import NiceRepr


class GroupwiseMLP(nn.Module):
    def __init__(self, num_class, input_dim, hidden_dim, output_dim,
                 num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            GroupWiseLinear(num_class, n, k)
            for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        if x.dim() == 4:
            resize_flag = True
            c0, b, k, d = x.shape
            x = x.flatten(0, 1)
        else:
            resize_flag = False

        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        if resize_flag:
            x = x.reshape(c0, b, k, -1)
        return x


class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, input_dim, output_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(num_class, input_dim, output_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(num_class, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            for j in range(self.input_dim):
                self.W[i][j].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[i].data.uniform_(-stdv, stdv)

    def forward(self, x: torch.FloatTensor):
        """

        Dim:
            - b: batch size
            - k: num_class
            - d: input dim
            - o: output dim

        Input:
            - x: shape(b,k,d) or (c0,b,k,d)

        Output:
            - x: shape(b,k,o) or (c0,b,k,o)
        """
        if x.dim() == 4:
            resize_flag = True
            c0, b, k, d = x.shape
            x = x.flatten(0, 1)
        else:
            resize_flag = False

        x = torch.einsum('bkd,kdo->bko', x, self.W)
        if self.bias:
            x = torch.einsum('bko,ko->bko', x, self.b)

        if resize_flag:
            x = x.reshape(c0, b, k, -1)
        return x


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@torch.no_grad()
def mask_sample(samples: NestedTensor, known_boxes):
    """[summary]

    Args:
        samples (NestedTensor): batch of imgs. B,3,H,W
        known_boxes (list of knownBox): [knownbox_each_img x B]

    Returns:
        [Tensor]: Masked imgs. B,3,H,W.
    """
    # print("HERE!!!!!!!!!")
    # import pdb; pdb.set_trace()
    boxes_flat = [
        box_ops.box_cxcywh_to_xyxy(kbs[:, :4])
        for idx, kbs in enumerate(known_boxes)
    ]
    img_shapes = samples.imgsize()
    device = samples.tensors.device
    # ! TODO:
    for idx, (shape, boxes) in enumerate(zip(img_shapes, boxes_flat)):
        h, w = shape.tolist()
        scale = torch.Tensor([w, h, w, h]).to(device)
        boxes = boxes * scale
        for box in boxes:
            x1, y1, x2, y2 = [int(i) for i in box.tolist()]
            samples.tensors[idx, :, y1:y2, x1:x2] = 0
    return samples


class AssignResult(NiceRepr):
    """Stores assignments between predicted and truth boxes.

    ! Borrow from mmdetection


    Attributes:
        num_gts (int): the number of truth boxes considered when computing this
            assignment

        gt_inds (LongTensor): for each predicted box indicates the 1-based
            index of the assigned truth box. 0 means unassigned and -1 means
            ignore.

        max_overlaps (FloatTensor): the iou between the predicted box and its
            assigned truth box.

        labels (None | LongTensor): If specified, for each predicted box
            indicates the category label of the assigned truth box.

    Example:
        >>> # An assign result between 4 predicted boxes and 9 true boxes
        >>> # where only two boxes were assigned.
        >>> num_gts = 9
        >>> max_overlaps = torch.LongTensor([0, .5, .9, 0])
        >>> gt_inds = torch.LongTensor([-1, 1, 2, 0])
        >>> labels = torch.LongTensor([0, 3, 4, 0])
        >>> self = AssignResult(num_gts, gt_inds, max_overlaps, labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(4,), max_overlaps.shape=(4,),
                      labels.shape=(4,))>
        >>> # Force addition of gt labels (when adding gt as proposals)
        >>> new_labels = torch.LongTensor([3, 4, 5])
        >>> self.add_gt_(new_labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(7,), max_overlaps.shape=(7,),
                      labels.shape=(7,))>
    """
    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        # Interface for possible user-defined properties
        self._extra_properties = {}

    @property
    def num_preds(self):
        """int: the number of predictions in this assignment"""
        return len(self.gt_inds)

    def set_extra_property(self, key, value):
        """Set user-defined new property."""
        assert key not in self.info
        self._extra_properties[key] = value

    def get_extra_property(self, key):
        """Get user-defined property."""
        return self._extra_properties.get(key, None)

    @property
    def info(self):
        """dict: a dictionary of info about the object"""
        basic_info = {
            'num_gts': self.num_gts,
            'num_preds': self.num_preds,
            'gt_inds': self.gt_inds,
            'max_overlaps': self.max_overlaps,
            'labels': self.labels,
        }
        basic_info.update(self._extra_properties)
        return basic_info

    def __nice__(self):
        """str: a "nice" summary string describing this assign result"""
        parts = []
        parts.append(f'num_gts={self.num_gts!r}')
        if self.gt_inds is None:
            parts.append(f'gt_inds={self.gt_inds!r}')
        else:
            parts.append(f'gt_inds.shape={tuple(self.gt_inds.shape)!r}')
        if self.max_overlaps is None:
            parts.append(f'max_overlaps={self.max_overlaps!r}')
        else:
            parts.append('max_overlaps.shape='
                         f'{tuple(self.max_overlaps.shape)!r}')
        if self.labels is None:
            parts.append(f'labels={self.labels!r}')
        else:
            parts.append(f'labels.shape={tuple(self.labels.shape)!r}')
        return ', '.join(parts)

    @classmethod
    def random(cls, **kwargs):
        """Create random AssignResult for tests or debugging.

        Args:
            num_preds: number of predicted boxes
            num_gts: number of true boxes
            p_ignore (float): probability of a predicted box assigned to an
                ignored truth
            p_assigned (float): probability of a predicted box not being
                assigned
            p_use_label (float | bool): with labels or not
            rng (None | int | numpy.random.RandomState): seed or state

        Returns:
            :obj:`AssignResult`: Randomly generated assign results.

        Example:
            >>> from mmdet.core.bbox.assigners.assign_result import *  # NOQA
            >>> self = AssignResult.random()
            >>> print(self.info)
        """
        from util.utils import ensure_rng
        rng = ensure_rng(kwargs.get('rng', None))

        num_gts = kwargs.get('num_gts', None)
        num_preds = kwargs.get('num_preds', None)
        p_ignore = kwargs.get('p_ignore', 0.3)
        p_assigned = kwargs.get('p_assigned', 0.7)
        p_use_label = kwargs.get('p_use_label', 0.5)
        num_classes = kwargs.get('p_use_label', 3)

        if num_gts is None:
            num_gts = rng.randint(0, 8)
        if num_preds is None:
            num_preds = rng.randint(0, 16)

        if num_gts == 0:
            max_overlaps = torch.zeros(num_preds, dtype=torch.float32)
            gt_inds = torch.zeros(num_preds, dtype=torch.int64)
            if p_use_label is True or p_use_label < rng.rand():
                labels = torch.zeros(num_preds, dtype=torch.int64)
            else:
                labels = None
        else:
            import numpy as np
            # Create an overlap for each predicted box
            max_overlaps = torch.from_numpy(rng.rand(num_preds))

            # Construct gt_inds for each predicted box
            is_assigned = torch.from_numpy(rng.rand(num_preds) < p_assigned)
            # maximum number of assignments constraints
            n_assigned = min(num_preds, min(num_gts, is_assigned.sum()))

            assigned_idxs = np.where(is_assigned)[0]
            rng.shuffle(assigned_idxs)
            assigned_idxs = assigned_idxs[0:n_assigned]
            assigned_idxs.sort()

            is_assigned[:] = 0
            is_assigned[assigned_idxs] = True

            is_ignore = torch.from_numpy(
                rng.rand(num_preds) < p_ignore) & is_assigned

            gt_inds = torch.zeros(num_preds, dtype=torch.int64)

            true_idxs = np.arange(num_gts)
            rng.shuffle(true_idxs)
            true_idxs = torch.from_numpy(true_idxs)
            gt_inds[is_assigned] = true_idxs[:n_assigned]

            gt_inds = torch.from_numpy(
                rng.randint(1, num_gts + 1, size=num_preds))
            gt_inds[is_ignore] = -1
            gt_inds[~is_assigned] = 0
            max_overlaps[~is_assigned] = 0

            if p_use_label is True or p_use_label < rng.rand():
                if num_classes == 0:
                    labels = torch.zeros(num_preds, dtype=torch.int64)
                else:
                    labels = torch.from_numpy(
                        # remind that we set FG labels to [0, num_class-1]
                        # since mmdet v2.0
                        # BG cat_id: num_class
                        rng.randint(0, num_classes, size=num_preds))
                    labels[~is_assigned] = 0
            else:
                labels = None

        self = cls(num_gts, gt_inds, max_overlaps, labels)
        return self

    def add_gt_(self, gt_labels):
        """Add ground truth as assigned results.

        Args:
            gt_labels (torch.Tensor): Labels of gt boxes
        """
        self_inds = torch.arange(1,
                                 len(gt_labels) + 1,
                                 dtype=torch.long,
                                 device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])

        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(len(gt_labels)), self.max_overlaps])

        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])

    def get_indices(self):
        inds_used = torch.where(self.gt_inds > 0)[0]
        tgt_inds = self.gt_inds[inds_used] - 1
        return inds_used, tgt_inds
