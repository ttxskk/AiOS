# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.utils import Registry, build_from_cfg

from .positional_encoding import (
    LearnedPositionalEncoding,
    SinePositionalEncoding,
)

TRANSFORMER = Registry('Transformer')
LINEAR_LAYERS = Registry('linear layers')
POSITIONAL_ENCODING = Registry('position encoding')

LINEAR_LAYERS.register_module('Linear', module=nn.Linear)
POSITIONAL_ENCODING.register_module('SinePositionalEncoding',
                                    module=SinePositionalEncoding)
POSITIONAL_ENCODING.register_module('LearnedPositionalEncoding',
                                    module=LearnedPositionalEncoding)


def build_transformer(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, TRANSFORMER, default_args)


def build_linear_layer(cfg, *args, **kwargs):
    """Build linear layer.
    Args:
        cfg (None or dict): The linear layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an linear layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding linear layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding linear layer.
    Returns:
        nn.Module: Created linear layer.
    """
    if cfg is None:
        cfg_ = dict(type='Linear')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in LINEAR_LAYERS:
        raise KeyError(f'Unrecognized linear type {layer_type}')
    else:
        linear_layer = LINEAR_LAYERS.get(layer_type)

    layer = linear_layer(*args, **kwargs, **cfg_)

    return layer


def build_positional_encoding(cfg, default_args=None):
    """Builder for Position Encoding."""
    return build_from_cfg(cfg, POSITIONAL_ENCODING, default_args)
