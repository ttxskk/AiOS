import os
import os.path as osp
import numpy as np

# from osx.common.utils.human_models import smpl_x

from humandata import HumanDataset
from config.config import cfg


class COCO_NA(HumanDataset):
    def __init__(self, transform, data_split):
        super(COCO_NA, self).__init__(transform, data_split)
        self.img_dir = 'data/datasets/coco_2017'
        self.annot_path = 'data/preprocessed_npz/multihuman_data/coco_wholebody_new_train_multi.npz'
        self.annot_path_cache = 'data/preprocessed_npz/cache/coco_train_cache_080824.npz'
        # osp.join(cfg.data_dir, 'cache', filename)
        self.keypoints2d = 'keypoints2d_ori'
        self.use_cache = getattr(cfg, 'use_cache', False)
        self.cam_param = {}

        # load data or cache
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
                cfg, f'{self.__class__.__name__}_train_sample_interval', 1))
            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)
