import os
import os.path as osp
import numpy as np

# from osx.common.utils.human_models import smpl_x

from humandata import HumanDataset
from config.config import cfg


class COCO_NA(HumanDataset):
    def __init__(self, transform, data_split):
        super(COCO_NA, self).__init__(transform, data_split)

        # pre_prc_file_train = 'spec_train_smpl.npz'
        # pre_prc_file_test = 'spec_test_smpl.npz'

        # if self.data_split == 'train':
        #     filename = getattr(cfg, 'filename', pre_prc_file_train)
        # else:
        #     raise ValueError('COCO_NA test set is not support')

        self.img_dir = 'data/datasets/coco_2017'
        # self.annot_path = osp.join(cfg.data_dir, 'preprocessed_datasets', filename)

        # self.annot_path_cache = osp.join(cfg.data_dir, 'cache', filename)

        self.annot_path = 'data/preprocessed_npz/multihuman_data/coco2017_neural_annot_wholebody_multi_data_train.npz'
        # osp.join(cfg.data_dir, 'preprocessed_datasets', filename)

        # self.annot_path_cache = 'data/preprocessed_npz/coco_whole_train_cache_230925_debug.npz'
        # .annot_path_cache = 'data/preprocessed_npz/coco_train_cache_230925.npz'
        self.annot_path_cache = 'data/preprocessed_npz/cache/coco_train_cache_1910_balance.npz'
        # osp.join(cfg.data_dir, 'cache', filename)
        self.keypoints2d = 'keypoints2d_ori'
        self.use_cache = getattr(cfg, 'use_cache', False)
        # self.use_cache = False
        # self.img_shape = (1080, 1920)  # (h, w)
        self.cam_param = {}

        # check image shape
        # img_path = osp.join(self.img_dir, np.load(self.annot_path)['image_path'][0])
        # img_shape = cv2.imread(img_path).shape[:2]
        # assert self.img_shape == img_shape, 'image shape is incorrect: {} vs {}'.format(self.img_shape, img_shape)

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
