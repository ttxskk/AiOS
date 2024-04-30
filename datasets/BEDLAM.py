import os.path as osp
from config.config import cfg
from humandata import HumanDataset


class BEDLAM(HumanDataset):
    def __init__(self, transform, data_split):
        super(BEDLAM, self).__init__(transform, data_split)

        self.img_dir = './data/datasets/bedlam/train_images/'
]        # self.annot_path_cache = 'data/preprocessed_npz/cache/bedlam_train_multi_fix_img_shape_cache.npz'

        self.annot_path = 'data/preprocessed_npz/multihuman_data/bedlam_train_multi_0915.npz'
        self.annot_path_cache = 'data/preprocessed_npz/cache/bedlam_train_cache_1211_sample5_balance_fix_exp.npz'        
        # self.annot_path_cache = 'data/preprocessed_npz/cache/bedlam_train_multi_fix_img_shape_cache_all.npz' 
        self.use_cache = getattr(cfg, 'use_cache', False)
        
        self.img_shape = None  #1024, 1024)  # (h, w)
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
                cfg, f'{self.__class__.__name__}_train_sample_interval', 5))
            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)
