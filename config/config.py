import os
import os.path as osp
import sys
import datetime
from mmcv import Config as MMConfig


# class Config():
#     def get_config_fromfile(self, config_path):
#         self.config_path = config_path
#         cfg = MMConfig.fromfile(self.config_path)
#         #import ipdb;ipdb.set_trace()
#         self.__dict__.update(dict(cfg))
#         # update dir
#         self.cur_dir = osp.dirname(os.path.abspath(__file__))
#         self.root_dir = osp.join(self.cur_dir, '..')
#         self.data_dir = osp.join(self.root_dir, 'dataset')
#         self.human_model_path = osp.join(self.root_dir, 'common', 'utils',
#                                          'human_model_files')
#         # import ipdb;ipdb.set_trace()
#         ## add some paths to the system root dir
#         # sys.path.insert(0, osp.join(self.root_dir, 'common'))
#         # sys.path.insert(0, osp.join(self.root_dir, 'united-perception_utils'))
#         # sys.path.insert(0, osp.join(self.cur_dir, 'humanbench_utils'))
#         # sys.path.insert(0, osp.join(self.cur_dir, 'dinov2_utils'))
#         # sys.path.insert(0, osp.join(self.cur_dir, 'lora_utils'))
#         # sys.path.insert(0, osp.join(self.cur_dir, 'vit_adapter_utils'))
#         from util.dir import add_pypath
#         add_pypath(osp.join(self.data_dir))
#         for dataset in os.listdir(osp.join(self.root_dir, 'data')):
#             if dataset not in ['humandata.py', '__pycache__', 'dataset.py']:
#                 add_pypath(osp.join(self.root_dir, 'data', dataset))
#         add_pypath(osp.join(self.root_dir, 'data'))
#         add_pypath(self.data_dir)

#     def prepare_dirs(self, exp_name):
#         time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
#         self.output_dir = osp.join(self.root_dir, f'{exp_name}_{time_str}')
#         self.model_dir = osp.join(self.output_dir, 'model_dump')
#         self.vis_dir = osp.join(self.output_dir, 'vis')
#         self.log_dir = osp.join(self.output_dir, 'log')
#         self.code_dir = osp.join(self.output_dir, 'code')
#         self.result_dir = osp.join(self.output_dir.split('/')[:-1])

#         from util.dir import make_folder
#         make_folder(self.model_dir)
#         make_folder(self.vis_dir)
#         make_folder(self.log_dir)
#         make_folder(self.code_dir)
#         make_folder(self.result_dir)

#         ## copy some code to log dir as a backup
#         copy_files = [
#             'main/train.py', 'main/test.py', 'common/base.py', 'main/OSX.py',
#             'common/nets', 'main/OSX_WoDecoder.py', 'data/dataset.py',
#             'data/MSCOCO/MSCOCO.py', 'data/AGORA/AGORA.py'
#         ]
#         for file in copy_files:
#             os.system(f'cp -r {self.root_dir}/{file} {self.code_dir}')

#     def update_test_config(self, testset, agora_benchmark, shapy_eval_split,
#                            pretrained_model_path, use_cache):
#         self.testset = testset
#         self.agora_benchmark = agora_benchmark
#         self.pretrained_model_path = pretrained_model_path
#         self.shapy_eval_split = shapy_eval_split
#         self.use_cache = use_cache

#     def update_config(self, num_gpus, exp_name):
#         self.num_gpus = num_gpus
#         self.exp_name = exp_name

#         self.prepare_dirs(self.exp_name)

#         # Save
#         cfg_save = MMConfig(self.__dict__)
#         cfg_save.dump(osp.join(self.code_dir, 'config_base.py'))


class Config(MMConfig):
    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        super().__init__(cfg_dict, cfg_text, filename)

    def get_config_fromfile(self, config_path):
        self.config_path = config_path

        cfg, _ = MMConfig._file2dict(self.config_path)
        # import ipdb;ipdb.set_trace()
        self.merge_from_dict(cfg)
        # #import ipdb;ipdb.set_trace()
        # self.__dict__.update(dict(cfg))
        # # update dir
        dir_dict = {}
        exp_name = 'exps62'
        time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_dict['cur_dir'] = osp.dirname(os.path.abspath(__file__))
        dir_dict['root_dir'] = osp.join(dir_dict['cur_dir'], '..')
        dir_dict['output_dir'] = osp.join(dir_dict['root_dir'], exp_name)
        dir_dict['result_dir'] = osp.join(dir_dict['output_dir'], 'result')
        dir_dict['data_dir'] = osp.join(dir_dict['root_dir'], 'dataset')
        dir_dict['human_model_path'] = osp.join('data/body_models')
        self.merge_from_dict(dir_dict)
        # # import ipdb;ipdb.set_trace()
        # ## add some paths to the system root dir
        sys.path.insert(0, osp.join(self.root_dir, 'common'))
        sys.path.insert(0, osp.join(self.root_dir, 'united-perception_utils'))
        sys.path.insert(0, osp.join(self.cur_dir, 'humanbench_utils'))
        sys.path.insert(0, osp.join(self.cur_dir, 'dinov2_utils'))
        sys.path.insert(0, osp.join(self.cur_dir, 'lora_utils'))
        sys.path.insert(0, osp.join(self.cur_dir, 'vit_adapter_utils'))
        from util.dir import add_pypath
        # add_pypath(osp.join(self.data_dir))
        for dataset in os.listdir('datasets'):
            if dataset not in ['humandata.py', '__pycache__', 'dataset.py']:
                add_pypath(osp.join(self.root_dir, 'data', dataset))
        add_pypath('datasets')
        add_pypath(self.data_dir)

    def prepare_dirs(self, exp_name):
        time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = osp.join(self.root_dir, f'{exp_name}_{time_str}')
        self.model_dir = osp.join(self.output_dir, 'model_dump')
        self.vis_dir = osp.join(self.output_dir, 'vis')
        self.log_dir = osp.join(self.output_dir, 'log')
        self.code_dir = osp.join(self.output_dir, 'code')
        self.result_dir = osp.join(self.output_dir.split('/')[:-1])
        from util.dir import make_folder
        make_folder(self.model_dir)
        make_folder(self.vis_dir)
        make_folder(self.log_dir)
        make_folder(self.code_dir)
        make_folder(self.result_dir)

        ## copy some code to log dir as a backup
        copy_files = [
            'main/train.py', 'main/test.py', 'common/base.py', 'main/OSX.py',
            'common/nets', 'main/OSX_WoDecoder.py', 'data/dataset.py',
            'data/MSCOCO/MSCOCO.py', 'data/AGORA/AGORA.py'
        ]
        for file in copy_files:
            os.system(f'cp -r {self.root_dir}/{file} {self.code_dir}')

    def update_test_config(self, testset, agora_benchmark, shapy_eval_split,
                           pretrained_model_path, use_cache):
        self.testset = testset
        self.agora_benchmark = agora_benchmark
        self.pretrained_model_path = pretrained_model_path
        self.shapy_eval_split = shapy_eval_split
        self.use_cache = use_cache

    def update_config(self, num_gpus, exp_name):
        self.num_gpus = num_gpus
        self.exp_name = exp_name

        self.prepare_dirs(self.exp_name)

        # Save
        cfg_save = MMConfig(self.__dict__)
        cfg_save.dump(osp.join(self.code_dir, 'config_base.py'))


cfg = Config()
cfg.get_config_fromfile('config/aios_smplx.py')
# import ipdb;ipdb.set_trace()
