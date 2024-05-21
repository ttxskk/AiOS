import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
import numpy as np
import torch

import util.misc as utils
from detrsmpl.data.datasets import build_dataloader
from mmcv.parallel import MMDistributedDataParallel
# from detrsmpl.core.conventions.keypoints_mapping import
from engine import evaluate, train_one_epoch, inference
# import models
from util.config import DictAction, cfg
from util.utils import ModelEma
# sys.path.append('./osx/main')
# from config import cfg
import shutil
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter   

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector',
                                     add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    # parser.add_argument('--exp_name', default='data/log/smplx_test', type=str)
    # dataset parameters
    
    # training parameters
    parser.add_argument('--output_dir',
                        default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device',
                        default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path',
                        help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--to_vid', action='store_true')
    parser.add_argument('--inference', action='store_true')
    # distributed training parameters


    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank',
                        type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument('--amp',
                        action='store_true',
                        help='Train with mixed precision')

    parser.add_argument('--inference_input', default=None, type=str)
    return parser


def build_model_main(args, cfg):
    print(args.modelname)
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors, postprocessors_aios = build_func(
        args, cfg)
    return model, criterion, postprocessors, postprocessors_aios


def main(args):
    
    utils.init_distributed_mode(args)
    
    # time.sleep(args.rank * 0.02)
    # dist.barrier()
    # debugpy.breakpoint()
    print('Loading config file from {}'.format(args.config_file))
    
    shutil.copy2(args.config_file,'config/aios_smplx.py')
    torch.distributed.barrier()
    from config.config import cfg
    from datasets.dataset import MultipleDatasets
    
    
    # cfg.fromfile(args.config_file)
    # cfg.get_config_fromfile(args.config_file)

 
    

    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, 'config_cfg.py')
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, 'config_args_raw.json')
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            continue
            raise ValueError('Key {} can used by args only'.format(k))

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False


    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'),
                          distributed_rank=args.rank,
                          color=False,
                          name='detr')
    logger.info('git:\n  {}\n'.format(utils.get_sha()))
    logger.info('Command: ' + ' '.join(sys.argv))
    writer = None
    if args.rank == 0:
        writer = SummaryWriter(args.output_dir)
        save_json_path = os.path.join(args.output_dir, 'config_args_all.json')
        # print("args:", vars(args))
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info('Full config saved to {}'.format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info('args: ' + str(args) + '\n')

    if args.frozen_weights is not None:
        assert args.masks, 'Frozen training is meant for segmentation only'

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors, postprocessors_aios = build_model_main(
        args, cfg)

    wo_class_error = False
    model.to(device)

    # ema
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    model_without_ddp = model
    if args.distributed:

        model = MMDistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    logger.info('number of params:' + str(n_parameters))
    logger.info('params:\n' + json.dumps(
        {n: p.numel()
         for n, p in model.named_parameters() if p.requires_grad},
        indent=2))

    # for name, param in model.named_parameters():
    #     if "smpl_cam_f_embed" not in name:
    #         param.requires_grad = False


    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)


    
    logger.info('Creating dataset...')
    if not args.eval:
        trainset= []
        for trainset_i,v in cfg.trainset_partition.items():
            exec('from datasets.' + trainset_i +
                ' import ' + trainset_i)
            trainset.append(
                eval(trainset_i)(transforms.ToTensor(), 'train'))
        trainset_loader = MultipleDatasets(trainset, make_same_len=False,partition=cfg.trainset_partition)
    
        data_loader_train = build_dataloader(
            trainset_loader,
            args.batch_size,
        0  if 'workers_per_gpu' in args else 2,
            dist=args.distributed)
    exec('from datasets.' + cfg.testset +
            ' import ' + cfg.testset)
    
    
    
    if not args.inference:
        dataset_val = eval(cfg.testset)(transforms.ToTensor(), "test")
    else:
        dataset_val = eval(cfg.testset)(args.inference_input,args.output_dir)
        
    data_loader_val = build_dataloader(
    dataset_val,
    args.batch_size,
    0  if 'workers_per_gpu' in args else 2,
    dist=args.distributed,
    shuffle=False)
        
        
    
    
    # import ipdb; ipdb.set_trace()

   

    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(data_loader_train),
            epochs=args.epochs,
            pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    


    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume,
                                                            map_location='cpu',
                                                            check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(
                    utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path,
                                map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        
        _tmp_st = OrderedDict({
            k: v
            for k, v in utils.clean_state_dict(checkpoint).items()
            if check_keep(k, _ignorekeywordlist)
        })
        logger.info('Ignore keys: {}'.format(json.dumps(ignorelist, indent=2)))
        # Change This
        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        print('loading')
        logger.info(str(_load_output))

        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)    
        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))


    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'

        if args.inference_input is not None and args.inference:
            inference(model,
                     criterion,
                     postprocessors,
                     data_loader_val,
                     device,
                     args.output_dir,
                     wo_class_error=wo_class_error,
                     args=args)            
        else:
            
            from config.config import cfg
            cfg.result_dir=args.output_dir
            cfg.exp_name=args.pretrain_model_path
            evaluate(model,
                     criterion,
                     postprocessors,
                     data_loader_val,
                     device,
                     args.output_dir,
                     wo_class_error=wo_class_error,
                     args=args)

        return

    print('Start training')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            wo_class_error=wo_class_error,
            lr_scheduler=lr_scheduler,
            args=args,
            logger=(logger if args.save_log else None),
            ema_m=ema_m,
            tf_writer=writer)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (
                    epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir /
                                        f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if args.use_ema:
                    weights.update({
                        'ema_model': ema_m.module.state_dict(),
                    })
                utils.save_on_master(weights, checkpoint_path)
        log_stats = {
            **{f'train_{k}': v
               for k, v in train_stats.items()},
        }

        ep_paras = {'epoch': epoch, 'n_parameters': n_parameters}
        log_stats.update(ep_paras)
        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass

        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / 'log.txt').open('a') as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script',
                                     parents=[get_args_parser()])
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
