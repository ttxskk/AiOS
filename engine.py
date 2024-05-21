import math
import os
import time
import datetime
import sys
from typing import Iterable
import os.path as osp
import torch
import util.misc as utils
from collections import OrderedDict
import mmcv
import torch
import numpy as np
import torch.distributed as dist
from mmcv.runner import get_dist_info
from detrsmpl.apis.test import collect_results_cpu, collect_results_gpu
from detrsmpl.utils.ffmpeg_utils import images_to_video
from torch.utils.tensorboard import SummaryWriter   
import json

def round_float(items):
    if isinstance(items, list):
        return [round_float(item) for item in items]
    elif isinstance(items, float):
        return round(items, 3)
    elif isinstance(items, np.ndarray):
        return round_float(float(items))
    elif isinstance(items, torch.Tensor):
        return round_float(items.detach().cpu().numpy())
    else:
        return items

def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    max_norm: float = 0,
                    wo_class_error=False,
                    lr_scheduler=None,
                    args=None,
                    logger=None,
                    ema_m=None,
                    tf_writer=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    # criterion_smpl.to(device)
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter(
            'class_error', utils.SmoothedValue(window_size=1,
                                               fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    # import pdb
    # pdb.set_trace()
    # metric_logger.log_every(data_loader, print_freq, header, logger=logger)
    
    for step_i, data_batch in enumerate(metric_logger.log_every(data_loader,
                                              print_freq,
                                              header,
                                              logger=logger)):
        # for data_batch in data_loader:
        
        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                # outputs = model(samples, targets)
                outputs, targets, data_batch_nc = model(data_batch)
            else:
                outputs, targets, data_batch_nc = model(data_batch)
            
            # torch.cuda.empty_cache() 
            
            ['hand_kp3d_4', 'face_kp3d_4', 'hand_kp2d_4',]
            loss_dict = criterion(outputs, targets, data_batch=data_batch_nc)
            weight_dict = criterion.weight_dict
            
            for k,v in weight_dict.items():
                for n in ['hand_kp3d_4', 'face_kp3d_4', 'hand_kp2d_4']:
                    if n in k:
                        weight_dict[k] = weight_dict[k]/10

            losses = sum(loss_dict[k] * weight_dict[k]
                         for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f'{k}_unscaled': v
            for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        # loss_smpl_sum = sum(loss_smpl.values())

        # loss_value_smpl = loss_smpl_sum.item()

        loss_value = losses_reduced_scaled.item()
        # loss_value = loss_value+loss_value_smpl
        for k,v in weight_dict.items():
            for n in ['hand_kp3d_4', 'face_kp3d_4', 'hand_kp2d_4']:
                if n in k:
                    weight_dict[k] = weight_dict[k]*10
        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)
        rank, _ = get_dist_info()

        if rank == 0:
            tf_writer.add_scalar(
                'loss', round_float(loss_value), step_i + len(data_loader) * epoch)
            for k, v in loss_dict_reduced_scaled.items():
                tf_writer.add_scalar(
                    k, round_float(v), step_i + len(data_loader) * epoch)
            for k, v in loss_dict_reduced_unscaled.items():
                tf_writer.add_scalar(
                    k, round_float(v), step_i + len(data_loader) * epoch)
        json_log = OrderedDict()
        json_log['now_time'] = str(datetime.datetime.now())
        json_log['epoch'] = epoch
        json_log['lr'] = optimizer.param_groups[0]['lr']
        json_log['loss'] = round_float(loss_value)
        for k, v in loss_dict_reduced_scaled.items():
            json_log[k] = round_float(v)

        for k, v in loss_dict_reduced_unscaled.items():
            json_log[k] = round_float(v)

        if rank == 0:
            log_path = os.path.join(args.output_dir, 'train.log.json')
            with open(log_path, 'a+') as f:
                mmcv.dump(json_log, f, file_format='json')
                f.write('\n')

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print('BREAK!' * 5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    resstat = {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items() if meter.count > 0
    }
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update(
            {f'weight_{k}': v
             for k, v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model,
             criterion,
             postprocessors,
             data_loader,
             device,
             output_dir,
             wo_class_error=False,
             tmpdir=None,
             gpu_collect=False,
             args=None,
             logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter='  ')
    if not wo_class_error:
        metric_logger.add_meter(
            'class_error', utils.SmoothedValue(window_size=1,
                                               fmt='{value:.2f}'))
    header = 'Test:'
    iou_types = tuple(k for k in ('bbox', 'keypoints'))
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print('useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'.format(
            useCats))

    _cnt = 0
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()

    if rank == 0:
        # Check if tmpdir is valid for cpu_collect
        if (not gpu_collect) and (tmpdir is not None and osp.exists(tmpdir)):
            raise OSError((f'The tmpdir {tmpdir} already exists.',
                           ' Since tmpdir will be deleted after testing,',
                           ' please make sure you specify an empty one.'))
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)
    # i=0
    cur_sample_idx = 0
    eval_result = {}
    # print()
    cur_eval_result_list = []
    rank, world_size = get_dist_info()

    for data_batch in metric_logger.log_every(
        data_loader, 10, header, logger=logger):
        # i = i+1
        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                # outputs = model(samples, targets)
                outputs, targets, data_batch_nc = model(data_batch)
            else:
                outputs,targets, data_batch_nc = model(data_batch)
        
        orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        result = postprocessors['bbox'](outputs, orig_target_sizes, targets, data_batch_nc,dataset = dataset)    
        
        # DOING SMPLer-X Test
        cur_eval_result = dataset.evaluate(result,cur_sample_idx)
        
        cur_eval_result_list.append(cur_eval_result)
        # for cur_eval_result in cur_eval_result_list:
        #     for k, v in cur_eval_result.items():
        #         if k in eval_result:
        #             eval_result[k] += v
        #         else:
        #             eval_result[k] = v
        cur_sample_idx += len(result)
    cur_eval_result_new = collect_results_cpu(cur_eval_result_list, len(dataset))
    
    if rank == 0:
        
        cntt = 0
        for res in cur_eval_result_new:

            for k,v in res.items():
                if len(v)>0:
                    if k != 'ann_idx' and k != 'img_path':                 
                        if k in eval_result:
                            eval_result[k].append(v)
                        else:
                            eval_result[k] = [v]

        for k,v in eval_result.items():
            
            # if k == 'mpvpe_all' or k == 'pa_mpvpe_all':
            eval_result[k] = np.concatenate(v)
            
            
        dataset.print_eval_result(eval_result)
        # print(len(cur_eval_result_new))
        
    # dataset.print_eval_result(eval_result)
        # if i==4:
        #     break
    # collect results from all ranks
    # if rank == 0:
    #     import pdb
    #     pdb.set_trace()  
    
    # return results




@torch.no_grad()
def inference(model,
             criterion,
             postprocessors,
             data_loader,
             device,
             output_dir,
             wo_class_error=False,
             tmpdir=None,
             gpu_collect=False,
             args=None,
             logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter='  ')
    if not wo_class_error:
        metric_logger.add_meter(
            'class_error', utils.SmoothedValue(window_size=1,
                                               fmt='{value:.2f}'))
    header = 'Test:'
    iou_types = tuple(k for k in ('bbox', 'keypoints'))
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print('useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'.format(
            useCats))

    _cnt = 0
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()

    if rank == 0:
        # Check if tmpdir is valid for cpu_collect
        if (not gpu_collect) and (tmpdir is not None and osp.exists(tmpdir)):
            raise OSError((f'The tmpdir {tmpdir} already exists.',
                           ' Since tmpdir will be deleted after testing,',
                           ' please make sure you specify an empty one.'))
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)
    # i=0
    cur_sample_idx = 0
    eval_result = {}
    for data_batch in metric_logger.log_every(
        data_loader, 10, header, logger=logger):
        # i = i+1
        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                # outputs = model(samples, targets)
                outputs, targets, data_batch_nc = model(data_batch)
            else:
                outputs,targets, data_batch_nc = model(data_batch)
        
        orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        result = postprocessors['bbox'](outputs, orig_target_sizes, targets, data_batch_nc)    

        eval_out = dataset.inference(result)
        eval_result.update(eval_out)
    eval_result.update(eval_out)
    # print('ssss',dataset.result_img_dir,dataset.out_path,dataset.img_name)
    time.sleep(3)
    if rank == 0 and args.to_vid:
        # img_tmp = dataset.img_path[0]
        if hasattr(dataset,'result_img_dir'):
            import shutil
            images_to_video(dataset.result_img_dir, os.path.join(dataset.output_path,dataset.img_name+'_demo.mp4'),remove_raw_file=False, fps=30)
            shutil.rmtree(dataset.result_img_dir)
            shutil.rmtree(dataset.tmp_dir)


