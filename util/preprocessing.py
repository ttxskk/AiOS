import numpy as np
import cv2
import random
from config.config import cfg
import math
from .human_models import smpl_x, smpl
from .transforms import cam2pixel, transform_joint_to_other_db, transform_joint_to_other_db_batch
from plyfile import PlyData, PlyElement
import torch
import torch.distributed as dist

def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError('Fail to read %s' % path)

    if order == 'RGB':
        img = img[:, :, ::-1].copy()

    img = img.astype(np.float32)
    return img


def get_bbox(joint_img, joint_valid, extend_ratio=1.2):
    x_img, y_img = joint_img[:, 0], joint_img[:, 1]
    x_img = x_img[joint_valid == 1]
    y_img = y_img[joint_valid == 1]
    xmin = min(x_img)
    ymin = min(y_img)
    xmax = max(x_img)
    ymax = max(y_img)

    x_center = (xmin + xmax) / 2.
    width = xmax - xmin
    xmin = x_center - 0.5 * width * extend_ratio
    xmax = x_center + 0.5 * width * extend_ratio

    y_center = (ymin + ymax) / 2.
    height = ymax - ymin
    ymin = y_center - 0.5 * height * extend_ratio
    ymax = y_center + 0.5 * height * extend_ratio

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox


def sanitize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w > 0 and h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2 - x1 + 1, y2 - y1 + 1])
    else:
        bbox = None

    return bbox
def resize(ori_shape, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple
    # import ipdb; ipdb.set_trace(context=15)
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if min_original_size ==0:
                print('min_original_size:',min_original_size)
            if max_original_size / (min_original_size) * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (w, h)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (ow, oh)

    def get_size(ori_shape, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(ori_shape, size, max_size)

    size = get_size(ori_shape, size, max_size)

    
    return size

def process_bbox(bbox, img_width, img_height, ratio=1.):
    
    bbox = np.array(bbox, dtype=np.float32)
    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    bbox[2] = w * ratio
    bbox[3] = h * ratio
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.

    bbox = sanitize_bbox(bbox, img_width, img_height)    
    return bbox


def get_aug_config(data_name):
    scale_factor = 0.25
    rot_factor = 30
    color_factor = 0.2
    crop_factor = 0.1
    
    if data_name == 'GTA_Human2':
        sample_ratio = 0.5
        sample_prob = 0.5
    elif data_name == 'AGORA_MM':
        sample_ratio = 0.5
        sample_prob = 0.7
    elif data_name == 'BEDLAM':
        sample_ratio = 0.6
        sample_prob = 0.7 
    elif data_name == 'COCO_NA':
        sample_ratio = 0.6
        sample_prob = 0.7
    elif data_name == 'CrowdPose':
        sample_ratio = 0.5
        sample_prob = 0.5
    elif data_name == 'PoseTrack':
        sample_ratio = 0.5
        sample_prob = 0.3
    elif data_name == 'UBody_MM':
        sample_ratio = 0.5
        sample_prob = 0.3        
    elif data_name == 'ARCTIC':
        sample_ratio = 0.5
        sample_prob = 0.3
    elif data_name == 'RICH':
        sample_ratio = 0.5
        sample_prob = 0.3
    elif data_name == 'EgoBody_Egocentric':
        sample_ratio = 0.5
        sample_prob = 0.3
    elif data_name == 'EgoBody_Kinect':
        sample_ratio = 0.5
        sample_prob = 0.3
    else:
        sample_ratio = 0.5
        sample_prob = 0.3
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([
        random.uniform(c_low, c_up),
        random.uniform(c_low, c_up),
        random.uniform(c_low, c_up)
    ])
    do_flip = random.random() < 0.5
    crop_hw = np.array([
        0.2 - np.random.rand() * crop_factor, 0.2 - np.random.rand() * crop_factor
    ])
    # crop_hw = np.array([
    #     0.3 - np.random.rand() * crop_factor, 0.3 - np.random.rand() * crop_factor
    # ])
    return scale, rot, color_scale, do_flip, crop_hw, sample_ratio, sample_prob

def augmentation_keep_size(img, bbox, data_split):
    ori_shape = img.shape[:2][::-1]
    if getattr(cfg, 'no_aug', False) and data_split == 'train':
        scale, rot, color_scale, do_flip,size,crop = 1.0, 0.0, np.array([1, 1, 1]), False, ori_shape, np.array([1,1])
        
        size = random.choice(cfg.train_sizes)
        max_size = cfg.train_max_size
    elif data_split == 'train':
        scale, rot, color_scale, do_flip, crop = get_aug_config()
        rot=0
        # scale, rot, do_flip, crop = 1.0, 0.0, False, np.array([1,1])
        size = random.choice(cfg.train_sizes)
        max_size = cfg.train_max_size
    else:
        scale, rot, color_scale, do_flip, crop = 1.0, 0.0, np.array([1, 1, 1]), False, np.array([1,1])
        size = random.choice(cfg.test_sizes)
        max_size = cfg.test_max_size
    
    crop_bbox_wh = (bbox[2:]*crop).astype(np.uint32)
    xy_range = img.shape[:2][::-1]-crop_bbox_wh
    crop_bbox_xywh = np.array([np.random.randint(0,xy_range[0]+1),np.random.randint(0,xy_range[1]+1),crop_bbox_wh[0],crop_bbox_wh[1]])
    reshape_size = resize(crop_bbox_xywh[2:], size, max_size)
    
    img, trans, inv_trans = generate_patch_image(img, crop_bbox_xywh, 1, rot, do_flip, reshape_size[::-1])
    img = np.clip(img * color_scale[None, None, :], 0, 255)
    return img, trans, inv_trans, rot, do_flip

def augmentation_instance_sample(img, bbox, data_split,data,dataname):
    ori_shape = img.shape[:2][::-1]
    
    if getattr(cfg, 'no_aug', False) and data_split == 'train':
        scale, rot, color_scale, do_flip,size,crop,sample_ratio,sample_prob = 1.0, 0.0, np.array([1, 1, 1]), False, ori_shape, np.array([1,1]), 0,0
        
        size = random.choice(cfg.train_sizes)
        max_size = cfg.train_max_size
    elif data_split == 'train':
        scale, rot, color_scale, do_flip, crop, sample_ratio,sample_prob = get_aug_config(dataname)
        rot=0
        # scale, rot, do_flip, crop = 1.0, 0.0, False, np.array([1,1])
        size = random.choice(cfg.train_sizes)
        max_size = cfg.train_max_size
    else:
        scale, rot, color_scale, do_flip, crop,sample_ratio,sample_prob = 1.0, 0.0, np.array([1, 1, 1]), False, np.array([1,1]),0,0
        size = random.choice(cfg.test_sizes)
        max_size = cfg.test_max_size
    
    
    if random.random() < sample_prob:
        crop_person_number = len(data['bbox'])
        
        if random.random() < sample_ratio:
            if random.random() < 0.6:
                crop_person_number_sample = 1
            else:
                crop_person_number_sample = np.random.randint(crop_person_number) + 1
        else:
            crop_person_number_sample = crop_person_number
        sample_ids = np.array(
            random.sample(list(range(crop_person_number)), crop_person_number_sample))
        
        bbox_xyxy = []

        bbox_xyxy = np.stack(data['bbox'],axis=0)[sample_ids]

        leftTop_ = bbox_xyxy[:, :2]
        leftTop_ = np.array([np.min(leftTop_[:, 0]), np.min(leftTop_[:, 1])])
        rightBottom_ = bbox_xyxy[:, 2:4]
        rightBottom_ = np.array(
            [np.max(rightBottom_[:, 0]),
                np.max(rightBottom_[:, 1])])
        crop_bbox_xyxy = np.concatenate([leftTop_, rightBottom_])
        crop_bbox_xywh = crop_bbox_xyxy.copy()
        crop_bbox_xywh[2:] = crop_bbox_xywh[2:]-crop_bbox_xywh[:2]
        crop_bbox_xywh = adjust_bounding_box(crop_bbox_xywh,ori_shape[0],ori_shape[1])
    else:
        crop_bbox_xywh = bbox.copy()
    reshape_size = resize(crop_bbox_xywh[2:], size, max_size)
    # try:
    #     reshape_size = resize(crop_bbox_xywh[2:], size, max_size)
    # except Exception as e:
    #     print(crop_bbox_xywh)
    #     print(size)
    #     print(max_size)
    #     raise e
    img, trans, inv_trans = generate_patch_image(img, crop_bbox_xywh, 1, rot, do_flip, reshape_size[::-1])
    img = np.clip(img * color_scale[None, None, :], 0, 255)
    return img, trans, inv_trans, rot, do_flip

def adjust_bounding_box(input_bbox,image_width, image_height):
    left_x, left_y, width, height = input_bbox
    # Calculate original bounding box center
    original_center_x = left_x + width / 2
    original_center_y = left_y + height / 2

    # Calculate target aspect ratio
    target_aspect_ratio = image_width / image_height

    # Adjust width and height to match target aspect ratio
    if width / height > target_aspect_ratio:
        # Bounding box is wider, adjust height
        new_height = width / target_aspect_ratio
        new_width = width
    else:
        # Bounding box is taller, adjust width
        new_width = height * target_aspect_ratio
        new_height = height

    # Calculate new bounding box center
    new_center_x = original_center_x
    new_center_y = original_center_y

    # Check if the adjusted bounding box is out of the image boundaries
    if new_center_x - new_width / 2 < 0:
        # Shift the bounding box to the right to fit within the image
        new_center_x = new_width / 2
    elif new_center_x + new_width / 2 > image_width:
        # Shift the bounding box to the left to fit within the image
        new_center_x = image_width - new_width / 2

    if new_center_y - new_height / 2 < 0:
        # Shift the bounding box down to fit within the image
        new_center_y = new_height / 2
    elif new_center_y + new_height / 2 > image_height:
        # Shift the bounding box up to fit within the image
        new_center_y = image_height - new_height / 2

    # Calculate adjusted left x and left y of the bounding box and convert to integers
    adjusted_left_x = int(new_center_x - new_width / 2)
    adjusted_left_y = int(new_center_y - new_height / 2)
    # Ensure width and height are integers as well
    adjusted_width = int(new_width)
    adjusted_height = int(new_height)

    # Return adjusted bounding box coordinates (left x, left y, width, height)
    return np.array([adjusted_left_x, adjusted_left_y, adjusted_width, adjusted_height])


def generate_patch_image(cvimg, bbox, scale, rot, do_flip, out_shape):
    img = cvimg.copy()
    
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height,
                                    out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img,
                               trans, (int(out_shape[1]), int(out_shape[0])),
                               flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x,
                                        bb_c_y,
                                        bb_width,
                                        bb_height,
                                        out_shape[1],
                                        out_shape[0],
                                        scale,
                                        rot,
                                        inv=True)

    return img_patch, trans, inv_trans


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x,
                            c_y,
                            src_width,
                            src_height,
                            dst_width,
                            dst_height,
                            scale,
                            rot,
                            inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32),
                            rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32),
                             rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans


def process_db_coord_batch_no_valid(joint_img, joint_cam, do_flip,
                           img_shape, flip_pairs, img2bb_trans, rot,
                           src_joints_name, target_joints_name,
                           input_img_shape):
    joint_img_original = joint_img.copy()
    joint_img, joint_cam = joint_img.copy(), joint_cam.copy()
    
    # flip augmentation
    if do_flip:
        joint_cam[:, :, 0] = -joint_cam[:, :, 0]
        joint_img[:, :, 0] = img_shape[1] - 1 - joint_img[:, :, 0]
        for pair in flip_pairs:
            joint_img[:, pair[0], :], joint_img[:, pair[
                1], :] = joint_img[:, pair[1], :].copy(
                ), joint_img[:, pair[0], :].copy()
            joint_cam[:, pair[0], :], joint_cam[:, pair[
                1], :] = joint_cam[:, pair[1], :].copy(
                ), joint_cam[:, pair[0], :].copy()
            
    # 3D data rotation augmentation
    rot_aug_mat = np.array(
        [[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
         [np.sin(np.deg2rad(-rot)),
          np.cos(np.deg2rad(-rot)), 0], [0, 0, 1]],
        dtype=np.float32)
    num_p, num_joints, joints_dim = joint_cam.shape
    joint_cam = joint_cam.reshape(num_p * num_joints, joints_dim)
    joint_cam[:,:-1] = np.dot(rot_aug_mat, joint_cam[:,:-1].transpose(1, 0)).transpose(1, 0)
    joint_cam = joint_cam.reshape(num_p, num_joints, joints_dim)
    
    # affine transformation
    joint_img_xy1 = \
        np.concatenate((joint_img[:, :, :2], np.ones_like(joint_img[:, :, :1])), 2)
    joint_img_xy1 = joint_img_xy1.reshape(num_p * num_joints, 3)

    joint_img[:, :, :2] = np.dot(img2bb_trans,
                                 joint_img_xy1.transpose(1, 0)).transpose(
                                     1, 0).reshape(num_p, num_joints, 2)

    joint_img[:, :,
              0] = joint_img[:, :,
                             0] / input_img_shape[1] * cfg.output_hm_shape[2]
    joint_img[:, :,
              1] = joint_img[:, :,
                             1] / input_img_shape[0] * cfg.output_hm_shape[1]

    # check truncation
    # TODO
    # remove 3rd 
    joint_trunc =  ((joint_img_original[:,:, 0] >= 0) * (joint_img[:,:, 0] >= 0) * (joint_img[:,:, 0] < cfg.output_hm_shape[2]) * \
                    (joint_img_original[:,:, 1] >= 0) *(joint_img[:,:, 1] >= 0) * (joint_img[:,:, 1] < cfg.output_hm_shape[1]) * \
                    joint_img[:,:, -1]
                    ).reshape(num_p, -1, 1).astype(np.float32)


    # transform joints to target db joints

    joint_img = transform_joint_to_other_db_batch(joint_img, src_joints_name,
                                                  target_joints_name)
    joint_cam_wo_ra = transform_joint_to_other_db_batch(
        joint_cam, src_joints_name, target_joints_name)
    
    joint_trunc = transform_joint_to_other_db_batch(joint_trunc,
                                                    src_joints_name,
                                                    target_joints_name)

    # root-alignment, for joint_cam input wo ra
    joint_cam_ra = joint_cam_wo_ra.copy()
    joint_cam_ra[:,:,:3] = joint_cam_ra[:,:,:3] - joint_cam_ra[:, smpl_x.root_joint_idx,
                                               None, :3]  # root-relative
    joint_cam_ra[:, smpl_x.joint_part[
        'lhand'], :3] = joint_cam_ra[:, smpl_x.joint_part[
            'lhand'], :3] - joint_cam_ra[:, smpl_x.lwrist_idx,
                                        None, :3]  # left hand root-relative
    joint_cam_ra[:, smpl_x.joint_part[
        'rhand'], :3] = joint_cam_ra[:, smpl_x.joint_part[
            'rhand'], :3] - joint_cam_ra[:, smpl_x.rwrist_idx,
                                        None, :3]  # right hand root-relative
    joint_cam_ra[:, smpl_x.
                 joint_part['face'], :3] = joint_cam_ra[:, smpl_x.joint_part[
                     'face'], :3] - joint_cam_ra[:, smpl_x.neck_idx,
                                                None, :3]  # face root-relative
    return joint_img, joint_cam_wo_ra, joint_cam_ra, joint_trunc



def process_human_model_output_batch_ubody(human_model_param,
                                     do_flip,
                                     rot,
                                     as_smplx,
                                     part_valid
                                     ):
    num_person = human_model_param['body_pose'].shape[0]
    human_model = smpl_x
    rotation_valid = np.ones((num_person,smpl_x.orig_joint_num), dtype=np.float32)
    coord_valid = np.ones((num_person,smpl_x.joint_num), dtype=np.float32)
    # expr_valid = np.ones((num_person), dtype=np.float32)
    # shape_valid = np.ones((num_person), dtype=np.float32)
    # root_pose, body_pose, shape, trans = human_model_param['root_pose'], human_model_param['body_pose'], \
    #                                         human_model_param['shape'], human_model_param['trans']
    
    if 'smplx_valid' in human_model_param:
        smplx_valid = human_model_param['smplx_valid']
        shape_valid = human_model_param['smplx_valid']
    else:
        smplx_valid = np.ones(num_person, dtype=np.bool8)
        shape_valid = np.ones(num_person, dtype=np.bool8)

    if 'expr_valid' in human_model_param:
        expr_valid = human_model_param['expr_valid']
    else:
        expr_valid = np.ones(num_person, dtype=np.bool8)
    expr_valid*=smplx_valid

    if 'face_valid' in human_model_param:
        face_valid = human_model_param['face_valid']
    else:
        face_valid = np.ones(num_person, dtype=np.bool8)
    face_valid *= smplx_valid

    # check lhand valid key exsits
    if 'lhand_valid' in human_model_param:  
        lhand_valid = human_model_param['lhand_valid']
    else:
        lhand_valid = np.ones(num_person, dtype=np.bool8)
    lhand_valid*=smplx_valid
    
    # check rhand valid key exsits
    if 'rhand_valid' in human_model_param:
        rhand_valid = human_model_param['rhand_valid']
    else:
        rhand_valid = np.ones(num_person, dtype=np.bool8)
    rhand_valid*=smplx_valid
    
    # check validation of the smplx parameters
    if 'body_pose' in human_model_param \
        and human_model_param['body_pose'] is not None:
        root_pose, body_pose = human_model_param['root_pose'], human_model_param['body_pose']
        shape, trans = human_model_param['shape'], human_model_param['trans']
        root_pose = torch.FloatTensor(root_pose).view(num_person, 1, 3)
        body_pose = torch.FloatTensor(body_pose).view(num_person, -1, 3)
        shape = torch.FloatTensor(shape).view(num_person, -1)
        trans = torch.FloatTensor(trans).view(num_person,-1)
    else:
        root_pose = np.zeros((num_person, 3), dtype=np.float32)
        body_pose = np.zeros((num_person, 3 * len(smpl_x.orig_joint_part['body'])), dtype=np.float32)
        shape = np.zeros((num_person, 10), dtype=np.float32)
        trans = np.zeros((num_person, 3), dtype=np.float32)
        rotation_valid[:, smpl_x.orig_joint_part['body']] = 0
        coord_valid[:, smpl_x.joint_part['body']] = 0
    body_pose*=smplx_valid[:, None, None]
    root_pose*=smplx_valid[:, None, None]
    shape*=smplx_valid[:, None]
    trans*=smplx_valid[:, None]
    rotation_valid[:, smpl_x.orig_joint_part['body']]*=smplx_valid[:,None]   
    coord_valid[:, smpl_x.joint_part['body']]*=smplx_valid[:,None]  
    
    # check validation of the smplx parameters
    if 'lhand_pose' in human_model_param \
        and human_model_param['lhand_pose'] is not None:
        lhand_pose = human_model_param['lhand_pose']
        lhand_pose = torch.FloatTensor(lhand_pose).view(num_person, -1, 3)
        # lhand_valid = part_valid['lhand']
        # rotation_valid[:, smpl_x.orig_joint_part['lhand']]*=lhand_valid[:,None]
        # coord_valid[:, smpl_x.joint_part['lhand']]*=lhand_valid[:,None]
    else:
        lhand_pose = np.zeros((num_person, 3 * len(smpl_x.orig_joint_part['lhand'])), dtype=np.float32)
        rotation_valid[:, smpl_x.orig_joint_part['lhand']] = 0
        coord_valid[:, smpl_x.joint_part['lhand']] = 0
    
    lhand_pose*=lhand_valid[:,None,None]    
    rotation_valid[:, smpl_x.orig_joint_part['lhand']]*=lhand_valid[:,None]   
    coord_valid[:, smpl_x.joint_part['lhand']]*=lhand_valid[:,None]  
    
    if 'rhand_pose' in human_model_param \
        and human_model_param['rhand_pose'] is not None:
        rhand_pose = human_model_param['rhand_pose']
        rhand_pose = torch.FloatTensor(rhand_pose).view(num_person, -1, 3)
        # rhand_valid = part_valid['rhand']
        # rotation_valid[:, smpl_x.orig_joint_part['rhand']]*=rhand_valid[:,None]
        # coord_valid[:, smpl_x.joint_part['rhand']]*=rhand_valid[:,None]
    else:
        rhand_pose = np.zeros((num_person, 3 * len(smpl_x.orig_joint_part['rhand'])), dtype=np.float32)
        rotation_valid[:, smpl_x.orig_joint_part['rhand']] = 0
        coord_valid[:, smpl_x.joint_part['rhand']] = 0
    rhand_pose*=rhand_valid[:,None,None]  
    rotation_valid[:, smpl_x.orig_joint_part['rhand']]*=rhand_valid[:,None]   
    coord_valid[:, smpl_x.joint_part['rhand']]*=rhand_valid[:,None]
    
    if 'expr' in human_model_param  and \
        human_model_param['expr'] is not None:
        expr = human_model_param['expr']
        # face_valid = part_valid['face']
        # expr_valid = expr_valid*face_valid
    else:
        expr = np.zeros((num_person, smpl_x.expr_code_dim), dtype=np.float32)
        expr_valid = expr_valid*0
    expr*=face_valid[:,None]   
    expr = torch.FloatTensor(expr).view(num_person,-1)
    expr_valid*=face_valid # expr is invalid if face_valid is 0
    
    if 'jaw_pose' in human_model_param and \
        human_model_param['jaw_pose'] is not None:
        jaw_pose = human_model_param['jaw_pose']
        # face_valid = part_valid['face']
        # rotation_valid[:, smpl_x.orig_joint_part['face']]*=face_valid[:,None]
        # coord_valid[:, smpl_x.joint_part['face']]*=face_valid[:,None]
    else:
        jaw_pose = np.zeros((num_person, 3), dtype=np.float32)
        rotation_valid[:,smpl_x.orig_joint_part['face']] = 0
        coord_valid[:,smpl_x.joint_part['face']] = 0
        
    jaw_pose*=face_valid[:,None]
    jaw_pose = torch.FloatTensor(jaw_pose).view(num_person, -1, 3)
    rotation_valid[:, smpl_x.orig_joint_part['face']]*=face_valid[:,None]
    coord_valid[:, smpl_x.joint_part['face']]*=face_valid[:,None]
    
    if 'gender' in human_model_param and \
        human_model_param['gender'] is not None:
        gender = human_model_param['gender']
    else:
        gender = 'neutral'
    
    if as_smplx == 'smpl':
        rotation_valid[:,:] = 0
        rotation_valid[:,:21] = 1
        expr_valid = expr_valid*0
        coord_valid[:,:] = 0
        coord_valid[:,smpl_x.joint_part['body']] = 1
    
    root_pose = torch.FloatTensor(root_pose).view(num_person, 1, 3)
    body_pose = torch.FloatTensor(body_pose).view(num_person, -1, 3)
    lhand_pose = torch.FloatTensor( lhand_pose).view(num_person, -1, 3)
    rhand_pose = torch.FloatTensor(rhand_pose).view(num_person, -1, 3)
    jaw_pose = torch.FloatTensor(jaw_pose).view(num_person, -1, 3)

    shape = torch.FloatTensor(shape).view(num_person, -1)
    expr = torch.FloatTensor(expr).view(num_person,-1)
    trans = torch.FloatTensor(trans).view(num_person,-1)

    
    

    pose = torch.cat((root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose),dim=1)

    ## so far, data augmentations are not applied yet
    ## now, apply data augmentations

    
    # x,y affine transform, root-relative depth
    
    # 3D data rotation augmentation
    # rot_aug_mat = np.array(
    #     [[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
    #      [np.sin(np.deg2rad(-rot)),
    #       np.cos(np.deg2rad(-rot)), 0], [0, 0, 1]],
    #     dtype=np.float32)
    
    # parameters
    # flip pose parameter (axis-angle)
    if do_flip:
        for pair in human_model.orig_flip_pairs:
            pose[:, pair[0], :], pose[:,
                                      pair[1], :] = pose[:, pair[1], :].clone(
                                      ), pose[:, pair[0], :].clone()
            rotation_valid[:,pair[0]], rotation_valid[:,pair[1]] = rotation_valid[:,pair[1]].copy(), rotation_valid[:,
                pair[0]].copy()
        pose[:,:, 1:3] *= -1  # multiply -1 to y and z axis of axis-angle
    # rotate root pose
    pose = pose.numpy()
    root_pose = pose[:, human_model.orig_root_joint_idx, :]
    
    # for pose_i in range(len(root_pose)):
    #     root_pose_mat = cv2.Rodrigues(root_pose[pose_i])[0]
    #     root_pose[pose_i] = cv2.Rodrigues(np.dot(rot_aug_mat,
    #                                              root_pose_mat))[0][:, 0]

    pose[:, human_model.orig_root_joint_idx] = root_pose.reshape(num_person, 3)

    # change to mean shape if beta is too far from it
    # shape[(shape.abs() > 3).any(dim=1)] = 0.
    shape = shape.numpy().reshape(num_person, -1)
    
    
    # shape_valid = shape.sum(-1)!=0

    # return results
    pose = pose.reshape(num_person, -1)
    expr = expr.numpy().reshape(num_person, -1)

        
    return pose, shape, expr, rotation_valid, coord_valid, expr_valid, shape_valid

def process_human_model_output_batch_simplify(human_model_param,
                                     do_flip,
                                     rot,
                                     as_smplx, data_name=None
                                     ):
    num_person = human_model_param['body_pose'].shape[0]
    human_model = smpl_x
    rotation_valid = np.ones((num_person,smpl_x.orig_joint_num), dtype=np.float32)
    coord_valid = np.ones((num_person,smpl_x.joint_num), dtype=np.float32)
    # expr_valid = np.ones((num_person), dtype=np.float32)
    # shape_valid = np.ones((num_person), dtype=np.float32)
    # shape, trans = human_model_param['shape'], human_model_param['trans']
    # check smplx valid key exsits
    if 'smplx_valid' in human_model_param:
        smplx_valid = human_model_param['smplx_valid']
        shape_valid = human_model_param['smplx_valid']
    else:
        smplx_valid = np.ones(num_person, dtype=np.bool8)
        shape_valid = np.ones(num_person, dtype=np.bool8)
        
    if 'expr_valid' in human_model_param:
        expr_valid = human_model_param['expr_valid']
    else:
        expr_valid = np.ones(num_person, dtype=np.bool8)
    expr_valid*=smplx_valid
    
    # check face valid key exsits
    if 'face_valid' in human_model_param:
        face_valid = human_model_param['face_valid']
    else:
        face_valid = np.ones(num_person, dtype=np.bool8)
    face_valid *= smplx_valid
    
    # check lhand valid key exsits
    if 'lhand_valid' in human_model_param:  
        lhand_valid = human_model_param['lhand_valid']
    else:
        lhand_valid = np.ones(num_person, dtype=np.bool8)
    lhand_valid*=smplx_valid
    
    # check rhand valid key exsits
    if 'rhand_valid' in human_model_param:
        rhand_valid = human_model_param['rhand_valid']
    else:
        rhand_valid = np.ones(num_person, dtype=np.bool8)
    rhand_valid*=smplx_valid
    
    # check validation of the smplx parameters
    if 'body_pose' in human_model_param \
        and human_model_param['body_pose'] is not None:
        root_pose, body_pose = human_model_param['root_pose'], human_model_param['body_pose']
        shape, trans = human_model_param['shape'], human_model_param['trans']
        root_pose = torch.FloatTensor(root_pose).view(num_person, 1, 3)
        body_pose = torch.FloatTensor(body_pose).view(num_person, -1, 3)
        shape = torch.FloatTensor(shape).view(num_person, -1)
        trans = torch.FloatTensor(trans).view(num_person,-1)
    else:
        root_pose = np.zeros((num_person, 3), dtype=np.float32)
        body_pose = np.zeros((num_person, 3 * len(smpl_x.orig_joint_part['body'])), dtype=np.float32)
        shape = np.zeros((num_person, 10), dtype=np.float32)
        trans = np.zeros((num_person, 3), dtype=np.float32)
        rotation_valid[:, smpl_x.orig_joint_part['body']] = 0
        coord_valid[:, smpl_x.joint_part['body']] = 0
    body_pose*=smplx_valid[:, None, None]
    root_pose*=smplx_valid[:, None, None]
    shape*=smplx_valid[:, None]
    trans*=smplx_valid[:, None]
    rotation_valid[:, smpl_x.orig_joint_part['body']]*=smplx_valid[:,None]   
    coord_valid[:, smpl_x.joint_part['body']]*=smplx_valid[:,None]  
    
    if 'lhand_pose' in human_model_param \
        and human_model_param['lhand_pose'] is not None:
        lhand_pose = human_model_param['lhand_pose']
        lhand_pose = torch.FloatTensor(lhand_pose).view(num_person, -1, 3)
    else:
        lhand_pose = np.zeros((num_person, 3 * len(smpl_x.orig_joint_part['lhand'])), dtype=np.float32)
        rotation_valid[:, smpl_x.orig_joint_part['lhand']] = 0
        coord_valid[:, smpl_x.joint_part['lhand']] = 0 
        
    lhand_pose*=lhand_valid[:,None,None]    
    rotation_valid[:, smpl_x.orig_joint_part['lhand']]*=lhand_valid[:,None]   
    coord_valid[:, smpl_x.joint_part['lhand']]*=lhand_valid[:,None]  

    if 'rhand_pose' in human_model_param \
        and human_model_param['rhand_pose'] is not None:
        rhand_pose = human_model_param['rhand_pose']
        rhand_pose = torch.FloatTensor(rhand_pose).view(num_person, -1, 3)
    else:
        rhand_pose = np.zeros((num_person, 3 * len(smpl_x.orig_joint_part['rhand'])), dtype=np.float32)
        rotation_valid[:, smpl_x.orig_joint_part['rhand']] = 0
        coord_valid[:, smpl_x.joint_part['rhand']] = 0
    rhand_pose*=rhand_valid[:,None,None]  
    rotation_valid[:, smpl_x.orig_joint_part['rhand']]*=rhand_valid[:,None]   
    coord_valid[:, smpl_x.joint_part['rhand']]*=rhand_valid[:,None]
    
    # face valid > expr valid > face kps valid, but for synbody and bedlam
    if 'expr' in human_model_param  and \
        human_model_param['expr'] is not None:
        expr = human_model_param['expr']
    else:
        expr = np.zeros((num_person, smpl_x.expr_code_dim), dtype=np.float32)
        expr_valid = expr_valid * 0
    expr*=face_valid[:,None]   
    expr = torch.FloatTensor(expr).view(num_person,-1)
    expr_valid*=face_valid # expr is invalid if face_valid is 0
    # for coco and ubody, if face is invalid, jaw pose and face kps2d should be false
    if 'jaw_pose' in human_model_param and \
        human_model_param['jaw_pose'] is not None:
        jaw_pose = human_model_param['jaw_pose']
    else:
        jaw_pose = np.zeros((num_person, 3), dtype=np.float32)
        rotation_valid[:,smpl_x.orig_joint_part['face']] = 0
        coord_valid[:,smpl_x.joint_part['face']] = 0
    jaw_pose*=face_valid[:,None]
    jaw_pose = torch.FloatTensor(jaw_pose).view(num_person, -1, 3)
    rotation_valid[:, smpl_x.orig_joint_part['face']]*=face_valid[:,None]
    coord_valid[:, smpl_x.joint_part['face']]*=face_valid[:,None]

    if 'gender' in human_model_param and \
        human_model_param['gender'] is not None:
        gender = human_model_param['gender']
    else:
        gender = 'neutral'

    if as_smplx == 'smpl':
        rotation_valid[:,:] = 0
        rotation_valid[:,:21] = 1
        expr_valid = expr_valid*0
        coord_valid[:,:] = 0
        coord_valid[:,smpl_x.joint_part['body']] = 1 
    # print(root_pose.shape, body_pose.shape, lhand_pose.shape, rhand_pose.shape, jaw_pose.shape)
    pose = torch.cat((root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose),dim=1)

    # parameters
    # flip pose parameter (axis-angle)
    if do_flip:
        for pair in human_model.orig_flip_pairs:
            pose[:, pair[0], :], pose[:,
                                      pair[1], :] = pose[:, pair[1], :].clone(
                                      ), pose[:, pair[0], :].clone()
            rotation_valid[:,pair[0]], rotation_valid[:,pair[1]] = rotation_valid[:,pair[1]].copy(), rotation_valid[:,
                pair[0]].copy()
        pose[:,:, 1:3] *= -1  # multiply -1 to y and z axis of axis-angle
    # rotate root pose
    pose = pose.numpy()
    root_pose = pose[:, human_model.orig_root_joint_idx, :]

    # for pose_i in range(len(root_pose)):
    #     root_pose_mat = cv2.Rodrigues(root_pose[pose_i])[0]
    #     root_pose[pose_i] = cv2.Rodrigues(np.dot(rot_aug_mat,
    #                                              root_pose_mat))[0][:, 0]

    pose[:, human_model.orig_root_joint_idx] = root_pose.reshape(num_person, 3)

    # change to mean shape if beta is too far from it
    # shape[(shape.abs() > 3).any(dim=1)] = 0.
    shape = shape.numpy().reshape(num_person, -1)
    # shape_valid = shape.sum(-1)!=0
    # return results
    pose = pose.reshape(num_person, -1)
    expr = expr.numpy().reshape(num_person, -1)

        
    return pose, shape, expr, rotation_valid, coord_valid, expr_valid, shape_valid

def load_obj(file_name):
    v = []
    obj_file = open(file_name)
    for line in obj_file:
        words = line.split(' ')
        if words[0] == 'v':
            x, y, z = float(words[1]), float(words[2]), float(words[3])
            v.append(np.array([x, y, z]))
    return np.stack(v)


def load_ply(file_name):
    plydata = PlyData.read(file_name)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    v = np.stack((x, y, z), 1)
    return v


def resize_bbox(bbox, scale=1.2):
    if isinstance(bbox, list):
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    else:
        x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    x_size, y_size = x2 - x1, y2 - y1
    x1_resize = x_center - x_size / 2.0 * scale
    x2_resize = x_center + x_size / 2.0 * scale
    y1_resize = y_center - y_size / 2.0 * scale
    y2_resize = y_center + y_size / 2.0 * scale
    bbox[0], bbox[1], bbox[2], bbox[
        3] = x1_resize, y1_resize, x2_resize, y2_resize
    return bbox
