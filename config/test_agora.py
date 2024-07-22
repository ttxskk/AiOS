_base_ = ['coco_transformer.py']

num_classes = 2
lr = 0.0001
param_dict_type = 'default'
lr_backbone = 1e-05
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult = 0.1
ddetr_lr_param = False
batch_size = 2
weight_decay = 0.0001
epochs = 200
lr_drop = 11
save_checkpoint_interval = 1
clip_max_norm = 0.1
onecyclelr = False
multi_step_lr = False
lr_drop_list = [30, 60]

modelname = 'aios_smplx'
frozen_weights = None
backbone = 'resnet50'
use_checkpoint = False

dilation = False
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
random_refpoints_xy = False
fix_refpoints_hw = -1
dec_layer_number = None
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
dln_xy_noise = 0.2
dln_hw_noise = 0.2
two_stage_type = 'standard'
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
two_stage_learn_wh = False
two_stage_default_hw = 0.05
two_stage_keep_all_tokens = False
rm_detach = None
num_select = 50
transformer_activation = 'relu'
batch_norm_type = 'FrozenBatchNorm2d'

masks = False
losses = ["smpl_pose", "smpl_beta", "smpl_kp2d","smpl_kp3d","smpl_kp3d_ra",'labels', 'boxes', "keypoints"]
# losses = ['labels', 'boxes', "keypoints"]
aux_loss = True
set_cost_class = 2.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
set_cost_keypoints = 10.0
set_cost_kpvis = 0.0
set_cost_oks = 4.0
cls_loss_coef = 2.0
# keypoints_loss_coef = 10.0
smpl_pose_loss_root_coef = 5 * 0.1
smpl_pose_loss_body_coef = 5 * 0.1
smpl_pose_loss_lhand_coef = 1 * 0.01
smpl_pose_loss_rhand_coef = 1 * 0.01
smpl_pose_loss_jaw_coef = 1 * 0.01
smpl_beta_loss_coef = 0.5

# smpl_kp3d_loss_coef = 10
smpl_body_kp3d_loss_coef = 5.0 * 0.1
smpl_face_kp3d_loss_coef = 1.0 * 0.1
smpl_lhand_kp3d_loss_coef = 1 * 0.1
smpl_rhand_kp3d_loss_coef = 1 * 0.1

# kp3d ra
smpl_body_kp3d_ra_loss_coef = 5 * 0.1
smpl_face_kp3d_ra_loss_coef = 1 * 0.1
smpl_lhand_kp3d_ra_loss_coef = 1 * 0.1
smpl_rhand_kp3d_ra_loss_coef = 1 * 0.1


# smpl_kp2d_ba_loss_coef = 1.0
smpl_body_kp2d_loss_coef = 5.0 * 0.1
smpl_lhand_kp2d_loss_coef = 1.0 * 0.1
smpl_rhand_kp2d_loss_coef = 1.0 * 0.1
smpl_face_kp2d_loss_coef = 1.0 * 0.1

smpl_body_kp2d_ba_loss_coef = 0
smpl_face_kp2d_ba_loss_coef = 0
smpl_lhand_kp2d_ba_loss_coef = 0
smpl_rhand_kp2d_ba_loss_coef = 0

bbox_loss_coef = 5.0
body_bbox_loss_coef = 5.0
lhand_bbox_loss_coef = 5.0
rhand_bbox_loss_coef = 5.0
face_bbox_loss_coef = 5.0

giou_loss_coef = 2.0
body_giou_loss_coef = 2.0
rhand_giou_loss_coef = 2.0
lhand_giou_loss_coef = 2.0
face_giou_loss_coef = 2.0

keypoints_loss_coef = 10.0
rhand_keypoints_loss_coef = 5.0
lhand_keypoints_loss_coef = 5.0
face_keypoints_loss_coef = 5.0
       
oks_loss_coef=4.0
rhand_oks_loss_coef = 4.0
lhand_oks_loss_coef = 4.0
face_oks_loss_coef = 4.0


enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.25
rm_self_attn_layers = None
indices_idx_list = [1, 2, 3, 4, 5, 6, 7]

decoder_sa_type = 'sa'
matcher_type = 'HungarianMatcher'
decoder_module_seq = ['sa', 'ca', 'ffn']
nms_iou_threshold = -1

dec_pred_bbox_embed_share = False
dec_pred_class_embed_share = False
dec_pred_pose_embed_share = False
body_only = True

# for dn
use_dn = True
dn_number = 100
dn_box_noise_scale = 0.4
dn_label_noise_ratio = 0.5
embed_init_tgt = False
dn_label_coef = 0.3
dn_bbox_coef = 0.5
dn_batch_gt_fuse = False
dn_attn_mask_type_list = ['match2dn', 'dn2dn', 'group2group']
dn_labelbook_size = 100

match_unstable_error = False

# for ema
use_ema = True
ema_decay = 0.9997
ema_epoch = 0

cls_no_bias = False
num_body_points = 17  # for coco
num_hand_points = 6 # for coco
num_face_points = 6  # for coco
num_group = 100
num_box_decoder_layers = 2
num_hand_face_decoder_layers = 4
no_mmpose_keypoint_evaluator = True
strong_aug = False


body_model_test=\
    dict(
        type='smplx',
        keypoint_src='smplx',
        num_expression_coeffs=10,
        num_betas=10,
        keypoint_dst='smplx_137',
        model_path='data/body_models/smplx',
        use_pca=False,
        use_face_contour=True)

body_model_train = \
    dict(
        type='smplx',
        keypoint_src='smplx',
        num_expression_coeffs=10,
        num_betas=10,
        keypoint_dst='smplx_137',
        model_path='data/body_models/smplx',
        use_pca=False,
        use_face_contour=True)

# will be update in exp
exp_name = 'output/exp1/dataset_debug'

# quick access
lr = 1e-3
end_epoch = 150
train_batch_size = 32

scheduler = 'step'
step_size = 20
gamma = 0.1

# continue
continue_train = True
pretrained_model_path = './data/checkpoint/edpose_r50_coco.pth'

# dataset setting
dataset_list = ['AGORA_MM']
trainset_3d = ['AGORA_MM']
trainset_2d = []
trainset_humandata = []
testset = 'AGORA_MM'
train_sizes=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
train_max_size=1333
test_sizes=[800]
test_max_size=1333
no_aug=True
# model
use_cache = True

## UBody setting
train_sample_interval = 10
test_sample_interval = 100
make_same_len = False

## input, output size
input_body_shape = (256, 192)
output_hm_shape = (16, 16, 12)
input_hand_shape = (256, 256)
output_hand_hm_shape = (16, 16, 16)
output_face_hm_shape = (8, 8, 8)
input_face_shape = (192, 192)
focal = (5000, 5000)  # virtual focal lengths
princpt = (input_body_shape[1] / 2, input_body_shape[0] / 2
           )  # virtual principal point position
body_3d_size = 2
hand_3d_size = 0.3
face_3d_size = 0.3
camera_3d_size = 2.5

bbox_ratio = 1.2

## directory
output_dir, model_dir, vis_dir, log_dir, result_dir, code_dir = None, None, None, None, None, None

agora_benchmark = 'na' # 'agora_model', 'test_only'
