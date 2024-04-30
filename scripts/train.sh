python -m torch.distributed.launch \
    --nproc_per_node 2 \
    main.py \
    --output_dir "data/log/exps_62_debug" \
    -c "config/edpose_smplx.cfg.pretrain.py" \
    --options batch_size=2 lr_drop=55 num_body_points=17 backbone="resnet50" \
    --pretrain_model_path "data/checkpoint/edpose_r50_coco.pth"