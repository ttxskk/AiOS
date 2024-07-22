python -m torch.distributed.launch \
    --nproc_per_node 8 \
    main.py \
    --output_dir "data/log/" \
    -c "config/aios_smplx_pretrain.py" \
    --options batch_size=2 backbone="resnet50" \
    --pretrain_model_path data/checkpoint/edpose_r50_coco.pth