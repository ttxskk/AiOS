python -m torch.distributed.launch \
    --nproc_per_node 1 \
    main.py \
    --output_dir "/mnt/AFS_sunqingping/log/exp7_" \
    -c "config/aios_smplx_debug.py" \
    --options batch_size=2 backbone="resnet50" \
    --pretrain_model_path data/log/exp0/checkpoint0026.pth