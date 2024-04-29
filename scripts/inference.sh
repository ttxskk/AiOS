
INPUT_VIDEO=$1
OUTPUT_DIR=$2

python -m torch.distributed.launch \
    --nproc_per_node 2 \
    main.py \
    -c "config/aios_smplx_inference.py" \
    --options batch_size=8 epochs=100 lr_drop=55 num_body_points=17 backbone="resnet50" \
    --resume "data/checkpoint/aios_checkpoint.pth" \
    --eval \
    --inference \
    --to_vid \
    --inference_input demo/${INPUT_VIDEO}.mp4 \
    --output_dir demo/${OUTPUT_DIR}
