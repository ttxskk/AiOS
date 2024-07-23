CHECKPOINT=$1
OUTPUT_DIR=$2
python -m torch.distributed.launch \
    --nproc_per_node 8 \
    main.py \
    -c "config/aios_smplx_agora_val.py" \
    --options batch_size=8 backbone="resnet50" \
    --resume ${CHECKPOINT} \
    --eval \
    --inference \
    --inference_input data/datasets/agora/3840x2160/validation \
    --output_dir test_result/${OUTPUT_DIR}
