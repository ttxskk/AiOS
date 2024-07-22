CHECKPOINT=$1
OUTPUT_DIR=$2
THRESHOLD=${3:-0.7}
GPU_NUM=${5:-8}
python -m torch.distributed.launch \
    --nproc_per_node ${GPU_NUM} \
    main.py \
    -c "config/aios_smplx_agora_val.py" \
    --options batch_size=8  backbone="resnet50" threshold=${THRESHOLD} \
    --resume ${CHECKPOINT} \
    --eval \
    --inference \
    --inference_input data/datasets/agora/3840x2160/test \
    --output_dir test_result/${OUTPUT_DIR}

zip -r test_result/${OUTPUT_DIR}/predictions.zip test_result/${OUTPUT_DIR}/predictions

