CHECKPOINT=$1
OUTPUT_DIR=$2
GPU_NUM=${3:-8}
THRESHOLD=${4:-0.7}

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
cd test_result/${OUTPUT_DIR}
zip -r predictions.zip predictions

