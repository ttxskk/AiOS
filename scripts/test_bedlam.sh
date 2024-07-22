CHECKPOINT=$1
OUTPUT_DIR=$2
THRESHOLD=${3:-0.7}
GPU_NUM=${4:-8}
python -m torch.distributed.launch \
    --nproc_per_node ${GPU_NUM} \
    main.py \
    -c "config/aios_smplx_bedlam.py" \
    --options batch_size=8 backbone="resnet50" threshold=${THRESHOLD} \
    --resume ${CHECKPOINT} \
    --eval \
    --inference \
    --inference_input data/datasets/bedlam/test_images \
    --output_dir test_result/${OUTPUT_DIR}
zip -r test_result/${OUTPUT_DIR}/predictions.zip test_result/${OUTPUT_DIR}/predictions
