CHECKPOINT=$1
OUTPUT_DIR=$2
GPU_NUM=${3:-8}
THRESHOLD=${4:-0.7}
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
cd test_result/${OUTPUT_DIR}
zip -r predictions.zip predictions
