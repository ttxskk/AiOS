#!/bin/bash
CHECKPOINT=$1
INPUT_VIDEO=$2
OUTPUT_DIR=$3
NUM_PERSON=${4:-1}
THRESHOLD=${5:-0.3}
GPU_NUM=${6:-8}
python -m torch.distributed.launch \
    --nproc_per_node ${GPU_NUM} \
    main.py \
    -c "config/aios_smplx_demo.py" \
    --options batch_size=8 backbone="resnet50" num_person=${NUM_PERSON} threshold=${THRESHOLD} \
    --resume ${CHECKPOINT} \
    --eval \
    --inference \
    --to_vid \
    --inference_input ${INPUT_VIDEO} \
    --output_dir demo/${OUTPUT_DIR}
