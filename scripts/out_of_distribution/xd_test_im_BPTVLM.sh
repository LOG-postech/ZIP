#!/bin/bash
# Change directory to project root
cd ../..

# Training configuration
TRAINER=BPTVLM
CFG=vit_b16
SHOTS=16
DATASETS=("imagenet_a" "imagenet_r" "imagenet_sketch" "imagenetv2")
DATA="Put your data path here"

# Training loop for different seeds
for SEED in 1 2 3; do
    # Training loop for different datasets
    for DATASET in "${DATASETS[@]}"; do
        DIR=output/xd/evaluation/${TRAINER}/${DATASET}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --model-dir output/basic/${TRAINER}/imagenet/seed${SEED} \
            --load-epoch 1000 \
            --eval-only \
            DATASET.SUBSAMPLE_CLASSES all \
            DATASET.NUM_SHOTS ${SHOTS}
        fi
    done
done