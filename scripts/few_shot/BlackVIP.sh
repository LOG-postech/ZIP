#!/bin/bash
# Change directory to project root
cd ../..

# Training configuration
TRAINER=BlackVIP
CFG=vit_b16
SHOTS=16
DATASET=$1
MAX_API_CALLS=5000
DATA="Put your data path here"

# Training loop for different seeds
for SEED in 1 2 3; do
    DIR=output/basic/${TRAINER}/${DATASET}/seed${SEED}
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
        DATASET.SUBSAMPLE_CLASSES all \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.BASIC.MAX_API_CALLS ${MAX_API_CALLS}
    fi
done