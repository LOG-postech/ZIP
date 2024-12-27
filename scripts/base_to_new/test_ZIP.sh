#!/bin/bash
# Change directory to project root
cd ../..

# Training configuration
TRAINER=ZIP
CFG=vit_b16
SHOTS=16
DATASET=$1
NCTX=8
INTRINSIC_DIM=500
RANK=5
DATA="Put your data path here"

# Training loop for different seeds
for SEED in 1 2 3; do
    COMMON_DIR=${TRAINER}/${DATASET}/nctx${NCTX}/seed${SEED}
    MODEL_DIR=output/base2new/base/${COMMON_DIR}
    DIR=output/base2new/new/${COMMON_DIR}
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
        --model-dir ${MODEL_DIR} \
        --load-epoch 1000 \
        --eval-only \
        DATASET.SUBSAMPLE_CLASSES new \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.ZIP.N_CTX ${NCTX} \
        TRAINER.ZIP.INTRINSIC_DIM ${INTRINSIC_DIM} \
        TRAINER.ZIP.RANK ${RANK}
    fi
done