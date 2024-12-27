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
MAX_API_CALLS=5000
DATA="Put your data path here"

# Training loop for different seeds
for SEED in 1 2 3; do
    DIR=output/base2new/base/${TRAINER}/${DATASET}/nctx${NCTX}/seed${SEED}
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
        DATASET.SUBSAMPLE_CLASSES base \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.BASIC.MAX_API_CALLS ${MAX_API_CALLS} \
        TRAINER.ZIP.N_CTX ${NCTX} \
        TRAINER.ZIP.INTRINSIC_DIM ${INTRINSIC_DIM} \
        TRAINER.ZIP.RANK ${RANK}
    fi
done