#!/bin/bash
# Change directory to project root
cd ../..

# Training configuration
TRAINER=ZIP
CFG=vit_b16
SHOTS=16
DATASETS=("caltech101" "oxford_pets" "oxford_flowers" "food101" "fgvc_aircraft" "sun397" "dtd" "eurosat" "ucf101" "svhn" "resisc45" "clevr")
NCTX=8
INTRINSIC_DIM=500
RANK=5
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
            --model-dir output/basic/${TRAINER}/imagenet/nctx${NCTX}/seed${SEED} \
            --load-epoch 1000 \
            --eval-only \
            DATASET.SUBSAMPLE_CLASSES all \
            DATASET.NUM_SHOTS ${SHOTS} \
            TRAINER.ZIP.N_CTX ${NCTX} \
            TRAINER.ZIP.INTRINSIC_DIM ${INTRINSIC_DIM} \
            TRAINER.ZIP.RANK ${RANK}
        fi
    done
done