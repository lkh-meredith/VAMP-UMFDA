#!/bin/bash

# custom config
DATA=./data
TRAINER=VAMP

CFG=vamp # config file

DATASET=officehome
EPOCH=50
for SHOTS in 3 6
do
      for SEED in 1 2 3 5 # set different seed
      do
          DIR=./results/${TRAINER}/vit16//${DATASET}/shot${SHOTS}/seed${SEED}
          echo "Run this job and save the output to ${DIR}"
          python train.py \
          --root ${DATA} \
          --seed ${SEED} \
          --trainer ${TRAINER} \
          --dataset-config-file configs/datasets/${DATASET}.yaml \
          --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
          --output-dir ${DIR} \
          --source-domains a p r \
          --target-domains c \
          OPTIM.LR ${LR} \
          OPTIM.MAX_EPOCH ${EPOCH} \
          DATASET.NUM_SHOTS ${SHOTS}
      done
      for SEED in 1 2 3 5
      do
          DIR=./results/${TRAINER}/vit16/${date}/${DATASET}/maple_md/shot${SHOTS}/epoch${EPOCH}/lr${LR}_sigma_${SIGMA}/bs_x${BATCH_SIZE_X}_bs_u${BATCH_SIZE_U_T}/lr_${LR}_alpha1_${ALPHA1}_alpha2_${ALPHA2}_cof${CONFI}/seed${SEED}
          echo "Run this job and save the output to ${DIR}"
          python train.py \
          --root ${DATA} \
          --seed ${SEED} \
          --trainer ${TRAINER} \
          --dataset-config-file configs/datasets/${DATASET}.yaml \
          --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
          --output-dir ${DIR} \
          --source-domains c p r \
          --target-domains a \
          OPTIM.LR ${LR} \
          OPTIM.MAX_EPOCH ${EPOCH} \
          DATASET.NUM_SHOTS ${SHOTS}
      done
      for SEED in 1 2 3 5
      do
          DIR=./results/${TRAINER}/vit16/${date}/${DATASET}/maple_md/shot${SHOTS}/epoch${EPOCH}/lr${LR}_sigma_${SIGMA}/bs_x${BATCH_SIZE_X}_bs_u${BATCH_SIZE_U_T}/lr_${LR}_alpha1_${ALPHA1}_alpha2_${ALPHA2}_cof${CONFI}/seed${SEED}
          echo "Run this job and save the output to ${DIR}"
          python train.py \
          --root ${DATA} \
          --seed ${SEED} \
          --trainer ${TRAINER} \
          --dataset-config-file configs/datasets/${DATASET}.yaml \
          --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
          --output-dir ${DIR} \
          --source-domains c a r \
          --target-domains p \
          OPTIM.LR ${LR} \
          OPTIM.MAX_EPOCH ${EPOCH} \
          DATASET.NUM_SHOTS ${SHOTS}
      done
      for SEED in 1 2 3 5
      do
          DIR=./results/${TRAINER}/vit16/${date}/${DATASET}/maple_md/shot${SHOTS}/epoch${EPOCH}/lr${LR}_sigma_${SIGMA}/bs_x${BATCH_SIZE_X}_bs_u${BATCH_SIZE_U_T}/lr_${LR}_alpha1_${ALPHA1}_alpha2_${ALPHA2}_cof${CONFI}/seed${SEED}
          echo "Run this job and save the output to ${DIR}"
          python train.py \
          --root ${DATA} \
          --seed ${SEED} \
          --trainer ${TRAINER} \
          --dataset-config-file configs/datasets/${DATASET}.yaml \
          --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
          --output-dir ${DIR} \
          --source-domains c a p \
          --target-domains r \
          OPTIM.LR ${LR} \
          OPTIM.MAX_EPOCH ${EPOCH} \
          DATASET.NUM_SHOTS ${SHOTS}
      done
done
