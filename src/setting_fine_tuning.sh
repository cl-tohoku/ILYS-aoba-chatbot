#! /bin/sh

# your own path (to be changed)
WORK_DIR="./fine_tuning"
DATA_DIR="/path/to/fine_tuning_data"
PRETRAINED_MODEL="/path/to/pre_trained_model_checkpoint.pt"

# path & extension
MODEL_DIR="${WORK_DIR}/fine_tuned_models"
TENSORBOARD_DIR="${WORK_DIR}/fine_tuned_tensorboard_log"
SRC_LANG="context"
TRG_LANG="response"

# model parameters
ENC_EMB=1024
ENC_FFN=8192
ENC_LAYER=2
ENC_HEAD=32
DEC_EMB=${ENC_EMB}
DEC_FFN=${ENC_FFN}
DEC_LAYER=16
DEC_HEAD=${ENC_HEAD}

# optimizer setting
GPU=0
MAX_TOKEN=2000
MAX_UPDATE=`expr 10000 + 400000`  # fine-tuning updates + pre-training updates
UFREQ=16
WARMUP_STEP=5000  # {100, 500, 1000, 5000}
INIT_LR=1e-07
LR=1e-04  # {1e-04, 5e-05, 1e-05, 5e-06}
MIN_LR=1e-09

# save & log
KEEP_LAST_EPOCH=1
KEEP_LAST_UPD=5
SAVE_UPD=2000
LOG_UPD=2000

# else
NUM_WORKERS=15
SEED=2020
