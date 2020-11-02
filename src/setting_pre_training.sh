#! /bin/sh

# your own path (to be changed)
WORK_DIR="./pre_training"
DATA_DIR="/path/to/pre_training_data"

# path & extension
MODEL_DIR="${WORK_DIR}/pre_trained_models"
TENSORBOARD_DIR="${WORK_DIR}/pre_trained_tensorboard_logs"
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
MAX_UPDATE=400000
UFREQ=16
WARMUP_STEP=3125  # {500, 1000, 2000, 3125, 5000}
INIT_LR=1e-07
LR=1e-03  # {1e-03, 5e-04, 2e-04, 1e-04, 5e-05, 2e-05}
MIN_LR=1e-09

# save & log
KEEP_LAST_EPOCH=1
KEEP_LAST_UPD=10
SAVE_UPD=20000
LOG_UPD=2500

# else
NUM_WORKERS=15
SEED=2020
