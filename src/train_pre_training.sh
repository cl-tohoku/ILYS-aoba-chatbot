#! /bin/sh

export LANG=C.UTF-8
export PYTHONIOENCODING=utf-8

date
hostname
uname -a
which python
python --version
pip list

. ${SETTING_FILE}

mkdir -p ${MODEL_DIR}

CUDA_VISIBLE_DEVICES=${GPU} fairseq-train ${DATA_DIR} \
  --save-dir ${MODEL_DIR} \
  --seed ${SEED} \
  --source-lang ${SRC_LANG} \
  --target-lang ${TRG_LANG} \
  --arch transformer_vaswani_wmt_en_de_big \
  --activation-fn gelu \
  --dropout 0.1 \
  --attention-dropout 0.0 \
  --relu-dropout 0.0 \
  --encoder-embed-dim ${ENC_EMB} \
  --encoder-ffn-embed-dim ${ENC_FFN} \
  --encoder-layers ${ENC_LAYER} \
  --encoder-attention-heads ${ENC_HEAD} \
  --encoder-normalize-before \
  --decoder-embed-dim ${DEC_EMB} \
  --decoder-ffn-embed-dim ${DEC_FFN} \
  --decoder-layers ${DEC_LAYER} \
  --decoder-attention-heads ${DEC_HEAD} \
  --decoder-normalize-before \
  --share-all-embeddings \
  --max-tokens ${MAX_TOKEN} \
  --max-update ${MAX_UPDATE} \
  --update-freq ${UFREQ} \
  --num-workers ${NUM_WORKERS} \
  --fp16 \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt \
  --warmup-updates ${WARMUP_STEP} \
  --warmup-init-lr ${INIT_LR} \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --weight-decay 0.0 \
  --clip-norm 0.1 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --log-format simple \
  --tensorboard-logdir ${TENSORBOARD_DIR} \
  --keep-last-epochs ${KEEP_LAST_EPOCH} \
  --keep-interval-updates ${KEEP_LAST_UPD} \
  --save-interval-updates ${SAVE_UPD} \
  --log-interval ${LOG_UPD} \
  | tee -a ${MODEL_DIR}/train.log
