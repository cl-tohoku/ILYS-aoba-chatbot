#!/bin/bash

# Usage: bash preprocess_fairseq.sh [TRAIN PREFIX] [DEV PREFIX] [OUT PATH] [SPM VOCAB PATH]

TRAIN=$(readlink -f $1)
DEV=$(readlink -f $2)
PRE_PROCESSED_DIR=$(readlink -f $3)
SPM_VOCAB=$(readlink -f $4)

SRC_LANG="context"
TRG_LANG="response"
FAIRSEQ_VOCAB=${PRE_PROCESSED_DIR}/fairseq_vocab.txt
N_WORKER=12

echo "Train-context:" ${TRAIN}.${SRC_LANG}
echo "Train-response:" ${TRAIN}.${TRG_LANG}
echo "Dev-context:" ${DEV}.${SRC_LANG}
echo "Dev-response:" ${DEV}.${TRG_LANG}
echo "Output:" ${PRE_PROCESSED_DIR}
echo
echo "sentencepiece vocab:" ${SPM_VOCAB}
echo

mkdir -p ${PRE_PROCESSED_DIR}
cut -f1 ${SPM_VOCAB} | tail -n +4 | sed "s/$/ 100/g" > ${FAIRSEQ_VOCAB}

echo "Create:" ${FAIRSEQ_VOCAB}
echo
echo "Your fairseq version:"
pip list | grep fairseq
echo

set -x

fairseq-preprocess \
  --source-lang ${SRC_LANG} \
  --target-lang ${TRG_LANG} \
  --trainpref ${TRAIN} \
  --validpref ${DEV} \
  --destdir ${PRE_PROCESSED_DIR} \
  --srcdict ${FAIRSEQ_VOCAB} \
  --joined-dictionary \
  --workers ${N_WORKER}