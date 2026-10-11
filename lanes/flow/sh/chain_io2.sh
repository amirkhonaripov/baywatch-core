#!/usr/bin/env bash
# IO chain 2 (one IO-heavy prep at a time): after the DETRAC test pass -> CNR-EXT subset -> fine-tuning set (DETRAC train + NDISPark + COCO).
set -u
F=${SEEN_FAST:-/home/amirkhon/seen_fast}/flow
until [ -f $F/detrac/test/index_s5.json ]; do sleep 20; done
python -m lanes.flow.prep_cnr --keep-every 2
python -m lanes.flow.prep_finetune --detrac-stride 10 --coco-n 2500
echo CHAIN_IO2_DONE
