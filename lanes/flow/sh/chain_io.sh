#!/usr/bin/env bash
# IO chain (one IO-heavy prep job at a time): NDISPark -> NVMe, then UA-DETRAC test frames -> NVMe (single sequential pass over the zip).
set -u
python -m lanes.flow.data ndispark
python -m lanes.flow.data detrac --split test --stride 5
echo CHAIN_IO_DONE
