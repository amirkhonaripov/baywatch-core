#!/usr/bin/env bash
set -u
python -m lanes.flow.router_data commons
python -m lanes.flow.router_data coco
python -m lanes.flow.router_data hf
echo ROUTER_DATA_STAGE1_DONE
