#!/usr/bin/env bash
# Detached box job that returns immediately (scripts/box.sh --bg keeps the ssh session attached because its env
# preamble is part of the backgrounded && chain). Same semantics otherwise.
# Usage: lanes/ident/tools/bgbox.sh NAME [--gpu N] [--lock] -- <command...>
set -euo pipefail
BOX=${SEEN_BOX:-rtx}; R=/mnt/bigdrive/amirkhon/seen
NAME=$1; shift
GPU=""; LOCK=0
while [ $# -gt 0 ]; do case "$1" in --gpu) GPU=$2; shift 2;; --lock) LOCK=1; shift;; --) shift; break;; *) break;; esac; done
CMD=$(printf '%q ' "$@")
PRE="cd $R/code && source /home/amirkhon/seen_fast/env/main/bin/activate && export SEEN_ROOT=$R SEEN_FAST=/home/amirkhon/seen_fast PYTHONPATH=$R/code HF_HOME=$R/tools/hf-cache UV_CACHE_DIR=$R/tools/uv-cache"
[ -n "$GPU" ] && PRE="$PRE && export CUDA_VISIBLE_DEVICES=$GPU"
RUN="$CMD"; [ "$LOCK" = 1 ] && RUN="flock $R/locks/gpu${GPU:-0}.lock $CMD"
INNER="$PRE && $RUN"
ssh -n "$BOX" "nohup setsid bash -c $(printf '%q' "$INNER") > $R/logs/$NAME.log 2>&1 < /dev/null & echo \$! > $R/logs/$NAME.pid; echo started $NAME pid \$(cat $R/logs/$NAME.pid)"
