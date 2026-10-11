#!/usr/bin/env bash
# Run a command on the box inside the project env, from the synced code dir.
# Usage: scripts/box.sh [--gpu N] [--lock] [--bg NAME] -- <command...>
#   --gpu N   : CUDA_VISIBLE_DEVICES=N (default: none set)
#   --lock    : serialise on locks/gpu<N>.lock (use for training on GPU0)
#   --bg NAME : run detached with nohup; log -> logs/NAME.log, pid -> logs/NAME.pid
set -euo pipefail
BOX=${SEEN_BOX:-rtx}
R=/mnt/bigdrive/amirkhon/seen
GPU=""; LOCK=0; BG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU=$2; shift 2;;
    --lock) LOCK=1; shift;;
    --bg) BG=$2; shift 2;;
    --) shift; break;;
    *) break;;
  esac
done
CMD=$(printf '%q ' "$@")
PRE="cd $R/code && source /home/amirkhon/seen_fast/env/main/bin/activate && export SEEN_ROOT=$R SEEN_FAST=/home/amirkhon/seen_fast PYTHONPATH=$R/code HF_HOME=$R/tools/hf-cache UV_CACHE_DIR=$R/tools/uv-cache"
[ -n "$GPU" ] && PRE="$PRE && export CUDA_VISIBLE_DEVICES=$GPU"
RUN="$CMD"
[ "$LOCK" = 1 ] && RUN="flock $R/locks/gpu${GPU:-0}.lock $CMD"
if [ -n "$BG" ]; then
  # Everything (env setup + command) runs inside the detached shell, so ssh returns immediately.
  INNER="$PRE && $RUN"
  ssh "$BOX" "nohup setsid bash -c $(printf '%q' "$INNER") > $R/logs/$BG.log 2>&1 < /dev/null & echo \$! > $R/logs/$BG.pid; echo started $BG pid \$(cat $R/logs/$BG.pid)"
else
  ssh "$BOX" "$PRE && $RUN"
fi
