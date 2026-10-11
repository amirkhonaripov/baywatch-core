#!/usr/bin/env bash
# Download everything BayWatch needs to run: our trained weights + the public base models.
#   scripts/download_weights.sh                    # into ./baywatch_data, including Qwen3-VL-8B (~17 GB from Hugging Face)
#   scripts/download_weights.sh --no-vlm           # skip Qwen3-VL (the pipeline then runs on the fast models only)
#   scripts/download_weights.sh --archive FILE     # use a copy of the weights archive you already have
#   scripts/download_weights.sh --dest DIR         # put everything somewhere else
# Steps: (1) our weights, one ~1.9 GB archive from Google Drive, sha256-checked, unpacked, every file sha256-checked;
# (2) the plate-detector cache linked into ~/.cache where its library looks; (3) SigLIP2 (+ Qwen3-VL) from Hugging Face,
# no account needed; (4) DEST/env.sh with the environment variables the code reads.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="$PWD/baywatch_data"; VLM=1; ARCHIVE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-vlm) VLM=0; shift ;;
    --archive) ARCHIVE="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
M=weights/manifest.json
get() { python3 -c "import json,sys; m=json.load(open('$M')); print(eval(sys.argv[1]))" "$1"; }
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
NAME=$(get "m['archive']['name']"); SHA=$(get "m['archive']['sha256']"); URL=$(get "m['archive']['url']")
mkdir -p "$DEST"; DEST=$(cd "$DEST" && pwd)
ARCHIVE=${ARCHIVE:-$DEST/$NAME}

# 1. our trained weights
if [[ ! -f "$ARCHIVE" ]]; then
  echo "downloading $NAME (~1.9 GB) from Google Drive"
  command -v gdown >/dev/null || pip install -q gdown
  gdown "$URL" -O "$ARCHIVE"
fi
echo "checking the archive (sha256)"
[[ "$(sha "$ARCHIVE")" == "$SHA" ]] || { echo "checksum mismatch: delete $ARCHIVE and run again"; exit 1; }
tar -xzf "$ARCHIVE" -C "$DEST"
python3 - "$DEST" "$M" <<'PY'
import hashlib, json, sys
from pathlib import Path
dest, m = Path(sys.argv[1]), json.load(open(sys.argv[2]))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
bad = [f["path"] for f in m["files"] if not (dest / f["path"]).is_file() or sha(dest / f["path"]) != f["sha256"]]
sys.exit(f"{len(bad)} files failed the checksum: {bad[:5]}") if bad else print(f"ok: {len(m['files'])} weight files verified")
PY

# 2. worldwide plate detector cache (open-image-models looks in ~/.cache)
mkdir -p "$HOME/.cache"
if [[ ! -e "$HOME/.cache/open-image-models" ]]; then
  ln -s "$DEST/cache/open-image-models" "$HOME/.cache/open-image-models"
  echo "linked ~/.cache/open-image-models"
fi

# 3. public base models from Hugging Face
command -v hf >/dev/null || pip install -q "huggingface-hub>=1.0"
hf download google/siglip2-so400m-patch14-384 >/dev/null && echo "ok: SigLIP2 (scene router)"
if [[ $VLM == 1 ]]; then hf download Qwen/Qwen3-VL-8B-Instruct >/dev/null && echo "ok: Qwen3-VL-8B (bay-state checks)"; fi

# 4. environment for the code
mkdir -p "$DEST/root"
cat > "$DEST/env.sh" <<ENV
export SEEN_ROOT="$DEST/root"            # runs/ and evidence/ are written here
export SEEN_ARTIFACTS="$DEST/artifacts"  # our trained weights
export SEEN_VLM=$([[ $VLM == 1 ]] && echo local || echo none)                 # local = Qwen3-VL in-process (~20 GB GPU memory)
ENV
echo
echo "done. Next:"
echo "  source $DEST/env.sh"
echo "  python -m serve.local --port 8008"
