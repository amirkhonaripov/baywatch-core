#!/usr/bin/env bash
# Live demo on our own GPU box (RTX PRO 5000): models stay loaded, no cold start, ~2x faster than the public Modal API.
# The published website (GitHub Pages) keeps using Modal; this only changes where the demo laptop sends requests.
#   scripts/demo_rtx.sh          # start API on the box (if needed) + ssh tunnel + local website, then open the browser
#   scripts/demo_rtx.sh stop     # stop the tunnel and the local website (the box API keeps running)
set -euo pipefail
cd "$(dirname "$0")/.."
PORT_API=8008 PORT_WEB=5180

if [[ "${1:-}" == "stop" ]]; then
  pkill -f "ssh -f -N -o ExitOnForwardFailure=yes -L ${PORT_API}:localhost:${PORT_API} rtx" || true
  pkill -f "vite preview --port ${PORT_WEB}" || true
  exit 0
fi

health() { ssh rtx "curl -s -m 3 localhost:${PORT_API}/health" 2>/dev/null | grep -q '"status":"ok"'; }
if ! health; then
  scripts/box.sh --gpu 1 --bg api_local -- bash -c "SEEN_VLM=local python -m serve.local --port ${PORT_API}"
  echo -n "loading models on the box"
  until health; do echo -n "."; sleep 5; done; echo " ready"
fi

lsof -iTCP:${PORT_API} -sTCP:LISTEN >/dev/null 2>&1 || ssh -f -N -o ExitOnForwardFailure=yes -L ${PORT_API}:localhost:${PORT_API} rtx
curl -s -m 5 "localhost:${PORT_API}/health" | grep -q '"status":"ok"' || { echo "tunnel up but API not answering"; exit 1; }

cd web
[[ -d node_modules ]] || npm ci
npm run build >/dev/null
lsof -iTCP:${PORT_WEB} -sTCP:LISTEN >/dev/null 2>&1 || (nohup npx vite preview --port ${PORT_WEB} --strictPort >/tmp/baywatch_web.log 2>&1 &)
sleep 2
URL="http://localhost:${PORT_WEB}/?api=http://localhost:${PORT_API}#/console"
echo "demo: $URL"
[[ -n "${NO_OPEN:-}" ]] || open "$URL"
