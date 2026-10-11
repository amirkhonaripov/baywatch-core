#!/usr/bin/env bash
# Keep the defence API (Modal app "seen-demo", L40S) warm: start 10 minutes before the talk, Ctrl+C afterwards.
# The site uses it with:  https://amirkhonaripov.github.io/baywatch/?api=https://sanjar--seen-demo-api.modal.run#/console
URL=${1:-https://sanjar--seen-demo-api.modal.run}
while true; do
  printf '%s ' "$(date +%H:%M:%S)"; curl -s -m 120 "$URL/health?warm=true" | head -c 120; echo
  sleep 240   # containers scale down after 15 min idle
done
