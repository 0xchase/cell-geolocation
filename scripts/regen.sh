#!/bin/bash
# Regenerate figures sequentially, logging pass/fail compactly.
# Usage: ./regen.sh obs16_shenzhen_testing_cluster obs18_westminster_test_lab ...
cd "$(dirname "$0")"
PY=../../venv/bin/python
OUT=/tmp/regen_results.log
: > "$OUT"
for s in "$@"; do
  start=$(date +%s)
  if $PY "$s.py" --preview "../plots/$s.png" > "/tmp/regen_$s.log" 2>&1; then
    echo "OK   $(( $(date +%s) - start ))s  $s" >> "$OUT"
  else
    echo "FAIL $(( $(date +%s) - start ))s  $s :: $(grep -oE '[A-Za-z.]*(Error|Exception)[^\n]{0,110}' "/tmp/regen_$s.log" | tail -1)" >> "$OUT"
  fi
done
echo "BATCH COMPLETE" >> "$OUT"
