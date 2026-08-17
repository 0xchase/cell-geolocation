#!/bin/bash
# Waits for cell.mil_cells / cell.emb_cells (built by /tmp/chain2.sh on the
# server once summary_full lands), then regenerates the last four figures.
#
# obs14 is regenerated for completeness, but its ranking is by observation count,
# which measures crawl cadence rather than cell lifetime — see REGENERATION.md.
# Its panel B title now says so explicitly.
cd "$(dirname "$0")"
HOST=${CELL_DB_HOST:-ckanipe@nominatim.cybre.io}

echo "$(date) waiting for SITES COMPLETE"
while ! ssh "$HOST" 'grep -q "SITES COMPLETE" /tmp/chain2.log' 2>/dev/null; do sleep 60; done
ssh "$HOST" 'tail -2 /tmp/chain2.log'

./regen.sh \
  obs11_india_military_mcc123 \
  obs12_fort_bragg_mcc553 \
  obs13_military_test_mccs \
  obs14_embassy_test_code_leads

echo "$(date) final results:"
cat /tmp/regen_results.log
ls -1 ../plots/*.png | wc -l
