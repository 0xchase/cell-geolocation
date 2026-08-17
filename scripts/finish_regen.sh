#!/bin/bash
# Waits for the server-side summary_full chain, then regenerates the figures that
# depend on it. Run detached:  nohup ./finish_regen.sh > /tmp/finish.log 2>&1 &
#
# Excluded on purpose:
#   obs11/obs12/obs13  need cell.mil_cells   (not built yet)
#   obs14              needs cell.emb_cells  (not built yet, and its conclusion
#                      is invalid regardless — see REGENERATION.md)
cd "$(dirname "$0")"
HOST=${CELL_DB_HOST:-ckanipe@nominatim.cybre.io}

echo "$(date) waiting for CHAIN COMPLETE on $HOST"
while ! ssh "$HOST" 'grep -q "CHAIN COMPLETE" /tmp/chain.log' 2>/dev/null; do sleep 60; done
echo "$(date) chain done; summary_full rows:"
ssh "$HOST" "clickhouse-client --password password --query 'SELECT count() FROM cell.summary_full'"

./regen.sh \
  obs01_gaza_network_collapse \
  obs03_crimea_operator_substitution \
  obs05_nagorno_karabakh_substitution \
  obs25_north_korea_cells \
  obs26_venezuela_geopolitical_economic \
  obs27_kyivstar_moscow_base \
  obs28_frontline_tracking \
  obs29_frontline_tracking_quarterly \
  russia_ukraine_crossborder_maps

echo "$(date) results:"
cat /tmp/regen_results.log
echo
echo "NOW INSPECT EACH FIGURE. Re-running is not sufficient — see REGENERATION.md."
echo "Known pending title fixes: obs05 'one-shot', obs25 'surge in Feb-Mar 2026'."
