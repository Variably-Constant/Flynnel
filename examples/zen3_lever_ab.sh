#!/bin/bash
# One lever on against the same code with it off, paired by trial.
#
# The three-arm rotation in lever_rounds.sh compares arms that ran at
# different times, so drift between them lands in the ratio and its null
# measures the drift: 11 per cent on a quiet 16-core guest. Pairing
# within a trial subtracts it, and the same shape on Windows resolves to
# 0.61 per cent on ten pairs. Every speed lever here is smaller than the
# rotation's floor and larger than this one's.
#
# Both arms name the switch value. An unset switch takes whatever the
# crate defaults to, and two levers default on, so an off arm left bare
# runs the lever on and the two arms become one arm.
#
# Writes the `ARM` / `ARM_END` / control / throughput / retained lines
# paired_arms_report.py reads, so a Linux rotation and a Windows one are
# read by the same tool.
#
# usage: zen3_lever_ab.sh <tree> <lever> <trials> <window_s> <load>
#                         <reps> <irregular> <duty_ms> <entry> <profile_ns>

set -u

tree="${1:?tree}"
lever="${2:?lever env var}"
trials="${3:-40}"
window_s="${4:-2}"
load="${5:-12}"
reps="${6:-4096}"
irregular="${7:-1}"
duty_ms="${8:-50}"
entry="${9:-indexed}"
profile="${10:-}"

bin="$tree/target/release/examples/throughput_under_load"
[ -x "$bin" ] || { echo "MISSING_BINARY $bin"; exit 9; }

short=$(echo "$lever" | sed 's/^FLYNNEL_LEVER_//' | tr 'A-Z' 'a-z')
ncpu=$(nproc)

# Busy cores over a short interval from the aggregate cpu line, counting
# every process rather than a name list. Fields after the label are user
# nice system idle iowait irq softirq steal.
busy_cores() {
    local a b ai bi at bt
    a=$(awk '/^cpu /{print $2+$3+$4+$5+$6+$7+$8+$9, $5+$6}' /proc/stat)
    sleep 0.25
    b=$(awk '/^cpu /{print $2+$3+$4+$5+$6+$7+$8+$9, $5+$6}' /proc/stat)
    at=${a% *}; ai=${a#* }
    bt=${b% *}; bi=${b#* }
    awk -v at="$at" -v ai="$ai" -v bt="$bt" -v bi="$bi" -v n="$ncpu" \
        'BEGIN { dt = bt - at; if (dt <= 0) { print -1; exit } \
                 printf "%.2f", n * (1 - (bi - ai) / dt) }'
}

echo "LEVER_START $(date -Is) tree=$tree lever=$lever trials=$trials window=${window_s}s load=$load reps=$reps irregular=$irregular duty_ms=$duty_ms entry=$entry profile=${profile:-unpinned}"
echo "HOST $(uname -sr) ncpu=$ncpu $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
echo "HEAD $(git -C "$tree" rev-parse --short HEAD 2>/dev/null || echo unknown)"

export FLYNNEL_OCCUPANCY=1
if [ -n "$profile" ]; then export FLYNNEL_HOST_PROFILE_NS="$profile"; else unset FLYNNEL_HOST_PROFILE_NS; fi

sh "$HOME/vm_presence.sh" claim $$ "flynnel paired lever A/B on $short, TIMINGS, needs a quiet box, Flynnel-Scholar" 2>/dev/null
trap 'sh "$HOME/vm_presence.sh" release '"$$"' 2>/dev/null' EXIT INT TERM

stamp=$(date +%Y%m%d-%H%M%S)
for t in $(seq 1 "$trials"); do
    if [ $((t % 2)) -eq 1 ]; then order="1 0"; else order="0 1"; fi
    pos=0
    for arm in $order; do
        pos=$((pos + 1))
        # A fresh calibration directory per process. A shared one lets a
        # record drawn under one arm serve the other, which makes the
        # thresholds a confound rather than a constant.
        cal="/tmp/leverab_${short}_${stamp}_t${t}_${arm}"
        mkdir -p "$cal"
        busy=$(busy_cores)
        echo "ARM $short=$arm trial=$t position=$pos busy_cores=$busy"
        env "$lever=$arm" FLYNNEL_CALIBRATION_DIR="$cal" \
            "$bin" "$window_s" "$load" 1 0 "$duty_ms" "$reps" "$irregular" "$entry" 2>&1
        # Settled before the exit reading so it measures the box rather
        # than this arm's own threads being reclaimed.
        sleep 0.6
        echo "ARM_END $short=$arm trial=$t busy_cores=$(busy_cores)"
    done
    echo "trial $t/$trials done" >&2
done

echo "LEVER_DONE $(date -Is)"
