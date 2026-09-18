#!/bin/bash
# Whether serving a confirmed calibration routes worse than drawing one,
# on the Zen3 guest.
#
# The same comparison the PowerShell harness runs on the other host, and
# it writes the same ARM / ARM_END / control / throughput / retained
# lines so the same reports read both without knowing which box produced
# which log.
#
# Linux needs none of the Windows machinery. A backgrounded process
# outlives its ssh session here, so there is no job object to escape and
# no WMI to escape it with, and nothing holds a core the way a live
# sshd connection does on the other host.
#
# Busy cores come from /proc/stat rather than a process-name count: a
# count of build tools reads near zero on a box busy with anything else.

set -u

TREE="$HOME/Flynnel-l1"
TRIALS="${1:-40}"
WINDOW_S="${2:-2}"
LOAD="${3:-12}"
REPS="${4:-512}"
WARMUP="${5:-8}"
SETTLE_CORES="${6:-1.4}"

LOG="$HOME/zen3_serve_vs_draw.log"
EXE="$TREE/target/release/examples/throughput_under_load"
NCPU=$(nproc)

# Busy cores over a short interval, from the aggregate cpu line. Fields
# after the label are user nice system idle iowait irq softirq steal;
# idle time is the fourth and fifth.
busy_cores() {
    local a b ai bi at bt
    a=$(awk '/^cpu /{print $2+$3+$4+$5+$6+$7+$8+$9, $5+$6}' /proc/stat)
    sleep 0.25
    b=$(awk '/^cpu /{print $2+$3+$4+$5+$6+$7+$8+$9, $5+$6}' /proc/stat)
    at=${a% *}; ai=${a#* }
    bt=${b% *}; bi=${b#* }
    awk -v at="$at" -v ai="$ai" -v bt="$bt" -v bi="$bi" -v n="$NCPU" \
        'BEGIN { dt = bt - at; if (dt <= 0) { print -1; exit } \
                 printf "%.2f", n * (1 - (bi - ai) / dt) }'
}

# Waits for the box rather than sleeping a fixed time, and writes the
# figure that decided either way.
settle() {
    local tries=0 busy
    while : ; do
        busy=$(busy_cores)
        tries=$((tries + 1))
        if awk -v b="$busy" -v c="$SETTLE_CORES" 'BEGIN{exit !(b <= c)}'; then break; fi
        if [ "$tries" -ge 12 ]; then break; fi
        sleep 0.7
    done
    echo "$busy"
}

[ -x "$EXE" ] || { echo "MISSING_EXE $EXE" > "$LOG"; exit 9; }

STAMP=$(date +%Y%m%d-%H%M%S)
SHARED="/tmp/abserve_$STAMP"
mkdir -p "$SHARED"

{
    echo "AB_START $(date -Is) tree=$TREE trials=$TRIALS window=${WINDOW_S}s load=$LOAD reps=$REPS shared=$SHARED"
    echo "SOURCE_STATE head=$(git -C "$TREE" rev-parse --short HEAD) dirty=$(git -C "$TREE" status --porcelain --untracked-files=no | wc -l) in $TREE"
    echo "HOST $(uname -sr) ncpu=$NCPU $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
} > "$LOG"

export FLYNNEL_OCCUPANCY=1
unset FLYNNEL_HOST_PROFILE_NS

# Warmed until the record actually serves. Two draws confirm only if they
# agree, so a fixed count leaves the first trials drawing.
export FLYNNEL_CALIBRATION_DIR="$SHARED"
warmed=0
for w in $(seq 1 "$WARMUP"); do
    echo "WARM $w" >> "$LOG"
    "$EXE" "$WINDOW_S" "$LOAD" 1 0 0 "$REPS" 0 indexed >> "$LOG" 2>&1
    if grep -q 'READ from the stored record' "$LOG"; then
        warmed=1; echo "WARM_CONFIRMED after $w draws" >> "$LOG"; break
    fi
done
[ "$warmed" = 1 ] || { echo "WARM_NEVER_CONFIRMED after $WARMUP draws" >> "$LOG"; exit 4; }

for t in $(seq 1 "$TRIALS"); do
    if [ $((t % 2)) -eq 1 ]; then order="serve draw"; else order="draw serve"; fi
    pos=0
    for arm in $order; do
        pos=$((pos + 1))
        if [ "$arm" = serve ]; then
            export FLYNNEL_CALIBRATION_DIR="$SHARED"
        else
            fresh="/tmp/abdraw_${STAMP}_t${t}"; mkdir -p "$fresh"
            export FLYNNEL_CALIBRATION_DIR="$fresh"
        fi
        busy=$(settle)
        foreign=$(pgrep -c -x 'cargo|rustc|cc|ld' 2>/dev/null || echo 0)
        echo "ARM $arm trial=$t position=$pos foreign=$foreign busy_cores=$busy" >> "$LOG"
        "$EXE" "$WINDOW_S" "$LOAD" 1 0 0 "$REPS" 0 indexed >> "$LOG" 2>&1
        sleep 0.6
        echo "ARM_END $arm trial=$t busy_cores=$(busy_cores)" >> "$LOG"
    done
    echo "trial $t/$TRIALS done"
done

arms=$(grep -c '^ARM ' "$LOG")
retained=$(grep -c '^retained ' "$LOG")
reads=$(grep -c 'READ from the stored record' "$LOG")
echo "ROW_CENSUS arms=$arms retained=$retained reads=$reads expected=$((TRIALS * 2))" >> "$LOG"
echo "AB_DONE $(date -Is)" >> "$LOG"

# The serve arm must have served, or the two arms are the same arm twice
# and a null here is about a mechanism that never ran.
[ "$reads" -ge "$TRIALS" ] || { echo "SERVE_ARM_DID_NOT_SERVE reads=$reads over $TRIALS trials" >> "$LOG"; exit 6; }
exit 0
