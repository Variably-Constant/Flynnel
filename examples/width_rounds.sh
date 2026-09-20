#!/bin/sh
# The allowed-width lever, measured by taking CPUs away from a running
# process.
#
# available_parallelism reads the affinity mask and the cgroup quota,
# and a busy neighbour moves neither, so a load-based harness never
# reaches this switch. What reaches it is a narrowing applied partway
# through a run: with the switch off the pool goes on chunking for the
# machine it started on, with it on the worker count follows within the
# recheck interval.
#
# Three arms per trial - off, a second off as the null, on - in an order
# that rotates through all six permutations, for the same reason the
# throughput rotation does.
#
# usage: width_rounds.sh <tree> <window_s> <windows> <cpus> <trials> <profile_ns>

export PATH="$HOME/.cargo/bin:$PATH"

tree="$1"
window="$2"
windows="$3"
cpus="$4"
trials="$5"
profile="$6"

if [ -z "$tree" ] || [ -z "$window" ] || [ -z "$windows" ] || [ -z "$cpus" ] \
    || [ -z "$trials" ] || [ -z "$profile" ]; then
    echo "usage: $0 <tree> <window_s> <windows> <cpus> <trials> <profile_ns>" >&2
    exit 2
fi

bin="$tree/target/release/examples/width_narrowing"
if [ ! -x "$bin" ]; then
    echo "MISSING_BINARY $bin - build it before measuring" >&2
    exit 9
fi

# Without a way to narrow the process every arm reports one width and
# exits 3, which is a run that measured nothing dressed as a clean
# negative. Say so here instead, where the reason is visible.
if ! command -v taskset >/dev/null 2>&1; then
    echo "NO_TASKSET - this host cannot narrow a running process from the shell; nothing here can be measured" >&2
    exit 9
fi

unset FLYNNEL_LEVER_ONCORE_SPREAD
unset FLYNNEL_LEVER_BATCH_WEIGHT
unset FLYNNEL_LEVER_SMT_WINDOW
unset FLYNNEL_LEVER_ALLOWED_WIDTH
unset FLYNNEL_LEVER_CALIBRATION_REFUSAL
export FLYNNEL_HOST_PROFILE_NS="$profile"

sh "$HOME/vm_presence.sh" claim $$ "flynnel allowed-width A/B, TIMINGS, needs a quiet box, Flynnel-Scholar"
trap 'sh "$HOME/vm_presence.sh" release '"$$" EXIT INT TERM

half=$(( windows / 2 ))
if [ "$half" -lt 1 ]; then
    echo "windows must be at least two so there are rows on both sides of the narrowing" >&2
    exit 2
fi
# One warm window, then the windows before the narrowing.
narrow_at=$(( 1 + half * window ))

echo "WIDTH_START $(date -u '+%Y-%m-%d %H:%M:%S') tree=$tree window=${window}s windows=$windows cpus=$cpus trials=$trials profile=$profile"
echo "HOST $(hostname) cores=$(nproc) load=$(cut -d' ' -f1-3 /proc/loadavg)"
echo "HEAD $(cd "$tree" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "NARROW_AFTER_WINDOW $half"

order_for() {
    case $(( ($1 - 1) % 6 )) in
        0) echo "off null on" ;;
        1) echo "off on null" ;;
        2) echo "null off on" ;;
        3) echo "null on off" ;;
        4) echo "on off null" ;;
        *) echo "on null off" ;;
    esac
}

work="${TMPDIR:-/tmp}/width_rounds.$$"
mkdir -p "$work" || exit 9
trap 'sh "$HOME/vm_presence.sh" release '"$$"'; rm -rf "'"$work"'"' EXIT INT TERM

failed=0

run_arm() {
    arm="$1"
    t="$2"
    out="$work/arm.out"
    err="$work/arm.err"
    : > "$out"
    : > "$err"
    echo "ARM ALLOWED_WIDTH-$arm t$t $(date -u '+%H:%M:%S')"
    # Both arms name the value. An arm whose meaning is the switch state
    # cannot express that state by leaving the switch unset: it then
    # takes whatever the crate defaults to, and this lever defaults on,
    # so an unset off arm would be the on arm twice.
    if [ "$arm" = on ]; then
        ( exec env FLYNNEL_LEVER_ALLOWED_WIDTH=1 "$bin" "$window" "$windows" > "$out" 2> "$err" ) &
    else
        ( exec env FLYNNEL_LEVER_ALLOWED_WIDTH=0 "$bin" "$window" "$windows" > "$out" 2> "$err" ) &
    fi
    child=$!
    sleep "$narrow_at"
    # The process prints its own pid as the first line. Both are
    # reported: a mismatch means the shell narrowed something other
    # than the binary, which would leave the rows unchanged and the
    # driver satisfied.
    said=$(awk '/^pid /{print $2; exit}' "$out" 2>/dev/null)
    echo "NARROW pid_shell=$child pid_said=${said:-none} cpus=$cpus $(date -u '+%H:%M:%S')"
    if [ "$said" != "$child" ]; then
        echo "PID_MISMATCH shell=$child said=${said:-none} - the narrowing may have missed the run"
    fi
    # A count of the threads re-pinned rather than taskset's line per
    # thread. The count is the evidence that the narrowing reached the
    # whole pool and not just the main thread; the raw lines are forty
    # per arm and say the same thing. On failure they are printed,
    # because then which thread refused is the question.
    pinned=$(taskset -acp "$cpus" "$child" 2>&1)
    prc=$?
    echo "NARROWED threads=$(printf '%s\n' "$pinned" | grep -c 'new affinity list') exit=$prc"
    if [ "$prc" -ne 0 ]; then
        printf '%s\n' "$pinned"
        echo "TASKSET_FAILED pid=$child"
    fi
    wait "$child"
    rc=$?
    cat "$out"
    cat "$err" >&2
    if [ "$rc" -ne 0 ]; then
        echo "ARM_FAILED ALLOWED_WIDTH-$arm t$t exit=$rc"
        failed=$((failed + 1))
    fi
}

t=1
while [ "$t" -le "$trials" ]; do
    for arm in $(order_for "$t"); do
        run_arm "$arm" "$t"
    done
    t=$((t + 1))
done

echo "HOST_END load=$(cut -d' ' -f1-3 /proc/loadavg)"
echo "WIDTH_DONE $(date -u '+%Y-%m-%d %H:%M:%S') failed_arms=$failed"
exit "$failed"
