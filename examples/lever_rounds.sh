#!/bin/sh
# One lever at a time, three arms per trial, arm order counterbalanced.
#
# Each lever is read against its own null: the same binary invoked a
# second time with the switch still off. Whatever two identical arms
# separate by is the floor the on/off ratio has to clear.
#
# Arm order rotates through all six permutations of three arms, so each
# arm takes each position in the rotation exactly twice per six trials.
# Position within a rotation moved throughput by 22 percent on this
# host, with the sign depending on load, so a fixed order measures
# position and arm as one quantity.
#
# The host profile is pinned identically on every arm. Unpinned, each
# process takes its own calibration draw, the draw sets the dispatch
# cost and collapse threshold, and those set the routing that
# throughput measures. The levers act on the call-site classifier, not
# on the host profile, so pinning removes a variable without touching
# what is under test.
#
# Output goes to this script's stdout, which the launcher redirects to
# a file on the guest. Nothing here streams back through a channel: a
# channel that tears down takes every row printed after it, while the
# run carries on and finishes looking exactly the same.
#
# The batch-weight arm runs its burners on a duty cycle rather than
# holding them. Every figure that weighting feeds divides a weighted
# total by a weighted count, so a share common to every batch cancels
# and steady contention moves nothing. The period is a parameter
# because it has to be long enough for a batch to fall inside one
# phase, which depends on the leaf size the workload produces.
#
# The workload shape is a parameter for the same kind of reason. Every
# lever here acts through the call-site classifier, and a workload whose
# leaves all cost the same gives the classifier a spread of zero and a
# class nothing can move. `reps` sets the per-item cost and `irregular`
# varies it deterministically with the index, so the work has a spread
# of its own for contention to be told apart from.
#
# usage: lever_rounds.sh <tree> <window_s> <trials> <load_threads> <duty_ms>
#                        <reps> <irregular> <profile_ns>

export PATH="$HOME/.cargo/bin:$PATH"

tree="$1"
window="$2"
trials="$3"
load_threads="$4"
duty_ms="$5"
reps="$6"
irregular="$7"
profile="$8"

if [ -z "$tree" ] || [ -z "$window" ] || [ -z "$trials" ] || [ -z "$load_threads" ] \
    || [ -z "$duty_ms" ] || [ -z "$reps" ] || [ -z "$irregular" ] || [ -z "$profile" ]; then
    echo "usage: $0 <tree> <window_s> <trials> <load_threads> <duty_ms> <reps> <irregular> <profile_ns>" >&2
    exit 2
fi

bin="$tree/target/release/examples/throughput_under_load"
if [ ! -x "$bin" ]; then
    echo "MISSING_BINARY $bin - build it before measuring" >&2
    exit 9
fi

# Inherited switches would apply to every arm including the off ones,
# and the describe() line each process prints would be the only record.
unset FLYNNEL_LEVER_ONCORE_SPREAD
unset FLYNNEL_LEVER_BATCH_WEIGHT
unset FLYNNEL_LEVER_SMT_WINDOW
unset FLYNNEL_LEVER_ALLOWED_WIDTH
unset FLYNNEL_LEVER_CALIBRATION_REFUSAL
# Not one of the five, and the same shape: a runtime switch defaulting
# off with the same criterion to meet. Its evidence is an outcome the
# pool already has rather than a load reading, so it is measured here
# beside them. FLYNNEL_SPIN_WINDOW_ROUNDS would pin the window and turn
# adaptation off, which is the other arm by accident.
unset FLYNNEL_ADAPTIVE_SPIN
unset FLYNNEL_SPIN_WINDOW_ROUNDS
export FLYNNEL_HOST_PROFILE_NS="$profile"

sh "$HOME/vm_presence.sh" claim $$ "flynnel per-lever throughput A/B, TIMINGS, needs a quiet box, about 20 minutes, Flynnel-Scholar"
trap 'sh "$HOME/vm_presence.sh" release '"$$" EXIT INT TERM

echo "LEVER_START $(date -u '+%Y-%m-%d %H:%M:%S') tree=$tree window=${window}s trials=$trials load=$load_threads duty_ms=$duty_ms reps=$reps irregular=$irregular profile=$profile"
echo "HOST $(hostname) cores=$(nproc) load=$(cut -d' ' -f1-3 /proc/loadavg)"
echo "HEAD $(cd "$tree" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# Two switches are absent from the table below, and the absences are
# printed so neither reads as an oversight.
#
# CALIBRATION_REFUSAL: prefers_incumbent returns None unless the
# incumbent record is trustworthy, trustworthiness needs a sample spread
# at or under 250 per mille, and no draw taken on this host reports
# below 1599.
#
# ALLOWED_WIDTH: the cap comes from available_parallelism, which reads
# the affinity mask and the cgroup quota. Burner threads move neither,
# so the pool here is sized against a width that is still correct
# whatever the load. width_rounds.sh narrows the process instead.
echo "SKIPPED CALIBRATION_REFUSAL - prefers_incumbent cannot fire while no draw clears the trust bound (flynnel-104)"
echo "SKIPPED ALLOWED_WIDTH - load does not move the affinity mask or the cgroup quota; see width_rounds.sh"

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

failed=0

run_arm() {
    lever="$1"
    var="$2"
    smt="$3"
    duty="$4"
    arm="$5"
    load="$6"
    t="$7"
    echo "ARM ${lever}-${arm} load${load} t${t} $(date -u '+%H:%M:%S')"
    if [ "$arm" = on ]; then
        env "$var=1" "$bin" "$window" "$load" 1 "$smt" "$duty" "$reps" "$irregular"
    else
        "$bin" "$window" "$load" 1 "$smt" "$duty" "$reps" "$irregular"
    fi
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "ARM_FAILED ${lever}-${arm} load${load} t${t} exit=$rc"
        failed=$((failed + 1))
    fi
}

# Name, switch, whether the plan carries the SMT prior, and the burner
# duty period. The prior is what effective_use_smt needs before it
# reads anything else, so the window switch is unreachable without it.
# A duty of zero holds the burners for the whole window.
for spec in \
    "ONCORE_SPREAD FLYNNEL_LEVER_ONCORE_SPREAD 0 0" \
    "BATCH_WEIGHT FLYNNEL_LEVER_BATCH_WEIGHT 0 $duty_ms" \
    "SMT_WINDOW FLYNNEL_LEVER_SMT_WINDOW 1 0" \
    "ADAPTIVE_SPIN FLYNNEL_ADAPTIVE_SPIN 0 $duty_ms"
do
    set -- $spec
    lever="$1"
    var="$2"
    smt="$3"
    duty="$4"
    for load in 0 "$load_threads"; do
        t=1
        while [ "$t" -le "$trials" ]; do
            for arm in $(order_for "$t"); do
                run_arm "$lever" "$var" "$smt" "$duty" "$arm" "$load" "$t"
            done
            t=$((t + 1))
        done
    done
done

echo "HOST_END load=$(cut -d' ' -f1-3 /proc/loadavg)"
echo "LEVER_DONE $(date -u '+%Y-%m-%d %H:%M:%S') failed_arms=$failed"
exit "$failed"
