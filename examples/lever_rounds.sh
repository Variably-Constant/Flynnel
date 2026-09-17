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
# Each load level is visited twice, with the levels rotated between the
# two passes so every one lands at a different position. A cell whose
# on/off ratio does not reproduce between its two visits was following
# the box rather than the load: an effect that tracks contention
# repeats wherever in the run the level is measured, and one that
# tracks drift does not.
#
# The curve's shape discriminates nothing on its own. A lever that
# genuinely helps only when cores are free and an artefact of an
# unloaded box both produce a ratio that moves with load.
# Repeatability under a changed visiting order is what separates them.
#
# usage: lever_rounds.sh <tree> <window_s> <trials> '<load levels>'
#                        <duty_ms> <reps> <irregular> <profile_ns>

export PATH="$HOME/.cargo/bin:$PATH"

tree="$1"
window="$2"
trials="$3"
levels="$4"
duty_ms="$5"
reps="$6"
irregular="$7"
profile="$8"

if [ -z "$tree" ] || [ -z "$window" ] || [ -z "$trials" ] || [ -z "$levels" ] \
    || [ -z "$duty_ms" ] || [ -z "$reps" ] || [ -z "$irregular" ] || [ -z "$profile" ]; then
    echo "usage: $0 <tree> <window_s> <trials> '<load levels>' <duty_ms> <reps> <irregular> <profile_ns>" >&2
    exit 2
fi

# Three or more levels, so the on/off ratio traces a curve rather than
# joining two points. Quoted as one argument: '0 6 12'.
set -- $levels
if [ $# -lt 3 ]; then
    echo "give at least three load levels; two visited once each cannot show whether a ratio repeats" >&2
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

echo "LEVER_START $(date -u '+%Y-%m-%d %H:%M:%S') tree=$tree window=${window}s trials=$trials levels='$levels' duty_ms=$duty_ms reps=$reps irregular=$irregular profile=$profile"
echo "HOST $(hostname) cores=$(nproc) load=$(cut -d' ' -f1-3 /proc/loadavg)"
echo "HEAD $(cd "$tree" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# Two switches are absent from the table below, and the absences are
# printed so neither reads as an oversight.
#
# CALIBRATION_REFUSAL: this harness pins the profile with
# FLYNNEL_HOST_PROFILE_NS, so nothing in it draws a calibration and the
# publish comparison the switch gates is never reached. That is a
# property of the harness, not of the code. A rotation that wanted this
# switch would have to let the calibration run, and would then need a
# stored record old enough to age out, since a fresh draw with nothing
# stored has no incumbent to compare against.
#
# ALLOWED_WIDTH: the cap comes from available_parallelism, which reads
# the affinity mask and the cgroup quota. Burner threads move neither,
# so the pool here is sized against a width that is still correct
# whatever the load. width_rounds.sh narrows the process instead.
echo "SKIPPED CALIBRATION_REFUSAL - this harness pins the profile with FLYNNEL_HOST_PROFILE_NS, so no calibration is drawn and the comparison the switch gates is never reached"
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

# The load levels for one pass, rotated by the pass number so every
# level is visited at a different position on the second pass.
#
# Two levels visited once each give one ratio per level and no way to
# tell a curve from two points that happen to differ. Adding levels
# without varying the sequence they are walked in does not help either:
# a box that drifts over the run produces a monotone curve for free
# when the levels always come in the same succession. Rotating means an
# effect that follows the box does not reproduce when a level is
# visited at a different position, while one that follows the load
# does.
#
# Rotation rather than reversal, because reversing leaves a middle
# level at the same position in both passes.
levels_for() {
    pass="$1"
    shift
    n=$#
    skip=$(( (pass - 1) % n ))
    i=0
    for lv in "$@"; do
        [ "$i" -ge "$skip" ] && echo "$lv"
        i=$((i + 1))
    done
    i=0
    for lv in "$@"; do
        [ "$i" -lt "$skip" ] && echo "$lv"
        i=$((i + 1))
    done
}

run_arm() {
    lever="$1"
    var="$2"
    smt="$3"
    duty="$4"
    arm="$5"
    load="$6"
    t="$7"
    pass="$8"
    echo "ARM ${lever}-${arm} load${load} p${pass} t${t} $(date -u '+%H:%M:%S')"
    if [ "$arm" = on ]; then
        env "$var=1" "$bin" "$window" "$load" 1 "$smt" "$duty" "$reps" "$irregular"
    else
        "$bin" "$window" "$load" 1 "$smt" "$duty" "$reps" "$irregular"
    fi
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "ARM_FAILED ${lever}-${arm} load${load} p${pass} t${t} exit=$rc"
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
    pass=1
    while [ "$pass" -le 2 ]; do
      for load in $(levels_for "$pass" $levels); do
        t=1
        while [ "$t" -le "$trials" ]; do
            for arm in $(order_for "$t"); do
                run_arm "$lever" "$var" "$smt" "$duty" "$arm" "$load" "$t" "$pass"
            done
            t=$((t + 1))
        done
      done
      pass=$((pass + 1))
    done
done

echo "HOST_END load=$(cut -d' ' -f1-3 /proc/loadavg)"
echo "LEVER_DONE $(date -u '+%Y-%m-%d %H:%M:%S') failed_arms=$failed"
exit "$failed"
