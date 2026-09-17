"""Which dispersion statistic tracks the condition a draw was taken in.

The calibration trust check reads (max - min) / median and refuses a
record above a bound. On every host measured so far no draw clears it,
while independent draws agree on their medians to a few percent. Those
two cannot both describe the dispersion of the same quantity.

This reads a set of draws taken across stated load levels and puts the
candidates side by side:

  across draws   how far the MEDIANS of independent draws fall apart.
                 This is the reproducibility the trust check exists to
                 protect, and it is the thing a bound should track.
  within a draw  the range and the interquartile range of the samples
                 behind one draw, which is what the check actually
                 reads.

A statistic worth gating on is small where the medians reproduce and
grows where they do not. One that stays large in both says nothing
about either.

Feed it a log carrying BOTH streams in order, since a draw is a pair of
sweep lines followed by its profile:

  FLYNNEL_PROFILE_SAMPLES=1 profile_draws 5 0 2>&1 | tee quiet.log
"""

import re
import sys
from collections import defaultdict

# Each pattern is coupled to one format string and nothing enforces the
# coupling, so the producer is named beside it. A pattern whose producer
# has been reworded matches nothing and reports a clean empty section,
# which is the same output as a run that measured nothing.
#
# src/sched/par_iter.rs, the crossover sweep under FLYNNEL_PROFILE_SAMPLES.
# The dispatch cost's own samples print as "profile point:" instead,
# carry no spread or iqr, and gate nothing, so they are not read here.
SWEEP = re.compile(
    r"^profile sweep:\s+min\s+(\d+)\s+median\s+(\d+)\s+max\s+(\d+)\s+"
    r"spread\s+(\d+)\s+iqr\s+(\d+)\s+per mille\s+samples\s+\[([0-9,\s]+)\]\s*$"
)
# examples/profile_draws.rs, one line per draw on stdout.
DRAW = re.compile(r"^draw\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+load=(\d+)\s*$")
# examples/profile_draws.rs, once per process. Nonzero means the stored
# record was handed back rather than measured, which a reader cannot
# tell from a steady host.
REPEATED = re.compile(r"^repeated_draws\s+(\d+)\s*$")
# src/sched/calibration_store.rs, at the publish comparison, under
# FLYNNEL_OCCUPANCY. The same quantity as a pair gap above, reported by
# a running process, which is the one place a live process holds two
# draws of its host at once. It carries no load label, because nothing
# at that site knows what else the box was doing.
FIELD = re.compile(
    r"^flynnel: two draws of this host disagree by (\d+),(\d+),(\d+) per mille"
)
FIGURES = ("dispatch", "collapse", "wake")


def median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def spread_per_mille(values):
    """(max - min) / median, in parts per thousand: the check's own form."""
    m = median(values)
    if not m:
        return None
    return (max(values) - min(values)) * 1000.0 / m


def pairwise_per_mille(values):
    """Typical disagreement within a pair of draws, per thousand.

    What a calibration could afford to measure about itself: draw
    twice, compare the medians, keep nothing else. Pairs are DISJOINT -
    (1,2), (3,4), (5,6) - so each one is what a single process taking
    two draws would have seen, and no draw contributes to two of them.
    Overlapping pairs would share a draw and correlate, which is not
    the thing being tested.

    The figure is the median over those pairs, so one unlucky pair does
    not set it - the same reason a draw keeps a median.

    Distinct from the spread over all of them, which needs as many
    draws as it has and is what this is being tested against.
    """
    if len(values) < 2:
        return None
    m = median(values)
    if not m:
        return None
    gaps = [abs(values[i + 1] - values[i]) for i in range(0, len(values) - 1, 2)]
    return median(gaps) * 1000.0 / m


def pair_gaps_per_mille(values):
    """Every disjoint pair's disagreement, per thousand, not their median.

    `pairwise_per_mille` answers what a typical pair looks like, which
    is the right figure for a separation table. A bound is set by the
    ends instead: the worst pair a quiet host produces is what it has
    to sit above, and the best pair a busy host produces is what it has
    to sit below. A median hides both ends, which are the whole of the
    question.
    """
    if len(values) < 2:
        return []
    m = median(values)
    if not m:
        return []
    return [
        abs(values[i + 1] - values[i]) * 1000.0 / m
        for i in range(0, len(values) - 1, 2)
    ]


def note(text):
    """One progress line, flushed.

    Flushed here rather than left to the caller passing `-u`: a reader
    pointed at a rotation's worth of logs is the case where progress
    matters, and that is also the case where it is redirected to a file
    and held in a buffer until the end. A script that reports only its
    result cannot be asked where it is while it runs.
    """
    print(text, file=sys.stderr, flush=True)


def parse(paths):
    by_load = defaultdict(lambda: {"draws": [], "sweeps": []})
    repeated = 0
    pending = []
    for n, path in enumerate(paths, 1):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.strip()
                s = SWEEP.match(line)
                if s:
                    pending.append(
                        {
                            "median": int(s.group(2)),
                            "spread": int(s.group(4)),
                            "iqr": int(s.group(5)),
                            "samples": [int(x) for x in s.group(6).split(",")],
                        }
                    )
                    continue
                r = REPEATED.match(line)
                if r:
                    repeated += int(r.group(1))
                    continue
                d = DRAW.match(line)
                if d:
                    load = int(d.group(5))
                    by_load[load]["draws"].append(
                        {
                            "dispatch": int(d.group(2)),
                            "collapse": int(d.group(3)),
                            "wake": int(d.group(4)),
                        }
                    )
                    # The sweeps printed since the last draw belong to
                    # this one. A draw with none means the samples were
                    # not printed, which the harness refuses to run
                    # without, so it is reported rather than filled in.
                    by_load[load]["sweeps"].append(pending)
                    pending = []
        drawn = sum(len(v["draws"]) for v in by_load.values())
        note(f"parsed {n}/{len(paths)} {path}: {drawn} draws so far")
    return by_load, repeated


def main(paths):
    by_load, repeated = parse(paths)
    if not by_load:
        print("NO DRAWS PARSED")
        return 1
    if repeated:
        print(
            f"{repeated} draw(s) returned the previous draw's figures: a stored record "
            "answering rather than a measurement. Everything below is short by that many."
        )

    print()
    print("ACROSS DRAWS - how far independent medians fall apart:")
    print(f"{'load':>5} {'n':>3} " + " ".join(f"{f:>12}" for f in FIGURES))
    print("-" * 50)
    for load in sorted(by_load):
        draws = by_load[load]["draws"]
        cells = []
        for f in FIGURES:
            v = spread_per_mille([d[f] for d in draws])
            cells.append("-" if v is None else f"{v:>12.0f}")
        print(f"{load:>5} {len(draws):>3} " + " ".join(cells))
    print("in parts per thousand of the median. This is the figure a")
    print("bound exists to protect.")

    print()
    print("WITHIN A DRAW - what the trust check reads, and its candidate:")
    print(f"{'load':>5} {'sweeps':>7} {'range':>10} {'iqr':>10} {'range/iqr':>10}")
    print("-" * 50)
    for load in sorted(by_load):
        flat = [s for per_draw in by_load[load]["sweeps"] for s in per_draw]
        if not flat:
            print(f"{load:>5} {'0':>7}   no sweep lines: the samples were not printed")
            continue
        rng = median([s["spread"] for s in flat])
        iqr = median([s["iqr"] for s in flat])
        ratio = "-" if not iqr else f"{rng / iqr:>10.1f}"
        print(f"{load:>5} {len(flat):>7} {rng:>10.0f} {iqr:>10.0f} {ratio:>10}")
    print("medians over every sweep at that load, in parts per thousand.")
    print("These are the two crossover sweeps, which is what the trust")
    print("check reads: it keeps the larger of the collapse and wake")
    print("spreads. The dispatch cost's own samples print under a")
    print("different label, carry no spread or iqr, and gate nothing.")

    separation(by_load)
    admitted = candidate_bound(by_load) or {}
    field_gaps(paths, admitted)

    print()
    print("READ IT THIS WAY. A statistic worth gating on is small where")
    print("the across-draw figures are small and grows with them. One")
    print("that is large at every load separates no condition, and a")
    print("bound on it refuses every record or none.")
    return 0


def separation(by_load):
    """How far each candidate moves between the quietest and busiest draws.

    The question is not which statistic looks reasonable but which one
    TRACKS. A bound exists to protect the reproducibility of the median,
    so the figure to match is how far independent medians fall apart;
    a statistic that barely moves while that figure moves a lot cannot
    tell the two conditions apart at any bound.

    Reported as a ratio between the extreme load levels, which needs at
    least two of them. Nothing is chosen here and no bound is proposed.
    """
    loads = sorted(by_load)
    print()
    print("SEPARATION between the quietest and busiest draws:")
    if len(loads) < 2:
        print("  one load level only; a separation needs at least two.")
        return
    lo, hi = loads[0], loads[-1]

    def ratio(a, b):
        """The separation, or why there is not one.

        A quiet figure of zero is not missing data: it is the pairs
        agreeing exactly, which is the strongest separation a figure
        can show and must not render the same as a column nobody
        filled. Returned as a string so the two cases stay apart in the
        table, since that collapse is the defect this whole exercise
        keeps turning up.
        """
        if a is None or b is None:
            return "-"
        if a == 0:
            return "exact" if b == 0 else "0 -> nonzero"
        return f"{b / a:.1f}"

    rows = []
    for f in FIGURES:
        a = spread_per_mille([d[f] for d in by_load[lo]["draws"]])
        b = spread_per_mille([d[f] for d in by_load[hi]["draws"]])
        rows.append((f"across draws, {f}", a, b, ratio(a, b)))
    # The one candidate a running calibration could afford. Five draws
    # is not something a process start can pay for; two is. If the
    # disagreement between one consecutive PAIR tracks the scatter of
    # all five above, then two draws carry the signal and the extra
    # three are not buying anything.
    for f in FIGURES:
        a = pairwise_per_mille([d[f] for d in by_load[lo]["draws"]])
        b = pairwise_per_mille([d[f] for d in by_load[hi]["draws"]])
        rows.append((f"two draws, {f}", a, b, ratio(a, b)))
    for key, label in (("spread", "within a draw, range"), ("iqr", "within a draw, iqr")):
        flat_lo = [s for per in by_load[lo]["sweeps"] for s in per]
        flat_hi = [s for per in by_load[hi]["sweeps"] for s in per]
        a = median([s[key] for s in flat_lo]) if flat_lo else None
        b = median([s[key] for s in flat_hi]) if flat_hi else None
        rows.append((label, a, b, ratio(a, b)))

    print(f"{'figure':<26} {f'load {lo}':>10} {f'load {hi}':>10} {'ratio':>14}")
    print("-" * 64)
    for label, a, b, r in rows:
        at = "-" if a is None else f"{a:>10.0f}"
        bt = "-" if b is None else f"{b:>10.0f}"
        print(f"{label:<26} {at} {bt} {r:>14}")
    print()
    print("'0 -> nonzero' is a figure that was exactly zero on the quiet")
    print("host and is not on the busy one: the pairs agreed to the")
    print("nanosecond and then stopped. That is the widest separation a")
    print("column can show, and a dash would have hidden it.")
    print()
    print("The top rows are what a bound is FOR. A candidate below them")
    print("whose ratio is near 1 does not separate these conditions and")
    print("cannot be made to by choosing a bound.")


def candidate_bound(by_load):
    """What bound the data admits, if it admits one.

    A bound on two-draw disagreement has to clear two things at once.
    It has to sit above every gap the quiet host produced, or a quiet
    host has its calibration refused and re-measures forever. It has to
    sit at or below every gap the busy host produced, or a contended
    draw is accepted and, because a passing record is permanent per
    stamp, stays accepted.

    So the data admits a bound exactly when the worst quiet pair is
    smaller than the best busy pair, and the admissible bounds are the
    interval between them. When those ranges overlap no number is
    proposed, because none exists: an overlap says the two conditions
    produce the same disagreement, and a bound picked anyway would
    refuse quiet hosts and accept busy ones at whatever rate the
    overlap dictates, while looking like a threshold someone chose.

    The pair counts print alongside, because an interval between two
    extremes is set by exactly one observation at each end. Read them
    before reading the interval: nothing here is refused for having too
    few pairs, so a bound resting on two of them prints the same as one
    resting on fifty and only the count says which it is.
    """
    admitted = {}
    loads = sorted(by_load)
    print()
    print("CANDIDATE BOUND on two-draw disagreement:")
    if len(loads) < 2:
        print("  one load level only; a bound needs a quiet side and a busy one.")
        return admitted
    lo, hi = loads[0], loads[-1]

    print(
        f"{'figure':<14} {'quiet pairs':>11} {'worst quiet':>12} "
        f"{'busy pairs':>11} {'best busy':>11} {'admits':>22}"
    )
    print("-" * 86)
    for f in FIGURES:
        quiet = pair_gaps_per_mille([d[f] for d in by_load[lo]["draws"]])
        busy = pair_gaps_per_mille([d[f] for d in by_load[hi]["draws"]])
        if not quiet or not busy:
            print(f"{f:<14} {len(quiet):>11} {'-':>12} {len(busy):>11} {'-':>11} {'no pairs':>22}")
            continue
        worst_quiet = max(quiet)
        best_busy = min(busy)
        if worst_quiet < best_busy:
            # Geometric rather than arithmetic: these are ratios, so a
            # midpoint should sit proportionally between the ends
            # rather than be dragged by the larger one.
            pick = (worst_quiet * best_busy) ** 0.5 if worst_quiet > 0 else best_busy / 2.0
            admits = f"{worst_quiet:.0f} < b <= {best_busy:.0f}, try {pick:.0f}"
            admitted[f] = (worst_quiet, best_busy, pick)
        else:
            admits = f"overlap by {worst_quiet - best_busy:.0f}"
            admitted[f] = (worst_quiet, best_busy, None)
        print(
            f"{f:<14} {len(quiet):>11} {worst_quiet:>12.0f} "
            f"{len(busy):>11} {best_busy:>11.0f} {admits:>22}"
        )
    print()
    print("'overlap by N' is the answer that matters: the quiet host's")
    print("worst pair disagreed by N more than the busy host's best, so")
    print("no bound tells them apart and the figure cannot carry the")
    print("refusal however it is tuned. A figure that admits an interval")
    print("is one where a bound is a reading rather than a preference.")
    return admitted


def field_gaps(paths, admitted):
    """The same quantity as reported by running processes, and how a
    candidate bound would have fallen against it.

    These carry no load label: the publish comparison knows what the two
    draws were, not what else the box was doing. So they cannot be split
    into a quiet side and a busy one and cannot derive a bound. What
    they can say is how often a bound derived above would have fired on
    traffic that actually happened, which is the question the derivation
    cannot answer about itself.

    A bound that fires on almost everything or on almost nothing is one
    whose interval was set by the sweep's conditions rather than by the
    host, and neither the interval nor its width shows that.
    """
    seen = {f: [] for f in FIGURES}
    for n, path in enumerate(paths, 1):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                m = FIELD.match(raw.strip())
                if m:
                    for f, g in zip(FIGURES, m.groups()):
                        seen[f].append(int(g))
        note(f"scanned {n}/{len(paths)} {path} for field gaps: {len(seen['dispatch'])} so far")

    total = sum(len(v) for v in seen.values())
    print()
    print("FIELD GAPS reported by running processes:")
    if total == 0:
        print("  none in these logs. The publish comparison reports only")
        print("  under FLYNNEL_OCCUPANCY, and only where a record was")
        print("  already stored for this host, so an empty section is a")
        print("  run that never met an incumbent rather than a run whose")
        print("  draws agreed.")
        return

    print(f"{'figure':<10} {'n':>5} {'min':>8} {'median':>8} {'max':>8} {'vs bound':>22}")
    print("-" * 66)
    for f in FIGURES:
        values = seen[f]
        if not values:
            print(f"{f:<10} {0:>5} {'-':>8} {'-':>8} {'-':>8} {'-':>22}")
            continue
        band = admitted.get(f)
        if band is None or band[2] is None:
            verdict = "no bound derived"
        else:
            pick = band[2]
            over = sum(1 for v in values if v > pick)
            verdict = f"{over}/{len(values)} over {pick:.0f}"
        print(
            f"{f:<10} {len(values):>5} {min(values):>8} "
            f"{median(values):>8.0f} {max(values):>8} {verdict:>22}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <draws.log> [more.log ...]", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
