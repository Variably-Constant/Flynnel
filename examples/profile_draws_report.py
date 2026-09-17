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

SWEEP = re.compile(
    r"^profile sweep:\s+min\s+(\d+)\s+median\s+(\d+)\s+max\s+(\d+)\s+"
    r"spread\s+(\d+)\s+iqr\s+(\d+)\s+per mille\s+samples\s+\[([0-9,\s]+)\]\s*$"
)
DRAW = re.compile(r"^draw\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+load=(\d+)\s*$")
REPEATED = re.compile(r"^repeated_draws\s+(\d+)\s*$")
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


def parse(paths):
    by_load = defaultdict(lambda: {"draws": [], "sweeps": []})
    repeated = 0
    pending = []
    for path in paths:
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

    separation(by_load)

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
        if a is None or b is None or not a:
            return None
        return b / a

    rows = []
    for f in FIGURES:
        a = spread_per_mille([d[f] for d in by_load[lo]["draws"]])
        b = spread_per_mille([d[f] for d in by_load[hi]["draws"]])
        rows.append((f"across draws, {f}", a, b, ratio(a, b)))
    for key, label in (("spread", "within a draw, range"), ("iqr", "within a draw, iqr")):
        flat_lo = [s for per in by_load[lo]["sweeps"] for s in per]
        flat_hi = [s for per in by_load[hi]["sweeps"] for s in per]
        a = median([s[key] for s in flat_lo]) if flat_lo else None
        b = median([s[key] for s in flat_hi]) if flat_hi else None
        rows.append((label, a, b, ratio(a, b)))

    print(f"{'figure':<26} {f'load {lo}':>10} {f'load {hi}':>10} {'ratio':>8}")
    print("-" * 58)
    for label, a, b, r in rows:
        at = "-" if a is None else f"{a:>10.0f}"
        bt = "-" if b is None else f"{b:>10.0f}"
        rt = "-" if r is None else f"{r:>8.1f}"
        print(f"{label:<26} {at} {bt} {rt}")
    print()
    print("The top rows are what a bound is FOR. A candidate below them")
    print("whose ratio is near 1 does not separate these conditions and")
    print("cannot be made to by choosing a bound.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <draws.log> [more.log ...]", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
