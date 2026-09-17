"""Does the newer scheduler retain as much under load, and cost nothing quiet?

Reads a log from a run that alternates two trees through one harness.
The harness measures a control window and a loaded window inside each
process, so the figure compared here is already a within-process ratio
and the two sides differ only in the library.

Three questions, reported separately because they fail separately:

  quiet cost      control dispatches, tip against base. The criterion is
                  never slower, so a tip below base here is a regression
                  whatever the loaded arm does.
  loaded cost     loaded dispatches, tip against base.
  retention       loaded as a share of quiet, tip against base. This is
                  what improving under load means: holding more of the
                  quiet rate when the machine is busy.

Nothing is called a win or a loss against a fixed threshold. The spread
across trials is printed beside each figure, because a difference
smaller than the spread is not a difference, and the reader is expected
to compare the two rather than be told.
"""

import re
import sys
from collections import defaultdict

SIDE = re.compile(
    r"^SIDE (\w+) load=(\d+) trial=(\d+) reps=(\d+) entry=(\w+) foreign=(\d+)\s*$"
)
THROUGHPUT = re.compile(r"^throughput (\d+) (\d+) (\d+) ([\d.]+)\s*$")
CONTROL = re.compile(r"^control (\d+) (\d+) (\d+) ([\d.]+)\s*$")
RETAINED = re.compile(r"^retained (\d+) (\d+) ([\d.]+) control_first=(\w+)\s*$")


def note(text):
    print(text, file=sys.stderr, flush=True)


def median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def spread(v):
    """Max over min, or None where a zero makes the ratio undefined."""
    if not v:
        return None
    lo, hi = min(v), max(v)
    return hi / lo if lo > 0 else None


def parse(paths):
    """Rows keyed by (side, load). A SIDE line names who produced the
    rows after it, so a log with no SIDE lines yields nothing rather
    than attributing every row to one tree."""
    rows = defaultdict(lambda: {"control": [], "loaded": [], "retained": [], "foreign": []})
    key = None
    for n, path in enumerate(paths, 1):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.strip()
                s = SIDE.match(line)
                if s:
                    # Keyed on reps as well as load: two per-item costs
                    # are two workloads, and pooling them would average a
                    # figure that is large at one and small at the other
                    # into one that is true at neither.
                    key = (s.group(1), int(s.group(2)), int(s.group(4)))
                    rows[key]["foreign"].append(int(s.group(6)))
                    continue
                if key is None:
                    continue
                c = CONTROL.match(line)
                if c:
                    rows[key]["control"].append(int(c.group(3)))
                    continue
                t = THROUGHPUT.match(line)
                if t:
                    rows[key]["loaded"].append(int(t.group(3)))
                    continue
                r = RETAINED.match(line)
                if r:
                    rows[key]["retained"].append(float(r.group(3)))
        note(f"parsed {n}/{len(paths)} {path}: {len(rows)} side/load/reps cells")
    return rows


def ratio(tip, base):
    if base is None or tip is None or base == 0:
        return None
    return tip / base


def main(paths):
    rows = parse(paths)
    if not rows:
        print("NO ROWS PARSED - the log carries no SIDE lines, so nothing")
        print("can be attributed to either tree.")
        return 1

    cells = sorted({(load, reps) for (_, load, reps) in rows})
    print()
    print("TIP AGAINST BASE. Above 1.00 is the tip doing more.")
    print(f"{'reps':>6} {'load':>5} {'n':>3} {'quiet':>8} {'loaded':>8} "
          f"{'retention':>10} {'base spread':>12} {'tip spread':>11}")
    print("-" * 71)

    regressions = []
    undecided = []
    for (load, reps) in cells:
        b = rows.get(("base", load, reps))
        t = rows.get(("tip", load, reps))
        if not b or not t:
            print(f"{reps:>6} {load:>5} {'-':>3} {'one side missing':>40}")
            continue

        quiet = ratio(median(t["control"]), median(b["control"]))
        loaded = ratio(median(t["loaded"]), median(b["loaded"]))
        retention = ratio(median(t["retained"]), median(b["retained"]))

        # The quiet arm's own variation between trials, and nothing else.
        # Pooling the control and loaded readings would put the load
        # effect into the spread, which is the quantity being measured
        # rather than the noise around it, and at any real load level
        # that pooled figure is large enough to excuse every regression.
        bs = spread(b["control"])
        ts = spread(t["control"])
        trials = len(t["retained"])

        def fmt(x):
            return "-" if x is None else f"{x:.3f}"

        print(f"{reps:>6} {load:>5} {trials:>3} {fmt(quiet):>8} {fmt(loaded):>8} "
              f"{fmt(retention):>10} {fmt(bs):>12} {fmt(ts):>11}")

        # Flagged, not judged, and only where there is something to judge
        # against. One trial has no spread to speak of: min equals max, so
        # the estimate is 1.000 and every difference however small clears
        # it. That is a verdict manufactured from a sample of one.
        # A spread wider than the difference it is meant to judge decides
        # nothing. The test "is the shortfall bigger than the noise"
        # passes automatically once the noise is large, so a cell like
        # that is not evidence of no regression; it is evidence that this
        # run could not have seen one. Derived from the data rather than
        # set: the comparison is the cell's own spread against the
        # cell's own deviation from parity.
        deviation = None if quiet is None else abs(1.0 - quiet)
        blind = (
            bs is not None
            and deviation is not None
            and (bs - 1.0) > max(deviation, 0.0)
        )

        if trials < 2:
            undecided.append(
                f"  reps {reps} load {load}: {trials} trial, so the base has no "
                f"trial-to-trial spread and nothing here can be called noise"
            )
        elif blind:
            undecided.append(
                f"  reps {reps} load {load}: base varies by {bs:.3f} between its own "
                f"trials while the tip differs by {deviation:.3f}, so the noise is "
                f"larger than the effect and this cell decides nothing"
            )
        elif quiet is not None and bs is not None and quiet < 1.0:
            worse_by = 1.0 / quiet
            if worse_by > bs:
                regressions.append(
                    f"  reps {reps} load {load}: quiet {quiet:.3f}, worse by "
                    f"{worse_by:.3f} against the base's own quiet spread of {bs:.3f}"
                )

    # What else was on the box while this ran. The arms alternate, so
    # background load reaches both trees and does not bias the
    # comparison; it widens the spread a difference has to clear, which
    # costs sensitivity. A run with foreign processes throughout and no
    # regression found has not shown there is none, only that none was
    # large enough to see through that much noise.
    foreign = [f for cell in rows.values() for f in cell["foreign"]]
    if foreign:
        busy = sum(1 for f in foreign if f > 0)
        print()
        print(f"BACKGROUND: {busy} of {len(foreign)} arms started with other build")
        print(f"  processes running, up to {max(foreign)} at once.")
        if busy:
            print("  The comparison survives this because the arms alternate, but")
            print("  the spread is wider than it would be on a quiet box, so a")
            print("  small regression could sit under it unseen.")

    print()
    if undecided:
        print("CELLS THAT DECIDE NOTHING:")
        for u in undecided:
            print(u)
        print()

    if regressions:
        print("SLOWER THAN THE BASE BY MORE THAN ITS OWN QUIET SPREAD:")
        for r in regressions:
            print(r)
        print()
        print("The criterion is never slower. A quiet figure below one that")
        print("also exceeds the base's trial-to-trial spread is a regression")
        print("rather than noise, and it stands whatever the loaded arm did.")
    elif not undecided:
        print("No quiet figure fell below the base by more than the base's own")
        print("trial-to-trial spread, and no cell was too noisy to decide. That")
        print("is the never-slower half. Whether the tip is FASTER under load is")
        print("the retention column, which is a separate claim and needs its own")
        print("reading.")
    else:
        print("NOTHING IS CONCLUDED ABOUT THE UNDECIDED CELLS ABOVE. A cell whose")
        print("noise exceeds the effect passes the never-slower test for the wrong")
        print("reason, and counting that as a pass is how a run that could not")
        print("have seen a regression gets reported as one that found none.")

    # Non-zero where something is wrong or unjudgeable, so a caller that
    # reads only the code does not take an undecided run for a clean one.
    if regressions:
        return 1
    if undecided:
        return 3
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <never_slower.log> [more.log ...]", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
