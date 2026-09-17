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

SIDE = re.compile(r"^SIDE (\w+) load=(\d+) trial=(\d+)\s*$")
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
    rows = defaultdict(lambda: {"control": [], "loaded": [], "retained": []})
    side = None
    load = None
    for n, path in enumerate(paths, 1):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.strip()
                s = SIDE.match(line)
                if s:
                    side, load = s.group(1), int(s.group(2))
                    continue
                if side is None:
                    continue
                c = CONTROL.match(line)
                if c:
                    rows[(side, load)]["control"].append(int(c.group(3)))
                    continue
                t = THROUGHPUT.match(line)
                if t:
                    rows[(side, load)]["loaded"].append(int(t.group(3)))
                    continue
                r = RETAINED.match(line)
                if r:
                    rows[(side, load)]["retained"].append(float(r.group(3)))
        note(f"parsed {n}/{len(paths)} {path}: {len(rows)} side/load cells")
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

    loads = sorted({load for (_, load) in rows})
    print()
    print("TIP AGAINST BASE, per load level. Above 1.00 is the tip doing more.")
    print(f"{'load':>5} {'n':>3} {'quiet':>8} {'loaded':>8} {'retention':>10} "
          f"{'base spread':>12} {'tip spread':>11}")
    print("-" * 64)

    regressions = []
    for load in loads:
        b = rows.get(("base", load))
        t = rows.get(("tip", load))
        if not b or not t:
            print(f"{load:>5} {'-':>3} {'one side missing':>40}")
            continue

        quiet = ratio(median(t["control"]), median(b["control"]))
        loaded = ratio(median(t["loaded"]), median(b["loaded"]))
        retention = ratio(median(t["retained"]), median(b["retained"]))

        bs = spread(b["control"] + b["loaded"])
        ts = spread(t["control"] + t["loaded"])

        def fmt(x):
            return "-" if x is None else f"{x:.3f}"

        print(f"{load:>5} {len(t['retained']):>3} {fmt(quiet):>8} {fmt(loaded):>8} "
              f"{fmt(retention):>10} {fmt(bs):>12} {fmt(ts):>11}")

        # Flagged, not judged: a figure below one is only a regression if
        # it sits outside what the same binary varies by between trials.
        if quiet is not None and bs is not None and quiet < 1.0:
            worse_by = 1.0 / quiet
            if worse_by > bs:
                regressions.append(
                    f"  load {load}: quiet {quiet:.3f}, worse by more than the "
                    f"base's own spread of {bs:.3f}"
                )

    print()
    if regressions:
        print("SLOWER THAN THE BASE BY MORE THAN ITS OWN SPREAD:")
        for r in regressions:
            print(r)
        print()
        print("The criterion is never slower. A quiet figure below one that")
        print("also exceeds the base's trial-to-trial spread is a regression")
        print("rather than noise, and it stands whatever the loaded arm did.")
    else:
        print("No quiet figure fell below the base by more than the base's own")
        print("trial-to-trial spread. That is the never-slower half. Whether")
        print("the tip is FASTER under load is the retention column, which is")
        print("a separate claim and needs its own reading.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <never_slower.log> [more.log ...]", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
