"""Pair each bench arm's forward reading against its reversed one.

    python examples/pair_orders.py <criterion log>

`simc_cooperative` and `seed_depth_stability` register every group
twice, forward and with the arms reversed, so each arm is measured at
two positions. A ratio that survives both orders is the code; one
present in one order and absent or reversed in the other is the host.

WHAT IT COMPARES AGAINST, and why it is not a threshold. An arm's raw
forward-to-reversed difference is an UNPAIRED quantity: the two
readings sit at opposite ends of a group and carry whatever the machine
did in between. Criterion's interval is a within-arm precision
estimate, so testing one against the other rejects arms that merely
drifted with everything around them.

The relationship BETWEEN arms is the paired quantity, because arms in
one order drift together. The `vsgrp` column divides an arm's own
movement by its group's median movement, which removes the drift the
pairing already cancels and leaves the arm moving differently from its
neighbors - which is what contamination looks like. That figure is
compared against the arm's own interval width, so nothing here is a
number somebody picked, and both are printed on every row so a verdict
can be overruled from the same line.

WHAT IT REFUSES TO DO. A timing line it cannot attribute to an arm
ends the run rather than being dropped: a parser that silently skips a
format it does not recognize reports a complete-looking table over half
the data. An arm present in only one order is named as UNPAIRED rather
than omitted. An unrecognized time unit is an error rather than a
guess.

Units are normalized to nanoseconds. Criterion writes ns, us in either
spelling of micro, ms and s, and a sweep spanning sizes mixes them
within one table. It also writes the benchmark id on its own line when
the name is long and on the same line as the timing when it is short,
so both shapes appear in one log.
"""
import re
import sys
from collections import OrderedDict

ID = re.compile(r"^([A-Za-z0-9_]+)/([A-Za-z0-9_]+)\s*$")
BRACKET = r"\[\s*([\d.]+)\s*(\S+)\s+([\d.]+)\s*(\S+)\s+([\d.]+)\s*(\S+)\s*\]"
TIME = re.compile(r"^\s*time:\s*" + BRACKET)
ID_AND_TIME = re.compile(
    r"^([A-Za-z0-9_]+)/([A-Za-z0-9_]+)\s+time:\s*" + BRACKET
)
# Anything announcing a measurement, counted so the two patterns above
# can be held to account for all of it.
ANY_TIME = re.compile(r"\btime:\s*\[")
# What `seed_depth_stability` prints per arm. The fields are read as
# key=value pairs rather than by position, so one being added or
# removed leaves the others readable instead of matching nothing and
# reporting the whole column as absent.
OCCUPANCY = re.compile(r"^occupancy\s+(\S+)/(\S+)\s+(.*)$")
FIELD = re.compile(r"([a-z_]+)=(\d+)")

SCALE = {"ns": 1.0, "us": 1e3, "µs": 1e3, "μs": 1e3, "ms": 1e6, "s": 1e9}


def to_ns(value, unit):
    if unit not in SCALE:
        raise SystemExit("unrecognized time unit %r - refusing to guess" % unit)
    return float(value) * SCALE[unit]


def split_order(group):
    """Separate a group name into its base and which order it ran in."""
    if group.endswith("_rev"):
        return group[:-4], "rev"
    return group, "fwd"


def parse(path):
    """Return timings and per-arm figures, keyed by (base_group, arm).

    Timings are {order: (low, median, high)} in ns. The figures are
    {order: (occupancy_pct, flips)} and are absent for a bench that does
    not print them, which is not an error.
    """
    times = OrderedDict()
    figures = {}
    current = None
    seen_time_lines = 0
    attributed = 0

    def record(group, arm, t, first):
        low = to_ns(t.group(first), t.group(first + 1))
        med = to_ns(t.group(first + 2), t.group(first + 3))
        high = to_ns(t.group(first + 4), t.group(first + 5))
        base, order = split_order(group)
        times.setdefault((base, arm), {})[order] = (low, med, high)

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if ANY_TIME.search(line):
                seen_time_lines += 1
            o = OCCUPANCY.match(line.strip())
            if o:
                base, order = split_order(o.group(1))
                fields = {k: int(v) for k, v in FIELD.findall(o.group(3))}
                if fields:
                    figures.setdefault((base, o.group(2)), {})[order] = fields
                continue
            both = ID_AND_TIME.match(line)
            if both:
                record(both.group(1), both.group(2), both, 3)
                attributed += 1
                current = None
                continue
            m = ID.match(line)
            if m:
                current = (m.group(1), m.group(2))
                continue
            t = TIME.match(line)
            if t and current:
                record(current[0], current[1], t, 1)
                attributed += 1
                current = None

    dropped = seen_time_lines - attributed
    if dropped:
        raise SystemExit(
            "%d of %d timing lines in %s could not be attributed to an arm - "
            "refusing to report a partial table as a complete one. Fix the "
            "pattern before trusting any of it."
            % (dropped, seen_time_lines, path))
    return times, figures


def human(ns):
    if ns >= 1e6:
        return "%.4f ms" % (ns / 1e6)
    if ns >= 1e3:
        return "%.4f us" % (ns / 1e3)
    return "%.1f ns" % ns


def group_drift(times):
    """Each group's median reversed-over-forward ratio across its arms.

    Dividing an arm's own ratio by this leaves what the arm did that its
    neighbors did not, which is the paired comparison.
    """
    out = {}
    per_group = {}
    for (base, _arm), orders in times.items():
        if "fwd" in orders and "rev" in orders:
            per_group.setdefault(base, []).append(
                orders["rev"][1] / orders["fwd"][1])
    for base, ratios in per_group.items():
        ratios.sort()
        mid = len(ratios) // 2
        out[base] = (ratios[mid] if len(ratios) % 2
                     else (ratios[mid - 1] + ratios[mid]) / 2.0)
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python examples/pair_orders.py <criterion log>")
    times, figures = parse(sys.argv[1])
    if not times:
        raise SystemExit("no criterion time lines found in %s" % sys.argv[1])
    drift = group_drift(times)

    total = len(times)
    print("read %d arm(s) from %s, %d with a reported figure"
          % (total, sys.argv[1], len(figures)), flush=True)
    print("%4s %-28s %-22s %12s %12s %7s %7s %7s  %-16s %s"
          % ("left", "group", "arm", "forward", "reversed", "raw%", "vsgrp%",
             "band%", "reading", "pct,flips fwd/rev"), flush=True)
    unpaired = []
    flipping = []
    for done, ((base, arm), orders) in enumerate(times.items(), start=1):
        left = total - done
        if "fwd" not in orders or "rev" not in orders:
            unpaired.append((base, arm, sorted(orders)))
            continue
        fl, fm, fh_ = orders["fwd"]
        rl, rm, rh = orders["rev"]
        diff = abs(fm - rm) / min(fm, rm) * 100.0
        # The wider of the two orders' own interval widths, as a
        # percentage of that order's median.
        band = max((fh_ - fl) / fm, (rh - rl) / rm) * 100.0
        d = drift.get(base, 1.0)
        rel = ((rm / fm) / d) if d else 1.0
        rel_pct = abs(rel - 1.0) * 100.0
        reading = "agree" if rel_pct <= band else "ORDERS DISAGREE"

        f = figures.get((base, arm), {})
        if f:
            note = "/".join(
                ",".join("%s=%s" % kv for kv in sorted(f[k].items()))
                if k in f else "-"
                for k in ("fwd", "rev"))
            if any(v.get("flips", 0) for v in f.values()):
                flipping.append((base, arm, f))
        else:
            note = ""

        print("%4d %-28s %-22s %12s %12s %6.1f%% %6.1f%% %6.1f%%  %-16s %s"
              % (left, base, arm, human(fm), human(rm), diff, rel_pct, band,
                 reading, note), flush=True)

    for base, arm, have in unpaired:
        print("UNPAIRED %s/%s - only %s" % (base, arm, ",".join(have)),
              flush=True)

    for base, arm, f in flipping:
        print("FLIPPING %s/%s - dispatches here seeded a different leaf count "
              "from the one before them: %s" % (base, arm, f), flush=True)

    print("done: %d paired, %d unpaired, %d with seed-depth flips"
          % (total - len(unpaired), len(unpaired), len(flipping)), flush=True)


if __name__ == "__main__":
    main()
