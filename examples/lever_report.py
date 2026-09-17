"""Per-lever throughput, each switch read against its own null.

Every lever has to clear the same two-part criterion: never slower with
the host quiet, faster with it contended, against the same code without
the lever. One binary runs both arms, so a row differs from its
neighbor in the switch and in nothing else.

The null arm is the same binary invoked a second time with the switch
still off. It measures what two identical arms do in this rotation, and
it is the floor the on/off ratio has to clear. Without it a 1.48 and a
1.07 look alike, which is how an earlier reading here was called and
then withdrawn.

The engagement section is read before the throughput table. A switch
that never reached its mechanism produces the same clean null as one
that reached it and did not help, and the difference is not visible in
a dispatch count.
"""

import re
import sys
from collections import defaultdict

ARM = re.compile(r"^ARM\s+([A-Z_]+)-(off|null|on)\s+load(\d+)\s+t(\d+)")
ROW = re.compile(r"^throughput\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s*$")
ENGAGE = re.compile(r"^engagement\s+(.*)$")
FAILED = re.compile(r"^ARM_FAILED\s+(\S+)\s+load(\d+)\s+t(\d+)\s+exit=(\d+)")

# The fields a lever acts through. Each is discrete, so a switch that
# moved none of them across its own arms moved no decision, whatever
# the dispatch counts did.
DECISIONS = ("class", "smt", "workers", "allowed")
# The fields those decisions are taken from. Counters and spreads,
# reported as medians because each arm is one process.
MEASURES = (
    "leaves",
    "oncore_items",
    "per_item_ns",
    "cv2_wall",
    "cv2_oncore",
    "cv2_window",
    "window_ticks",
)


def median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def numeric_median(values):
    """Median over the readings that are readings.

    A dash is the harness saying it had nothing to report, which is not
    a zero and must not be averaged with one. The count of dashes is
    returned beside the median so a column built entirely out of them
    cannot read as a measurement.
    """
    nums = [float(v) for v in values if v not in ("-", "")]
    return median(nums), len(values) - len(nums)


def parse(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    pending = None
    cells = defaultdict(list)
    engage = defaultdict(list)
    failures = []
    skipped = []
    by_position = defaultdict(list)
    seen_in_trial = defaultdict(int)

    for line in lines:
        if line.startswith("SKIPPED "):
            skipped.append(line[len("SKIPPED "):])
            continue
        f = FAILED.match(line)
        if f:
            failures.append(line)
            pending = None
            continue
        m = ARM.match(line)
        if m:
            pending = (m.group(1), m.group(2), int(m.group(3)), int(m.group(4)))
            continue
        e = ENGAGE.match(line)
        if e and pending is not None:
            lever, arm, load, _ = pending
            fields = {}
            for token in e.group(1).split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    fields[key] = value
            engage[(lever, load, arm)].append(fields)
            continue
        r = ROW.match(line)
        if r and pending is not None:
            lever, arm, load, trial = pending
            if int(r.group(1)) != load:
                raise SystemExit(
                    f"MISMATCH: label load{load}, row load {r.group(1)}"
                )
            value = float(r.group(4))
            cells[(lever, load, arm)].append(value)
            key = (lever, load, trial)
            seen_in_trial[key] += 1
            by_position[(load, seen_in_trial[key])].append(value)
    return cells, engage, failures, skipped, by_position


def report_engagement(engage, levers, loads):
    print()
    print("ENGAGEMENT, read before the throughput table:")
    print("-" * 78)
    for lever in levers:
        for load in loads:
            arms = {
                arm: engage.get((lever, load, arm), []) for arm in ("off", "null", "on")
            }
            if not arms["off"] or not arms["on"]:
                print(f"{lever} load{load}: no engagement line on one arm")
                continue
            moved = []
            for field in DECISIONS:
                off_values = {f.get(field) for f in arms["off"]}
                on_values = {f.get(field) for f in arms["on"]}
                if off_values != on_values:
                    moved.append(f"{field} {sorted(off_values)} -> {sorted(on_values)}")
            print(f"{lever} load{load}:")
            if moved:
                for line in moved:
                    print(f"    moved   {line}")
            else:
                held = ", ".join(
                    f"{field}={sorted({f.get(field) for f in arms['off']})}"
                    for field in DECISIONS
                )
                print(f"    no decision moved between its own arms: {held}")
                print("    the throughput row below is not a reading of this mechanism")
            for field in MEASURES:
                off_med, off_blank = numeric_median([f.get(field, "-") for f in arms["off"]])
                on_med, on_blank = numeric_median([f.get(field, "-") for f in arms["on"]])
                if off_med is None and on_med is None:
                    print(f"    {field:<14} never reported on either arm")
                    continue
                off_text = "-" if off_med is None else f"{off_med:.1f}"
                on_text = "-" if on_med is None else f"{on_med:.1f}"
                note = ""
                if off_blank or on_blank:
                    note = f"  ({off_blank} and {on_blank} arms had no reading)"
                print(f"    {field:<14} off {off_text:>12}   on {on_text:>12}{note}")
            print()


def main(path):
    cells, engage, failures, skipped, by_position = parse(path)

    if not cells:
        print("NO ROWS PARSED")
        return 1
    total = sum(len(v) for v in cells.values())
    print(f"parsed {total} rows from {path}", flush=True)

    for line in skipped:
        print(f"not measured: {line}")
    for line in failures:
        print(line)
    if failures:
        print(
            f"{len(failures)} arm(s) exited non-zero. Their cells are short by that "
            "many rows, which a median hides."
        )

    levers = sorted({k[0] for k in cells})
    loads = sorted({k[1] for k in cells})

    report_engagement(engage, levers, loads)

    print()
    print(f"{'lever':<22} {'load':>5} {'on/off':>8} {'null/off':>9} {'verdict':<28}")
    print("-" * 78)
    for lever in levers:
        for load in loads:
            off = median(cells.get((lever, load, "off"), []))
            null = median(cells.get((lever, load, "null"), []))
            on = median(cells.get((lever, load, "on"), []))
            if not off or not null or not on:
                print(f"{lever:<22} {load:>5}   incomplete cell")
                continue
            r_on, r_null = on / off, null / off
            # The null's distance from 1.0 is the floor. A lever is only
            # read as moving anything when it is further from 1.0 than
            # the null is, and in the direction the criterion wants.
            floor = abs(r_null - 1.0)
            effect = r_on - 1.0
            if abs(effect) <= floor:
                verdict = "inside the null floor"
            elif load == 0:
                verdict = "REGRESSION" if effect < 0 else "faster (unloaded)"
            else:
                verdict = "SLOWER under load" if effect < 0 else "faster under load"
            print(f"{lever:<22} {load:>5} {r_on:>8.3f} {r_null:>9.3f} {verdict:<28}")

    # Position, read before the levers are believed. If throughput
    # drifts with position within a rotation then position is a variable
    # the arms carry, and a design that did not rotate them measured it.
    print()
    print("THROUGHPUT BY POSITION IN THE ROTATION, arm ignored:")
    print(f"{'load':>5} {'pos':>4} {'n':>4} {'median/s':>12} {'vs pos 1':>9}")
    print("-" * 40)
    drift = 0.0
    for load in loads:
        first = median(by_position.get((load, 1), []))
        for pos in sorted(p for (l, p) in by_position if l == load):
            v = by_position[(load, pos)]
            m = median(v)
            rel = (m / first) if first else float("nan")
            drift = max(drift, abs(rel - 1.0))
            print(f"{load:>5} {pos:>4} {len(v):>4} {m:>12.2f} {rel:>9.3f}")
    print()
    if drift > 0.05:
        print(f"Position moves throughput by up to {drift * 100:.0f} percent.")
        print("Read every verdict above against that. If the arms were not")
        print("rotated across trials, position and arm are one variable and")
        print("the table is measuring their sum.")
    else:
        print(f"Position moves throughput by at most {drift * 100:.1f} percent here,")
        print("so the rotation is not carrying the result.")

    print()
    print("on/off and null/off are medians of the same binary in one")
    print("rotation. A verdict of 'inside the null floor' means the")
    print("switch moved the number by no more than two identical arms")
    print("moved it, which is not a measurement of the switch.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
