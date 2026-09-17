"""The allowed-width lever, read either side of the narrowing.

Each window carries the width it was allowed, and the split is made on
that rather than on the window index: the driver narrows on a schedule
and a process that starts slowly puts the narrowing inside a window
rather than between two. NARROW_AFTER_WINDOW is kept as a cross-check
and a disagreement is reported. The criterion splits the same way:
before the narrowing the switch must not be slower, after it the switch
must be faster.

The width and worker columns are the engagement evidence, and they come
from the same rows. A run whose allowed width never moved narrowed
nothing, and its windows look exactly like a lever that did not help.
"""

import re
import sys
from collections import defaultdict

ARM = re.compile(r"^ARM\s+ALLOWED_WIDTH-(off|null|on)\s+t(\d+)")
WINDOW = re.compile(
    r"^window\s+(\d+)\s+(\d+)\s+([\d.]+)\s+allowed=(\d+)\s+workers=(\d+)\s*$"
)
SPLIT = re.compile(r"^NARROW_AFTER_WINDOW\s+(\d+)\s*$")
SEEN = re.compile(r"^widths_seen\s+(\d+)\s*$")
FAILED = re.compile(r"^ARM_FAILED\s+(\S+)\s+t(\d+)\s+exit=(\d+)")
MISMATCH = re.compile(r"^PID_MISMATCH\s")
NARROWED = re.compile(r"^NARROWED\s+threads=(\d+)\s+exit=(\d+)\s*$")


def median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def parse(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    split = None
    arm = None
    rows = defaultdict(list)
    seen = defaultdict(list)
    problems = []
    for line in lines:
        s = SPLIT.match(line)
        if s:
            split = int(s.group(1))
            continue
        if FAILED.match(line) or MISMATCH.match(line) or line.startswith("TASKSET_FAILED"):
            problems.append(line)
            continue
        # A narrowing that reached one thread is not a narrowing: the
        # pool's workers each carry their own mask, and the main thread
        # is not one of them.
        n = NARROWED.match(line)
        if n and (int(n.group(1)) < 2 or n.group(2) != "0"):
            problems.append(f"{line}  - the narrowing did not reach the pool")
            continue
        a = ARM.match(line)
        if a:
            arm = a.group(1)
            continue
        k = SEEN.match(line)
        if k and arm is not None:
            seen[arm].append(int(k.group(1)))
            continue
        w = WINDOW.match(line)
        if w and arm is not None:
            rows[arm].append(
                {
                    "i": int(w.group(1)),
                    "per_s": float(w.group(3)),
                    "allowed": int(w.group(4)),
                    "workers": int(w.group(5)),
                }
            )
    return split, rows, seen, problems


def half(rows, full_width, after):
    """Split on the width each window actually saw, not on its index.

    The driver narrows on a schedule, and the schedule and the windows
    drift against each other: a process that starts slowly puts the
    narrowing inside a window rather than between two. Each row carries
    the width it was allowed when it ended, so the rows say where the
    boundary fell and the schedule is only a cross-check.
    """
    return [r for r in rows if (r["allowed"] < full_width) == after]


def describe(rows, field):
    return sorted({r[field] for r in rows})


def main(path):
    split, rows, seen, problems = parse(path)
    if not rows:
        print("NO WINDOW ROWS PARSED")
        return 1

    # The widest reading anywhere is the machine before anything was
    # taken away. Rows at that width are the before half.
    full_width = max(r["allowed"] for arm_rows in rows.values() for r in arm_rows)
    if split is None:
        print("no NARROW_AFTER_WINDOW marker; splitting on the width alone")
    else:
        print(f"narrowing scheduled after window {split}")
    print(f"full width {full_width}, rows below it are the narrowed half")
    for line in problems:
        print(line)

    print()
    print("engagement, from the same rows:")
    for arm in ("off", "null", "on"):
        if arm not in rows:
            print(f"  {arm:<5} no rows")
            continue
        widths = seen.get(arm, [])
        if widths and max(widths) < 2:
            print(
                f"  {arm:<5} every run saw one allowed width, so nothing was narrowed "
                "and these rows measure nothing"
            )
        before = half(rows[arm], full_width, False)
        after = half(rows[arm], full_width, True)
        # Where the boundary actually fell, against where it was
        # scheduled. A process that started slowly puts the narrowing
        # inside a window, and then the schedule names the wrong split.
        landed = min((r["i"] for r in after), default=None)
        if split is not None and landed is not None and landed != split + 1:
            print(
                f"  {arm:<5} the narrowing was scheduled after window {split} and first "
                f"shows in window {landed}"
            )
        print(
            f"  {arm:<5} allowed before {describe(before, 'allowed')} after "
            f"{describe(after, 'allowed')}   workers before "
            f"{describe(before, 'workers')} after {describe(after, 'workers')}"
        )

    print()
    print(f"{'phase':<22} {'on/off':>8} {'null/off':>9} {'verdict':<28}")
    print("-" * 72)
    for label, after in (("before the narrowing", False), ("after the narrowing", True)):
        med = {}
        for arm in ("off", "null", "on"):
            med[arm] = median(
                [r["per_s"] for r in half(rows.get(arm, []), full_width, after)]
            )
        if not all(med.values()):
            print(f"{label:<22}   incomplete")
            continue
        r_on = med["on"] / med["off"]
        r_null = med["null"] / med["off"]
        floor = abs(r_null - 1.0)
        effect = r_on - 1.0
        if abs(effect) <= floor:
            verdict = "inside the null floor"
        elif after:
            verdict = "SLOWER narrowed" if effect < 0 else "faster narrowed"
        else:
            verdict = "REGRESSION" if effect < 0 else "faster (full width)"
        print(f"{label:<22} {r_on:>8.3f} {r_null:>9.3f} {verdict:<28}")

    print()
    print("Before the narrowing the switch must not be slower; after it")
    print("the switch must be faster. A verdict of 'inside the null floor'")
    print("means the switch moved the number by no more than two identical")
    print("arms moved it.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
