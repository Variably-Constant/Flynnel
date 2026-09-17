"""The allowed-width lever, read either side of the narrowing.

Windows up to and including NARROW_AFTER_WINDOW ran on the full
machine; the ones after ran on the CPUs the process was left with. The
criterion splits the same way: before the narrowing the switch must not
be slower, after it the switch must be faster.

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


def half(rows, split, after):
    return [r for r in rows if (r["i"] > split) == after]


def describe(rows, field):
    return sorted({r[field] for r in rows})


def main(path):
    split, rows, seen, problems = parse(path)
    if split is None:
        print("NO NARROW_AFTER_WINDOW MARKER - the log cannot be split")
        return 1
    if not rows:
        print("NO WINDOW ROWS PARSED")
        return 1

    print(f"narrowing applied after window {split}")
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
        before = half(rows[arm], split, False)
        after = half(rows[arm], split, True)
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
            med[arm] = median([r["per_s"] for r in half(rows.get(arm, []), split, after)])
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
