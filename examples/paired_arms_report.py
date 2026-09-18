"""Two arms compared trial by trial, where both ran inside one trial.

Comparing the arms' medians throws the pairing away. Both arms run
within seconds of each other inside a trial, so whatever the box was
doing is close to common between them, and the difference taken inside
a trial removes it. Across trials that difference is what varies.

Reports the per-trial ratio of the arm under test to the reference, on
the loaded window and on the retained ratio. The sign convention: above
one means the arm under test was FASTER.

Prints how large an effect this many trials could have resolved, so a
null reads as a bound rather than as an absence. A run that cannot see
a ten per cent regression must not be quoted as evidence there is none.
"""

import re
import sys
from collections import defaultdict

# Trailing fields are tolerated rather than anchored out. A harness that
# starts recording one more thing about its conditions must not turn
# every row into an unparsed line, which reads the same as a run that
# produced none.
ARM = re.compile(
    r"^ARM (\S+) trial=(\d+) position=(\d+) foreign=(\d+)"
    r"(?:\s+busy_cores=([\d.-]+))?"
)
ROWS = {
    "ctl": re.compile(r"^control (\d+) (\d+) (\d+) ([\d.]+)\s*$"),
    "thr": re.compile(r"^throughput (\d+) (\d+) (\d+) ([\d.]+)\s*$"),
    "ret": re.compile(r"^retained (\d+) (\d+) ([\d.]+)"),
}


def median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def parse(path):
    per = defaultdict(dict)
    cur = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            m = ARM.match(line)
            if m:
                cur = (m.group(1), int(m.group(2)))
                if m.group(5) is not None:
                    per[int(m.group(2))].setdefault(m.group(1), {})["busy"] = float(
                        m.group(5)
                    )
                continue
            if cur is None:
                continue
            arm, trial = cur
            for key, pattern in ROWS.items():
                m = pattern.match(line)
                if not m:
                    continue
                value = float(m.group(3))
                per[trial].setdefault(arm, {})[key] = value
                break
    return per


def report(per, under_test, reference, max_busy=None):
    print()
    print(f"{under_test} against {reference}, per trial. Above 1.000 means "
          f"{under_test} was faster.")
    if max_busy is not None:
        kept = {t: a for t, a in per.items()
                if all(a.get(n, {}).get("busy", 1e9) <= max_busy
                       for n in (under_test, reference))}
        print(f"Trials where either arm started above {max_busy} busy cores are "
              f"dropped: {len(per) - len(kept)} of {len(per)} gone.")
        print("A quiet gate is checked at entry and says nothing about the rest")
        print("of the run, so a trial can begin clear and be measured through a")
        print("storm. This is the same voiding a neighbour's bench does on its")
        print("own regions, applied here after the fact because the harness")
        print("records the figure per arm but does not yet act on it.")
        per = kept
    print()
    print(f"{'metric':<10} {'pairs':>5} {'median':>8} {'min':>8} {'max':>8} "
          f"{'wins':>6} {'resolvable':>11}")
    print("-" * 62)

    for key, label in (("thr", "loaded"), ("ret", "retained"), ("ctl", "control")):
        ratios = []
        for _, arms in sorted(per.items()):
            a, b = arms.get(under_test), arms.get(reference)
            if not a or not b or key not in a or key not in b or not b[key]:
                continue
            ratios.append(a[key] / b[key])
        if not ratios:
            print(f"{label:<10} {'no pairs':>5}")
            continue
        med = median(ratios)
        wins = sum(1 for r in ratios if r > 1.0)
        # Spread of the per-trial ratios says what this many pairs could
        # have seen. A median difference smaller than this is a number,
        # not a finding.
        spread = sorted(ratios)
        lo, hi = spread[len(spread) // 10], spread[-1 - len(spread) // 10]
        resolvable = (hi - lo) / (len(ratios) ** 0.5)
        print(f"{label:<10} {len(ratios):>5} {med:>8.4f} {min(ratios):>8.4f} "
              f"{max(ratios):>8.4f} {wins:>3}/{len(ratios):<2} {resolvable:>10.2%}")

    print()
    print("A median inside the resolvable column is not a difference. The")
    print("column is the middle-eight-tenths spread of the per-trial ratios")
    print("over the root of the pair count, so it falls as trials are added")
    print("and says what the run could have caught rather than what it saw.")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"usage: {sys.argv[0]} <log> <arm_under_test> <reference_arm> "
              f"[max_busy_cores]", file=sys.stderr)
        sys.exit(2)
    data = parse(sys.argv[1])
    if not data:
        print("NO ARM LINES PARSED. A log from a run that died before its")
        print("first arm reads the same as one that never wrote them.")
        sys.exit(1)
    bound = float(sys.argv[4]) if len(sys.argv) > 4 else None
    report(data, sys.argv[2], sys.argv[3], bound)
