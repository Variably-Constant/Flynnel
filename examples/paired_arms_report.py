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
# Busy cores on the way out of an arm. Kept beside the entry figure
# rather than replacing it: a window is judged on the worse of its two
# ends, since a trial that began clear and finished in a storm was
# measured through the storm.
ARM_END = re.compile(r"^ARM_END (\S+) trial=(\d+) busy_cores=([\d.-]+)")
ROWS = {
    "ctl": re.compile(r"^control (\d+) (\d+) (\d+) ([\d.]+)\s*$"),
    "thr": re.compile(r"^throughput (\d+) (\d+) (\d+) ([\d.]+)\s*$"),
    "ret": re.compile(r"^retained (\d+) (\d+) ([\d.]+)"),
}
ENGAGE = re.compile(r"^engagement\s+(.*)$")

# Fields that are decisions rather than measurements. A switch that
# moved none of these across its own arms moved no decision, and the
# ratio below is then a comparison of one arm with itself.
DECISIONS = {
    "class",
    "smt",
    "workers",
    "allowed",
    "spin_adaptive",
    "spin_window",
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
            m = ARM_END.match(line)
            if m:
                slot = per[int(m.group(2))].setdefault(m.group(1), {})
                slot["busy"] = max(slot.get("busy", 0.0), float(m.group(3)))
                continue
            if cur is None:
                continue
            arm, trial = cur
            e = ENGAGE.match(line)
            if e:
                fields = {}
                for token in e.group(1).split():
                    if "=" in token:
                        key, value = token.split("=", 1)
                        fields[key] = value
                per[trial].setdefault(arm, {})["engage"] = fields
                continue
            for key, pattern in ROWS.items():
                m = pattern.match(line)
                if not m:
                    continue
                value = float(m.group(3))
                per[trial].setdefault(arm, {})[key] = value
                break
    return per


def report_engagement(per, under_test, reference):
    """Which decisions moved between the two arms, before any ratio.

    A mechanism that never ran produces the same clean rows as one that
    ran and did not help, so a ratio is only a reading of a switch once
    something the switch controls is seen to differ.
    """
    seen = defaultdict(lambda: defaultdict(set))
    for arms in per.values():
        for name in (under_test, reference):
            for key, value in arms.get(name, {}).get("engage", {}).items():
                seen[name][key].add(value)
    if not seen:
        print("no engagement line in this log; what the switch had to work "
              "with is unrecorded and the ratio below is unattributable")
        return
    moved = [k for k in sorted(DECISIONS)
             if seen[under_test].get(k) != seen[reference].get(k)
             and (seen[under_test].get(k) or seen[reference].get(k))]
    print()
    if moved:
        print("decisions that differ between the arms:")
        for k in moved:
            print(f"  {k:<14} {under_test}={sorted(seen[under_test].get(k, []))} "
                  f"{reference}={sorted(seen[reference].get(k, []))}")
    else:
        print("NO DECISION MOVED between these arms. Every field the switch "
              "controls reads alike on both, so the table below compares "
              "something with itself.")


def report(per, under_test, reference, max_busy=None):
    report_engagement(per, under_test, reference)
    print()
    print(f"{under_test} against {reference}, per trial. Above 1.000 means "
          f"{under_test} was faster.")
    if max_busy is not None:
        kept = {t: a for t, a in per.items()
                if all(a.get(n, {}).get("busy", 1e9) <= max_busy
                       for n in (under_test, reference))}
        print(f"Trials where either arm reached {max_busy} busy cores at either "
              f"end are dropped: {len(per) - len(kept)} of {len(per)} gone.")
        print("A window is judged on the worse of its two ends. A quiet gate is")
        print("checked once at entry and says nothing about the rest of a run, so")
        print("a trial can begin clear and be measured through a storm; a log")
        print("without ARM_END rows is judged on its entry figure alone and")
        print("cannot see that case.")
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
