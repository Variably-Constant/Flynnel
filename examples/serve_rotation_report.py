"""Serve-policy arms compared, from a rotation log.

Reads what pc2_serve_rotation.ps1 writes: an ARM line naming the arm,
its trial, its position in that trial's order and the foreign process
count, followed by the harness's own control, throughput and retained
rows.

Three things this reports that an eye over the log does not.

COLLAPSES BY TRIAL INDEX, not by arm. The arms run interleaved inside a
trial, so a collapse that lands on every arm at the same trial is
something that happened to the box. Reading collapses per arm without
that check attributed them to the serving policies once, and the arm
that serves nothing collapsed at the same index.

POSITION, because a window measured first in its trial is worth a few
per cent more than one measured last. The rotation exists to cancel
that, and this says whether it did: an arm that drew position one more
often than the others had an advantage the rotation was meant to
remove.

THE SPREAD BESIDE THE MEDIAN. A median difference smaller than the
within-arm range is not a difference, and every figure this campaign
reported and then withdrew was a small-n extreme read without one.
"""

import re
import sys
from collections import defaultdict

ARM = re.compile(
    r"^ARM (\S+) trial=(\d+) position=(\d+) foreign=(\d+)\s*$"
)
CONTROL = re.compile(r"^control (\d+) (\d+) (\d+) ([\d.]+)\s*$")
RETAINED = re.compile(r"^retained (\d+) (\d+) ([\d.]+)")


def note(text):
    print(text, file=sys.stderr, flush=True)


def median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def parse(paths):
    arms = defaultdict(lambda: {"ctl": [], "ret": [], "pos": [], "foreign": []})
    by_trial = defaultdict(dict)
    cur = None
    for n, path in enumerate(paths, 1):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.strip()
                m = ARM.match(line)
                if m:
                    cur = (m.group(1), int(m.group(2)))
                    arms[m.group(1)]["pos"].append(int(m.group(3)))
                    arms[m.group(1)]["foreign"].append(int(m.group(4)))
                    continue
                if cur is None:
                    continue
                arm, trial = cur
                m = CONTROL.match(line)
                if m:
                    value = int(m.group(3))
                    arms[arm]["ctl"].append(value)
                    by_trial[trial][arm] = value
                    continue
                m = RETAINED.match(line)
                if m:
                    arms[arm]["ret"].append(float(m.group(3)))
        note(f"parsed {n}/{len(paths)} {path}: {len(arms)} arms")
    return arms, by_trial


def main(paths):
    arms, by_trial = parse(paths)
    if not arms:
        print("NO ARM LINES PARSED. Without them no row can be attributed to")
        print("an arm, and a log from a run that died before its first arm")
        print("reads the same as one that never wrote them.")
        return 1

    print()
    print(f"{'arm':<20} {'n':>3} {'ctl med':>9} {'ctl min':>9} {'ctl max':>9} "
          f"{'range%':>7} {'ret med':>8} {'pos avg':>8}")
    print("-" * 80)
    for arm in sorted(arms):
        d = arms[arm]
        if not d["ctl"]:
            print(f"{arm:<20} {'no control rows':>20}")
            continue
        m = median(d["ctl"])
        rng = 100.0 * (max(d["ctl"]) - min(d["ctl"])) / m if m else 0.0
        ret = median(d["ret"])
        pos = sum(d["pos"]) / len(d["pos"]) if d["pos"] else 0.0
        print(f"{arm:<20} {len(d['ctl']):>3} {int(m):>9} {min(d['ctl']):>9} "
              f"{max(d['ctl']):>9} {rng:>7.1f} "
              f"{'-' if ret is None else round(ret, 4):>8} {pos:>8.2f}")

    # A collapse is judged against the arm's own median, so an arm that
    # is slower throughout is not counted as collapsing.
    meds = {a: median(d["ctl"]) for a, d in arms.items() if d["ctl"]}
    shared, lone = [], []
    for trial, per_arm in sorted(by_trial.items()):
        low = [a for a, v in per_arm.items() if meds.get(a) and v < meds[a] * 0.5]
        if not low:
            continue
        if len(low) == len(per_arm) and len(per_arm) > 1:
            shared.append(trial)
        else:
            lone.append((trial, sorted(low)))

    print()
    if shared:
        print(f"COLLAPSED ON EVERY ARM at trials {shared}. The arms run")
        print("interleaved inside a trial, so this is the box rather than any")
        print("policy, and it must not be counted against the arms.")
    if lone:
        print("COLLAPSED ON SOME ARMS ONLY:")
        for trial, who in lone:
            print(f"  trial {trial}: {', '.join(who)}")
        print("  One of these is one observation. Two arms differ only when")
        print("  the rate differs over enough trials to be a rate.")
    if not shared and not lone:
        print("No arm fell below half its own median in any trial.")

    foreign = [f for d in arms.values() for f in d["foreign"]]
    if foreign:
        busy = sum(1 for f in foreign if f > 0)
        print()
        print(f"BACKGROUND: {busy} of {len(foreign)} arms started with other")
        print(f"build processes running, up to {max(foreign)} at once.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <serve_rotation.log> [more.log ...]", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
