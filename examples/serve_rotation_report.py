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

A collapse counts when EITHER window falls, the control or the loaded
one. They fail independently: a build storm arriving inside the loaded
window leaves the control untouched, so judging on the control alone
scores that arm clean and hands the trial to whichever arms happened to
be running when the storm also caught their control. That reads as a
policy difference and is a clock.

POSITION, because a window measured first in its trial is worth a few
percent more than one measured last. The rotation exists to cancel
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

# Trailing fields are tolerated rather than anchored out. A harness that
# starts recording one more thing about its conditions must not turn
# every row into an unparsed line, which reads the same as a run that
# produced none.
ARM = re.compile(
    r"^ARM (\S+) trial=(\d+) position=(\d+) foreign=(\d+)"
    r"(?:\s+busy_cores=([\d.-]+))?"
)
# Busy cores on the way out of an arm, so a window is judged on the
# worse of its two ends rather than on how the box looked when it
# opened.
ARM_END = re.compile(r"^ARM_END (\S+) trial=(\d+) busy_cores=([\d.-]+)")
CONTROL = re.compile(r"^control (\d+) (\d+) (\d+) ([\d.]+)\s*$")
THROUGHPUT = re.compile(r"^throughput (\d+) (\d+) (\d+) ([\d.]+)\s*$")
RETAINED = re.compile(r"^retained (\d+) (\d+) ([\d.]+)")

# Which series a collapse is judged against, and the fraction of an
# arm's own median a window has to fall below to count as one. Judged
# per series against that series' median so an arm whose loaded window
# is legitimately half its control is not scored as collapsing.
SERIES = ("ctl", "thr")
COLLAPSE_FRACTION = 0.5


def note(text):
    print(text, file=sys.stderr, flush=True)


def median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def parse(paths):
    arms = defaultdict(
        lambda: {
            "ctl": [], "thr": [], "ret": [], "pos": [],
            "foreign": [], "busy": [], "busy_end": [],
        }
    )
    by_trial = defaultdict(lambda: defaultdict(dict))
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
                    if m.group(5) is not None:
                        arms[m.group(1)]["busy"].append(float(m.group(5)))
                    continue
                m = ARM_END.match(line)
                if m:
                    arms[m.group(1)]["busy_end"].append(float(m.group(3)))
                    continue
                if cur is None:
                    continue
                arm, trial = cur
                for key, pattern in (("ctl", CONTROL), ("thr", THROUGHPUT)):
                    m = pattern.match(line)
                    if m:
                        value = int(m.group(3))
                        arms[arm][key].append(value)
                        by_trial[trial][arm][key] = value
                        break
                else:
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
    # The loaded window sits beside the control because the criterion is
    # about throughput under load; the control is what says whether a
    # difference in it belongs to the arm or to the box.
    print(f"{'arm':<20} {'n':>3} {'ctl med':>9} {'thr med':>9} {'ctl min':>9} "
          f"{'thr min':>9} {'range%':>7} {'ret med':>8} {'pos avg':>8}")
    print("-" * 90)
    for arm in sorted(arms):
        d = arms[arm]
        if not d["ctl"]:
            print(f"{arm:<20} {'no control rows':>20}")
            continue
        m = median(d["ctl"])
        tm = median(d["thr"])
        rng = 100.0 * (max(d["ctl"]) - min(d["ctl"])) / m if m else 0.0
        ret = median(d["ret"])
        pos = sum(d["pos"]) / len(d["pos"]) if d["pos"] else 0.0
        print(f"{arm:<20} {len(d['ctl']):>3} {int(m):>9} "
              f"{'-' if tm is None else int(tm):>9} {min(d['ctl']):>9} "
              f"{'-' if not d['thr'] else min(d['thr']):>9} {rng:>7.1f} "
              f"{'-' if ret is None else round(ret, 4):>8} {pos:>8.2f}")

    # A collapse is judged against the arm's own median for the same
    # series, so an arm that is slower throughout is not counted as
    # collapsing, and a loaded window that always sits near half its
    # control is not either.
    meds = {
        key: {a: median(d[key]) for a, d in arms.items() if d[key]}
        for key in SERIES
    }
    shared, lone = [], []
    for trial, per_arm in sorted(by_trial.items()):
        low = []
        for arm, windows in per_arm.items():
            fell = [
                key
                for key, value in windows.items()
                if meds[key].get(arm)
                and value < meds[key][arm] * COLLAPSE_FRACTION
            ]
            if fell:
                low.append((arm, sorted(fell)))
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
            named = ", ".join(f"{arm} ({'+'.join(fell)})" for arm, fell in who)
            print(f"  trial {trial}: {named}")
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

    # The figure that says whether a run got the box it claims. The count
    # above names four processes and reads near zero on a host saturated
    # by anything else; this one is derived from idle time and counts
    # everything.
    cores = [c for d in arms.values() for c in d["busy"]]
    if cores:
        cores.sort()
        print()
        print(f"BUSY CORES at arm start: median {cores[len(cores) // 2]:.2f}, "
              f"min {cores[0]:.2f}, max {cores[-1]:.2f} over {len(cores)} arms.")
        ends = sorted(c for d in arms.values() for c in d["busy_end"])
        if ends:
            print(f"BUSY CORES at arm end:   median {ends[len(ends) // 2]:.2f}, "
                  f"min {ends[0]:.2f}, max {ends[-1]:.2f} over {len(ends)} arms.")
        else:
            print("Arm ends were not sampled, so a window that began clear and")
            print("finished in a storm is not distinguishable here from one that")
            print("stayed clear throughout.")
    else:
        print()
        print("BUSY CORES not recorded. The process count above cannot say")
        print("whether the box was quiet, so a quiet-host claim from this log")
        print("rests on nothing.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <serve_rotation.log> [more.log ...]", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
