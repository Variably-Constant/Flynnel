"""Per-leaf dispatch overhead from a trace dump, as the minimum gap.

Reads the CSV `flynnel::sched::trace::dump_to_stderr` writes:

    TRACE,<thread>,<event_kind_num>,<payload>,<tsc>

LeafStart is 3 and LeafEnd is 4. Within one thread the gap from a
LeafEnd to the next LeafStart is what the dispatch machinery spent
between two pieces of work, which is where per-leaf overhead lives.

THE MINIMUM IS THE FIGURE. Background load, preemption and cache misses
can only make a gap longer; nothing makes it shorter than the work
actually done. So over hundreds of leaves the smallest gap is the
overhead with the interference removed, and it is stable on a busy box
in a way a throughput window is not. The median and the maximum are
printed beside it to show how much interference there was, not as
competing estimates.

Gaps are counted per thread. A gap taken across two threads is not a
gap, it is the offset between two TSCs that were never synchronised.
"""

import re
import sys
from collections import defaultdict

TRACE = re.compile(r"^TRACE,([^,]+),(\d+),(\d+),(\d+)\s*$")
LEAF_START = 3
LEAF_END = 4


def note(text):
    print(text, file=sys.stderr, flush=True)


def median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def parse(paths):
    """Per-thread event sequences, in the order the thread recorded them."""
    threads = defaultdict(list)
    rows = 0
    for n, path in enumerate(paths, 1):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                m = TRACE.match(raw.strip())
                if m:
                    rows += 1
                    threads[m.group(1)].append((int(m.group(2)), int(m.group(4))))
        note(f"parsed {n}/{len(paths)} {path}: {rows} trace rows, {len(threads)} threads")
    return threads


def gaps_and_bodies(threads):
    """Inter-leaf gaps and leaf body durations, in TSC ticks.

    A gap needs a LeafEnd followed by a LeafStart on the SAME thread with
    nothing between them. Any other pairing is a different interval and
    is not counted, so a thread whose events interleave with something
    else contributes fewer gaps rather than wrong ones.
    """
    gaps = []
    bodies = []
    for name, events in threads.items():
        for i in range(len(events) - 1):
            (kind, tsc), (next_kind, next_tsc) = events[i], events[i + 1]
            if kind == LEAF_END and next_kind == LEAF_START:
                gaps.append(next_tsc - tsc)
            elif kind == LEAF_START and next_kind == LEAF_END:
                bodies.append(next_tsc - tsc)
        note(f"  thread {name}: {len(events)} events")
    return gaps, bodies


def report(label, values):
    if not values:
        print(f"{label:<22} {'no samples':>12}")
        return
    print(
        f"{label:<22} {len(values):>7} {min(values):>10} "
        f"{median(values):>12.0f} {max(values):>12}"
    )


def rank(values, k):
    """The k-th smallest, or None where there are fewer than k."""
    if len(values) < k:
        return None
    return sorted(values)[k - 1]


def compare(base_log, tip_log):
    """Two logs against each other at matched ranks.

    The minimum of a sample falls as the sample grows, so comparing one
    side's min with the other's is a comparison of sample sizes wherever
    they differ. The k-th smallest at a k both sides reach is comparable;
    the plain minimum is not.

    The leaf body is the control. This change does not touch the body, so
    whatever difference appears there is artifact, and a difference in
    the gap only means something if it exceeds it.
    """
    base_gaps, base_bodies = gaps_and_bodies(parse([base_log]))
    tip_gaps, tip_bodies = gaps_and_bodies(parse([tip_log]))

    print()
    print(f"base gaps n={len(base_gaps)} bodies n={len(base_bodies)}")
    print(f"tip  gaps n={len(tip_gaps)} bodies n={len(tip_bodies)}")
    print()
    print("RANK-MATCHED, tip over base. The body is unchanged code, so its")
    print("column is the artifact floor and the gap column only says")
    print("something where it exceeds it.")
    print(f"{'rank':>6} {'base gap':>10} {'tip gap':>10} {'gap x':>8} "
          f"{'base body':>10} {'tip body':>10} {'body x':>8}")
    print("-" * 68)

    verdict_rows = []
    for k in (1, 5, 10, 25, 50, 100):
        bg, tg = rank(base_gaps, k), rank(tip_gaps, k)
        bb, tb = rank(base_bodies, k), rank(tip_bodies, k)
        if bg is None or tg is None:
            continue
        gx = tg / bg if bg else None
        bx = (tb / bb) if (bb and tb) else None

        def fmt(x, w=8):
            return f"{'-':>{w}}" if x is None else f"{x:>{w}.3f}"

        print(f"{k:>6} {bg:>10} {tg:>10} {fmt(gx)} "
              f"{bb if bb else '-':>10} {tb if tb else '-':>10} {fmt(bx)}")
        if gx is not None and bx is not None:
            verdict_rows.append((k, gx, bx))

    print()
    if not verdict_rows:
        print("Not enough matched ranks to compare.")
        return 3

    exceeds = [(k, gx, bx) for (k, gx, bx) in verdict_rows if gx > bx]
    if not exceeds:
        print("At every matched rank the gap ratio is within the body ratio, so")
        print("nothing here separates the two schedulers from the artifact of")
        print("measuring them. This is not a clean bill: it is the statement")
        print("that a difference, if any, is smaller than this probe resolves.")
        return 0

    print("GAP RATIO EXCEEDS THE BODY RATIO at these ranks:")
    for (k, gx, bx) in exceeds:
        print(f"  rank {k}: gap {gx:.3f}x against body {bx:.3f}x")
    print()
    print("That is the residue the control does not explain. It is evidence")
    print("of added per-leaf cost, and its size is the gap ratio divided by")
    print("the body ratio rather than the gap ratio itself.")
    return 1


def main(paths):
    if len(paths) == 2:
        return compare(paths[0], paths[1])
    threads = parse(paths)
    if not threads:
        print("NO TRACE ROWS PARSED. dump_to_stderr writes nothing when")
        print("FLYNNEL_TRACE is unset, and an empty trace reads exactly like")
        print("a dispatch that cost nothing.")
        return 1

    gaps, bodies = gaps_and_bodies(threads)

    print()
    print("TSC TICKS, per thread, same-thread pairs only:")
    print(f"{'interval':<22} {'n':>7} {'min':>10} {'median':>12} {'max':>12}")
    print("-" * 66)
    report("leaf body", bodies)
    report("gap between leaves", gaps)

    print()
    if not gaps:
        print("No LeafEnd followed by a LeafStart on one thread, so there is no")
        print("inter-leaf interval here to measure. A dispatch with one leaf per")
        print("thread produces this, and so does a trace where the workers never")
        print("flushed.")
        return 3

    print(f"THE FIGURE IS min={min(gaps)} ticks over {len(gaps)} gaps.")
    print("Load can only lengthen a gap, so the smallest is the overhead with")
    print("the interference removed. The median and max say how much")
    print("interference there was; they are not competing estimates.")
    if median(gaps) and min(gaps) > 0:
        print(f"Interference factor at the median: {median(gaps) / min(gaps):.1f}x")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <trace.log> [more.log ...]", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
