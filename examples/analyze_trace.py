"""Parse trace_heavy_dispatch.exe stderr output, isolate the measured
dispatch window via TSC, compute per-worker timeline statistics.

Usage:
    python examples/analyze_trace.py /tmp/flynnel_trace.csv

Output: per-worker busy-time / idle-time breakdown + concurrency
histogram showing how many workers were actively executing a leaf
at each microsecond of the measured dispatch.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

# Event kind ids (mirror src/sched/trace.rs::TraceEvent)
DISPATCH_ENTER = 1
DISPATCH_EXIT = 2
LEAF_START = 3
LEAF_END = 4
JOIN_PUSH = 5
JOIN_WAIT_BEGIN = 6
JOIN_WAIT_END = 7
WORKER_WAKE = 8
STEAL_HIT = 9

EVENT_NAMES = {
    DISPATCH_ENTER: 'DispatchEnter',
    DISPATCH_EXIT: 'DispatchExit',
    LEAF_START: 'LeafStart',
    LEAF_END: 'LeafEnd',
    JOIN_PUSH: 'JoinPush',
    JOIN_WAIT_BEGIN: 'JoinWaitBegin',
    JOIN_WAIT_END: 'JoinWaitEnd',
    WORKER_WAKE: 'WorkerWake',
    STEAL_HIT: 'StealHit',
}

# Assume 3.0 GHz TSC -> 1 ns = 3 cycles. Zen+ R7 2700 = 3.2 GHz base.
TSC_PER_NS = 3.0


def parse_trace(path, prefix='TRACE,'):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.startswith(prefix):
                continue
            parts = line.strip().split(',')
            if len(parts) != 5:
                continue
            _, thread, ev, payload, tsc = parts
            try:
                rows.append((thread, int(ev), int(payload), int(tsc)))
            except ValueError:
                continue
    return rows


def parse_rayon_dispatch_window(path):
    """Find the measured-rayon-dispatch TSC window. Approach: the
    comment line `# rayon dispatch elapsed: <us> us` marks the END
    of the measured rayon dispatch. We don't have an exact start
    TSC for it; estimate by finding the LAST cluster of rayon push
    events (event 5) that precedes the elapsed comment, and use
    its earliest event as start and latest as end."""
    tsc_start = 0
    tsc_end = 0
    # Find the elapsed line - it's printed AFTER the measured dispatch
    # by the example binary. We use it as a sanity check rather than
    # for the window itself.
    return tsc_start, tsc_end


def rayon_summary(path):
    """Summarize rayon trace events: wake counts, sleep counts,
    worker engagement.
    """
    rows = parse_trace(path, prefix='RAYON_TRACE,')
    if not rows:
        return
    print(f'\n# RAYON trace ({len(rows)} events):')
    by_event = defaultdict(int)
    by_worker_pushes = defaultdict(int)
    by_worker_wakes = defaultdict(int)
    by_worker_sleeps = defaultdict(int)
    for thread, ev, payload, tsc in rows:
        by_event[ev] += 1
        if ev == JOIN_PUSH:
            by_worker_pushes[thread] += 1
        elif ev == 102:  # RayonWorkerWake
            by_worker_wakes[thread] += 1
        elif ev == 101:  # RayonWorkerSleep
            by_worker_sleeps[thread] += 1
    print(f'  event counts: {dict(by_event)}')
    print(f'  worker_wake events: {sum(by_worker_wakes.values())} across '
          f'{len(by_worker_wakes)} workers')
    print(f'  worker_sleep events: {sum(by_worker_sleeps.values())} across '
          f'{len(by_worker_sleeps)} workers')
    print(f'  push events: {sum(by_worker_pushes.values())} across '
          f'{len(by_worker_pushes)} workers')
    # Producer-side: count wake_producer (event 100) requests
    wake_prod = [r for r in rows if r[1] == 100]
    if wake_prod:
        wake_amounts = [r[2] for r in wake_prod]
        print(f'  producer-side wake requests: {len(wake_prod)}, '
              f'total threads requested: {sum(wake_amounts)}')


def find_measured_window(rows):
    """Identify the measured dispatch window. Main thread's buffer
    contains MEASURED (index 0) then WAKE-UP (index 1) pairs. The
    warm-up was cleared via reset_current_thread() so it's not in
    the buffer. Returns (tsc_start, tsc_end) of the MEASURED pair.
    """
    main_events = [r for r in rows if r[0] == 'main']
    enters = [r[3] for r in main_events if r[1] == DISPATCH_ENTER]
    exits = [r[3] for r in main_events if r[1] == DISPATCH_EXIT]
    if not enters or not exits:
        return 0, 0
    # MEASURED is the first pair; WAKE-UP is second (if present).
    return enters[0], exits[0]


def per_worker_stats(rows, t_start, t_end):
    """Compute, for each worker, time spent in leaves vs idle inside
    the measured dispatch window."""
    by_worker = defaultdict(list)
    for thread, ev, payload, tsc in rows:
        if t_start <= tsc <= t_end and thread != 'main':
            by_worker[thread].append((ev, tsc))

    worker_stats = {}
    for worker, events in by_worker.items():
        events.sort(key=lambda e: e[1])
        leaf_busy_cycles = 0
        leaf_count = 0
        in_leaf_start = None
        first_event_tsc = None
        last_event_tsc = None
        for ev, tsc in events:
            if first_event_tsc is None:
                first_event_tsc = tsc
            last_event_tsc = tsc
            if ev == LEAF_START:
                in_leaf_start = tsc
            elif ev == LEAF_END and in_leaf_start is not None:
                leaf_busy_cycles += tsc - in_leaf_start
                leaf_count += 1
                in_leaf_start = None
        worker_stats[worker] = {
            'leaf_count': leaf_count,
            'leaf_busy_cycles': leaf_busy_cycles,
            'first_event_tsc': first_event_tsc,
            'last_event_tsc': last_event_tsc,
            'total_events': len(events),
        }
    return worker_stats


def print_report(rows, t_start, t_end, worker_stats):
    total_cycles = t_end - t_start
    total_ns = total_cycles / TSC_PER_NS
    print(f'\n# Measured dispatch window:')
    print(f'  duration: {total_cycles} cycles = {total_ns/1000:.1f} us '
          f'({total_ns/1000000:.3f} ms)')
    print(f'  workers active: {len(worker_stats)}')

    print(f'\n# Per-worker breakdown (within the measured window):')
    print(f'  {"worker":<24} {"leaves":>6} {"busy_us":>9} {"first_us":>9} '
          f'{"last_us":>9} {"span_us":>9} {"idle_us":>9} {"events":>7}')
    workers_sorted = sorted(worker_stats.items(),
                             key=lambda kv: -kv[1]['leaf_count'])
    total_busy_us = 0.0
    for worker, s in workers_sorted:
        busy_us = s['leaf_busy_cycles'] / TSC_PER_NS / 1000
        total_busy_us += busy_us
        first_us = (s['first_event_tsc'] - t_start) / TSC_PER_NS / 1000 if s['first_event_tsc'] else 0
        last_us = (s['last_event_tsc'] - t_start) / TSC_PER_NS / 1000 if s['last_event_tsc'] else 0
        span_us = last_us - first_us if s['first_event_tsc'] else 0
        idle_us = span_us - busy_us
        print(f'  {worker:<24} {s["leaf_count"]:>6} {busy_us:>9.1f} '
              f'{first_us:>9.1f} {last_us:>9.1f} {span_us:>9.1f} '
              f'{idle_us:>9.1f} {s["total_events"]:>7}')

    n_workers = len(worker_stats)
    ideal_per_worker_us = total_busy_us / n_workers if n_workers else 0
    print(f'\n# Aggregate:')
    print(f'  total worker busy time: {total_busy_us:.1f} us')
    print(f'  ideal per-worker (busy/N): {ideal_per_worker_us:.1f} us')
    print(f'  wall-clock dispatch: {total_ns/1000:.1f} us')
    print(f'  parallel efficiency: {ideal_per_worker_us/total_ns*1000:.1%}')

    # Cold-start gap: time from dispatch start (t_start) to FIRST worker
    # event. This is the cascade wake-up latency.
    earliest_worker_event = min((s['first_event_tsc'] for s in worker_stats.values()
                                 if s['first_event_tsc']), default=None)
    if earliest_worker_event:
        cold_start_us = (earliest_worker_event - t_start) / TSC_PER_NS / 1000
        print(f'  cold-start latency (t_dispatch to first worker event): '
              f'{cold_start_us:.1f} us')

    # Late-finish: time from last LEAF_END to t_end (dispatch exit).
    latest_leaf_end = 0
    for thread, ev, payload, tsc in rows:
        if ev == LEAF_END and t_start <= tsc <= t_end and thread != 'main':
            if tsc > latest_leaf_end:
                latest_leaf_end = tsc
    if latest_leaf_end:
        tail_us = (t_end - latest_leaf_end) / TSC_PER_NS / 1000
        print(f'  tail latency (last LeafEnd to dispatch exit): '
              f'{tail_us:.1f} us')


def main():
    if len(sys.argv) != 2:
        print('usage: python examples/analyze_trace.py <trace.csv>',
              file=sys.stderr)
        sys.exit(1)
    rows = parse_trace(sys.argv[1])
    print(f'# parsed {len(rows)} flynnel trace rows')
    t_start, t_end = find_measured_window(rows)
    if not t_start or not t_end:
        print('# could not identify dispatch window', file=sys.stderr)
        sys.exit(1)
    stats = per_worker_stats(rows, t_start, t_end)
    print_report(rows, t_start, t_end, stats)
    rayon_summary(sys.argv[1])


if __name__ == '__main__':
    main()
