"""The SMT-from-window switch, read across a counterbalanced rotation.

Each arm prints one row:

    smt_recovery <smt_before> <smt_after> <returned_at> <pre_n> <post_n>

`smt_before` is the decision on a quiet host, `smt_after` the decision
once a load has been applied and lifted, and `returned_at` the window
the answer came back in, or a dash when it never did.

WHAT THE ROWS CAN AND CANNOT SAY. On uniform work the correct answer is
false: siblings contest the same execution unit when every leaf is the
same shape. So the LEVEL is the finding and the recovery column
discriminates nothing when neither arm's decision moves. Both are
reported, and a column where every arm agrees is called out as such
rather than being read as a clean negative.

The null arm is the same binary invoked a second time with the switch
still off. A difference between off and null is what two identical arms
do in this rotation, and the on/off difference has to clear it.

An arm whose windows ran no dispatches is not evidence either way: the
decision cannot move in a window that never dispatched. Those are
counted and named.
"""

import re
import sys
from collections import defaultdict

ARM = re.compile(r"^ARM\s+([A-Z_]+)-(off|null|on)\s+t(\d+)")
ROW = re.compile(
    r"^smt_recovery\s+(true|false)\s+(true|false)\s+(\d+|-)\s+(\d+)\s+(\d+)\s*$"
)
FAILED = re.compile(r"^ARM_FAILED\s+(\S+)\s+t(\d+)\s+exit=(\d+)")


def main(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    pending = None
    rows = defaultdict(list)
    failures = []
    for line in lines:
        f = FAILED.match(line)
        if f:
            failures.append(line)
            pending = None
            continue
        m = ARM.match(line)
        if m:
            pending = m.group(2)
            continue
        r = ROW.match(line)
        if r and pending is not None:
            rows[pending].append(
                {
                    "before": r.group(1) == "true",
                    "after": r.group(2) == "true",
                    "returned": None if r.group(3) == "-" else int(r.group(3)),
                    "pre_n": int(r.group(4)),
                    "post_n": int(r.group(5)),
                }
            )
            pending = None

    if not rows:
        print("NO ROWS PARSED")
        return 1

    for line in failures:
        print(line)
    if failures:
        print(
            f"{len(failures)} arm(s) exited non-zero. Exit 2 is the harness refusing "
            "to run because the plan's SMT prior is false, which is a run that measured "
            "nothing rather than a negative result."
        )

    print()
    print(f"{'arm':<6} {'n':>3} {'before true':>12} {'after true':>11} {'returned':>9} {'empty':>6}")
    print("-" * 56)
    for arm in ("off", "null", "on"):
        v = rows.get(arm, [])
        if not v:
            print(f"{arm:<6}   no rows")
            continue
        before = sum(1 for x in v if x["before"])
        after = sum(1 for x in v if x["after"])
        returned = sum(1 for x in v if x["returned"] is not None)
        empty = sum(1 for x in v if x["pre_n"] == 0 or x["post_n"] == 0)
        print(
            f"{arm:<6} {len(v):>3} {f'{before}/{len(v)}':>12} {f'{after}/{len(v)}':>11} "
            f"{f'{returned}/{len(v)}':>9} {empty:>6}"
        )

    print()
    off = rows.get("off", [])
    null = rows.get("null", [])
    on = rows.get("on", [])
    if off and null and on:
        def level(v):
            return sum(1 for x in v if x["before"]), len(v)

        o, n, p = level(off), level(null), level(on)
        if o[0] == n[0] and p[0] != o[0]:
            print(
                f"THE LEVEL SEPARATES. off and its null both answer true in "
                f"{o[0]}/{o[1]}, on answers true in {p[0]}/{p[1]}."
            )
        elif o[0] == n[0] == p[0]:
            print(
                f"No arm's decision differs: all three answer true in {o[0]} of their "
                "trials. The switch moved nothing here."
            )
        else:
            print(
                f"off {o[0]}/{o[1]}, null {n[0]}/{n[1]}, on {p[0]}/{p[1]}. The null "
                "differs from off, so this rotation's floor covers the on/off gap and "
                "nothing is called."
            )

    empties = sum(
        1 for v in rows.values() for x in v if x["pre_n"] == 0 or x["post_n"] == 0
    )
    if empties:
        print()
        print(
            f"{empties} arm(s) had a window that dispatched nothing. A decision cannot "
            "move in a window that never dispatched, so those rows are not evidence "
            "either way; raise the window length."
        )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <smt.log>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
