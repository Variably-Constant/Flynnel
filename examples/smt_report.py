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
ENGAGE = re.compile(r"^smt_engagement\s+(.*)$")
LEVERS = re.compile(r"^levers:\s+(.*)$")


def main(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    pending = None
    rows = defaultdict(list)
    engage = defaultdict(list)
    switches = defaultdict(set)
    failures = []
    # `pending` guards the row, which must follow its own ARM line.
    # `arm` outlives it, because the engagement line comes AFTER the
    # row and would otherwise have nothing to attach to.
    arm = None
    for line in lines:
        f = FAILED.match(line)
        if f:
            failures.append(line)
            pending = None
            arm = None
            continue
        m = ARM.match(line)
        if m:
            pending = m.group(2)
            arm = m.group(2)
            continue
        s = LEVERS.match(line)
        if s and arm is not None:
            for token in s.group(1).split():
                if token.startswith("smt_window="):
                    switches[arm].add(token.split("=", 1)[1])
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
            continue
        e = ENGAGE.match(line)
        if e and arm is not None:
            fields = {}
            for token in e.group(1).split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    fields[key] = value
            engage[arm].append(fields)

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

    if engage:
        print()
        # Whether each arm ran the switch its label claims. An off arm
        # that leaves the variable unset takes the crate's default, and
        # two levers ship on, so a rotation can compare an arm with
        # itself and still produce a clean null.
        print("arm states:")
        if not switches:
            print("  no levers line in this log; what each arm ran is unrecorded")
        else:
            for arm_name, want in (("off", "false"), ("null", "false"), ("on", "true")):
                seen = sorted(switches.get(arm_name, []))
                if not seen:
                    print(f"  {arm_name:<5} no smt_window reported")
                elif seen != [want]:
                    print(f"  {arm_name:<5} smt_window={seen}, and this arm means {want}"
                          f" - do not read its rows")
                else:
                    print(f"  {arm_name:<5} smt_window={seen}, as labelled")
        print()
        print("engagement, from the same arms:")
        for arm in ("off", "null", "on"):
            v = engage.get(arm, [])
            if not v:
                print(f"  {arm:<5} no engagement line")
                continue
            ticks = sorted({f.get("window_ticks", "-") for f in v})
            win = sorted({f.get("cv2_window", "-") for f in v})
            # cv2_window is the latest of the ticks counted in
            # window_ticks, and within one run that figure spans the
            # whole range the classifier can express. The extremes are
            # what say which regimes the switch was consulted across.
            lo = sorted({f.get("cv2_window_min", "-") for f in v})
            hi = sorted({f.get("cv2_window_max", "-") for f in v})
            life = sorted({f.get("cv2_lifetime", "-") for f in v})
            sw = sorted({f.get("smt_switch", "-") for f in v})
            print(f"  {arm:<5} window_ticks={ticks} cv2_window={win}")
            print(f"        cv2_window_min={lo} cv2_window_max={hi}")
            print(f"        cv2_lifetime={life} smt_switch={sw}")
        # The switch chooses between the window and the lifetime
        # figure. With no window classified it falls back to the
        # lifetime one, which is what the off arm reads anyway, so both
        # arms consult the same number and agreeing rows say nothing
        # about the mechanism.
        on = engage.get("on", [])
        if on and all(f.get("window_ticks") == "0" for f in on):
            print()
            print("  NO WINDOW WAS EVER CLASSIFIED on the on arm, so the switch fell")
            print("  back to the lifetime figure the off arm already uses. The rows")
            print("  above are not a reading of this mechanism.")
        elif on and all(f.get("smt_switch") == "false" for f in on):
            print()
            print("  The switch reports itself OFF on the arm labelled on; the driver")
            print("  did not set it and no row here measures it.")

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
