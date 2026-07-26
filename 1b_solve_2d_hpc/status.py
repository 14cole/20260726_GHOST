#!/usr/bin/env python3
"""
STEP 1b -- HOW FAR ALONG IS THE RUN?

    python3 status.py            the newest run
    python3 status.py <run_dir>  one specific run

How many units are done, which are still outstanding, and whether any log looks
like a failure.  A missing unit is either still queued or it died -- the
difference is in runs/<run>/logs/.

Re-running a generated submit_job*.slurm is safe: an exactly attested unit is
skipped, so a partial run resumes without trusting an unverified file.
"""

import os
import re
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(os.path.dirname(HERE), "Backend")
sys.path.insert(0, BACKEND)

from hpc_common import (latest_run_dir, require_hpc_output_attestations, # noqa: E402
                        require_hpc_run_provenance, run_status)

_FAIL = re.compile(r"Traceback|Error|error:|CANCELLED|OOM|Killed|failed=[1-9]")


def report(run_dir):
    st = run_status(run_dir)
    man = st["manifest"]
    try:
        require_hpc_run_provenance(man, "ghost.hpc.2d-run.v1")
        if not st["pending"]:
            require_hpc_output_attestations(run_dir, man)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"run provenance/attestation failure: {exc}") from exc
    print(f"\n{Path(run_dir).name}   {st['n_done']}/{st['n_units']} units"
          + ("  COMPLETE" if not st["pending"]
             else f"  {st['pending']} outstanding"))
    print(f"  solver {man.get('solver', '?')}, "
          f"{len(man['frequencies_ghz'])} freq, pols {man['polarizations']}, "
          f"{man['n_slots']} slot(s)")
    have = {p.name for p in st["done"]}
    missing = [u for u in man["units"]
               if f"{u['polarization']}_{float(u['frequency_ghz']):.3f}GHz_"
                  f"{u['geometry_stem']}.grim" not in have]
    for u in missing[:8]:
        print(f"    outstanding: {u['polarization']} {u['frequency_ghz']:g}GHz "
              f"{u['geometry_stem']}")
    if len(missing) > 8:
        print(f"    ... and {len(missing) - 8} more")
    logs = sorted((Path(run_dir) / "logs").glob("*"))
    bad = [p for p in logs
           if p.is_file() and _FAIL.search(p.read_text(errors="ignore")[-20000:])]
    if bad:
        print(f"  {len(bad)} log(s) mention a failure -- read them:")
        for p in bad[:4]:
            print(f"    {p}")
    elif logs:
        print(f"  {len(logs)} log file(s), none mentioning a failure")
    else:
        print("  no logs yet (nothing has started)")


def main():
    if len(sys.argv) > 1:
        report(sys.argv[1])
        return
    runs = os.path.join(HERE, "runs")
    if not os.path.isdir(runs) or not list(Path(runs).glob("run_*")):
        raise SystemExit("no runs yet -- run submit.py first.")
    report(latest_run_dir(runs))


if __name__ == "__main__":
    main()
