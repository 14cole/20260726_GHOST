#!/usr/bin/env python3
"""
STEP 2b (part 1) -- SUBMIT THE BODY SOLVE TO SLURM
==================================================

WHAT IT DOES
  The cluster version of step 2a: same body, same sweep, and it writes the same
  two artefacts into the same Body/ folder, so steps 3 and 4 cannot tell which
  one you ran.  Put exactly ONE body .geo in this folder; materials come from
  the .geo itself.

  It uses the same material-aware BoR dispatch as 2a.  Floating-point ordering
  can still differ across BLAS/thread configurations, but there is no separate
  bare-PEC production engine.

  One unit is one (frequency, polarization).  ALL ASPECTS of a unit are solved
  in a single call -- each azimuthal mode is factored once and extra aspects
  are just more right-hand sides -- so the aspect sweep is nearly free and
  units are the parallel grain.  That also means a BoR unit parallelizes
  INTERNALLY (threads across modes and streaming tiles), which is why the node
  is divided into a few workers of WORKERS_PER_UNIT threads each rather than
  one process per core.

  The driver keeps its settings in a CONFIG block of module-level constants,
  and the SLURM script it generates execs THE SAME FILE with --worker.  So this
  writes a CONFIGURED COPY into runs/ and submits THAT -- overriding the
  constants in your own process would never reach the nodes.  Keep the copy
  with the results; it is the record of what actually ran.

  ASPECTS_DEG are angles from the rotation axis (0 = nose-on, 90 = broadside,
  180 = tail-on).  There is NO second body angle: a body of revolution is
  axisymmetric, so [0, 180] characterises the whole sphere.  Radar azimuth and
  elevation appear in steps 3 and 4, when an attitude maps each radar look onto
  one of these aspects.

INPUTS
  <one>.geo              your body of revolution, drawn in GEOMETRY_UNITS

OUTPUTS
  driver_configured.py   the copy that was submitted
  runs/run_<stamp>/      manifest, frozen geometry/material inputs, scripts,
                         results, and logs
  submitted.txt          the run directory, read by collect.py

KNOBS (below)
  FREQUENCIES_GHZ, ASPECT_STEP_DEG, GEOMETRY_UNITS, CFIE_ALPHA,
  N_NODES, N_JOBS, WORKERS_PER_UNIT, SLURM_*, SUBMIT

  FREQUENCIES_GHZ must cover what you ask for in step 3 -- the body cannot be
  interpolated onto a frequency it was never solved at.

THEN  wait for the queue, then:  python3 status.py  ->  python3 collect.py
      Re-submitting skips units whose .grim already exists, so a partial run
      resumes rather than starting over.

NO CLUSTER HERE?  Leave SUBMIT = False and run one slot exactly as SLURM would
      (run the GENERATED SCRIPT, not the driver directly -- the script is what
      sets the cd and the PYTHONPATH that puts Backend/ on the worker's path):
          SLURM_ARRAY_TASK_ID=0 bash runs/run_<stamp>/submit_job0.slurm

    python3 submit.py
"""

import glob
import math
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BACKEND = os.path.join(ROOT, "Backend")
sys.path.insert(0, BACKEND)
sys.path.insert(0, ROOT)

# ─── KNOBS ──────────────────────────────────────────────────────────────────
FREQUENCIES_GHZ = [3.0, 6.0]
# None solves the exact aspects used by grid.py.  A positive value requests a
# legacy uniform sweep, which is suitable only if it explicitly contains every
# production output-grid aspect.
ASPECT_STEP_DEG = None
POLARIZATIONS = ["VV", "HH"]  # BOTH -- steps 3 and 4 need both channels
GEOMETRY_UNITS = "meters"     # units the .geo is drawn in
CFIE_ALPHA = 0.5              # closed PEC bodies -> CFIE (matches step 2a)

N_NODES, N_JOBS = 1, 1        # slots = N_NODES x N_JOBS; units go round-robin
WORKERS_PER_UNIT = 4          # threads INSIDE one BoR solve (modes + tiles)
SLURM_PARTITION = "compute"
SLURM_ACCOUNT = None          # e.g. "my_project"; None to omit
SLURM_TIME = None             # None = no walltime limit; or "HH:MM:SS"
SUBMIT = False                # True = actually sbatch
# ────────────────────────────────────────────────────────────────────────────

from hpc_common import (BOR_DRIVER, configure_driver,                 # noqa: E402
                        latest_run_dir)
from grid import AZIMUTHS_DEG, ELEVATIONS_DEG                         # noqa: E402
from frame import AXIS_AZ_DEG, AXIS_EL_DEG                            # noqa: E402
from feature_sum import (radar_grid_aspects,                          # noqa: E402
                         geometry_input_fingerprint)


def _validate_config():
    """Reject incomplete or filename-ambiguous production BoR sweeps."""
    frequencies = [float(value) for value in FREQUENCIES_GHZ]
    pols = [str(value).strip().upper() for value in POLARIZATIONS]
    if (
        not frequencies
        or not all(math.isfinite(value) and value > 0.0
                   for value in frequencies)
        or len(set(frequencies)) != len(frequencies)
        or len({f"{value:.3f}" for value in frequencies})
        != len(frequencies)
    ):
        raise SystemExit(
            "FREQUENCIES_GHZ must be non-empty, finite, positive, unique, "
            "and distinct after the 0.001 GHz output-name precision."
        )
    if set(pols) != {"VV", "HH"} or len(pols) != 2:
        raise SystemExit(
            "POLARIZATIONS must contain exactly VV and HH. The radar-frame "
            "body product requires both physical channels."
        )
    if (
        int(N_NODES) < 1
        or int(N_JOBS) < 1
        or int(WORKERS_PER_UNIT) < 1
    ):
        raise SystemExit(
            "N_NODES, N_JOBS, and WORKERS_PER_UNIT must be positive integers."
        )
    alpha = float(CFIE_ALPHA)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise SystemExit("CFIE_ALPHA must be finite and lie in [0, 1].")


def stage(geo):
    """Stage an immutable content-addressed geometry + material-table set."""
    digest = geometry_input_fingerprint(geo, GEOMETRY_UNITS)
    root = os.path.join(
        HERE, "runs", "staging",
        f"{os.path.splitext(os.path.basename(geo))[0]}_{digest[:16]}")
    frd, opn = os.path.join(root, "FRD"), os.path.join(root, "OPN")
    os.makedirs(frd, exist_ok=True)
    os.makedirs(opn, exist_ok=True)
    staged_geo = os.path.join(frd, os.path.basename(geo))
    shutil.copy2(geo, staged_geo)
    for name in sorted(os.listdir(os.path.dirname(geo))):
        src = os.path.join(os.path.dirname(geo), name)
        if name.startswith("mat.") and os.path.isfile(src):
            shutil.copy2(src, os.path.join(frd, name))
    if geometry_input_fingerprint(staged_geo, GEOMETRY_UNITS) != digest:
        raise RuntimeError("staged geometry/material fingerprint changed while "
                           "copying; refusing a non-atomic solver input.")
    return frd, opn


def main():
    _validate_config()
    if SUBMIT and not shutil.which("sbatch"):
        raise SystemExit("SUBMIT = True but there is no sbatch on PATH -- run "
                         "this on the cluster login node, or set SUBMIT = False.")
    geos = sorted(glob.glob(os.path.join(HERE, "*.geo")))
    if len(geos) != 1:
        raise SystemExit(f"put exactly one body .geo in {HERE} -- found "
                         f"{[os.path.basename(g) for g in geos]}.")
    geo = geos[0]
    required_aspects = radar_grid_aspects(
        AZIMUTHS_DEG, ELEVATIONS_DEG, AXIS_AZ_DEG, AXIS_EL_DEG)
    if ASPECT_STEP_DEG is None:
        aspects = [float(a) for a in required_aspects]
    else:
        step = float(ASPECT_STEP_DEG)
        if not np.isfinite(step) or step <= 0.0:
            raise SystemExit("ASPECT_STEP_DEG must be None or a positive number.")
        aspects = [float(a) for a in
                   np.arange(0.0, 180.0 + 0.5 * step, step)]
        missing_aspects = [
            float(q) for q in required_aspects
            if not np.any(np.isclose(aspects, q, rtol=0.0, atol=1e-9))
        ]
        if missing_aspects:
            raise SystemExit(
                "The requested uniform ASPECT_STEP_DEG does not contain the "
                f"current grid.py aspects ({len(missing_aspects)} missing; "
                f"first {missing_aspects[:5]}). Use ASPECT_STEP_DEG = None. "
                "Production complex body fields are not coarsely interpolated.")

    frd, opn = stage(geo)
    driver = configure_driver(BOR_DRIVER,
                              os.path.join(HERE, "driver_configured.py"), {
        "FRD_DIR": frd,
        "OPN_DIR": opn,
        "OUTPUT_DIR": os.path.join(HERE, "runs"),
        "FREQUENCIES_GHZ": [float(f) for f in FREQUENCIES_GHZ],
        "ASPECTS_DEG": aspects,
        "POLARIZATIONS": list(POLARIZATIONS),
        "GEOMETRY_UNITS": GEOMETRY_UNITS,
        "CFIE_ALPHA": float(CFIE_ALPHA),
        "N_NODES": int(N_NODES),
        "N_JOBS": int(N_JOBS),
        "WORKERS_PER_UNIT": int(WORKERS_PER_UNIT),
        "SLURM_PARTITION": SLURM_PARTITION,
        "SLURM_ACCOUNT": SLURM_ACCOUNT,
        "SLURM_TIME": SLURM_TIME,
        # steps 3 and 4 build the radar frame themselves, from the aspect
        # sweep plus the fixed attitude in Backend/frame.py -- so the driver's
        # own az/el product is dead weight here.
        "AZEL_ENABLE": False,
        "SUBMIT": bool(SUBMIT),
    })

    n_units = len(FREQUENCIES_GHZ) * len(POLARIZATIONS)
    print("STEP 2b  body solve on SLURM")
    print(f"         body: {os.path.basename(geo)}")
    print(f"         {n_units} unit(s) = {len(FREQUENCIES_GHZ)} freq x "
          f"{len(POLARIZATIONS)} pol, {len(aspects)} aspects each (solved in "
          f"one call), over {N_NODES * N_JOBS} slot(s)")
    print(f"         driver copy: {os.path.relpath(driver, HERE)}")
    print(f"         SUBMIT = {SUBMIT}\n")

    sys.argv = [str(driver)]                    # the driver parses argv itself
    ns = {"__file__": str(driver), "__name__": "__main__"}
    with open(driver, encoding="utf-8") as stream:
        driver_source = stream.read()
    exec(compile(driver_source, str(driver), "exec"), ns)        # noqa: S102

    run_dir = latest_run_dir(os.path.join(HERE, "runs"))
    with open(os.path.join(HERE, "submitted.txt"), "w") as fh:
        fh.write(str(run_dir) + "\n")
    print(f"\n         run directory: {run_dir}")
    print("         wrote submitted.txt")
    if SUBMIT:
        print("\nNEXT     watch the queue (squeue or python3 status.py), then "
              "python3 collect.py")
    else:
        print("\nNEXT     SUBMIT = False, so nothing was queued.  Run one slot "
              "here exactly as\n         SLURM would (the script sets the cd and "
              "the PYTHONPATH):\n"
              f"           SLURM_ARRAY_TASK_ID=0 bash "
              f"{run_dir}/submit_job0.slurm")


if __name__ == "__main__":
    main()
