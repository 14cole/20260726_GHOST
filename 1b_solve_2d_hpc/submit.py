#!/usr/bin/env python3
"""
STEP 1b (part 1) -- SUBMIT THE 2-D MONOSTATIC SOLVES TO SLURM
=============================================================

WHAT IT DOES
  The cluster version of step 1a -- identical physics, identical output names,
  identical geometries/FRD + geometries/OPN layout.  One unit is one
  (coupon, frequency, polarization), so a library of seam designs is
  n_coupons x n_freqs x 2 units that all run AT ONCE.  That count is the whole
  reason this step exists: coupons are individually cheap, there are just a lot
  of them.

  Cluster work has a real shape, so it is two scripts: submit.py queues it, you
  wait, collect.py gathers it.

  The drivers keep their settings in a CONFIG block of module-level constants,
  and the SLURM script they generate execs THE SAME FILE with --worker.  So
  this writes a CONFIGURED COPY of the driver into runs/ and submits THAT --
  overriding the constants in your own process would never reach the nodes.
  Keep the copy with the results; it is the record of what actually ran.

INPUTS
  geometries/FRD/<base>.geo    one clean cross-section per variation
  geometries/OPN/<base>.geo    the same plank WITH the feature, same file name

OUTPUTS
  driver_configured.py         staging copy used to create the run
  runs/run_<stamp>/            manifest, immutable driver/geometry/material
                               inputs, submit scripts, results, and logs
  submitted.txt                the run directory, read by collect.py

KNOBS (below)
  FREQUENCIES_GHZ, ANGLES_DEG (90 = normal incidence on the outer face),
  POLARIZATIONS, GEOMETRY_UNITS, N_NODES, N_JOBS, SLURM_*, SUBMIT

THEN  wait for the queue, then:  python3 status.py  ->  python3 collect.py
      Re-submitting skips units whose .grim already exists, so a partial run
      resumes rather than starting over.

NO CLUSTER HERE?  Leave SUBMIT = False and run one slot exactly as SLURM would
      (the generated script sets the cd and the PYTHONPATH for you):
          SLURM_ARRAY_TASK_ID=0 bash runs/run_<stamp>/submit_job0.slurm

    python3 submit.py
"""

import glob
import hashlib
import math
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(os.path.dirname(HERE), "Backend")
sys.path.insert(0, BACKEND)

# ─── KNOBS ──────────────────────────────────────────────────────────────────
FREQUENCIES_GHZ = [3.0, 6.0]
ANGLES_DEG = [float(a) for a in range(0, 181, 5)]   # 90 = onto the outer face
POLARIZATIONS = ["TM", "TE"]        # BOTH -- a delta needs both channels
GEOMETRY_UNITS = "meters"           # units the .geo files are drawn in

N_NODES, N_JOBS = 1, 1              # total parallel compute = N_NODES x N_JOBS
SLURM_PARTITION = "compute"
SLURM_ACCOUNT = None                # e.g. "my_project"; None to omit
SLURM_TIME = None                   # None = no walltime limit; or "HH:MM:SS"
SUBMIT = False                      # True = actually sbatch
# ────────────────────────────────────────────────────────────────────────────

from hpc_common import (TWOD_DRIVER, configure_driver,                # noqa: E402
                        latest_run_dir)
from feature_sum import geometry_input_fingerprint                    # noqa: E402


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_config():
    """Reject an HPC request that cannot produce a complete seam dataset."""
    frequencies = [float(value) for value in FREQUENCIES_GHZ]
    angles = [float(value) for value in ANGLES_DEG]
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
    if (
        not angles
        or not all(math.isfinite(value) for value in angles)
        or len(set(angles)) != len(angles)
    ):
        raise SystemExit("ANGLES_DEG must be non-empty, finite, and unique.")
    if set(pols) != {"TM", "TE"} or len(pols) != 2:
        raise SystemExit(
            "POLARIZATIONS must contain exactly TM and TE. A production "
            "feature delta cannot silently zero-fill a missing channel."
        )
    if int(N_NODES) < 1 or int(N_JOBS) < 1:
        raise SystemExit("N_NODES and N_JOBS must both be positive integers.")


def stage(geos):
    """Copy the coupons into runs/geometries/<ROLE>/<base>_<ROLE>.geo.

    The driver names each output <POL>_<FREQ>GHz_<geometry stem>.grim, and
    geometries/FRD/x.geo and geometries/OPN/x.geo have the SAME stem -- so
    without the role in the file name the two halves of a pair would write to
    one path and the clean solve would be silently skipped as "already done".
    Appending the marker here also hands step 1c the names it pairs on, so the
    role lives in the folder for you and in the file name for the tools.
    """
    fingerprints = {
        os.path.abspath(path):
            geometry_input_fingerprint(path, GEOMETRY_UNITS)
        for path in sorted(geos)
    }
    bundle_hash = hashlib.sha256(
        "\n".join(
            f"{path}:{fingerprints[path]}" for path in sorted(fingerprints)
        ).encode("utf-8")
    ).hexdigest()
    root = os.path.join(HERE, "runs", "staging", bundle_hash[:20])
    for role in ("FRD", "OPN"):
        d = os.path.join(root, role)
        os.makedirs(d, exist_ok=True)
    staged_pairs = []
    for p in geos:
        role = os.path.basename(os.path.dirname(p))
        base = os.path.basename(p)[: -len(".geo")]
        if base.endswith(("_FRD", "_OPN")):
            base = base[:-4]
        staged = os.path.join(root, role, f"{base}_{role}.geo")
        shutil.copy2(p, staged)
        staged_pairs.append((os.path.abspath(p), staged))
        for name in sorted(os.listdir(os.path.dirname(p))):
            src = os.path.join(os.path.dirname(p), name)
            if name.startswith("mat.") and os.path.isfile(src):
                dst = os.path.join(root, role, name)
                if os.path.exists(dst):
                    with open(src, "rb") as lhs, open(dst, "rb") as rhs:
                        if lhs.read() != rhs.read():
                            raise SystemExit(
                                f"{role} geometries use conflicting {name} "
                                "tables in one solve bundle.")
                else:
                    shutil.copy2(src, dst)
    for original, staged in staged_pairs:
        expected = fingerprints[original]
        if (
            geometry_input_fingerprint(original, GEOMETRY_UNITS) != expected
            or _sha256_file(original) != _sha256_file(staged)
        ):
            raise SystemExit(
                "a geometry or material table changed while the immutable HPC "
                "input bundle was being staged; no run was submitted."
            )
    for role in ("FRD", "OPN"):
        sources = {
            name: os.path.join(HERE, "geometries", role, name)
            for name in os.listdir(os.path.join(HERE, "geometries", role))
            if name.startswith("mat.")
            and os.path.isfile(
                os.path.join(HERE, "geometries", role, name)
            )
        }
        staged_role = os.path.join(root, role)
        staged_tables = {
            name: os.path.join(staged_role, name)
            for name in os.listdir(staged_role)
            if name.startswith("mat.")
            and os.path.isfile(os.path.join(staged_role, name))
        }
        if set(sources) != set(staged_tables) or any(
            _sha256_file(sources[name])
            != _sha256_file(staged_tables[name])
            for name in sources
        ):
            raise SystemExit(
                f"the staged {role} material-table inventory is stale or "
                "changed; no run was submitted."
            )
    return os.path.join(root, "FRD"), os.path.join(root, "OPN")


def main():
    _validate_config()
    if SUBMIT and not shutil.which("sbatch"):
        raise SystemExit("SUBMIT = True but there is no sbatch on PATH -- run "
                         "this on the cluster login node, or set SUBMIT = False.")

    frd = os.path.join(HERE, "geometries", "FRD")
    opn = os.path.join(HERE, "geometries", "OPN")
    geos = sorted(glob.glob(os.path.join(frd, "*.geo")) +
                  glob.glob(os.path.join(opn, "*.geo")))
    if not geos:
        raise SystemExit(
            "no geometries/FRD/*.geo or geometries/OPN/*.geo.\n"
            "  Put the clean cross-section of each variation in geometries/FRD/\n"
            "  and the featured one, SAME FILE NAME, in geometries/OPN/.")

    bases = {}
    for p in geos:
        role = os.path.basename(os.path.dirname(p))
        base = os.path.basename(p)[: -len(".geo")]
        bases.setdefault(base[:-4] if base.endswith(("_FRD", "_OPN")) else base,
                         set()).add(role)
    for base, roles in sorted(bases.items()):
        if roles != {"FRD", "OPN"}:
            print(f"         WARNING  {base} has only {sorted(roles)} -- step 1c "
                  f"needs BOTH to subtract")

    staged_frd, staged_opn = stage(geos)
    driver = configure_driver(TWOD_DRIVER,
                              os.path.join(HERE, "driver_configured.py"), {
        "FRD_DIR": staged_frd,
        "OPN_DIR": staged_opn,
        "OUTPUT_DIR": os.path.join(HERE, "runs"),
        "FREQUENCIES_GHZ": [float(f) for f in FREQUENCIES_GHZ],
        "AZIMUTHS_DEG": [float(a) for a in ANGLES_DEG],
        "POLARIZATIONS": list(POLARIZATIONS),
        "GEOMETRY_UNITS": GEOMETRY_UNITS,
        "N_NODES": int(N_NODES),
        "N_JOBS": int(N_JOBS),
        "SLURM_PARTITION": SLURM_PARTITION,
        "SLURM_ACCOUNT": SLURM_ACCOUNT,
        "SLURM_TIME": SLURM_TIME,
        "SUBMIT": bool(SUBMIT),
    })

    n_units = len(geos) * len(FREQUENCIES_GHZ) * len(POLARIZATIONS)
    print("STEP 1b  2-D monostatic solves on SLURM")
    print(f"         {len(bases)} variation(s), {len(geos)} coupon file(s)")
    print(f"         {n_units} unit(s) = {len(geos)} geom x "
          f"{len(FREQUENCIES_GHZ)} freq x {len(POLARIZATIONS)} pol, "
          f"{len(ANGLES_DEG)} angles each, over {N_NODES * N_JOBS} slot(s)")
    print(f"         driver copy: {os.path.relpath(driver, HERE)}")
    print(f"         SUBMIT = {SUBMIT}\n")

    sys.argv = [str(driver)]
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
              f"the PYTHONPATH):\n           SLURM_ARRAY_TASK_ID=0 bash "
              f"{run_dir}/submit_job0.slurm")


if __name__ == "__main__":
    main()
