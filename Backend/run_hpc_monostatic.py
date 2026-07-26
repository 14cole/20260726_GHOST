#!/usr/bin/env python3
"""
HPC monostatic RCS sweep driver (SLURM).

Edit the CONFIG block below and run:

    python run_hpc_monostatic.py

Workflow:
- Discover geometry files under FRD_DIR + OPN_DIR.
- Expand into a (geometry × frequency × polarization) unit list. All azimuths
  for a unit are solved in a single solver call (matrix factored once).
- Distribute units round-robin across N_NODES × N_JOBS parallel slots.
- Write N_JOBS sbatch scripts (each one a job array of size N_NODES) and
  submit them. Each array task runs on one node and parallelizes its assigned
  units across the cores SLURM allocated.
- As each unit finishes, its result is exported immediately to
  "<POL>_<FREQ:.3f>GHz_<geometry_stem>.grim" in <run_dir>/results/.

Restartable: a unit whose .grim file already exists is skipped, so you can
cancel and resubmit one job's slice (e.g., move it to a different partition)
without re-doing finished work. The manifest's slot partitioning is fixed
once written, so the other in-flight submissions stay correct.

Internal worker invocation (called by SLURM, not by the user):
    python run_hpc_monostatic.py --worker <run_dir> <job_index> <node_index>
"""

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import workflow_provenance as _workflow_provenance
from workflow_provenance import (
    backend_source_fingerprint,
    manifest_solve_spec_fingerprint,
    runtime_environment_fingerprint,
    stable_json_fingerprint,
    unit_solve_spec_fingerprint,
    verify_output_attestation,
    write_output_attestation,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — the only section most users need to edit
# ═══════════════════════════════════════════════════════════════════════════════

# Input geometry folders. Every *.geo file found under these paths
# (recursively) is added to the sweep. Source folder is NOT injected into
# output filenames — the geometry filename is preserved verbatim.
FRD_DIR = "geometries/FRD"
OPN_DIR = "geometries/OPN"

# Requested sweep.
FREQUENCIES_GHZ = [2.0, 4.0, 6.0, 8.0, 10.0]
AZIMUTHS_DEG    = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
POLARIZATIONS   = ["VV", "HH"]          # any subset of: VV, HH, TM, TE

# Output root. A new run_YYYYMMDD_HHMMSS/ subfolder is created inside.
OUTPUT_DIR = "rcs_runs"

# --- Multi-node / multi-submission parallelism -----------------------------
# Total parallel compute = N_NODES × N_JOBS nodes. Units are split round-robin
# across that many slots. N_JOBS separate sbatch submissions are produced
# (each one a job array of size N_NODES). This lets you put e.g. 2 nodes on
# partition A and 2 nodes on partition B without overlapping work — the
# submissions don't need to talk to each other; the partitioning is
# deterministic from the manifest.
N_NODES = 1
N_JOBS  = 1

# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED — fine tuning (SLURM resources, solver knobs, env setup)
# ═══════════════════════════════════════════════════════════════════════════════

# --- SLURM resources (per array task = one node) ---------------------------
SLURM_PARTITION = "compute"
SLURM_ACCOUNT   = None            # e.g. "my_project"; None to omit
SLURM_QOS       = None
SLURM_TIME      = None            # None = no walltime limit; or "HH:MM:SS"
CORES_PER_NODE  = None            # None = request whole node via --exclusive
                                  # (pool size auto-detected from SLURM env).
                                  # Or set an integer, e.g. 32.
MEM_PER_NODE    = "0"             # "0" = ALL memory of the node (SLURM idiom;
                                  # recommended with --exclusive). None = omit
                                  # the directive -> cluster default applies
                                  # (often DefMemPerCPU ~3.5G x CPUs, which can
                                  # be far less than node RAM and OOM-kill
                                  # workers). Or an explicit value, e.g. "64G".
MAX_WORKERS_PER_NODE = None       # None = one worker per allocated core. Set a
                                  # smaller integer when units are memory-heavy:
                                  # peak node RAM ~ pool_size x per-unit peak
                                  # (dense solve ~ 5*16*N^2 bytes, N = boundary
                                  # nodes ~ 20 x perimeter/lambda). An OOM KILL
                                  # (cgroup) can hang the Pool, unlike a Python
                                  # MemoryError which is caught and logged.
SLURM_MAIL_TYPE = None            # e.g. "END,FAIL"
SLURM_MAIL_USER = None
SLURM_EXTRA_SBATCH = []  # type: List[str]  # raw extra lines, e.g. "--constraint=intel"

JOB_PROLOGUE = []  # type: List[str]

# --- Solver knobs (mirror run_monostatic.py) -------------------------------
GEOMETRY_UNITS          = "inches"       # "inches" or "meters"
SOLVER_METHOD           = "auto"         # "auto" | "direct"
# The present 2-D SLP formulations do not implement a distinct CFIE operator.
# Keep this at zero; nonzero values are rejected instead of acting as a dead
# production control.
CFIE_ALPHA              = 0.0
MAX_PANELS              = 50_000
BLAS_THREADS_PER_WORKER = 1              # keeps N workers × BLAS threads sane

# --- Geometry discovery & submission ---------------------------------------
GEOMETRY_EXTS = (".geo",)
PYTHON_EXE    = sys.executable           # interpreter used inside the job
SUBMIT        = True                     # False → write .slurm files but don't sbatch

# ═══════════════════════════════════════════════════════════════════════════════

_SBATCH = shutil.which("sbatch") or "sbatch"


# ─── shared helpers ────────────────────────────────────────────────────────

def _solver_source_fingerprint():
    # type: () -> str
    backend_dir = str(Path(_workflow_provenance.__file__).resolve().parent)
    return backend_source_fingerprint(
        backend_dir,
        {"driver_configured.py": str(Path(__file__).resolve())},
    )


def _verify_run_provenance(manifest):
    # type: (Dict[str, Any]) -> None
    expected_source = str(manifest.get("solver_source_sha256", ""))
    expected_runtime = str(manifest.get("runtime_environment_sha256", ""))
    if not expected_source or not expected_runtime:
        raise RuntimeError(
            "HPC run manifest lacks exact solver-source/runtime provenance; "
            "legacy runs must be regenerated before reuse."
        )
    current_source = _solver_source_fingerprint()
    current_runtime = runtime_environment_fingerprint()
    if current_source != expected_source:
        raise RuntimeError(
            "Solver source/native artifacts differ from the HPC run manifest; "
            "no cached or new field will be used from this mixed source state."
        )
    if current_runtime != expected_runtime:
        raise RuntimeError(
            "Python/platform/NumPy/SciPy/BLAS runtime differs from the HPC run "
            "manifest; start a new run in this numerical environment."
        )


def _unit_attestation_fields(manifest, unit):
    # type: (Dict[str, Any], Dict[str, Any]) -> Dict[str, Any]
    return {
        "run_id": str(manifest["run_id"]),
        "solver_source_sha256": str(manifest["solver_source_sha256"]),
        "runtime_environment_sha256":
            str(manifest["runtime_environment_sha256"]),
        "geometry_input_sha256":
            str(unit["geometry_input_sha256"]),
        "run_solve_spec_sha256":
            manifest_solve_spec_fingerprint(manifest),
        "unit_solve_spec_sha256":
            unit_solve_spec_fingerprint(unit),
        "solver_config_sha256":
            stable_json_fingerprint(manifest.get("solver_config", {})),
        "angular_grid_kind": "azimuths_deg",
        "angular_grid_deg":
            [float(value) for value in unit["azimuths_deg"]],
        "polarization": str(unit["polarization"]),
        "frequency_ghz": float(unit["frequency_ghz"]),
    }


def _verify_unit_input(unit, manifest):
    # type: (Dict[str, Any], Dict[str, Any]) -> None
    from feature_sum import geometry_input_fingerprint
    current = geometry_input_fingerprint(
        str(unit["geometry"]),
        str(manifest["solver_config"]["geometry_units"]),
    )
    if current != unit.get("geometry_input_sha256"):
        raise RuntimeError(
            f"Frozen geometry/material input changed during the HPC unit: "
            f"{unit['geometry']}"
        )


def _discover_geometries():
    # type: () -> List[Path]
    """Return every geometry file under FRD_DIR/OPN_DIR (deduplicated)."""
    found = []   # type: List[Path]
    seen = set()  # type: set
    for d in (FRD_DIR, OPN_DIR):
        root = Path(d)
        if not root.is_dir():
            print(f"  [warn] dir not found: {root}", file=sys.stderr)
            continue
        for ext in GEOMETRY_EXTS:
            for p in sorted(root.rglob(f"*{ext}")):
                rp = p.resolve()
                if rp in seen:
                    continue
                seen.add(rp)
                found.append(p)
    return found


def _pin_blas_threads(n):
    # type: (int) -> None
    """Pin BLAS threads via env vars. Called in parent and in each pool worker."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)


def _detect_cores():
    # type: () -> int
    """Cores actually allocated to this process. Prefers SLURM, falls back to OS."""
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.environ.get(var, "").strip()
        if v.isdigit() and int(v) > 0:
            return int(v)
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except Exception:
            pass
    return max(1, os.cpu_count() or 1)


def _unit_output_path(run_dir, unit):
    # type: (Path, Dict[str, Any]) -> Path
    pol  = unit["polarization"]
    freq = float(unit["frequency_ghz"])
    stem = unit["geometry_stem"]
    return run_dir / "results" / f"{pol}_{freq:.3f}GHz_{stem}.grim"


def _solve_and_export(unit, snapshot, material_base, run_dir_str):
    # type: (Dict[str, Any], Dict[str, Any], str, str) -> Tuple[str, str]
    """Pool-worker entry point: solve one unit, export .grim. Idempotent."""
    run_dir = Path(run_dir_str)
    out_path = _unit_output_path(run_dir, unit)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    _verify_run_provenance(manifest)
    _verify_unit_input(unit, manifest)
    attestation = _unit_attestation_fields(manifest, unit)
    if out_path.exists():
        verify_output_attestation(str(out_path), attestation)
        return ("skipped", str(out_path))

    from rcs_solver import solve_monostatic_rcs_2d
    result = solve_monostatic_rcs_2d(
        geometry_snapshot=snapshot,
        frequencies_ghz=[float(unit["frequency_ghz"])],
        elevations_deg=[float(a) for a in unit["azimuths_deg"]],
        polarization=unit["polarization"],
        geometry_units=GEOMETRY_UNITS,
        material_base_dir=material_base,
        max_panels=MAX_PANELS,
        cfie_alpha=CFIE_ALPHA,
        solver_method=SOLVER_METHOD,
        strict_quality_gate=True,
        compute_condition_number=True,
    )
    _verify_run_provenance(manifest)
    _verify_unit_input(unit, manifest)

    from grim_io import export_result_to_grim
    written = export_result_to_grim(
        result, str(out_path),
        source_path=str(snapshot.get("source_path", "") or ""),
        history=(f"run_hpc_monostatic.py pol={unit['polarization']} "
                 f"freq={unit['frequency_ghz']}GHz"),
    )
    actual_path = str(written[0]) if written else str(out_path)
    write_output_attestation(actual_path, attestation)
    _verify_run_provenance(manifest)
    _verify_unit_input(unit, manifest)
    return ("written", actual_path)


def _solve_and_export_star(args):
    # type: (tuple) -> tuple
    """Pool.imap_unordered entry point: unpack args and catch exceptions in-band.

    The full traceback string is returned (not just str(exc)) so the SLURM log
    shows where the failure happened, not just the message.
    """
    u, snap, mat_base, run_dir_str = args
    try:
        status, path = _solve_and_export(u, snap, mat_base, run_dir_str)
        return ("ok", status, path, u)
    except Exception:
        return ("err", traceback.format_exc(), "", u)


# ─── submit mode (user-invoked) ────────────────────────────────────────────

def _build_slurm(script_path, run_dir, job_index):
    # type: (Path, Path, int) -> str
    n_array = N_NODES
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=rcs_{run_dir.name}_j{job_index}",
        f"#SBATCH --array=0-{n_array - 1}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --partition={SLURM_PARTITION}",
        f"#SBATCH --output={run_dir}/logs/job{job_index}_task_%A_%a.out",
        f"#SBATCH --error={run_dir}/logs/job{job_index}_task_%A_%a.err",
    ]
    # Cores: explicit count or --exclusive (whole node; pool auto-detects via SLURM env).
    if CORES_PER_NODE is not None:
        lines.append(f"#SBATCH --cpus-per-task={CORES_PER_NODE}")
    else:
        lines.append("#SBATCH --exclusive")
    # Memory and time are optional; omitting them means no limit.
    if MEM_PER_NODE:
        lines.append(f"#SBATCH --mem={MEM_PER_NODE}")
    if SLURM_TIME:
        lines.append(f"#SBATCH --time={SLURM_TIME}")
    if SLURM_ACCOUNT:   lines.append(f"#SBATCH --account={SLURM_ACCOUNT}")
    if SLURM_QOS:       lines.append(f"#SBATCH --qos={SLURM_QOS}")
    if SLURM_MAIL_TYPE: lines.append(f"#SBATCH --mail-type={SLURM_MAIL_TYPE}")
    if SLURM_MAIL_USER: lines.append(f"#SBATCH --mail-user={SLURM_MAIL_USER}")
    for extra in SLURM_EXTRA_SBATCH:
        e = extra.strip()
        if not e:
            continue
        lines.append(e if e.startswith("#SBATCH") else f"#SBATCH {e}")

    lines += [
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(str(script_path.parent))}",
        *JOB_PROLOGUE,
        (f"exec {shlex.quote(PYTHON_EXE)} {shlex.quote(str(script_path))} "
         f"--worker {shlex.quote(str(run_dir))} {job_index} "
         f"${{SLURM_ARRAY_TASK_ID}}"),
        "",
    ]
    return "\n".join(lines)


def submit():
    # type: () -> None
    geometries = _discover_geometries()
    if not geometries:
        sys.exit("ERROR: no geometry files (*.geo) found under FRD_DIR or OPN_DIR.")

    pols = [p.strip().upper() for p in POLARIZATIONS if p and p.strip()]
    if not pols:            sys.exit("ERROR: POLARIZATIONS is empty.")
    if not FREQUENCIES_GHZ: sys.exit("ERROR: FREQUENCIES_GHZ is empty.")
    if not AZIMUTHS_DEG:    sys.exit("ERROR: AZIMUTHS_DEG is empty.")
    if (
        len(set(pols)) != len(pols)
        or not set(pols).issubset({"TM", "TE", "VV", "HH"})
    ):
        sys.exit(
            "ERROR: POLARIZATIONS must be a unique subset of TM/TE (VV/HH "
            "aliases are accepted)."
        )
    frequencies = [float(value) for value in FREQUENCIES_GHZ]
    if (
        not all(math.isfinite(value) and value > 0.0
                for value in frequencies)
        or len(set(frequencies)) != len(frequencies)
        or len({f"{value:.3f}" for value in frequencies})
        != len(frequencies)
    ):
        sys.exit(
            "ERROR: frequencies must be finite, positive, unique, and "
            "distinct at the 0.001 GHz output-name precision."
        )
    azimuths = [float(value) for value in AZIMUTHS_DEG]
    if (
        not all(math.isfinite(value) for value in azimuths)
        or len(set(azimuths)) != len(azimuths)
    ):
        sys.exit("ERROR: AZIMUTHS_DEG must be finite and unique.")
    if str(SOLVER_METHOD).strip().lower() not in {"auto", "direct"}:
        sys.exit(
            "ERROR: SOLVER_METHOD must be 'auto' or 'direct'; the legacy "
            "FMM/GMRES route is not a supported production solver."
        )
    if not math.isfinite(float(CFIE_ALPHA)) or float(CFIE_ALPHA) != 0.0:
        sys.exit(
            "ERROR: CFIE_ALPHA must be 0 for the current 2-D formulations; "
            "a nonzero value would not select a different operator."
        )
    if int(MAX_PANELS) < 1 or int(BLAS_THREADS_PER_WORKER) < 1:
        sys.exit("ERROR: MAX_PANELS and BLAS_THREADS_PER_WORKER must be >= 1.")
    if int(N_NODES) < 1 or int(N_JOBS) < 1:
        sys.exit("ERROR: N_NODES and N_JOBS must be >= 1.")

    stems = [g.stem for g in geometries]
    if len(stems) != len(set(stems)):
        sys.exit("ERROR: geometry stems must be unique; per-unit result names "
                 "would otherwise overwrite one another.")

    run_id  = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir = Path(OUTPUT_DIR).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir()
    (run_dir / "results").mkdir()

    # Freeze every geometry and adjacent mat.<flag> table inside this run before
    # any worker can start.  A later submission may replace its staging bundle;
    # queued/archived runs must remain immutable.
    frozen_geometries = []
    for index, geom in enumerate(geometries):
        inp = run_dir / "inputs" / f"{index:04d}_{geom.stem}"
        inp.mkdir(parents=True, exist_ok=False)
        frozen = inp / geom.name
        shutil.copy2(str(geom), str(frozen))
        for table in sorted(geom.parent.glob("mat.*")):
            if table.is_file():
                shutil.copy2(str(table), str(inp / table.name))
        frozen_geometries.append((geom, frozen))

    units = []  # type: List[Dict[str, Any]]
    from feature_sum import geometry_input_fingerprint
    for original, geom in frozen_geometries:
        input_fingerprint = geometry_input_fingerprint(
            str(geom), GEOMETRY_UNITS
        )
        for pol in pols:
            for f in FREQUENCIES_GHZ:
                units.append({
                    "geometry":      str(geom.resolve()),
                    "geometry_stem": geom.stem,
                    "geometry_original": str(original.resolve()),
                    "geometry_input_sha256": input_fingerprint,
                    "polarization":  pol,
                    "frequency_ghz": float(f),
                    "azimuths_deg":  [float(a) for a in AZIMUTHS_DEG],
                })

    source_driver = Path(__file__).resolve()
    manifest = {
        "schema":          "ghost.hpc.2d-run.v1",
        "run_id":          run_id,
        "created":         datetime.now().isoformat(),
        "frd_dir":         str(Path(FRD_DIR).resolve()),
        "opn_dir":         str(Path(OPN_DIR).resolve()),
        "output_dir":      str(run_dir),
        "frequencies_ghz": list(FREQUENCIES_GHZ),
        "azimuths_deg":    list(AZIMUTHS_DEG),
        "polarizations":   pols,
        "n_nodes":         int(N_NODES),
        "n_jobs":          int(N_JOBS),
        "n_slots":         int(N_NODES) * int(N_JOBS),
        "n_units":         len(units),
        "solver_source_sha256": _solver_source_fingerprint(),
        "runtime_environment_sha256":
            runtime_environment_fingerprint(),
        "solver_config": {
            "geometry_units":          GEOMETRY_UNITS,
            "solver_method":           SOLVER_METHOD,
            "cfie_alpha":              CFIE_ALPHA,
            "max_panels":              MAX_PANELS,
            "blas_threads_per_worker": BLAS_THREADS_PER_WORKER,
            "cores_per_node":          CORES_PER_NODE,
        },
        "units": units,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    script_path = run_dir / "driver_configured.py"
    shutil.copy2(str(source_driver), str(script_path))
    slurm_paths = []  # type: List[Path]
    for j in range(int(N_JOBS)):
        sp = run_dir / f"submit_job{j}.slurm"
        sp.write_text(_build_slurm(script_path, run_dir, j))
        sp.chmod(0o755)
        slurm_paths.append(sp)

    print("=" * 70)
    print("HPC monostatic RCS sweep")
    print("=" * 70)
    print(f"  Run dir       : {run_dir}")
    print(f"  Geometries    : {len(geometries)}")
    print(f"  Polarizations : {', '.join(pols)}")
    print(f"  Frequencies   : {len(FREQUENCIES_GHZ)}  "
          f"({min(FREQUENCIES_GHZ):g}-{max(FREQUENCIES_GHZ):g} GHz)")
    print(f"  Azimuths      : {len(AZIMUTHS_DEG)}")
    print(f"  Units total   : {len(units)}  (geom × freq × pol)")
    print(f"  Slots         : {N_JOBS} job(s) × {N_NODES} node(s) "
          f"= {int(N_JOBS) * int(N_NODES)} parallel nodes")
    cores_str = str(CORES_PER_NODE) if CORES_PER_NODE is not None else "auto (--exclusive)"
    mem_str   = str(MEM_PER_NODE) if MEM_PER_NODE else "unlimited"
    time_str  = str(SLURM_TIME) if SLURM_TIME else "unlimited"
    print(f"  Per node      : {cores_str} cores, {mem_str} RAM, "
          f"{time_str} walltime")
    print(f"  Slurm scripts : {len(slurm_paths)} files in {run_dir}")

    if not SUBMIT:
        print("\n  SUBMIT=False — submit manually with:")
        for sp in slurm_paths:
            print(f"    sbatch {sp}")
        return

    if shutil.which("sbatch") is None:
        print("\n  [warn] sbatch not on PATH. Submit manually:")
        for sp in slurm_paths:
            print(f"    sbatch {sp}")
        return

    for sp in slurm_paths:
        print(f"\n  Submitting: sbatch {sp.name}")
        res = subprocess.run(
            [_SBATCH, str(sp)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if res.returncode != 0:
            sys.exit(f"sbatch failed (exit {res.returncode}):\n"
                     f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}")
        print(f"  {res.stdout.strip()}")

    print(f"\nMonitor with:  squeue -u $USER")
    print(f"Outputs in:    {run_dir}/results/")


# ─── worker mode (invoked by SLURM) ────────────────────────────────────────

def _slot_units(units, job_index, node_index, n_nodes, n_jobs):
    # type: (List[Dict[str, Any]], int, int, int, int) -> List[Dict[str, Any]]
    n_slots = n_nodes * n_jobs
    slot = job_index * n_nodes + node_index
    if slot < 0 or slot >= n_slots:
        raise ValueError(f"slot {slot} out of range (0..{n_slots - 1})")
    return [u for i, u in enumerate(units) if i % n_slots == slot]


def worker(run_dir_str, job_index, node_index):
    # type: (str, int, int) -> None
    _pin_blas_threads(BLAS_THREADS_PER_WORKER)
    from geometry_io import parse_geometry, build_geometry_snapshot

    run_dir  = Path(run_dir_str).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    _verify_run_provenance(manifest)
    units    = manifest["units"]
    n_nodes  = int(manifest.get("n_nodes", 1))
    n_jobs   = int(manifest.get("n_jobs", 1))

    my_units = _slot_units(units, job_index, node_index, n_nodes, n_jobs)
    cores    = _detect_cores()
    worker_cap = cores if MAX_WORKERS_PER_NODE is None else max(1, int(MAX_WORKERS_PER_NODE))
    pool_size = max(1, min(cores, worker_cap, len(my_units))) if my_units else 1

    slot_id = job_index * n_nodes + node_index
    print("=" * 70)
    print(f"  Slot {slot_id}/{n_nodes * n_jobs - 1}  "
          f"(job={job_index}, node={node_index})")
    print(f"  Units assigned: {len(my_units)} of {len(units)} total")
    print(f"  Cores detected: {cores}   pool size: {pool_size}   "
          f"(BLAS threads/worker: {BLAS_THREADS_PER_WORKER})")
    print("=" * 70, flush=True)

    if not my_units:
        print("  No work for this slot.")
        return

    # Parse each geometry once; share snapshots across all units that use it.
    snapshots = {}  # type: Dict[str, Tuple[Dict[str, Any], str]]
    from feature_sum import geometry_input_fingerprint
    for u in my_units:
        gpath = u["geometry"]
        if gpath in snapshots:
            continue
        p = Path(gpath)
        if not p.is_file():
            sys.exit(f"Geometry missing on compute node: {p}")
        expected_input = str(u.get("geometry_input_sha256", ""))
        actual_input = geometry_input_fingerprint(
            str(p), str(manifest["solver_config"]["geometry_units"])
        )
        if not expected_input or actual_input != expected_input:
            sys.exit(
                f"Frozen geometry/material input fingerprint mismatch for "
                f"{p}; no field was solved."
            )
        title, segments, ibcs, dielectrics = parse_geometry(p.read_text())
        snap = build_geometry_snapshot(title, segments, ibcs, dielectrics)
        snap["source_path"] = str(p)
        snapshots[gpath] = (snap, str(p.parent))

    # Pin BLAS in parent; Pool initializer pins it in each worker too (works on
    # all start methods, unlike ProcessPoolExecutor which only got initializer
    # support in Python 3.7).
    t0 = time.time()
    n_done = n_skipped = n_failed = 0
    total = len(my_units)
    args_list = [
        (u, snapshots[u["geometry"]][0], snapshots[u["geometry"]][1], str(run_dir))
        for u in my_units
    ]
    # maxtasksperchild=1: each worker process is replaced after every unit, so
    # memory fragmentation / allocator growth from a big solve can never
    # accumulate across the hundreds of units of a long sweep.
    with Pool(processes=pool_size,
              initializer=_pin_blas_threads,
              initargs=(BLAS_THREADS_PER_WORKER,),
              maxtasksperchild=1) as pool:
        for idx, result in enumerate(
            pool.imap_unordered(_solve_and_export_star, args_list, chunksize=1),
            start=1,
        ):
            kind, a, b, u = result
            tag = (f"{u['polarization']} {u['frequency_ghz']:7.3f}GHz "
                   f"{u['geometry_stem']}")
            if kind == "ok":
                status, path = a, b
                if status == "skipped":
                    n_skipped += 1
                else:
                    n_done += 1
                print(f"  [{idx:3d}/{total}] {status:7s}  {tag}  -> "
                      f"{Path(path).name}", flush=True)
            else:
                n_failed += 1
                print(f"  [{idx:3d}/{total}] FAILED   {tag}", flush=True)
                for line in str(a).rstrip().splitlines():
                    print(f"      {line}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Slot complete. wrote={n_done}, skipped={n_skipped}, "
          f"failed={n_failed}.  {elapsed:.1f} s elapsed.")
    if n_failed:
        raise SystemExit(1)


# ─── entry point ───────────────────────────────────────────────────────────

def main():
    # type: () -> None
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument(
        "--worker", nargs=3, metavar=("RUN_DIR", "JOB_INDEX", "NODE_INDEX"),
        help="Internal: execute one array-task slice. Invoked by SLURM.",
    )
    args = ap.parse_args()
    if args.worker:
        worker(args.worker[0], int(args.worker[1]), int(args.worker[2]))
    else:
        submit()


if __name__ == "__main__":
    main()
