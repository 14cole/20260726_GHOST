#!/usr/bin/env python3
"""
Run the line-expansion / feature-signature validation suite and summarise.

    cd tests && python3 run_feature_gates.py          # feature suite (default)
    cd tests && python3 run_feature_gates.py --all     # + foundational batteries
    cd tests && python3 run_feature_gates.py --fast     # skip the slow BoR sweeps

Each gate is a standalone script that prints its own PASS/FAIL lines and exits
0 (all passed) or 1 (something failed).  This runner invokes them in order,
fast first, and prints one summary table with per-gate status and wall time.
Exit code is nonzero if any gate fails.

BoR solves inside the gates are cached to hidden .pkl files after the first
run, so a second pass of the whole suite is quick.
"""

import subprocess
import sys
import time
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# (script, label, slow?)  slow = does uncached BoR solves on first run
FEATURE = [
    ("test_feature_pipeline_safety.py", "feature/body production safety regressions", False),
    ("test_component_schema_safety.py", "component/delta/pattern schema safety",       False),
    ("test_grim_io_safety.py",          "GRIM field/schema fail-closed safety",        False),
    ("test_workflow_cache_provenance.py", "workflow cache/provenance safety",          False),
    ("validate_line_polarization.py",   "signed seam Jones projection",              False),
    ("validate_wing.py",                "wing anchor (flat plate, analytic)",        False),
    ("validate_corner.py",              "wing-body dihedral corner estimate",        False),
    ("validate_point_scatterer.py",     "3D-MoM delta pattern placed at a point",    False),
    ("validate_occluder.py",            "STL body-shadowing (geometric occlusion)",  False),
    ("validate_delta_library.py",       "filename-indexed delta library + arcs",     False),
    ("validate_join_datasets.py",       "production filenames: join + OPN/FRD pair", False),
    ("validate_grim_compat.py",         "GRIM_Revised_2 interop (skips if absent)",  False),
    ("validate_line_expansion.py",      "ring gate (core physics vs BoR truth)",     True),
    ("validate_feature_sum.py",         "pipeline G0-G6 (delta/sum/export/wing)",    True),
    ("validate_full_workflow.py",       "full chain -> multi-freq radar .grim",      True),
    ("validate_line_expansion_band.py", "sampled scaled-frequency checks (1/6/18 GHz)", True),
    ("validate_line_expansion_size.py", "electrical-size robustness (groove sweep)", True),
    ("validate_line_expansion_coated.py", "coated-feature cross-formulation anchor",  True),
    ("validate_coupon_bakeoff.py",      "coupon geometry sensitivity",                True),
]

# the solver gates the feature work rests on (longer; run with --all)
FOUNDATION = [
    ("test_2d_safety_regressions.py", "2D dispatch/preflight safety regressions",      False),
    ("test_2d_material_safety_regressions.py",
                                          "2D passive-material/null safety",           False),
    ("test_2d_numerical_safety_regressions.py",
                                          "2D material-mesh/FMM convergence safety",   False),
    ("test_2d_quality_fail_closed.py", "2D residual/quality fail-closed safety",        False),
    ("test_bor_safety_regressions.py", "BoR dispatch/material/mode safety",             False),
    ("test_bor_az_el_grid_safety.py", "BoR radar-grid basis/support safety",             False),
    ("validate_mie.py",            "2D Mie regression (PEC/dielectric/lossy/coated)", False),
    ("validate_2d_resonance.py",   "2D interior-resonance Mie regression",             False),
    ("validate_mixed.py",          "2D mixed-material limiting cases",                 False),
    ("validate_orientation_check.py", "2D geometry orientation checks",               False),
    ("validate_preflight_holes.py", "2D topology/hole preflight",                      False),
    ("validate_mie_sphere.py",     "BoR Mie sphere series",                           False),
    ("validate_bor_phase1.py",     "BoR PEC EFIE + 2D<->BoR strip cross-check",        True),
    ("validate_bor_phase2.py",     "BoR CFIE + IBC",                                   True),
    ("validate_bor_phase3.py",     "BoR dielectric / coated (PMCHWT)",                 True),
    ("validate_bor_phase4.py",     "BoR integration / az-el / adaptive sweep",         True),
    ("validate_bor_streaming.py",  "BoR streamed/table/native equivalence + Mie",       True),
    ("validate_bor_multilayer.py", "BoR multilayer spheres + independent Mie",          True),
    ("validate_bor_junctions.py",  "BoR material-junction limiting cases",              True),
    ("validate_bor_banded.py",     "BoR banded/partial-coating convergence",            True),
]


def _reported_failure_lines(text):
    """Failure summaries printed by legacy validators that may still exit 0."""
    found = []
    for line in str(text).splitlines():
        if re.search(r"\[\s*FAIL\s*\]", line, flags=re.IGNORECASE) \
                or re.search(r"\bGATES?\s+FAILED\b", line,
                             flags=re.IGNORECASE) \
                or re.search(r"\bFAILED\s*:", line, flags=re.IGNORECASE):
            found.append(line)
            continue
        count = re.search(r"\b(\d+)\s+FAILED\b", line,
                          flags=re.IGNORECASE)
        if count and int(count.group(1)) > 0:
            found.append(line)
    return found


def run(scripts, skip_slow):
    results = []
    for script, label, slow in scripts:
        if slow and skip_slow:
            print(f"  SKIP (--fast)  {script}")
            results.append((script, label, "skip", 0.0))
            continue
        print(f"  running        {script} ...", flush=True)
        t0 = time.time()
        p = subprocess.run(
            [sys.executable, "-u", os.path.join(HERE, script)],
            cwd=HERE, capture_output=True, text=True)
        dt = time.time() - t0
        reported_failures = _reported_failure_lines(p.stdout + "\n" + p.stderr)
        ok = p.returncode == 0 and not reported_failures
        # surface the gate's own final summary line(s)
        tail = [ln for ln in p.stdout.splitlines()
                if "PASS" in ln or "FAIL" in ln or "GATES" in ln]
        status = "PASS" if ok else "FAIL"
        results.append((script, label, status, dt))
        print(f"  {status}  ({dt:.0f} s)  {script}")
        if not ok:
            # show the failing lines to make the summary actionable
            for ln in (reported_failures or tail)[-8:]:
                print(f"       {ln}")
            if not tail:
                print("       (no PASS/FAIL lines captured; stderr tail:)")
                for ln in p.stderr.splitlines()[-6:]:
                    print(f"       {ln}")
    return results


def main():
    args = set(sys.argv[1:])
    skip_slow = "--fast" in args
    suite = FEATURE + (FOUNDATION if "--all" in args else [])

    print("=" * 74)
    print("Feature-signature validation suite" + ("  (+ foundation)" if "--all" in args else ""))
    print("=" * 74)
    t0 = time.time()
    results = run(suite, skip_slow)

    print("\n" + "=" * 74)
    print("Summary")
    print("=" * 74)
    n_pass = n_fail = n_skip = 0
    for script, label, status, dt in results:
        mark = {"PASS": "PASS", "FAIL": "FAIL", "skip": "----"}[status]
        n_pass += status == "PASS"
        n_fail += status == "FAIL"
        n_skip += status == "skip"
        t = f"{dt:5.0f}s" if status != "skip" else "     "
        print(f"  [{mark}] {t}  {label}")
    print("-" * 74)
    print(f"  {n_pass} passed, {n_fail} failed, {n_skip} skipped   "
          f"({time.time() - t0:.0f} s total)")
    print("=" * 74)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
