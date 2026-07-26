#!/usr/bin/env python3
"""
Frequency-band robustness of the line-expansion calibration.

The ring gate (validate_line_expansion.py) measured the local-coefficient
calibration constants psi_TE / psi_TM and the +/-20 deg validity window at ONE frequency
(6 GHz).  A reusable delta is only trustworthy across a wide band if those
numbers carry NO hidden frequency dependence.  This gate re-runs the ring-gate
physics at 1, 6 and 18 GHz.

To isolate the question, the geometry is scaled to a FIXED ELECTRICAL SIZE at
each frequency (body radius = 1.2 lambda, groove = lambda/3 x lambda/8, mesh
lambda/25).  So the electrical problem is identical and the ONLY variable is
absolute frequency:

  * psi is a phase offset between the 2D and BoR far-field conventions -> must
    be frequency-independent if the conventions have no explicit f-phase.
  * the 2D<->BoR MAGNITUDE match (currently <2.5 dB) tests that the two solvers'
    far-field k-power normalizations agree -> a real, non-trivial check.

PASS => the constants baked into line_expand.py (PSI_VV_DEG, PSI_HH_DEG) and
the validity window work at the three sampled frequencies for this
electrically scaled PEC benchmark. It is not proof for every frequency or
material. (Electrical-size robustness—a fixed PHYSICAL feature whose electrical
size changes across the band—is a separate question, not tested here.)

Reuses validate_line_expansion.py's tested geometry builders by setting its
module globals per frequency.  BoR solves are cached per frequency.

Run from tests/:  python3 validate_line_expansion_band.py
"""

import math
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "..")

import validate_line_expansion as RG                                   # noqa: E402
from bor_solver import solve_bor                                       # noqa: E402
from line_expand import (PSI_HH_DEG, PSI_VV_DEG, expand_perimeter,     # noqa: E402
                         seam_coefficients_from_2d,
                         surface_of_revolution_normal)
from validation_cache import bor_solver_sources, cache_path            # noqa: E402

C0 = 299_792_458.0
FREQS_GHZ = [1.0, 6.0, 18.0]
ASPECTS = np.arange(40.0, 140.1, 10.0)
CORE = (70.0, 110.0)
COUPON_LAM = 12.0     # match the coupon the baked psi constants were calibrated on
N_MODES = 22
PHI = np.arange(0.0, 180.1, 2.5)

MAG_TOL_DB = 2.5
PHASE_TOL_DEG = 25.0

_fails = []


def gate(label, ok, note=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {note}")
    if not ok:
        _fails.append(label)


def _wrap(x):
    return np.degrees(np.angle(np.exp(1j * np.radians(x))))


def _set_geometry_for(freq_ghz):
    """Scale the ring-gate builders to a fixed electrical size at this freq."""
    lam = C0 / (freq_ghz * 1e9)
    RG.LAM = lam
    RG.A_BODY = 1.2 * lam
    RG.L_BODY = 6.0 * lam
    RG.W_GROOVE = lam / 3.0
    RG.H_GROOVE = lam / 8.0
    RG.DS = lam / 25.0
    RG.Z0 = 0.0
    return lam


def _cached_bor(freq_ghz):
    lam = _set_geometry_for(freq_ghz)
    backend = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    key = cache_path(
        "band_bor",
        {
            "frequency_ghz": freq_ghz,
            "aspects_deg": ASPECTS,
            "coupon_lam": COUPON_LAM,
            "n_modes": N_MODES,
            "generatrix_smooth": RG.cylinder_gen(False),
            "generatrix_groove": RG.cylinder_gen(True),
        },
        bor_solver_sources(backend),
    )
    if os.path.exists(key):
        with open(key, "rb") as fh:
            return lam, pickle.load(fh)
    bor = {}
    for tag, grooved in (("smooth", False), ("groove", True)):
        t1 = time.time()
        gen = RG.cylinder_gen(grooved)
        bor[tag] = solve_bor(gen, freq_ghz * 1e9, ASPECTS, formulation="cfie",
                             cfie_alpha=0.5, n_modes=N_MODES, workers=4)
        print(f"     {tag}: {bor[tag]['n_unknowns']} unk, {bor[tag]['modes_used']} modes, "
              f"res {bor[tag]['linear_residual']:.1e}, {time.time() - t1:.0f} s")
    with open(key, "wb") as fh:
        pickle.dump(bor, fh)
    return lam, bor


print("=" * 78)
print("Line-expansion frequency-band robustness (fixed electrical size)")
print(f"  frequencies: {FREQS_GHZ} GHz   body a=1.2 lambda (ka~7.5), groove lam/3 x lam/8")
print(f"  baked constants: psi_VV={PSI_VV_DEG:+.1f}  psi_HH={PSI_HH_DEG:+.1f}  (from 6 GHz gate)")
print("=" * 78)

rows = []
t0 = time.time()
for f in FREQS_GHZ:
    print(f"\n--- {f:g} GHz ---")
    lam, bor = _cached_bor(f)
    _set_geometry_for(f)                       # ensure globals match this freq
    dF = {"vv": np.asarray(bor["groove"]["amp_vv"]) - np.asarray(bor["smooth"]["amp_vv"]),
          "hh": np.asarray(bor["groove"]["amp_hh"]) - np.asarray(bor["smooth"]["amp_hh"])}

    coef = seam_coefficients_from_2d(
        RG.coupon(COUPON_LAM, True, thickness=0.9 * lam),
        RG.coupon(COUPON_LAM, False, thickness=0.9 * lam),
        f, PHI, geometry_units="meters", label=f"{f:g}GHz coupon")

    per = RG.ring_perimeter(RG.A_BODY, 0.0)
    nfn = surface_of_revolution_normal(RG.cylinder_gen(False))
    th = np.radians(ASPECTS)
    dirs = np.column_stack([np.sin(th), np.zeros_like(th), np.cos(th)])
    exp = expand_perimeter(
        per, coef, nfn, dirs,
        psi_tm_deg=PSI_HH_DEG, psi_te_deg=PSI_VV_DEG)

    core = (ASPECTS >= CORE[0]) & (ASPECTS <= CORE[1])
    row = {"f": f}
    for ch in ("vv", "hh"):
        r = dF[ch][core] / exp[f"F_{ch}"][core]
        mag = np.abs(20 * np.log10(np.abs(r)))
        phase = np.abs(np.degrees(np.angle(r)))
        # widest symmetric window (both mag and phase) about broadside
        win = 0.0
        for half in np.arange(5.0, 55.0, 5.0):
            sel = np.abs(ASPECTS - 90.0) <= half + 1e-9
            rr = dF[ch][sel] / exp[f"F_{ch}"][sel]
            if (np.abs(20 * np.log10(np.abs(rr))).max() <= MAG_TOL_DB
                    and np.abs(np.degrees(np.angle(rr))).max() <= PHASE_TOL_DEG):
                win = half
            else:
                break
        row[ch] = {"phase": phase.max(), "mag": mag.max(), "win": win}
    rows.append(row)
    print(f"     VV: phase<={row['vv']['phase']:5.1f}  mag<={row['vv']['mag']:.2f}dB  "
          f"window +/-{row['vv']['win']:.0f}   |   "
          f"HH: phase<={row['hh']['phase']:5.1f}  mag<={row['hh']['mag']:.2f}dB  "
          f"window +/-{row['hh']['win']:.0f}")

print(f"\n{'=' * 78}")
print("Summary")
print(f"  {'freq':>6}  {'VV phase':>8} {'VV mag':>7} {'VV win':>7}   "
      f"{'HH phase':>8} {'HH mag':>7} {'HH win':>7}")
for r in rows:
    print(f"  {r['f']:6g}  {r['vv']['phase']:8.1f} {r['vv']['mag']:6.2f}dB {r['vv']['win']:6.0f}   "
          f"{r['hh']['phase']:8.1f} {r['hh']['mag']:6.2f}dB {r['hh']['win']:6.0f}")

print("\nVerdict")
for ch, name in (("vv", "VV"), ("hh", "HH")):
    gate(f"{name} calibrated residual phase <= {PHASE_TOL_DEG:g} deg "
         "at all sampled frequencies",
         max(r[ch]["phase"] for r in rows) < PHASE_TOL_DEG,
         f"(worst {max(r[ch]['phase'] for r in rows):.1f} deg)")
    gate(f"{name} magnitude match <= {MAG_TOL_DB} dB at all freqs",
         max(r[ch]["mag"] for r in rows) < MAG_TOL_DB,
         f"(worst {max(r[ch]['mag'] for r in rows):.2f} dB)")
    gate(f"{name} validity window >= 15 deg at all freqs",
         min(r[ch]["win"] for r in rows) >= 15.0,
         f"(min +/-{min(r[ch]['win'] for r in rows):.0f} deg)")

print(f"\n{'=' * 78}")
print("ALL GATES PASSED" if not _fails else f"{len(_fails)} FAILED",
      f"  ({time.time() - t0:.0f} s)")
for fl in _fails:
    print(f"   FAILED: {fl}")
print("=" * 78)
sys.exit(1 if _fails else 0)
