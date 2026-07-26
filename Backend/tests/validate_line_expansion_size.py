#!/usr/bin/env python3
"""
Electrical-SIZE robustness of the line-expansion delta (companion to the
frequency-band gate).

A real feature is a fixed PHYSICAL size, so across a wide band its ELECTRICAL
size changes: a panel gap that is ~lambda/12 at the low end is ~lambda or more
at the high end.  This gate holds the frequency fixed (6 GHz) and sweeps the
groove width across that electrical range, mapping where the delta stays
reliable:

  * the production-calibrated residual phase must stay bounded;
  * the 2D<->BoR magnitude match and the validity window map the usable range.

Expectations, and what the sweep is FOR: at very small electrical size the
delta approaches the numerical floor (accuracy noisy but the feature is also
negligible); at ~lambda and above the groove stops being a clean local edge
(its own cavity/curvature structure appears) and the flat-coupon substitute
degrades.  Somewhere in between is the reliable band -- this prints it.

Ground truth is again the axisymmetric BoR groove (exact).  The clean body is
identical for every groove size, so it is solved ONCE.

Run from tests/:  python3 validate_line_expansion_size.py
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
FREQ_GHZ = 6.0
LAM = C0 / (FREQ_GHZ * 1e9)
# fixed body (same electrical body as the ring/band gates)
A_BODY = 1.2 * LAM
L_BODY = 6.0 * LAM
DS = LAM / 25.0
# groove widths to sweep, in wavelengths; depth keeps the base aspect ratio
WIDTHS_LAM = [0.08, 0.17, 0.33, 0.67, 1.0]
DEPTH_PER_WIDTH = (1.0 / 8.0) / (1.0 / 3.0)          # base was lam/3 x lam/8
ASPECTS = np.arange(40.0, 140.1, 10.0)
CORE = (70.0, 110.0)
COUPON_LAM = 12.0
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


def _set_globals(w_lam):
    RG.LAM = LAM
    RG.A_BODY = A_BODY
    RG.L_BODY = L_BODY
    RG.DS = DS
    RG.Z0 = 0.0
    RG.W_GROOVE = w_lam * LAM
    RG.H_GROOVE = w_lam * LAM * DEPTH_PER_WIDTH


def _cached(tag, builder):
    backend = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generatrix = builder()
    key = cache_path(
        f"gsweep_{tag}",
        {
            "frequency_ghz": FREQ_GHZ,
            "aspects_deg": ASPECTS,
            "n_modes": N_MODES,
            "generatrix": generatrix,
        },
        bor_solver_sources(backend),
    )
    if os.path.exists(key):
        with open(key, "rb") as fh:
            return pickle.load(fh)
    t1 = time.time()
    res = solve_bor(generatrix, FREQ_GHZ * 1e9, ASPECTS, formulation="cfie",
                    cfie_alpha=0.5, n_modes=N_MODES, workers=4)
    print(f"     {tag}: {res['n_unknowns']} unk, {res['modes_used']} modes, "
          f"res {res['linear_residual']:.1e}, {time.time() - t1:.0f} s")
    with open(key, "wb") as fh:
        pickle.dump(res, fh)
    return res


print("=" * 82)
print("Line-expansion electrical-SIZE robustness (fixed 6 GHz, sweep groove width)")
print(f"  body a=1.2 lambda, groove depth = {DEPTH_PER_WIDTH:.3f} x width")
print(f"  baked constants: psi_VV={PSI_VV_DEG:+.1f}  psi_HH={PSI_HH_DEG:+.1f}")
print("=" * 82)

t0 = time.time()
_set_globals(WIDTHS_LAM[0])
print("\nClean body (solved once, shared by every groove size)")
smooth = _cached("smooth", lambda: RG.cylinder_gen(False))
body_vv = 10 * np.log10(np.asarray(smooth["sigma_vv"]) + 1e-30)
nfn = surface_of_revolution_normal(RG.cylinder_gen(False))
th = np.radians(ASPECTS)
dirs = np.column_stack([np.sin(th), np.zeros_like(th), np.cos(th)])
core = (ASPECTS >= CORE[0]) & (ASPECTS <= CORE[1])

rows = []
for w in WIDTHS_LAM:
    print(f"\n--- groove width {w:g} lambda ---")
    _set_globals(w)
    groove = _cached(f"g{w:.3f}", lambda: RG.cylinder_gen(True))
    dF = {"vv": np.asarray(groove["amp_vv"]) - np.asarray(smooth["amp_vv"]),
          "hh": np.asarray(groove["amp_hh"]) - np.asarray(smooth["amp_hh"])}
    feat_dbsm = 10 * np.log10(4 * math.pi * np.abs(dF["vv"]) ** 2 + 1e-30)

    coef = seam_coefficients_from_2d(
        RG.coupon(COUPON_LAM, True, thickness=0.9 * LAM),
        RG.coupon(COUPON_LAM, False, thickness=0.9 * LAM),
        FREQ_GHZ, PHI, geometry_units="meters", label=f"{w:g}lam groove")
    per = RG.ring_perimeter(A_BODY, 0.0)
    exp = expand_perimeter(
        per, coef, nfn, dirs,
        psi_tm_deg=PSI_HH_DEG, psi_te_deg=PSI_VV_DEG)

    row = {"w": w, "feat": feat_dbsm[core].max()}
    for ch in ("vv", "hh"):
        r = dF[ch][core] / exp[f"F_{ch}"][core]
        mag = np.abs(20 * np.log10(np.abs(r))).max()
        phase = np.abs(np.degrees(np.angle(r))).max()
        win = 0.0
        for half in np.arange(5.0, 55.0, 5.0):
            sel = np.abs(ASPECTS - 90.0) <= half + 1e-9
            rr = dF[ch][sel] / exp[f"F_{ch}"][sel]
            if (np.abs(20 * np.log10(np.abs(rr))).max() <= MAG_TOL_DB
                    and np.abs(np.degrees(np.angle(rr))).max() <= PHASE_TOL_DEG):
                win = half
            else:
                break
        row[ch] = {"phase": phase, "mag": mag, "win": win}
    rows.append(row)
    print(f"     feature {row['feat']:+.1f} dBsm (body "
          f"{body_vv[core].min():+.1f}..{body_vv[core].max():+.1f})   "
          f"VV phase<={row['vv']['phase']:5.1f} mag<={row['vv']['mag']:.2f} win+/-{row['vv']['win']:.0f}  "
          f"HH phase<={row['hh']['phase']:5.1f} mag<={row['hh']['mag']:.2f} win+/-{row['hh']['win']:.0f}")

print(f"\n{'=' * 82}")
print("Summary  (feature level vs body, and calibration vs groove electrical size)")
print(f"  {'w/lam':>6} {'feat dBsm':>10}   {'VVphase':>8} {'VVmag':>6} {'VVwin':>6}   "
      f"{'HHphase':>8} {'HHmag':>6} {'HHwin':>6}   {'reliable?':>9}")
for r in rows:
    ok = (r['vv']['phase'] < PHASE_TOL_DEG
          and r['hh']['phase'] < PHASE_TOL_DEG
          and r['vv']['mag'] < MAG_TOL_DB and r['hh']['mag'] < MAG_TOL_DB
          and min(r['vv']['win'], r['hh']['win']) >= 20.0)
    print(f"  {r['w']:6g} {r['feat']:+10.1f}   "
          f"{r['vv']['phase']:8.1f} {r['vv']['mag']:6.2f} {r['vv']['win']:6.0f}   "
          f"{r['hh']['phase']:8.1f} {r['hh']['mag']:6.2f} {r['hh']['win']:6.0f}   "
          f"{'YES' if ok else 'no':>9}")

reliable = [r["w"] for r in rows
            if r['vv']['phase'] < PHASE_TOL_DEG
            and r['hh']['phase'] < PHASE_TOL_DEG
            and r['vv']['mag'] < MAG_TOL_DB and r['hh']['mag'] < MAG_TOL_DB
            and min(r['vv']['win'], r['hh']['win']) >= 20.0]

print("\nVerdict")
gate("a usable electrical-size band exists (>= 3 widths reliable)", len(reliable) >= 3,
     f"(reliable widths: {reliable} lambda)")
gate("the base calibration size (lam/3) is reliable",
     any(abs(w - 1.0 / 3.0) < 0.02 for w in reliable),
     f"(0.33 lambda {'in' if any(abs(w - 1/3) < 0.02 for w in reliable) else 'NOT in'} band)")
if reliable:
    print(f"\n   ==> reliable groove electrical size: {min(reliable):g} .. {max(reliable):g} lambda "
          f"(at 6 GHz)")

print(f"\n{'=' * 82}")
print("ALL GATES PASSED" if not _fails else f"{len(_fails)} FAILED",
      f"  ({time.time() - t0:.0f} s)")
for fl in _fails:
    print(f"   FAILED: {fl}")
print("=" * 82)
sys.exit(1 if _fails else 0)
