#!/usr/bin/env python3
"""
Wing / fin line-expansion anchor gate: a flat rectangular plate.

A wing uses the SAME line-expansion integral as a surface feature, but the
coefficient is the FULL 2D amplitude of the airfoil cross-section (not a
featured-minus-smooth delta) and the line is the OPEN span (root -> tip).  The
clean anchor is an electrically large flat plate, whose standard physical-
optics broadside reference is

    sigma = 4 pi A^2 / lambda^2 ,   A = span L x chord c   (physical optics)

and whose in-span pattern is a sinc^2.  Three checks:

  A  line-expansion IDENTITY: sigma_3d(expansion) == (2 L^2 / lambda) sigma_2d
     of the SAME 2D strip solve.  This is exact by construction (expand's
     1/(4pi) prefactor produces the 2L^2/lambda strip factor) and is
     psi-independent -- it validates the normalization for a full-object
     coefficient.
  B  ANALYTIC anchor: sigma_3d(broadside) ~= 4 pi A^2 / lambda^2, to physical-
     optics accuracy for an electrically large plate.
  C  in-span PATTERN: sigma(beta) == sigma_broadside * sinc^2(k L sin beta),
     validating the span phase integral (sidelobes / nulls).

No BoR solve is needed -- the plate anchor is analytic.  Fast.

Run from tests/:  python3 validate_wing.py
"""

import math
import sys

import numpy as np

sys.path.insert(0, "..")

from line_expand import coefficients_from_2d, expand_perimeter          # noqa: E402
from rcs_solver import solve_monostatic_rcs_2d                          # noqa: E402

C0 = 299_792_458.0
FREQ_GHZ = 6.0
LAM = C0 / (FREQ_GHZ * 1e9)
CHORD = 4.0 * LAM               # c
SPAN = 8.0 * LAM                # L
THICK = LAM / 50.0              # plate thickness (approx zero-thickness strip)
DS = LAM / 25.0
K = 2.0 * math.pi / LAM

_fails = []


def gate(label, ok, note=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {note}")
    if not ok:
        _fails.append(label)


def plate_cross_section():
    """Thin PEC rectangle, chord along x (width CHORD), thickness THICK along y,
    outer faces at +/-THICK/2.  Drawn CW so left-of-travel normals face air.
    The +y face normal is the broadside (normal-incidence) direction."""
    c, t = CHORD, THICK
    nx = int(math.ceil(c / DS))
    xs = np.linspace(-c / 2, c / 2, nx + 1)
    nt = max(2, int(math.ceil(t / DS)))
    ys = np.linspace(t / 2, -t / 2, nt + 1)
    pts = [(x, t / 2) for x in xs]                       # top face  L->R (+x)
    pts += [(c / 2, y) for y in ys[1:]]                  # right edge down
    pts += [(x, -t / 2) for x in xs[::-1][1:]]           # bottom face R->L
    pts += [(-c / 2, y) for y in ys[::-1][1:-1]]         # left edge up (skip close)
    pairs = [{"x1": pts[i][0], "y1": pts[i][1],
              "x2": pts[(i + 1) % len(pts)][0], "y2": pts[(i + 1) % len(pts)][1]}
             for i in range(len(pts))]
    seg = {"name": "plate", "seg_type": "2",
           "properties": ["2", "0", "0", "0", "0"], "point_pairs": pairs}
    return {"title": "plate", "segments": [seg], "ibcs": [], "dielectrics": []}


print("=" * 74)
print(f"Wing anchor gate — flat plate {SPAN / LAM:g}x{CHORD / LAM:g} lambda "
      f"@ {FREQ_GHZ} GHz")
area = SPAN * CHORD
sig_po = 4.0 * math.pi * area ** 2 / LAM ** 2
print(f"  area {area:.4f} m^2   analytic 4piA^2/lam^2 = {sig_po:.3f} m^2 "
      f"({10 * math.log10(sig_po):+.2f} dBsm)")
print("=" * 74)

snap = plate_cross_section()

print("\n1. 2D strip cross-section solve")
phi = np.arange(40.0, 140.1, 2.0)
coef = coefficients_from_2d(snap, FREQ_GHZ, phi, geometry_units="meters",
                            label="flat plate strip")
# sigma_2d at broadside (phi = 90) per pol, directly from the solver
s2d = {}
for pol in ("TM", "TE"):
    r = solve_monostatic_rcs_2d(snap, [FREQ_GHZ], [90.0], pol, geometry_units="meters")
    s2d[pol] = float(r["samples"][0]["rcs_linear"])
print(f"   sigma_2d(broadside): TM {s2d['TM']:.4e}  TE {s2d['TE']:.4e} m "
      f"(strip PO k*c^2 = {K * CHORD ** 2:.4e})")

print("\n2. Line expansion along the span")
per = np.array([[[0.0, 0.0, -SPAN / 2], [0.0, 0.0, SPAN / 2]]])
def nfn(pts): return np.tile([0.0, 1.0, 0.0], (len(np.atleast_2d(pts)), 1))
d_bs = np.array([[0.0, 1.0, 0.0]])                       # broadside look
F = expand_perimeter(per, coef, nfn, d_bs)               # psi=0 (magnitude only)
sig3d = {"VV": 4 * math.pi * abs(F["F_vv"][0]) ** 2,
         "HH": 4 * math.pi * abs(F["F_hh"][0]) ** 2}
# VV<->TM, HH<->TE for this plate orientation (span || global z)
strip = {"VV": (2 * SPAN ** 2 / LAM) * s2d["TM"],
         "HH": (2 * SPAN ** 2 / LAM) * s2d["TE"]}
print(f"   {'chan':>4} {'sigma_3d':>12} {'(2L^2/lam)s2d':>14} {'4piA^2/lam^2':>13}")
for ch in ("VV", "HH"):
    print(f"   {ch:>4} {sig3d[ch]:12.4e} {strip[ch]:14.4e} {sig_po:13.4e}  "
          f"({10 * math.log10(sig3d[ch]):+.2f} dBsm)")

print("\n3. Verdict")
for ch in ("VV", "HH"):
    d_id = 10 * math.log10(sig3d[ch] / strip[ch])
    gate(f"{ch} expansion == (2L^2/lam) sigma_2d (line-expansion identity)",
         abs(d_id) < 0.05, f"({d_id:+.4f} dB)")
    d_po = 10 * math.log10(sig3d[ch] / sig_po)
    gate(f"{ch} broadside == 4piA^2/lambda^2 (analytic, PO)",
         abs(d_po) < 1.0, f"({d_po:+.2f} dB vs PO)")

print("\n4. In-span pattern (sinc^2)")
betas = np.arange(0.0, 30.1, 2.0)
d_span = np.column_stack([np.zeros_like(betas), np.cos(np.radians(betas)),
                          np.sin(np.radians(betas))])
Fp = expand_perimeter(per, coef, nfn, d_span)
sig_vv = 4 * math.pi * np.abs(Fp["F_vv"]) ** 2
x = K * SPAN * np.sin(np.radians(betas))
sinc2 = (np.sin(x) / np.where(x == 0, 1.0, x)) ** 2
sinc2[0] = 1.0
model = sig_vv[0] * sinc2
# Pure sinc^2 is the single-coefficient (uniform line) result; it holds over the
# main lobe + first sidelobe.  In the DEEP sidelobes (> ~20 dB down) the VV
# channel deviates because e_vv rotates off broadside and mixes the TM and TE
# strip coefficients (different phases) -- real physics the pure sinc^2 ignores.
# The span PHASE integral itself is confirmed by the null positions (m*lam/2L).
mask = model > sig_vv[0] * 10 ** (-2.0)             # main lobe + first sidelobe
err = np.abs(10 * np.log10(np.where(mask, sig_vv / np.where(model > 0, model, 1), 1.0)))
worst = float(err[mask].max())
# null positions: analytic nulls at sin(beta) = m*lam/(2L)
nulls_pred = [math.degrees(math.asin(m * LAM / (2 * SPAN)))
              for m in range(1, 5) if m * LAM / (2 * SPAN) < 1]
print(f"   {'beta':>6} {'sigma_VV dBsm':>13} {'sinc^2 model':>13} {'err dB':>8}")
for i, b in enumerate(betas):
    tag = "  *" if mask[i] else ("  null" if any(abs(b - n) <= 1.0 for n in nulls_pred) else "")
    print(f"   {b:6.0f} {10 * math.log10(sig_vv[i] + 1e-30):13.2f} "
          f"{10 * math.log10(model[i] + 1e-30):13.2f} "
          f"{err[i]:8.2f}{tag}")
print(f"   predicted nulls (m*lam/2L): {[f'{n:.1f}' for n in nulls_pred]} deg")
gate("in-span main lobe + first sidelobe match sinc^2",
     worst < 0.6, f"(worst {worst:.3f} dB within 20 dB of peak)")
# each predicted null has a local minimum within +/-1 deg
dips_ok = all(np.min(np.abs(betas - n)) <= 1.0 and
              10 * math.log10(sig_vv[int(np.argmin(np.abs(betas - n)))] + 1e-30)
              < 10 * math.log10(sig_vv[0]) - 12
              for n in nulls_pred)
gate("span pattern has nulls at m*lambda/2L (phase integral correct)", dips_ok,
     f"(nulls near {[f'{n:.1f}' for n in nulls_pred]} deg)")

print(f"\n{'=' * 74}")
print("ALL GATES PASSED" if not _fails else f"{len(_fails)} FAILED")
for f in _fails:
    print(f"   FAILED: {f}")
print("=" * 74)
sys.exit(1 if _fails else 0)
