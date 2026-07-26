#!/usr/bin/env python3
"""
Point-scatterer gate: a precomputed 3-D delta pattern placed at one coordinate
(feature_sum.point_scatterer_amplitude) -- e.g. a blind cavity solved by an
external 3-D MoM as featured - clean.

No external solver is needed: SYNTHETIC delta patterns with known properties
exercise the mechanics (the absolute cavity physics is the user's MoM's job;
this validates placement, orientation, polarization rotation, shadowing,
interpolation, and integration).

  P0  placement phase: translating the cavity by dr shifts phase by 2k d.dr.
  P1  shadowing: looks behind the aperture (d.n <= 0) contribute nothing.
  P2  ISOTROPIC co-pol delta -> energy ||S||_F preserved through the rotation
      AND no spurious cross-pol (co-pol in -> co-pol out).
  P3  diag(1,-1) delta -> cross-pol VH appears off-alignment (real depolarization
      survives the rotation), energy still preserved.
  P4  interpolation: a non-constant pattern read at a grid node returns the
      stored value.
  P5  integration: a point scatterer added to sum_features / export_radar_grim
      superposes onto the body (full == body + point).

Run from tests/:  python3 validate_point_scatterer.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, "..")

from feature_sum import (_attitude, _direction, export_radar_grim,       # noqa: E402
                         point_pattern_convention_metadata,
                         point_scatterer_amplitude, sum_features)

C0 = 299_792_458.0
FREQ = 6.0
K = 2 * math.pi * FREQ * 1e9 / C0
_fails = []


def gate(label, ok, note=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {note}")
    if not ok:
        _fails.append(label)


def pattern(vv, hh, vh, az=np.arange(0, 360.1, 10.0), el=np.arange(0, 90.1, 10.0)):
    """Build a synthetic delta pattern; vv/hh/vh are scalars or (naz,nel) arrays."""
    naz, nel = len(az), len(el)
    amp = np.zeros((naz, nel, 1, 3), complex)
    amp[:, :, 0, 0] = vv
    amp[:, :, 0, 1] = hh
    amp[:, :, 0, 2] = vh
    return {"azimuths": az, "elevations": el, "frequencies": [FREQ],
            "polarizations": ["VV", "HH", "VH"], "amp": amp,
            **point_pattern_convention_metadata()}


def frob(F):
    return (np.abs(F["F_vv"]) ** 2 + np.abs(F["F_hh"]) ** 2 + 2 * np.abs(F["F_vh"]) ** 2)


print("=" * 74)
print("Point-scatterer (precomputed 3-D delta pattern) gate")
print("=" * 74)

# a cavity with aperture facing +y, at a location off the axis
NORMAL = np.array([0.0, 1.0, 0.0])
LOC = np.array([0.10, 0.05, 0.03])
iso = pattern(1.0 + 0j, 1.0 + 0j, 0.0)                     # isotropic co-pol
# lit look directions (into the aperture hemisphere, d.n > 0)
th = np.radians([70, 80, 90, 100, 110])
dirs = np.column_stack([np.sin(th) * 0.6, np.abs(np.cos(th)) + 0.5, np.sin(th) * 0.3])
dirs = dirs / np.linalg.norm(dirs, axis=1)[:, None]

print("\nP0 placement phase")
F1 = point_scatterer_amplitude(iso, LOC, NORMAL, dirs, FREQ)
dr = np.array([0.0, 0.0, 0.04])
F2 = point_scatterer_amplitude(iso, LOC + dr, NORMAL, dirs, FREQ)
got = np.angle(F2["F_vv"] / F1["F_vv"])
pred = 2 * K * (dirs @ dr)
dphi = np.abs(np.angle(np.exp(1j * (got - pred))))
gate("P0 translation shifts phase by 2k d.dr", float(np.nanmax(dphi)) < 1e-6,
     f"(worst {np.nanmax(dphi):.1e} rad)")

print("\nP1 shadowing")
back = np.array([[0.3, -1.0, 0.1], [-0.2, -0.8, 0.5]])     # d.n < 0
Fb = point_scatterer_amplitude(iso, LOC, NORMAL, back, FREQ)
gate("P1 looks behind the aperture return zero",
     float(np.max(np.abs(Fb["F_vv"]) + np.abs(Fb["F_hh"]) + np.abs(Fb["F_vh"]))) < 1e-30,
     "(shadowed = 0)")

print("\nP2 isotropic co-pol: energy preserved, no spurious cross-pol")
gate("P2a Frobenius ||S||_F == 2 (rotation is unitary)",
     float(np.max(np.abs(frob(F1) - 2.0))) < 1e-9, f"(worst dev {np.max(np.abs(frob(F1)-2.0)):.1e})")
gate("P2b isotropic co-pol -> no cross-pol", float(np.max(np.abs(F1["F_vh"]))) < 1e-9,
     f"(max |VH| {np.max(np.abs(F1['F_vh'])):.1e})")

print("\nP3 diag(1,-1) delta: cross-pol appears, energy preserved")
dih = pattern(1.0 + 0j, -1.0 + 0j, 0.0)
Fd = point_scatterer_amplitude(dih, LOC, NORMAL, dirs, FREQ)
gate("P3a cross-pol VH is generated off alignment", float(np.max(np.abs(Fd["F_vh"]))) > 0.1,
     f"(max |VH| {np.max(np.abs(Fd['F_vh'])):.2f})")
gate("P3b Frobenius still == 2 (rotation preserves energy)",
     float(np.max(np.abs(frob(Fd) - 2.0))) < 1e-9, f"(worst dev {np.max(np.abs(frob(Fd)-2.0)):.1e})")

print("\nP4 interpolation returns stored values at grid nodes")
az = np.arange(0, 360.1, 30.0); el = np.arange(0, 90.1, 15.0)
naz, nel = len(az), len(el)
vv = (np.arange(naz)[:, None] + 2 * np.arange(nel)[None, :]).astype(float)   # unique per node
vv[-1, :] = vv[0, :]     # 0/360 are the same physical azimuth seam
pat = pattern(vv, 0.5 * vv, 0.0, az=az, el=el)
# query the exact node az=30 (i=1), el=45 (j=3): aperture +z so el is angle from +z; put d in cavity frame
NORMAL2 = np.array([0.0, 0.0, 1.0]); ROLL = np.array([1.0, 0.0, 0.0])
a0, e0 = 30.0, 45.0
d_node = _direction(a0, e0)                                # cavity frame == body frame (normal=+z, roll=+x)
Fn = point_scatterer_amplitude(pat, np.zeros(3), NORMAL2, d_node[None, :], FREQ, roll_ref=ROLL)
# at this node with M=I (aligned frames on the meridian) the co-pol equals the stored VV
stored = vv[1, 3]
gate("P4 node lookup returns the stored amplitude",
     abs(abs(Fn["F_vv"][0]) - stored) < 1e-6 * stored, f"(got {abs(Fn['F_vv'][0]):.3f}, stored {stored:.3f})")

print("\nP5 integration: point scatterer superposes onto the body via sum_features")
# a trivial 'body': flat complex amp over aspect
body = {"theta_deg": list(np.arange(0, 180.1, 10.0)),
        "amp_vv": [0.02 + 0.01j] * 19, "amp_hh": [0.015 - 0.005j] * 19}
gen = np.column_stack([np.linspace(0.05, 0.05, 5), np.linspace(0.15, -0.15, 5)])  # dummy side
pt = {"pattern": iso, "location": LOC, "aperture_normal": NORMAL}
d3 = np.array([_direction(a, e) for a in (0, 60, 120) for e in (-10, 0, 10)])
full = sum_features(body, [], d3, FREQ, generatrix=None, normal_fn=lambda p: np.tile([0, 1, 0.], (len(np.atleast_2d(p)), 1)),
                    mode="coherent", points=[pt])
bod = sum_features(body, [], d3, FREQ, normal_fn=lambda p: np.tile([0, 1, 0.], (len(np.atleast_2d(p)), 1)),
                   mode="coherent")
ptonly = sum_features(None, [], d3, FREQ, normal_fn=lambda p: np.tile([0, 1, 0.], (len(np.atleast_2d(p)), 1)),
                      mode="coherent", points=[pt])
err = float(np.max(np.abs(full["amp_vv"] - (bod["amp_vv"] + ptonly["amp_vv"]))))
gate("P5 sum_features: full == body + point (superposition)", err < 1e-12,
     f"(max diff {err:.1e})")

print(f"\n{'=' * 74}")
print("ALL GATES PASSED" if not _fails else f"{len(_fails)} FAILED")
for f in _fails:
    print(f"   FAILED: {f}")
print("=" * 74)
sys.exit(1 if _fails else 0)
