#!/usr/bin/env python3
"""Focused polarization gate for ``line_expand.expand_perimeter``.

Synthetic constant seam coefficients isolate the Jones-basis mechanics from
the 2D solver and the BoR calibration:

  P0  local seam axes remain finite and orthonormal at end-on degeneracy;
  P1  an isotropic local coefficient stays co-polar under +/-45 degree frame
      rotations (no coordinate-generated VH);
  P2  reversing a segment's point order leaves its physical Jones response
      unchanged;
  P3  a mirrored anisotropic seam preserves co-pol and reverses cross-pol;
  P4  aligned seams map TM->VV and TE->HH with the documented convention;
  P5  isotropy is preserved for conical incidence with d.t != 0.

Run from any directory:

    python3 Backend/tests/validate_line_polarization.py
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from line_expand import (SeamCoefficients, _transverse_seam_basis,  # noqa: E402
                         expand_perimeter)

FREQ_GHZ = 1.0
LENGTH = 0.041
PREF_LENGTH = LENGTH / (4.0 * math.pi)
PHI = np.array([0.0, 90.0, 180.0])
LOOK_BROADSIDE = np.array([[0.0, 1.0, 0.0]])
_fails = []


def gate(label, ok, note=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {note}")
    if not ok:
        _fails.append(label)


def normal_y(points):
    return np.tile([0.0, 1.0, 0.0], (len(np.atleast_2d(points)), 1))


def centered_segment(tangent, reverse=False):
    t = np.asarray(tangent, dtype=float).copy()
    t /= np.linalg.norm(t)
    a, b = -0.5 * LENGTH * t, 0.5 * LENGTH * t
    if reverse:
        a, b = b, a
    return np.asarray([[a, b]])


def coefficients(tm, te):
    return SeamCoefficients(
        FREQ_GHZ,
        PHI,
        np.full(PHI.shape, complex(tm)),
        np.full(PHI.shape, complex(te)),
        label="synthetic constant Jones coefficient",
    )


def expand(tangent, coef, look=LOOK_BROADSIDE, reverse=False):
    return expand_perimeter(
        centered_segment(tangent, reverse=reverse),
        coef,
        normal_y,
        np.asarray(look, dtype=float),
        max_piece_wavelengths=10.0,
    )


def channel_vector(result):
    return np.array([result["F_vv"][0], result["F_hh"][0], result["F_vh"][0]])


print("=" * 74)
print("Line-expansion signed-polarization gate")
print("=" * 74)

print("\nP0. End-on seam-basis degeneracy")
t = np.array([[0.0, 0.0, 1.0]])
b = np.array([[-1.0, 0.0, 0.0]])
d = np.array([0.0, 0.0, 1.0])
e_tm, e_te = _transverse_seam_basis(t, b, d)
finite = np.all(np.isfinite(e_tm)) and np.all(np.isfinite(e_te))
orth_err = max(
    abs(float(np.dot(e_tm[0], d))),
    abs(float(np.dot(e_te[0], d))),
    abs(float(np.dot(e_tm[0], e_te[0]))),
    abs(float(np.linalg.norm(e_tm[0])) - 1.0),
    abs(float(np.linalg.norm(e_te[0])) - 1.0),
)
gate("degenerate local axes are finite", finite)
gate("degenerate local axes are transverse and orthonormal",
     orth_err < 1e-13, f"(worst error {orth_err:.2e})")

print("\nP1. Isotropic coefficient under mirrored 45-degree seam rotations")
iso_value = 1.3 - 0.7j
iso = coefficients(iso_value, iso_value)
t_plus = np.array([1.0, 0.0, 1.0]) / math.sqrt(2.0)
t_minus = np.array([1.0, 0.0, -1.0]) / math.sqrt(2.0)
f_plus = expand(t_plus, iso)
f_minus = expand(t_minus, iso)
expected_iso = iso_value * PREF_LENGTH
scale_iso = max(abs(expected_iso), 1e-30)
for tag, result in (("+45", f_plus), ("-45", f_minus)):
    co_err = max(abs(result["F_vv"][0] - expected_iso),
                 abs(result["F_hh"][0] - expected_iso)) / scale_iso
    cross_ratio = abs(result["F_vh"][0]) / scale_iso
    gate(f"{tag} isotropic seam preserves equal co-pol",
         co_err < 1e-12, f"(relative error {co_err:.2e})")
    gate(f"{tag} isotropic seam creates no VH",
         cross_ratio < 1e-12, f"(|VH|/|co| {cross_ratio:.2e})")
mirror_err = np.max(np.abs(channel_vector(f_plus) - channel_vector(f_minus))) / scale_iso
gate("mirrored isotropic seams have identical Jones response",
     mirror_err < 1e-12, f"(relative error {mirror_err:.2e})")

print("\nP2. Segment point-order invariance")
aniso = coefficients(2.0 + 0.4j, -0.3 + 1.1j)
f_forward = expand(t_plus, aniso)
f_reverse = expand(t_plus, aniso, reverse=True)
order_scale = max(float(np.max(np.abs(channel_vector(f_forward)))), 1e-30)
order_err = np.max(np.abs(channel_vector(f_forward)
                          - channel_vector(f_reverse))) / order_scale
gate("reversing segment endpoints preserves VV/HH/VH",
     order_err < 1e-12, f"(relative error {order_err:.2e})")

print("\nP3. Mirror parity for an anisotropic local coefficient")
fa_plus = expand(t_plus, aniso)
fa_minus = expand(t_minus, aniso)
co_mirror_err = max(
    abs(fa_plus["F_vv"][0] - fa_minus["F_vv"][0]),
    abs(fa_plus["F_hh"][0] - fa_minus["F_hh"][0]),
) / order_scale
cross_odd_err = abs(fa_plus["F_vh"][0] + fa_minus["F_vh"][0]) / order_scale
cross_strength = abs(fa_plus["F_vh"][0]) / order_scale
gate("mirroring preserves anisotropic co-pol",
     co_mirror_err < 1e-12, f"(relative error {co_mirror_err:.2e})")
gate("mirroring reverses anisotropic VH sign",
     cross_odd_err < 1e-12, f"(relative error {cross_odd_err:.2e})")
gate("anisotropy still produces physical cross-pol",
     cross_strength > 1e-3, f"(|VH|/peak {cross_strength:.3f})")

print("\nP4. Aligned seam convention")
tm_value, te_value = 2.0 + 1.0j, -0.5 + 3.0j
aligned = expand([0.0, 0.0, 1.0], coefficients(tm_value, te_value))
vv_err = abs(aligned["F_vv"][0] - tm_value * PREF_LENGTH) / abs(tm_value * PREF_LENGTH)
hh_err = abs(aligned["F_hh"][0] - te_value * PREF_LENGTH) / abs(te_value * PREF_LENGTH)
vh_ratio = abs(aligned["F_vh"][0]) / max(abs(tm_value * PREF_LENGTH),
                                         abs(te_value * PREF_LENGTH))
gate("span-aligned broadside VV selects 2D TM",
     vv_err < 1e-12, f"(relative error {vv_err:.2e})")
gate("span-aligned broadside HH selects 2D TE",
     hh_err < 1e-12, f"(relative error {hh_err:.2e})")
gate("aligned diagonal coefficient has zero VH",
     vh_ratio < 1e-12, f"(|VH|/peak {vh_ratio:.2e})")

print("\nP5. Conical incidence (nonzero d.t)")
look_conical = np.array([[0.0, 0.8, 0.6]])
conical = expand([0.0, 0.0, 1.0], iso, look=look_conical)
conical_scale = max(abs(conical["F_vv"][0]), abs(conical["F_hh"][0]), 1e-30)
conical_co_err = abs(conical["F_vv"][0] - conical["F_hh"][0]) / conical_scale
conical_cross = abs(conical["F_vh"][0]) / conical_scale
gate("isotropic conical response preserves VV == HH",
     conical_co_err < 1e-12, f"(relative error {conical_co_err:.2e})")
gate("isotropic conical response creates no VH",
     conical_cross < 1e-12, f"(|VH|/|co| {conical_cross:.2e})")

# Exact end-on incidence is unilluminated (d.n == 0), but must not propagate
# NaN/Inf through the fallback polarization frame.
end_on = expand([0.0, 0.0, 1.0], iso, look=np.array([[0.0, 0.0, 1.0]]))
end_vec = channel_vector(end_on)
gate("exact end-on expansion is finite and grazing-zero",
     np.all(np.isfinite(end_vec)) and np.max(np.abs(end_vec)) == 0.0)

print(f"\n{'=' * 74}")
print("ALL GATES PASSED" if not _fails else f"{len(_fails)} FAILED")
for failure in _fails:
    print(f"   FAILED: {failure}")
print("=" * 74)
sys.exit(1 if _fails else 0)
