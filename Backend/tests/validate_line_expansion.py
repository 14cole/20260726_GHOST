#!/usr/bin/env python3
"""
Ring gate for the line-expanded feature machinery (line_expand.py).

Ground truth without a 3-D MoM: a CIRCUMFERENTIAL groove on a cylinder is
axisymmetric, so the BoR solver solves it rigorously at every aspect.  The
feature's true differential amplitude is

    dF_BoR(theta) = F_grooved(theta) - F_smooth(theta)          [complex]

which is exactly the quantity the line expansion claims to reproduce from a 2D
cross-section of the same groove.  Comparing them:

  * jointly pins the local TM/TE ``psi`` phase mappings before their projected
    contributions are added, then bounds the residual phase of the actual
    production HH/VV sum;
  * checks the absolute normalisation (|A|^2/(4k) -> 4pi|F|^2), which at
    broadside is the 2L^2/lambda strip relation already gated in
    validate_bor_phase1.py;
  * sweeps the seam's in-plane incidence angle through its whole range, since
    the stationary-phase point of the ring sits at 2D angle 180-theta.

What it does NOT probe: the oblique/cone-angle approximation.  A ring's
stationary-phase point is at alpha = 0, where d.t_hat = 0 identically — the
dominant contributor is always broadside to the local tangent.  That is a
property of ANY closed perimeter (stationary phase occurs where the tangent is
perpendicular to the projected look direction), so it is the right proxy for a
door outline, and the wrong proxy for a straight finite seam viewed off
broadside.

Run from tests/:  python3 validate_line_expansion.py
"""

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, "..")

from bor_solver import solve_bor                      # noqa: E402
from line_expand import (PSI_HH_DEG, PSI_VV_DEG,               # noqa: E402
                         VALIDITY_HALF_ANGLE_DEG, SeamCoefficients,
                         expand_perimeter, seam_coefficients_from_2d,
                         surface_of_revolution_normal)

C0 = 299792458.0

# ── case ─────────────────────────────────────────────────────────────────────
FREQ_GHZ = 6.0
LAM = C0 / (FREQ_GHZ * 1e9)
A_BODY = 0.060          # cylinder radius, m           (ka = 7.5)
L_BODY = 0.300          # cylinder length, m
Z0 = 0.0                # groove plane
W_GROOVE = LAM / 3.0    # groove axial width, m
H_GROOVE = LAM / 8.0    # groove depth, m
# Five-degree samples resolve the edge of the advertised validity sector.  A
# ten-degree grid can only say that the boundary lies somewhere between two
# samples, which is not adequate for calibrating a production error bound.
ASPECTS = np.arange(40.0, 140.1, 5.0)
# The criteria are evaluated over the near-broadside window and the gate also
# MEASURES how wide the good window actually is (see section 4). The local TM
# and TE coefficients require separate phases and are calibrated jointly
# because a finite ring mixes both into each vehicle channel away from exact
# broadside.
CORE = (90.0 - VALIDITY_HALF_ANGLE_DEG,
        90.0 + VALIDITY_HALF_ANGLE_DEG)
RESIDUAL_PHASE_TOL_DEG = 25.0
MAG_TOL_DB = 3.5        # dB  — |F_exp| must track |dF_BoR| this well across CORE
CALIBRATION_TOL_DEG = 0.25

# UNIFORM mesh in both solvers.  A graded mesh is tempting here but the BoR
# azimuthal FFT length is floored by the smallest far-pair gap
# (bor_kernels.n_xi_for_pairs), so a locally fine generatrix inflates every
# table build.  Uniform spacing, shared by the coupon, also makes the two
# solvers see the SAME polygonisation of the groove.
DS = LAM / 25.0
N_MODES = 22

_fails = []


def gate(label, ok, note=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {note}")
    if not ok:
        _fails.append(label)


# ── geometry builders ────────────────────────────────────────────────────────

def _groove_profile(s):
    """Inward displacement of the skin at along-surface coordinate s (m)."""
    d = np.zeros_like(s)
    inside = np.abs(s) <= W_GROOVE / 2.0
    d[inside] = H_GROOVE * np.cos(math.pi * s[inside] / W_GROOVE) ** 2
    return d


def _uniform_axis():
    """Side-wall z samples, uniform at DS, with a node exactly on the groove
    centre so both solvers polygonise the same profile."""
    n_half = int(math.ceil((L_BODY / 2.0 - Z0) / DS))
    up = Z0 + np.arange(1, n_half + 1) * DS
    n_half2 = int(math.ceil((L_BODY / 2.0 + Z0) / DS))
    dn = Z0 - np.arange(1, n_half2 + 1) * DS
    zs = np.concatenate([up[::-1], [Z0], dn])
    return np.clip(zs, -L_BODY / 2.0, L_BODY / 2.0)


def cylinder_gen(grooved: bool):
    """Closed cylinder generatrix, +z cap centre -> side -> -z cap centre."""
    z_side = _uniform_axis()
    rho = np.full_like(z_side, A_BODY)
    if grooved:
        rho = rho - _groove_profile(z_side - Z0)
    n_rad = max(2, int(math.ceil(A_BODY / DS)))
    top = np.column_stack([np.linspace(0.0, A_BODY, n_rad + 1),
                           np.full(n_rad + 1, L_BODY / 2.0)])
    bot = np.column_stack([np.linspace(A_BODY, 0.0, n_rad + 1),
                           np.full(n_rad + 1, -L_BODY / 2.0)])
    side = np.column_stack([rho, z_side])
    return np.vstack([top[:-1], side, bot[1:]])


def coupon(width_lam: float, grooved: bool, thickness: float = 0.9 * LAM):
    """Closed 2D cross-section of the joint: a "capsule" of PEC skin.

    Two phase-center / edge subtleties are handled here, because both leak
    directly into the expansion:

      * OUTER FACE AT y = 0.  The differential's phase centre is the seam on
        the outer surface, and the expansion places r(s) on that same surface.
        Referencing the coupon to mid-plank instead injects a spurious
        exp(k*t*sin(phi)) that reads as an aspect-dependent psi.
      * ROUNDED END CAPS (radius t/2, tangent to both faces).  A rectangular
        plank's PEC end edges ring the groove field back and forth; that
        multipath is width-dependent and does NOT cancel in the difference
        (the smooth coupon has no groove to feed it).  Rounded caps have no
        edge diffraction, so the only residual width dependence is a decaying
        creeping wave — which is what the width-convergence gate checks.
    """
    W = width_lam * LAM
    r_cap = thickness / 2.0
    n_half = int(math.ceil((W / 2.0) / DS))
    xf = np.arange(-n_half, n_half + 1) * DS          # outer face, left->right
    yf = -_groove_profile(xf) if grooved else np.zeros_like(xf)
    pts = [(float(a), float(b)) for a, b in zip(xf, yf)]

    n_cap = max(4, int(math.ceil(math.pi * r_cap / DS)))
    ang = np.linspace(math.pi / 2.0, -math.pi / 2.0, n_cap + 1)[1:]   # right cap
    pts += [(W / 2.0 + r_cap * math.cos(a), -r_cap + r_cap * math.sin(a))
            for a in ang]
    xb = np.arange(n_half, -n_half - 1, -1) * DS       # bottom face, right->left
    pts += [(float(x), -thickness) for x in xb]
    ang = np.linspace(-math.pi / 2.0, -3.0 * math.pi / 2.0, n_cap + 1)[1:]  # left cap
    pts += [(-W / 2.0 + r_cap * math.cos(a), -r_cap + r_cap * math.sin(a))
            for a in ang]

    # drop any point within DS/100 of the previous, then wrap-close
    clean = [pts[0]]
    for p in pts[1:]:
        if math.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > DS / 100.0:
            clean.append(p)
    if math.hypot(clean[-1][0] - clean[0][0], clean[-1][1] - clean[0][1]) <= DS / 100.0:
        clean.pop()
    pts = clean
    pairs = [{"x1": pts[i][0], "y1": pts[i][1],
              "x2": pts[(i + 1) % len(pts)][0], "y2": pts[(i + 1) % len(pts)][1]}
             for i in range(len(pts))]
    seg = {"name": "coupon", "seg_type": "2",
           "properties": ["2", "0", "0", "0", "0"], "point_pairs": pairs}
    return {"title": "coupon", "segments": [seg], "ibcs": [], "dielectrics": []}


def ring_perimeter(radius: float, z: float, n: int = 720):
    al = np.linspace(0.0, 2.0 * math.pi, n + 1)
    pts = np.column_stack([radius * np.cos(al), radius * np.sin(al),
                           np.full_like(al, z)])
    return np.stack([pts[:-1], pts[1:]], axis=1)


# The two BoR solves dominate runtime and never change while the 2D/expansion
# code is iterated (or when another gate imports this module for its geometry
# builders); cache them keyed on the physics.  Delete the .pkl to force.
import pickle  # noqa: E402
from validation_cache import bor_solver_sources, cache_path  # noqa: E402

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE = cache_path(
    "ring_gate_bor",
    {
        "frequency_ghz": FREQ_GHZ,
        "n_modes": N_MODES,
        "aspects_deg": ASPECTS,
        "generatrix_smooth": cylinder_gen(False),
        "generatrix_groove": cylinder_gen(True),
    },
    bor_solver_sources(_BACKEND),
)


def load_cached_bor(verbose: bool = True):
    """Return {'smooth','groove'} BoR results, from cache or by solving."""
    if os.path.exists(_CACHE):
        with open(_CACHE, "rb") as fh:
            bor = pickle.load(fh)
        if verbose:
            print(f"   [cached BoR ground truth from {_CACHE}]")
        return bor
    bor = {}
    for tag, gen in (("smooth", cylinder_gen(False)), ("groove", cylinder_gen(True))):
        t1 = time.time()
        bor[tag] = solve_bor(gen, FREQ_GHZ * 1e9, ASPECTS, formulation="cfie",
                             cfie_alpha=0.5, n_modes=N_MODES, workers=4)
        if verbose:
            print(f"   {tag}: {bor[tag]['n_unknowns']} unknowns, "
                  f"{bor[tag]['modes_used']} modes, "
                  f"residual {bor[tag]['linear_residual']:.1e}, {time.time() - t1:.0f} s")
    with open(_CACHE, "wb") as fh:
        pickle.dump(bor, fh)
    return bor


# lazy singleton so `import validate_line_expansion` is cheap yet exposes `bor`
bor = load_cached_bor(verbose=False) if os.path.exists(_CACHE) else None


def main():
    global bor
    print("=" * 74)
    print(f"Line-expansion ring gate — {FREQ_GHZ} GHz, lambda = {LAM * 1000:.1f} mm")
    print(f"  cylinder a={A_BODY * 1000:.0f} mm (ka={2 * math.pi * A_BODY / LAM:.2f}), "
          f"L={L_BODY * 1000:.0f} mm")
    print(f"  groove   w={W_GROOVE * 1000:.1f} mm (lam/{LAM / W_GROOVE:.1f}), "
          f"h={H_GROOVE * 1000:.1f} mm (lam/{LAM / H_GROOVE:.1f})")
    print("=" * 74)

    t0 = time.time()
    print("\n1. BoR ground truth (grooved - smooth, complex)")
    gen_smooth = cylinder_gen(False)
    print(f"   generatrix nodes: {len(gen_smooth)} (identical mesh both runs)")
    bor = load_cached_bor()

    dF = {"F_vv": np.asarray(bor["groove"]["amp_vv"]) - np.asarray(bor["smooth"]["amp_vv"]),
          "F_hh": np.asarray(bor["groove"]["amp_hh"]) - np.asarray(bor["smooth"]["amp_hh"])}
    lvl = {p: 10 * np.log10(4 * math.pi * np.abs(dF[p]) ** 2 + 1e-30) for p in dF}
    body_db = 10 * np.log10(np.asarray(bor["smooth"]["sigma_vv"]) + 1e-30)
    print(f"   feature dsigma  VV {lvl['F_vv'].min():+6.1f} .. {lvl['F_vv'].max():+6.1f} dBsm"
          f"   (smooth body {body_db.min():+.1f} .. {body_db.max():+.1f} dBsm)")

    print("\n2. Seam coefficients from the 2D coupon (featured - smooth)")
    # Include the grazing endpoints.  The production expansion is fail-closed:
    # a lit incidence outside coupon support is not silently treated as zero.
    phi = np.arange(0.0, 180.1, 2.5)
    coefs = {}
    for wl in (8.0, 12.0):
        t1 = time.time()
        coefs[wl] = seam_coefficients_from_2d(coupon(wl, True), coupon(wl, False),
                                              FREQ_GHZ, phi, geometry_units="meters",
                                              label=f"{wl:g}lam coupon")
        print(f"   {wl:g} lambda coupon: {len(phi)} angles x 2 pol, {time.time() - t1:.0f} s")

    d6, d9 = coefs[8.0], coefs[12.0]
    for pol, k in (("TM", "dA_tm"), ("TE", "dA_te")):
        a6, a9 = getattr(d6, k), getattr(d9, k)
        # Compare only over the near-normal support the ring expansion actually
        # consumes: over the advertised validity window the sampled cut angle
        # is phi = 180-theta.  (Residual convergence at
        # wider phi is coupon-cap creeping-wave ring-down, which the featured-
        # minus-smooth difference largely cancels and never enters the expansion.)
        band = np.abs(phi - 90.0) <= VALIDITY_HALF_ANGLE_DEG
        rel = np.abs(a6 - a9)[band] / max(np.max(np.abs(a9[band])), 1e-30)
        gate(f"coupon width convergence {pol} (8 vs 12 lambda, "
             f"|phi-90|<={VALIDITY_HALF_ANGLE_DEG:g})",
             float(rel.max()) < 0.12, f"(worst {100 * rel.max():.1f} % of peak)")

    print("\n3. Line expansion around the groove ring")
    per = ring_perimeter(A_BODY, Z0)
    nfn = surface_of_revolution_normal(cylinder_gen(False))
    th = np.radians(ASPECTS)
    dirs = np.column_stack([np.sin(th), np.zeros_like(th), np.cos(th)])
    # Resolve the two local-polarization contributions separately.  A finite
    # ring is not exactly polarization-pure away from broadside: HH contains a
    # smaller TE contribution and VV a smaller TM contribution.  Therefore the
    # coefficient phases must be calibrated jointly before those contributions
    # are added.  Applying a phase measured from the already-summed HH/VV
    # channel back to only one coefficient is not algebraically equivalent.
    zero = np.zeros_like(d9.dA_tm)
    tm_only = SeamCoefficients(
        d9.frequency_ghz, d9.phi_deg, d9.dA_tm, zero, label="TM-only")
    te_only = SeamCoefficients(
        d9.frequency_ghz, d9.phi_deg, zero, d9.dA_te, label="TE-only")
    exp_tm = expand_perimeter(per, tm_only, nfn, dirs)
    exp_te = expand_perimeter(per, te_only, nfn, dirs)
    exp9 = {p: exp_tm[p] + exp_te[p] for p in ("F_vv", "F_hh", "F_vh")}

    core = (ASPECTS >= CORE[0]) & (ASPECTS <= CORE[1])
    print(f"\n   {'aspect':>7} {'|dF_BoR|':>10} {'|F_exp|':>10} {'ratio dB':>9} "
          f"{'arg(ratio)':>11}   {'|dF_BoR|':>10} {'|F_exp|':>10} {'ratio dB':>9} "
          f"{'arg(ratio)':>11}")
    print(f"   {'':>7} {'------------- VV -------------':^42}   "
          f"{'------------- HH -------------':^42}")
    ratios = {}
    for pol in ("F_vv", "F_hh"):
        r = np.where(np.abs(exp9[pol]) > 0, dF[pol] / np.where(np.abs(exp9[pol]) > 0,
                                                               exp9[pol], 1.0), np.nan)
        ratios[pol] = r
    for i, a in enumerate(ASPECTS):
        row = f"   {a:7.1f}"
        for pol in ("F_vv", "F_hh"):
            r = ratios[pol][i]
            row += (f" {np.abs(dF[pol][i]):10.3e} {np.abs(exp9[pol][i]):10.3e} "
                    f"{20 * np.log10(np.abs(r)):9.2f} {np.degrees(np.angle(r)):11.1f}")
        print(row + ("  *" if core[i] else ""))

    print("\n4. Verdict  (* rows = core sector used for the criteria)")

    def _wrap(x):
        return np.degrees(np.angle(np.exp(1j * np.radians(x))))

    def _circular_ratio_phase(num, den):
        """Circular mean phase of num/den over CORE, excluding true nulls."""
        n = np.asarray(num)[core]
        d = np.asarray(den)[core]
        keep = (np.abs(n) > 1.0e-30) & (np.abs(d) > 1.0e-30)
        if not np.any(keep):
            raise RuntimeError("calibration channel has no nonzero samples.")
        r = n[keep] / d[keep]
        return float(np.degrees(np.angle(np.mean(r / np.abs(r)))))

    # Alternating unit-magnitude phase fit.  TE is initialized from the
    # TE-dominant VV channel; then each update subtracts the other calibrated
    # local-polarization contribution before measuring the remaining phase.
    psi_te = _circular_ratio_phase(dF["F_vv"], exp_te["F_vv"])
    psi_tm = 0.0
    for _ in range(5):
        psi_tm = _circular_ratio_phase(
            dF["F_hh"] - np.exp(1j * np.radians(psi_te)) * exp_te["F_hh"],
            exp_tm["F_hh"])
        psi_te = _circular_ratio_phase(
            dF["F_vv"] - np.exp(1j * np.radians(psi_tm)) * exp_tm["F_vv"],
            exp_te["F_vv"])

    exp_baked = expand_perimeter(
        per, d9, nfn, dirs,
        psi_tm_deg=PSI_HH_DEG, psi_te_deg=PSI_VV_DEG)

    def _window(pol):
        """Widest symmetric aspect window about 90 deg where the production
        coefficient-level calibration satisfies both error limits."""
        best = 0.0
        for half in np.arange(5.0, 55.0, 5.0):
            sel = np.abs(ASPECTS - 90.0) <= half + 1e-9
            r = dF[pol][sel] / exp_baked[pol][sel]
            if np.any(~np.isfinite(r)):
                break
            mag = np.abs(20 * np.log10(np.abs(r)))
            ph = np.abs(np.degrees(np.angle(r)))
            if mag.max() <= MAG_TOL_DB and ph.max() <= RESIDUAL_PHASE_TOL_DEG:
                best = half
            else:
                break
        return best

    for pol, name in (("F_vv", "VV"), ("F_hh", "HH")):
        r = dF[pol][core] / exp_baked[pol][core]
        mag_db = 20 * np.log10(np.abs(r))
        ph_c = np.degrees(np.angle(r))
        gate(f"{name} calibrated magnitude tracks truth over "
             f"{CORE[0]:.0f}-{CORE[1]:.0f} deg",
             float(np.abs(mag_db).max()) < MAG_TOL_DB,
             f"(median {np.median(mag_db):+.2f} dB, worst {mag_db[np.argmax(np.abs(mag_db))]:+.2f} dB)")
        gate(f"{name} calibrated residual phase over CORE",
             float(np.abs(ph_c).max()) < RESIDUAL_PHASE_TOL_DEG,
             f"(worst {np.abs(ph_c).max():.1f} deg)")

    for coefficient, measured, baked in (
            ("TE/VV", psi_te, PSI_VV_DEG),
            ("TM/HH", psi_tm, PSI_HH_DEG)):
        calibration_error = abs(_wrap(baked - measured))
        gate(f"baked psi_{coefficient} matches joint coefficient calibration",
             calibration_error <= CALIBRATION_TOL_DEG,
             f"(baked {baked:+.1f}, measured {measured:+.1f}, "
             f"error {calibration_error:.2f} deg)")

    print("\n   Local-coefficient calibration constants (measured jointly, reused):")
    print(f"     psi_TE/VV = {psi_te:+7.1f} deg    validity window = "
          f"broadside +/- {_window('F_vv'):.0f} deg")
    print(f"     psi_TM/HH = {psi_tm:+7.1f} deg    validity window = "
          f"broadside +/- {_window('F_hh'):.0f} deg")
    dpsi = _wrap(psi_te - psi_tm)
    print(f"     (TE-TM offset {dpsi:+.1f} deg; finite-ring channel mixing is "
          "included before the residual is evaluated)")
    gate(f"validity window reaches broadside +/- "
         f"{VALIDITY_HALF_ANGLE_DEG:g} deg",
         min(_window("F_vv"), _window("F_hh")) >= VALIDITY_HALF_ANGLE_DEG,
         f"(VV {_window('F_vv'):.0f} deg, HH {_window('F_hh'):.0f} deg)")

    print(f"\n{'=' * 74}")
    print("ALL GATES PASSED" if not _fails else f"{len(_fails)} failed",
          f"  ({time.time() - t0:.0f} s total)")
    for f in _fails:
        print(f"   FAILED: {f}")
    print("=" * 74)
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
