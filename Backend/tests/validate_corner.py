#!/usr/bin/env python3
"""
Wing-body dihedral corner double-bounce ESTIMATE gate (feature_sum.corner_amplitude).

Anchors the corner term to the standard right-dihedral corner reflector:

  A  PEAK magnitude at the bisector == 8 pi a^2 b^2 / lambda^2 (analytic).
  B  RETROREFLECTION signature: broad in the plane perpendicular to the fold
     (still > half-peak at +/-30 deg) but narrow ALONG the fold (sinc^2, first
     null near lambda/2b) -- the defining dihedral behavior.
  C  POLARIZATION: fold aligned to the look's V/H basis -> co-pol (VH ~ 0);
     fold at 45 deg -> pure cross-pol (VH dominates).  This is the exact ideal-
     dihedral Jones matrix, and is what makes a corner return distinctive.
  D  SHADOWING: a look that does not illuminate BOTH faces returns zero.

and the NON-RIGHT (canted / dihedral / anhedral) extension:

  E  REGRESSION: a right dihedral (delta = 0) keeps the original magnitude
     pattern. The field now retains the signed sinc aperture factor, so
     successive along-fold sidelobes correctly differ by pi in phase.
  F  PEAK ROLLOFF: the peak falls monotonically as the dihedral goes off square
     (delta = 0, 10, 20, 30 deg), and equals sigma_0 cos^2(2 delta) while the
     deflected lobe centre is still illuminated.
  G  LOBE DEFLECTION: the peak-return direction moves off the bisector by 2
     delta (both bounce senses -> a symmetric pair), leaving a cos^2(2 delta)
     dip ON the bisector.

No solver is needed -- the dihedral is analytic.  Fast.

Run from tests/:  python3 validate_corner.py
"""

import math
import sys

import numpy as np

sys.path.insert(0, "..")

from feature_sum import corner_amplitude, _pol_unit_vectors               # noqa: E402

C0_ = 299_792_458.0


def _corner_amplitude_v0(fold, n_wing, n_body, face_width, directions, frequency_ghz,
                         internal_phase_deg=0.0, retro_halfwidth_deg=45.0, occluder=None):
    """The ORIGINAL right-dihedral-only corner model, copied verbatim, kept here
    as the delta = 0 magnitude regression reference (no angle physics: no lobe
    deflection, no peak rolloff). It used sqrt(sinc^2)=abs(sinc), so it cannot
    be a complex-field phase reference beyond the first aperture null."""
    fold = np.asarray(fold, dtype=float)
    if fold.ndim == 3:
        p0, p1 = fold[0, 0], fold[-1, 1]
    else:
        p0, p1 = fold[0], fold[-1]
    f = p1 - p0
    Lf = float(np.linalg.norm(f))
    fhat = f / Lf
    r_c = 0.5 * (p0 + p1)
    nw = np.asarray(n_wing, float); nw = nw / np.linalg.norm(nw)
    nb = np.asarray(n_body, float); nb = nb / np.linalg.norm(nb)
    bhat = nw + nb
    bhat = bhat / np.linalg.norm(bhat)
    k = 2.0 * math.pi * frequency_ghz * 1e9 / C0_
    lam = C0_ / (frequency_ghz * 1e9)
    sigma0 = 8.0 * math.pi * face_width ** 2 * Lf ** 2 / lam ** 2
    retro = math.radians(retro_halfwidth_deg)
    intph = math.radians(internal_phase_deg)
    dirs = np.atleast_2d(np.asarray(directions, float))
    dirs = dirs / np.linalg.norm(dirs, axis=1)[:, None]
    e_vv, e_hh = _pol_unit_vectors(dirs)
    F = {c: np.zeros(len(dirs), complex) for c in ("F_vv", "F_hh", "F_vh")}
    for i, d in enumerate(dirs):
        if (d @ nw) <= 0.0 or (d @ nb) <= 0.0:
            continue
        df = float(d @ fhat)
        x = k * Lf * df
        a_fold = float(np.sinc(x / math.pi)) ** 2
        d_perp = d - df * fhat
        npn = float(np.linalg.norm(d_perp))
        if npn < 1e-9:
            continue
        phi = math.acos(np.clip((d_perp / npn) @ bhat, -1.0, 1.0))
        if phi > retro:
            continue
        a_perp = math.cos(phi) ** 2
        m = math.sqrt(max(sigma0 * a_fold * a_perp, 0.0) / (4.0 * math.pi))
        phat = fhat - df * d
        if np.linalg.norm(phat) < 1e-9:
            continue
        phat = phat / np.linalg.norm(phat)
        qhat = np.cross(d, phat)
        R = np.array([[phat @ e_vv[i], phat @ e_hh[i]],
                      [qhat @ e_vv[i], qhat @ e_hh[i]]])
        Svh = R.T @ np.array([[1.0, 0.0], [0.0, -1.0]]) @ R
        s = m * np.exp(2j * k * float(d @ r_c)) * np.exp(1j * intph)
        F["F_vv"][i] = s * Svh[0, 0]
        F["F_hh"][i] = s * Svh[1, 1]
        F["F_vh"][i] = s * Svh[0, 1]
    return F

C0 = 299_792_458.0
FREQ = 6.0
LAM = C0 / (FREQ * 1e9)
K = 2 * math.pi / LAM
A_FACE = 3.0 * LAM               # perpendicular face width a
B_FOLD = 5.0 * LAM               # fold length b
_fails = []


def gate(label, ok, note=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {note}")
    if not ok:
        _fails.append(label)


def sig(F, ch):
    return 4 * math.pi * np.abs(F[ch]) ** 2


def total_power(F):
    """Pol-invariant total polarimetric power 4pi||S||_F^2 (magnitude pattern
    independent of how it splits across V/H)."""
    return (4 * math.pi * (np.abs(F["F_vv"]) ** 2 + np.abs(F["F_hh"]) ** 2
                           + 2 * np.abs(F["F_vh"]) ** 2))


print("=" * 74)
print(f"Corner (dihedral) estimate gate — a={A_FACE/LAM:g}lam  b(fold)={B_FOLD/LAM:g}lam "
      f"@ {FREQ} GHz")
sig0 = 8 * math.pi * A_FACE ** 2 * B_FOLD ** 2 / LAM ** 2
print(f"  analytic peak 8pi a^2 b^2 / lam^2 = {sig0:.2f} m^2 "
      f"({10*math.log10(sig0):+.2f} dBsm)")
print("=" * 74)

# right dihedral: fold along +z, faces are the +x plane (n=+x) and +y plane (n=+y)
fold = np.array([[[0.0, 0.0, -B_FOLD / 2], [0.0, 0.0, B_FOLD / 2]]])
n_wing = np.array([1.0, 0.0, 0.0])
n_body = np.array([0.0, 1.0, 0.0])
bis = (n_wing + n_body) / np.linalg.norm(n_wing + n_body)   # (0.707,0.707,0)

print("\nA. peak magnitude at the bisector")
# fold||z projects onto V here, so VV is the co-pol; per-pol peak == 8pi a^2b^2/lam^2
Fp = corner_amplitude(fold, n_wing, n_body, A_FACE, bis[None, :], FREQ)
peak = float(sig(Fp, "F_vv")[0])
peak_db = 10 * math.log10(peak)
gate("bisector co-pol (VV) == 8pi a^2 b^2/lam^2",
     abs(10 * math.log10(peak / sig0)) < 0.01, f"({peak_db:+.2f} dBsm, ref {10*math.log10(sig0):+.2f})")

print("\nB. retroreflection signature (broad perpendicular, narrow along fold)")
# perpendicular-plane sweep: rotate the look about the fold (z) away from bisector
perp = []
for phi in (0.0, 15.0, 30.0):
    ca, sa = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    # rotate bisector about z by +phi
    d = np.array([bis[0] * ca - bis[1] * sa, bis[0] * sa + bis[1] * ca, 0.0])
    F = corner_amplitude(fold, n_wing, n_body, A_FACE, d[None, :], FREQ)
    perp.append(float(total_power(F)[0]))
gate("perpendicular plane is BROAD (>half-peak at +/-30 deg from bisector)",
     perp[2] > 0.5 * perp[0], f"(0deg {10*math.log10(perp[0]):+.1f}, "
     f"30deg {10*math.log10(perp[2]):+.1f} dBsm)")
# along-fold sweep: tilt the look toward +z (out of the perpendicular plane)
along = []
for beta in (0.0, 3.0, 6.0):
    d = math.cos(math.radians(beta)) * bis + math.sin(math.radians(beta)) * np.array([0, 0, 1.0])
    d = d / np.linalg.norm(d)
    F = corner_amplitude(fold, n_wing, n_body, A_FACE, d[None, :], FREQ)
    along.append(float(total_power(F)[0]))
null_beta = math.degrees(math.asin(LAM / (2 * B_FOLD)))     # first sinc null
gate("along-fold is NARROW (sinc^2; first null near lam/2b)",
     along[2] < 0.25 * along[0],
     f"(0deg {10*math.log10(along[0]):+.1f}, 6deg {10*math.log10(along[2]):+.1f} dBsm; "
     f"null~{null_beta:.1f}deg)")

print("\nC. polarization (fold aligned -> co-pol;  fold at 45 deg -> cross-pol)")
# fold along z, look in the x-y plane (bisector): fold projects to V (theta-pol)
Fc = corner_amplitude(fold, n_wing, n_body, A_FACE, bis[None, :], FREQ)
copol = sig(Fc, "F_vv")[0] + sig(Fc, "F_hh")[0]
cross = sig(Fc, "F_vh")[0]
gate("fold aligned to V/H -> co-pol (VH negligible)",
     cross < 1e-6 * copol, f"(VH/co = {cross/(copol+1e-30):.1e})")
# now a 45-deg fold: rotate the whole dihedral 45 deg about the bisector's normal.
# simplest: tilt the fold so it projects at 45 deg to V.  Use faces in planes
# whose fold is along (0, s, c) with the look still on the bisector of the faces.
ang = math.radians(45.0)
# fold direction f=(0, sin, cos) -> build two faces perpendicular to f and to
# each other; keep the same look on their bisector
f45 = np.array([0.0, math.sin(ang), math.cos(ang)])
# two orthonormal vectors perpendicular to f45 -> face normals at +/-45 of bisector
u = np.array([1.0, 0.0, 0.0])
u = u - (u @ f45) * f45; u /= np.linalg.norm(u)
w = np.cross(f45, u)
nw2 = (u + w) / math.sqrt(2)          # rotate so normals straddle the bisector u... use u,w as the two faces
nb2 = (u - w) / math.sqrt(2)
fold45 = np.array([[-f45 * B_FOLD / 2, f45 * B_FOLD / 2]])
look = (nw2 + nb2); look /= np.linalg.norm(look)
F45 = corner_amplitude(fold45, nw2, nb2, A_FACE, look[None, :], FREQ)
co45 = sig(F45, "F_vv")[0] + sig(F45, "F_hh")[0]
cr45 = sig(F45, "F_vh")[0]
gate("fold at 45 deg to V/H -> cross-pol dominant",
     cr45 > 3.0 * (co45 + 1e-30), f"(2|VH|^2/co = {2*cr45/(co45+1e-30):.1f})")

print("\nD. shadowing (both faces must be lit)")
# a look from behind the wing face (d.n_wing < 0) must return nothing
d_shadow = np.array([-0.5, 0.5, 0.0]); d_shadow /= np.linalg.norm(d_shadow)
Fs = corner_amplitude(fold, n_wing, n_body, A_FACE, d_shadow[None, :], FREQ)
tot = float(sig(Fs, "F_vv")[0] + sig(Fs, "F_hh")[0] + sig(Fs, "F_vh")[0])
gate("un-illuminated corner returns zero", tot < 1e-25, f"(total {tot:.1e} m^2)")

print("\nE. regression: right-dihedral magnitude retained; signed-sinc phase fixed")
# off-origin fold (exercises the placement phase) + a full-sphere direction grid
# (exercises the along-fold sinc, the retro cutoff and the shadow branch)
fold_off = np.array([[[0.021, -0.013, -B_FOLD / 2], [0.021, -0.013, B_FOLD / 2]]])
grid = []
for th_deg in range(10, 171, 20):
    st, ct = math.sin(math.radians(th_deg)), math.cos(math.radians(th_deg))
    for ph_deg in range(0, 360, 15):
        grid.append([st * math.cos(math.radians(ph_deg)), st * math.sin(math.radians(ph_deg)), ct])
grid = np.asarray(grid)
Fn = corner_amplitude(fold_off, n_wing, n_body, A_FACE, grid, FREQ, internal_phase_deg=37.0)
F0 = _corner_amplitude_v0(fold_off, n_wing, n_body, A_FACE, grid, FREQ, internal_phase_deg=37.0)
dmax = max(float(np.max(np.abs(Fn[c] - F0[c]))) for c in ("F_vv", "F_hh", "F_vh"))
magmax = max(float(np.max(np.abs(np.abs(Fn[c]) - np.abs(F0[c]))))
             for c in ("F_vv", "F_hh", "F_vh"))
nnz = int(np.count_nonzero(np.abs(F0["F_vv"]) + np.abs(F0["F_hh"]) + np.abs(F0["F_vh"])))
gate("delta=0 magnitude pattern unchanged over 216 looks",
     magmax < 1e-14,
     f"(max ||F|-|F_v0|| = {magmax:.1e}, phase-corrected max|dF| = "
     f"{dmax:.1e}, {nnz}/{len(grid)} looks non-zero)")
gate("delta=0 raises no warning", "warning" not in Fn, "")


def canted(delta_deg):
    """Right dihedral (fold ||z, faces n=+x/+y) with the WING face rotated
    delta_deg about the fold -> interior angle 90+delta, n_wing.n_body=sin(delta)."""
    g = math.radians(delta_deg)
    nw = np.array([math.cos(g), math.sin(g), 0.0])
    nb = np.array([0.0, 1.0, 0.0])
    bh = nw + nb
    return nw, nb, bh / np.linalg.norm(bh)


def perp_sweep(delta_deg, phis_deg):
    """Total polarimetric power sweeping the look in the plane PERPENDICULAR to
    the fold, phi measured from the bisector (positive about +z = n_wing x n_body)."""
    nw, nb, bh = canted(delta_deg)
    out = np.empty(len(phis_deg))
    for i, p in enumerate(phis_deg):
        ca, sa = math.cos(math.radians(p)), math.sin(math.radians(p))
        d = np.array([bh[0] * ca - bh[1] * sa, bh[0] * sa + bh[1] * ca, 0.0])
        out[i] = float(total_power(corner_amplitude(fold, nw, nb, A_FACE, d[None, :], FREQ))[0])
    return out


print("\nF. peak rolls off monotonically as the dihedral goes off square")
phis = np.arange(-80.0, 80.001, 0.25)
peaks = []
DELTAS = (0.0, 10.0, 20.0, 30.0, 40.0)
for dd in DELTAS:
    pk = float(np.max(perp_sweep(dd, phis)))
    peaks.append(pk)
    print(f"      delta={dd:4.0f} deg  peak {10*math.log10(max(pk,1e-30)):+7.2f} dBsm  "
          f"({10*math.log10(max(pk/peaks[0], 1e-30)):+6.2f} dB vs square)   "
          f"[cos^2(2 delta) = {10*math.log10(max(math.cos(2*math.radians(dd))**2,1e-30)):+6.2f} dB]")
gate("peak monotonically decreasing over delta = 0,10,20,30,40 deg",
     all(peaks[i + 1] < peaks[i] for i in range(len(DELTAS) - 1)), "")
# The rolloff is exactly sigma_0 cos^2(2 delta) while the deflected centre is
# still illuminated.  Opening the corner WIDENS the lit window to +/-(45+delta/2)
# about the bisector while the centre moves to 2 delta, so the centre stays lit
# up to delta = 30 deg; past that the lit window (and the retro cutoff) clip the
# lobe and the peak drops below the plain rolloff.
for i, dd in list(enumerate(DELTAS))[1:]:
    lim = 2 * sig0 * math.cos(2 * math.radians(dd)) ** 2
    rdb = 10 * math.log10(peaks[i] / lim)
    lit = 2 * dd <= 45.0 + dd / 2
    ok = abs(rdb) < 0.01 if lit else peaks[i] < lim
    gate(f"delta={dd:.0f} peak {'==' if lit else '<'} sigma_0 cos^2(2 delta) "
         f"(deflected centre {'lit' if lit else 'outside the lit window'})",
         ok, f"({rdb:+.2f} dB vs the unclipped rolloff)")
Fw = corner_amplitude(fold, *canted(30.0)[:2], A_FACE, canted(30.0)[2][None, :], FREQ)
gate("off-square corner warns", "warning" in Fw, f"({Fw.get('warning','')[:58]}...")

print("\nG. lobe centre deflects 2*delta off the bisector (symmetric pair)")
for dd in (5.0, 10.0, 15.0):
    sw = perp_sweep(dd, phis)
    phi_pk = float(phis[int(np.argmax(sw))])
    i_p = int(np.argmin(np.abs(phis - 2 * dd)))
    i_m = int(np.argmin(np.abs(phis + 2 * dd)))
    i_0 = int(np.argmin(np.abs(phis)))
    sym = abs(sw[i_p] - sw[i_m]) / max(sw[i_p], 1e-30)
    dip = sw[i_0] / max(sw[i_p], 1e-30)
    gate(f"delta={dd:.0f}: peak at |phi| = 2*delta = {2*dd:.0f} deg",
         abs(abs(phi_pk) - 2 * dd) <= 0.25, f"(peak at phi = {phi_pk:+.2f} deg)")
    gate(f"delta={dd:.0f}: +2delta and -2delta lobes equal (both bounce senses)",
         sym < 1e-12, f"(rel diff {sym:.1e})")
    gate(f"delta={dd:.0f}: bisector dip == cos^2(2 delta)",
         abs(dip - math.cos(2 * math.radians(dd)) ** 2) < 1e-9,
         f"(dip {10*math.log10(dip):+.2f} dB, ref "
         f"{10*math.log10(math.cos(2*math.radians(dd))**2):+.2f} dB)")

print("\nH. anhedral (corner closed, alpha<90) mirrors dihedral (corner opened)")
# the deflected pair is symmetric, so the sign of the cant only matters through
# the ILLUMINATED wedge: opening the corner widens it (+/-(45+delta/2)), closing
# it narrows it (+/-(45-delta/2)) -- so a closed corner is equal at small cant
# and LOWER once its narrower wedge clips the deflected lobe.
p_open, p_shut = float(np.max(perp_sweep(10.0, phis))), float(np.max(perp_sweep(-10.0, phis)))
gate("delta=10: opened and closed corners identical (wedge still holds the lobe)",
     abs(p_open - p_shut) / p_open < 1e-12, f"(rel diff {abs(p_open-p_shut)/p_open:.1e})")
q_open, q_shut = float(np.max(perp_sweep(25.0, phis))), float(np.max(perp_sweep(-25.0, phis)))
gate("delta=25: closed corner LOWER (narrower lit wedge clips the deflected lobe)",
     q_shut < q_open, f"(opened {10*math.log10(q_open):+.2f}, closed "
     f"{10*math.log10(q_shut):+.2f} dBsm)")

print(f"\n{'=' * 74}")
print("ALL GATES PASSED" if not _fails else f"{len(_fails)} FAILED")
for f in _fails:
    print(f"   FAILED: {f}")
print("=" * 74)
sys.exit(1 if _fails else 0)
