#!/usr/bin/env python3
"""
Coated (RAM-on-PEC) ring gate for the line-expansion feature machinery.

The user's routine case: a PEC body with a thickness of RAM on the outer face,
and a seam feature in that coated skin.  This is the coated analog of
validate_line_expansion.py: it proves the SAME pipeline works when the body is
coated, and checks whether the calibration constants (PSI_VV_DEG/PSI_HH_DEG,
measured on bare PEC) carry over to this coated surface. That is a testable
model property, not assumed a priori.

Ground truth WITHOUT a 3-D MoM: a circumferential groove in the outer coating
of a COATED cylinder is axisymmetric, so solve_bor_coated_pec solves it exactly.
dF_BoR = F(coated+groove) - F(coated+smooth) is what the expansion must
reproduce from a 2-D coated coupon (TYPE-3 coating over TYPE-4 PEC core).

Feature modelled: a groove in the OUTER coating surface (a coating recess) over
a smooth PEC core -- locally thinner RAM at the seam.  Core stays smooth.

Run from tests/:  python3 validate_line_expansion_coated.py
"""

import math
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "..")

from bor_solver import solve_bor_coated_pec                              # noqa: E402
from line_expand import (PSI_HH_DEG, PSI_VV_DEG, SeamCoefficients,       # noqa: E402
                         expand_perimeter, seam_coefficients_from_2d,
                         surface_of_revolution_normal)
from validation_cache import bor_solver_sources, cache_path              # noqa: E402

C0 = 299_792_458.0
FREQ_GHZ = 6.0
LAM = C0 / (FREQ_GHZ * 1e9)
A_CORE = 0.060                 # PEC core radius
T_RAM = LAM / 8.0              # coating thickness
A_OUT = A_CORE + T_RAM         # outer coating radius
L_BODY = 0.300                 # core length
GROOVE_W = LAM / 3.0           # seam width (in the coating outer surface)
GROOVE_H = 0.5 * T_RAM         # seam depth (recess into the coating)
DS = LAM / 25.0
EPS = complex(4.0, -2.0)       # lossy RAM (e^{+jwt}: negative imag)
MU = 1.0
N_MODES = 22
ASPECTS = np.arange(40.0, 140.1, 10.0)
CORE = (70.0, 110.0)
COUPON_LAM = 12.0
PHI = np.arange(0.0, 180.1, 2.5)
PHASE_TOL_DEG = 25.0
MAG_TOL_DB = 2.5

_fails = []


def gate(label, ok, note=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {note}")
    if not ok:
        _fails.append(label)


def _wrap(x):
    return np.degrees(np.angle(np.exp(1j * np.radians(x))))


def _groove_z(z):
    d = np.zeros_like(z)
    m = np.abs(z) <= GROOVE_W / 2.0
    d[m] = GROOVE_H * np.cos(math.pi * z[m] / GROOVE_W) ** 2
    return d


def _axis(hl):
    n = int(math.ceil(hl / DS))
    up = np.arange(1, n + 1) * DS
    return np.concatenate([up[::-1], [0.0], -up])       # +hl .. 0 .. -hl, node at 0


# ── coated cylinder generatrices (outer coating + PEC core) ──────────────────

def outer_gen(grooved):
    hl = L_BODY / 2.0 + T_RAM
    z = np.clip(_axis(hl), -hl, hl)
    rho = np.full_like(z, A_OUT)
    if grooved:
        rho = rho - _groove_z(z)
    nr = max(2, int(math.ceil(A_OUT / DS)))
    top = np.column_stack([np.linspace(0, A_OUT, nr + 1), np.full(nr + 1, hl)])
    bot = np.column_stack([np.linspace(A_OUT, 0, nr + 1), np.full(nr + 1, -hl)])
    return np.vstack([top[:-1], np.column_stack([rho, z]), bot[1:]])


def core_gen():
    hl = L_BODY / 2.0
    z = np.clip(_axis(hl), -hl, hl)
    rho = np.full_like(z, A_CORE)
    nr = max(2, int(math.ceil(A_CORE / DS)))
    top = np.column_stack([np.linspace(0, A_CORE, nr + 1), np.full(nr + 1, hl)])
    bot = np.column_stack([np.linspace(A_CORE, 0, nr + 1), np.full(nr + 1, -hl)])
    return np.vstack([top[:-1], np.column_stack([rho, z]), bot[1:]])


# ── coated 2-D coupon (TYPE-3 coating over TYPE-4 PEC core) ───────────────────

def _capsule_outline(Wl, thick, inset, grooved):
    W = Wl * LAM
    r = thick / 2.0 - inset
    n = int(math.ceil((W / 2.0) / DS))
    xf = np.arange(-n, n + 1) * DS
    yf = np.full_like(xf, -inset)
    if grooved and inset == 0.0:
        m = np.abs(xf) <= GROOVE_W / 2.0
        yf[m] -= GROOVE_H * np.cos(math.pi * xf[m] / GROOVE_W) ** 2
    pts = [(float(a), float(b)) for a, b in zip(xf, yf)]
    nc = max(4, int(math.ceil(math.pi * r / DS)))
    for a in np.linspace(math.pi / 2, -math.pi / 2, nc + 1)[1:]:
        pts.append((W / 2 + r * math.cos(a), -thick / 2 + r * math.sin(a)))
    for x in np.arange(n, -n - 1, -1) * DS:
        pts.append((float(x), -thick + inset))
    for a in np.linspace(-math.pi / 2, -3 * math.pi / 2, nc + 1)[1:]:
        pts.append((-W / 2 + r * math.cos(a), -thick / 2 + r * math.sin(a)))
    clean = [pts[0]]
    for p in pts[1:]:
        if math.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > DS / 100:
            clean.append(p)
    if math.hypot(clean[-1][0] - clean[0][0], clean[-1][1] - clean[0][1]) <= DS / 100:
        clean.pop()
    return [{"x1": clean[i][0], "y1": clean[i][1],
             "x2": clean[(i + 1) % len(clean)][0], "y2": clean[(i + 1) % len(clean)][1]}
            for i in range(len(clean))]


def coated_coupon(Wl, grooved, thick=0.9 * LAM):
    outer = _capsule_outline(Wl, thick, 0.0, grooved)
    core = _capsule_outline(Wl, thick, T_RAM, False)
    seg3 = {"name": "coat", "seg_type": "3", "properties": ["3", "0", "0", "1", "0"],
            "point_pairs": outer}
    seg4 = {"name": "core", "seg_type": "4", "properties": ["4", "0", "0", "1", "0"],
            "point_pairs": core}
    return {"title": "coated", "segments": [seg3, seg4], "ibcs": [],
            "dielectrics": [["1", str(EPS.real), str(EPS.imag), str(MU), "0"]]}


# ── run ──────────────────────────────────────────────────────────────────────

print("=" * 76)
print(f"COATED ring gate — {FREQ_GHZ} GHz, PEC core a={A_CORE*1000:.0f}mm + "
      f"RAM t={T_RAM*1000:.1f}mm (lam/{LAM/T_RAM:.0f})")
print(f"  RAM eps={EPS} mu={MU};  seam: coating groove w={GROOVE_W/LAM:.2f}lam "
      f"h={GROOVE_H*1000:.1f}mm")
print(f"  baked (bare-PEC) constants: psi_VV={PSI_VV_DEG:+.1f} psi_HH={PSI_HH_DEG:+.1f}")
print("=" * 76)

t0 = time.time()
_ck = (FREQ_GHZ, A_CORE, T_RAM, L_BODY, GROOVE_W, GROOVE_H, DS, N_MODES,
       EPS, MU, tuple(ASPECTS.tolist()))
_backend = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cache = cache_path(
    "coated_gate_bor",
    {
        "physics": _ck,
        "outer_smooth": outer_gen(False),
        "outer_groove": outer_gen(True),
        "core": core_gen(),
    },
    bor_solver_sources(_backend),
)

print("\n1. Coated BoR ground truth (groove - smooth, complex)")
core = core_gen()
if os.path.exists(_cache):
    with open(_cache, "rb") as fh:
        bor = pickle.load(fh)
    print(f"   [cached from {_cache}]")
else:
    bor = {}
    for tag, gr in (("smooth", False), ("groove", True)):
        t1 = time.time()
        bor[tag] = solve_bor_coated_pec(outer_gen(gr), core, FREQ_GHZ * 1e9, ASPECTS,
                                        EPS, MU, n_modes=N_MODES, workers=4)
        print(f"   {tag}: {bor[tag]['n_unknowns']} unk, {bor[tag]['modes_used']} modes, "
              f"res {bor[tag]['linear_residual']:.1e}, {time.time()-t1:.0f} s")
    with open(_cache, "wb") as fh:
        pickle.dump(bor, fh)

dF = {"vv": np.asarray(bor["groove"]["amp_vv"]) - np.asarray(bor["smooth"]["amp_vv"]),
      "hh": np.asarray(bor["groove"]["amp_hh"]) - np.asarray(bor["smooth"]["amp_hh"])}
body_db = 10 * np.log10(np.asarray(bor["smooth"]["sigma_vv"]) + 1e-30)
feat_db = 10 * np.log10(4 * math.pi * np.abs(dF["vv"]) ** 2 + 1e-30)
print(f"   coated feature dsigma VV {feat_db.min():+.1f}..{feat_db.max():+.1f} dBsm "
      f"(coated body {body_db.min():+.1f}..{body_db.max():+.1f})")

print("\n2. Coated coupon seam coefficient (featured - clean, TYPE-3 over TYPE-4)")
t1 = time.time()
coef = seam_coefficients_from_2d(coated_coupon(COUPON_LAM, True),
                                 coated_coupon(COUPON_LAM, False),
                                 FREQ_GHZ, PHI, geometry_units="meters", label="coated coupon")
print(f"   {COUPON_LAM:g} lambda coated coupon: {len(PHI)} angles x 2 pol, {time.time()-t1:.0f} s")

print("\n3. Line expansion around the coating-groove ring (at the OUTER surface)")
per = np.stack([np.column_stack([A_OUT * np.cos(np.linspace(0, 2*math.pi, 721)[:-1]),
                                 A_OUT * np.sin(np.linspace(0, 2*math.pi, 721)[:-1]),
                                 np.zeros(720)]),
                np.column_stack([A_OUT * np.cos(np.linspace(0, 2*math.pi, 721)[1:]),
                                 A_OUT * np.sin(np.linspace(0, 2*math.pi, 721)[1:]),
                                 np.zeros(720)])], axis=1)
nfn = surface_of_revolution_normal(outer_gen(False))
th = np.radians(ASPECTS)
dirs = np.column_stack([np.sin(th), np.zeros_like(th), np.cos(th)])
exp = expand_perimeter(
    per, coef, nfn, dirs,
    psi_tm_deg=PSI_HH_DEG, psi_te_deg=PSI_VV_DEG)

core_sel = (ASPECTS >= CORE[0]) & (ASPECTS <= CORE[1])
print(f"\n   {'aspect':>7} {'ratio dB VV':>12} {'arg VV':>8}   {'ratio dB HH':>12} {'arg HH':>8}")
ratios = {}
for ch in ("vv", "hh"):
    ratios[ch] = dF[ch] / np.where(np.abs(exp[f"F_{ch}"]) > 0, exp[f"F_{ch}"], 1.0)
for i, a in enumerate(ASPECTS):
    rv, rh = ratios["vv"][i], ratios["hh"][i]
    print(f"   {a:7.1f} {20*np.log10(np.abs(rv)):12.2f} {np.degrees(np.angle(rv)):8.1f}   "
          f"{20*np.log10(np.abs(rh)):12.2f} {np.degrees(np.angle(rh)):8.1f}"
          + ("  *" if core_sel[i] else ""))

print("\n4. Verdict")
for ch, name in (("vv", "VV"), ("hh", "HH")):
    r = ratios[ch][core_sel]
    mag = np.abs(20 * np.log10(np.abs(r)))
    phase = np.abs(np.degrees(np.angle(r))).max()
    gate(f"{name} magnitude tracks coated truth over CORE",
         float(mag.max()) < MAG_TOL_DB, f"(worst {mag.max():.2f} dB)")
    gate(f"{name} bare-PEC calibration leaves < {PHASE_TOL_DEG:g} deg "
         "residual phase on the coated anchor",
         float(phase) < PHASE_TOL_DEG,
         f"(worst {phase:.1f} deg)")

print(f"\n{'=' * 76}")
print("ALL GATES PASSED" if not _fails else f"{len(_fails)} FAILED",
      f"  ({time.time()-t0:.0f} s)")
for f in _fails:
    print(f"   FAILED: {f}")
print("=" * 76)
sys.exit(1 if _fails else 0)
