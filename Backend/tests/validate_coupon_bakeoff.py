#!/usr/bin/env python3
"""
Coupon bake-off: capsule vs floating TYPE-1 sheets vs on-body IBC taper.

All three characterise the SAME groove and are scored against the SAME BoR
ground truth (the ring gate's cached grooved/smooth cylinder).  Question: which
coupon reproduces the truth best -- especially the HH (TM) magnitude that was
the capsule's weak point -- and how SHORT can it be?

Coupon families (each a differential featured-minus-clean pair):
  capsule   : flat top y=0 + groove, rounded end caps, flat bottom   (current)
  floating  : closed half-disc PEC body + two floating TYPE-1 resistive sheets
              off the corners, tapering PEC(corner)->eta0(free end)
  onbody    : closed body whose flat top is PEC center + IBC taper wings
              (eta0 outer -> PEC inner), flat bottom

Scored at matched TOTAL footprints (8 lambda and 12 lambda full width), each
split half-body / quarter-absorber for the two absorber designs.

Metrics per coupon x width: production-calibrated VV/HH magnitude and residual
phase versus truth over broadside +/-20, and the validity window. Uses the
cached BoR truth, so only 2D coupon solves cost time.

Run from tests/:  python3 validate_coupon_bakeoff.py
"""

import math
import sys
import time

import numpy as np

sys.path.insert(0, "..")

import validate_line_expansion as RG                                   # noqa: E402
from line_expand import (PSI_HH_DEG, PSI_VV_DEG, expand_perimeter,     # noqa: E402
                         seam_coefficients_from_2d,
                         surface_of_revolution_normal)

LAM = RG.LAM
DS = RG.DS
ETA0 = 376.730313668
FREQ = RG.FREQ_GHZ
PHI = np.arange(0.0, 180.1, 2.5)
ASPECTS = RG.ASPECTS
CORE = (70.0, 110.0)
WIDTHS_LAM = [8.0, 12.0]
MAG_TOL = 2.5


def _wrap(x):
    return np.degrees(np.angle(np.exp(1j * np.radians(x))))


def _groove_y(x, grooved):
    return -RG._groove_profile(np.asarray(x, float)) if grooved else np.zeros(len(np.atleast_1d(x)))


def _flat_top(x0, x1, grooved):
    n = max(2, int(math.ceil((x1 - x0) / DS)))
    xs = np.linspace(x0, x1, n + 1)
    ys = _groove_y(xs, grooved)
    return list(zip(xs.tolist(), np.atleast_1d(ys).tolist()))


def _pairs(pts):
    return [{"x1": pts[i][0], "y1": pts[i][1], "x2": pts[i + 1][0], "y2": pts[i + 1][1]}
            for i in range(len(pts) - 1)]


# ── coupon builders (each takes total full-width in wavelengths) ──────────────

def capsule(total_lam, grooved):
    return RG.coupon(total_lam, grooved, thickness=0.9 * LAM)


def floating(total_lam, grooved):
    Wb = (total_lam / 2.0) * LAM / 2.0            # body half-width = total/2
    Ws = (total_lam / 4.0) * LAM                  # each sheet = total/4
    top = _flat_top(-Wb, Wb, grooved)
    nb = max(8, int(math.ceil(math.pi * Wb / DS)))
    ang = np.linspace(0.0, -math.pi, nb + 1)
    bottom = [(Wb * math.cos(a), Wb * math.sin(a)) for a in ang]
    body = {"name": "body", "seg_type": "2", "properties": ["2", "0", "0", "0", "0"],
            "point_pairs": _pairs(top + bottom[1:])}
    nsh = max(4, int(math.ceil(Ws / DS)))
    xl = np.linspace(-Wb, -Wb - Ws, nsh + 1)
    xr = np.linspace(Wb, Wb + Ws, nsh + 1)
    sL = {"name": "sL", "seg_type": "1", "properties": ["1", "0", "1", "0", "0"],
          "point_pairs": _pairs([(x, 0.0) for x in xl])}
    sR = {"name": "sR", "seg_type": "1", "properties": ["1", "0", "1", "0", "0"],
          "point_pairs": _pairs([(x, 0.0) for x in xr])}
    return {"title": "floating", "segments": [body, sL, sR],
            "ibcs": [["1", "cosine", "0", "0", str(ETA0), "0"]], "dielectrics": []}


def onbody(total_lam, grooved, thick=0.9):
    Wp = (total_lam / 2.0) * LAM / 2.0            # PEC center half-width = total/2
    Ww = (total_lam / 4.0) * LAM                  # each wing = total/4
    W = Wp + Ww
    t = thick * LAM
    xlw = np.linspace(-W, -Wp, max(4, int(math.ceil(Ww / DS))) + 1)
    xrw = np.linspace(Wp, W, max(4, int(math.ceil(Ww / DS))) + 1)
    lw = {"name": "lw", "seg_type": "2", "properties": ["2", "0", "1", "0", "0"],
          "point_pairs": _pairs([(x, 0.0) for x in xlw])}
    ctr = {"name": "ctr", "seg_type": "2", "properties": ["2", "0", "0", "0", "0"],
           "point_pairs": _pairs(_flat_top(-Wp, Wp, grooved))}
    rw = {"name": "rw", "seg_type": "2", "properties": ["2", "0", "2", "0", "0"],
          "point_pairs": _pairs([(x, 0.0) for x in xrw])}
    nb = max(2, int(math.ceil(2 * W / DS)))
    xb = np.linspace(W, -W, nb + 1)
    close = [(W, 0.0), (W, -t)] + [(x, -t) for x in xb[1:]] + [(-W, 0.0)]
    cl = {"name": "cl", "seg_type": "2", "properties": ["2", "0", "0", "0", "0"],
          "point_pairs": _pairs(close)}
    return {"title": "onbody", "segments": [lw, ctr, rw, cl],
            "ibcs": [["1", "cosine", str(ETA0), "0", "0", "0"],
                     ["2", "cosine", "0", "0", str(ETA0), "0"]], "dielectrics": []}


COUPONS = [("capsule", capsule), ("floating", floating), ("onbody", onbody)]

# ── BoR ground truth + ring expansion setup (cached) ─────────────────────────
print("=" * 90)
print(f"Coupon bake-off — groove w={RG.W_GROOVE/LAM:.2f}lam h={RG.H_GROOVE/LAM:.2f}lam "
      f"@ {FREQ} GHz, scored vs BoR truth")
print("=" * 90)
bor = RG.load_cached_bor()
dF = {"vv": np.asarray(bor["groove"]["amp_vv"]) - np.asarray(bor["smooth"]["amp_vv"]),
      "hh": np.asarray(bor["groove"]["amp_hh"]) - np.asarray(bor["smooth"]["amp_hh"])}
per = RG.ring_perimeter(RG.A_BODY, 0.0)
nfn = surface_of_revolution_normal(RG.cylinder_gen(False))
th = np.radians(ASPECTS)
dirs = np.column_stack([np.sin(th), np.zeros_like(th), np.cos(th)])
core = (ASPECTS >= CORE[0]) & (ASPECTS <= CORE[1])


def score(coef):
    exp = expand_perimeter(
        per, coef, nfn, dirs,
        psi_tm_deg=PSI_HH_DEG, psi_te_deg=PSI_VV_DEG)
    out = {}
    for ch in ("vv", "hh"):
        r = dF[ch][core] / exp[f"F_{ch}"][core]
        mag = float(np.abs(20 * np.log10(np.abs(r))).max())
        phase = float(np.abs(np.degrees(np.angle(r))).max())
        win = 0.0
        for half in np.arange(5.0, 55.0, 5.0):
            sel = np.abs(ASPECTS - 90.0) <= half + 1e-9
            rr = dF[ch][sel] / exp[f"F_{ch}"][sel]
            if (np.abs(20 * np.log10(np.abs(rr))).max() <= MAG_TOL
                    and np.abs(np.degrees(np.angle(rr))).max() <= 25.0):
                win = half
            else:
                break
        out[ch] = {"phase": phase, "mag": mag, "win": win}
    return out


rows = []
t0 = time.time()
for name, fn in COUPONS:
    for wl in WIDTHS_LAM:
        t1 = time.time()
        try:
            coef = seam_coefficients_from_2d(fn(wl, True), fn(wl, False), FREQ, PHI,
                                             geometry_units="meters", label=f"{name}{wl}")
            s = score(coef)
            rows.append((name, wl, s, time.time() - t1))
            print(f"  {name:9s} {wl:4.0f}lam   "
                  f"VV mag {s['vv']['mag']:.2f} phase {s['vv']['phase']:5.1f} win{s['vv']['win']:3.0f}   "
                  f"HH mag {s['hh']['mag']:.2f} phase {s['hh']['phase']:5.1f} win{s['hh']['win']:3.0f}   "
                  f"({time.time()-t1:.0f}s)")
        except Exception as e:
            rows.append((name, wl, None, time.time() - t1))
            print(f"  {name:9s} {wl:4.0f}lam   FAILED: {type(e).__name__}: {str(e)[:80]}")

print("\n" + "=" * 90)
print("Verdict — HH (TM) magnitude was the capsule's weak point; lower mag + wider window = better")
print("=" * 90)
print(f"  {'coupon':9s} {'width':>6}   {'VV mag':>7} {'VV win':>7}   {'HH mag':>7} {'HH win':>7}   "
      f"{'VVphase':>7} {'HHphase':>8}")
for name, wl, s, dt in rows:
    if s is None:
        print(f"  {name:9s} {wl:5.0f}l   FAILED")
        continue
    print(f"  {name:9s} {wl:5.0f}l   {s['vv']['mag']:6.2f}dB {s['vv']['win']:6.0f}   "
          f"{s['hh']['mag']:6.2f}dB {s['hh']['win']:6.0f}   "
          f"{s['vv']['phase']:7.1f} {s['hh']['phase']:8.1f}")

# pick best HH accuracy at the SHORTEST width
ok = [(n, wl, s) for (n, wl, s, _) in rows if s is not None]
if ok:
    best = min(ok, key=lambda r: (r[2]["hh"]["mag"], r[1]))
    print(f"\n  best HH magnitude match: {best[0]} at {best[1]:.0f} lambda "
          f"(HH {best[2]['hh']['mag']:.2f} dB, VV {best[2]['vv']['mag']:.2f} dB)")
    cap8 = next((s for (n, wl, s, _) in rows if n == "capsule" and wl == 8.0 and s), None)
    if cap8:
        print(f"  (capsule @ 8 lambda for reference: HH {cap8['hh']['mag']:.2f} dB, "
              f"VV {cap8['vv']['mag']:.2f} dB)")
print(f"\n  total {time.time()-t0:.0f}s")
print("=" * 90)
