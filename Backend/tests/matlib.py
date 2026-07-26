"""Shared helpers for material-handling checks (no top-level test execution)."""
import math, sys
import numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy import special as sp
from rcs_solver import solve_monostatic_rcs_2d
import mie_reference as mie

C0 = 299_792_458.0
ETA0 = 376.730313668


def circle_pairs(radius, nseg=64, cw=True):
    pairs = []
    for i in range(nseg):
        th0 = 2 * math.pi * i / nseg
        th1 = 2 * math.pi * (i + 1) / nseg
        if cw:
            th0, th1 = -th0, -th1
        pairs.append({
            "x1": radius * math.cos(th0), "y1": radius * math.sin(th0),
            "x2": radius * math.cos(th1), "y2": radius * math.sin(th1),
        })
    return pairs


def snapshot(segments, ibcs=None, diels=None):
    return {"title": "test", "segment_count": len(segments),
            "segments": segments, "ibcs": ibcs or [], "dielectrics": diels or []}


def run(snap, pol, freq_ghz=0.3):
    res = solve_monostatic_rcs_2d(
        geometry_snapshot=snap, frequencies_ghz=[freq_ghz],
        elevations_deg=[0.0], polarization=pol, geometry_units="meters")
    return res["samples"][0]["rcs_linear"], res["metadata"]["formulation"]


def sigma_impedance_cylinder(radius, freq_hz, Zs, pol):
    k = 2 * math.pi * freq_hz / C0
    ka = k * radius
    N = mie._nmax_for_ka(ka)
    n_arr = np.arange(-N, N + 1)
    Jn = sp.jn(n_arr, ka); Jnp = sp.jvp(n_arr, ka, 1)
    Hn = sp.hankel2(n_arr, ka); Hnp = sp.h2vp(n_arr, ka, 1)
    if pol == "TM":
        if abs(Zs) < 1e-12:
            a_n = -Jn / Hn
        else:
            cok = 1j * ETA0 / Zs
            a_n = -(Jnp - cok * Jn) / (Hnp - cok * Hn)
    else:
        cok = 1j * Zs / ETA0
        a_n = -(Jnp - cok * Jn) / (Hnp - cok * Hn)
    amp = np.sum(a_n * (-1.0) ** n_arr)
    return float((4.0 / k) * abs(amp) ** 2)


def db(x):
    return 10 * math.log10(max(x, 1e-30))
