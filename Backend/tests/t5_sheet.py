"""Closed circular resistive sheet: solver vs analytic jump-BC Mie series."""
import sys
import numpy as np
from scipy import special as sp

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rcs_solver as R

C0 = 299_792_458.0
ETA0 = 376.730313668

def sigma_sheet_cylinder(radius, freq_hz, pol, Zs):
    """Cylindrical resistive/reactive sheet at rho=a, e^{+jwt}, H^(2) outgoing."""
    k = 2*np.pi*freq_hz/C0
    ka = k*radius
    om_eps = k/ETA0   # omega*eps0
    om_mu = k*ETA0    # omega*mu0
    N = int(np.ceil(ka + 4.05*ka**(1/3) + 12))
    ns = np.arange(-N, N+1)
    a_out = np.zeros(ns.size, dtype=complex)
    for i, n in enumerate(ns):
        J, Jp = sp.jv(n, ka), sp.jvp(n, ka, 1)
        H, Hp = sp.hankel2(n, ka), sp.h2vp(n, ka, 1)
        if pol == 'TM':
            # E_z continuous: J + a H = c J
            # H_phi jump:  (k/(j om_mu)) [Jp + a Hp - c Jp] = (1/Zs)(J + a H)
            # unknowns (a, c):
            A = np.array([[H, -J],
                          [(k/(1j*om_mu))*Hp - (H/Zs), -(k/(1j*om_mu))*Jp]], dtype=complex)
            b = np.array([-J, -(k/(1j*om_mu))*Jp + J/Zs], dtype=complex)
        else:
            # E_phi continuous: k[Jp + a Hp] = k c Jp
            # H_z jump: cJ - (J + aH) = -(k/(j om_eps Zs)) [Jp + a Hp]
            A = np.array([[Hp, -Jp],
                          [-H + (k/(1j*om_eps*Zs))*Hp, J]], dtype=complex)
            b = np.array([-Jp, J - (k/(1j*om_eps*Zs))*Jp], dtype=complex)
        sol = np.linalg.solve(A, b)
        a_out[i] = sol[0]
    amp = np.sum(a_out * (-1.0)**ns)
    return (4.0/k)*abs(amp)**2

def circle_pairs(radius, n_seg, cw=True):
    th = np.linspace(0.0, 2.0*np.pi, n_seg+1)
    if cw:
        th = th[::-1]
    xs = radius*np.cos(th); ys = radius*np.sin(th)
    return [{"x1": float(xs[i]), "y1": float(ys[i]), "x2": float(xs[i+1]), "y2": float(ys[i+1])}
            for i in range(n_seg)]

radius, freq, nseg = 0.5, 0.3, 128
Zs = complex(150.0, -80.0)
snap = {"segments": [{"name": "card", "properties": ["1", "1", "1", "0", "0"],
                      "point_pairs": circle_pairs(radius, nseg, cw=True)}],
        "ibcs": [["1", "constant", str(Zs.real), str(Zs.imag), "0", "0"]],
        "dielectrics": []}

for pol in ("TM", "TE"):
    try:
        res = R.solve_monostatic_rcs_2d(geometry_snapshot=snap, frequencies_ghz=[freq],
                                        elevations_deg=[0.0], polarization=pol,
                                        geometry_units="meters")
        sig = res["samples"][0]["rcs_linear"]
        form = res["metadata"].get("formulation", "?")
        ref = sigma_sheet_cylinder(radius, freq*1e9, pol, Zs)
        print(f"{pol}: solver {sig:.6f}   analytic {ref:.6f}   rel err {abs(sig-ref)/ref:.3e}  [{form}]")
    except Exception as e:
        print(f"{pol}: FAILED {type(e).__name__}: {e}")
