"""IBC (impedance) cylinder: solver vs analytic Robin-BC Mie series, both alpha signs."""
import sys, math
import numpy as np
from scipy import special as sp

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rcs_solver as R

C0 = 299_792_458.0
ETA0 = 376.730313668

def sigma_ibc_cylinder(radius, freq_hz, pol, Zs, sign=+1):
    """Analytic impedance-cylinder backscatter, e^{+jwt}, H^(2) outgoing.

    BC with outward normal (into air), rho-derivative at rho=a:
      TM: dEz/drho = +j k (eta0/Zs) Ez      (E_z = Zs * H_phi_tan relation)
      TE: dHz/drho = +j k (Zs/eta0) Hz
    'sign' flips the RHS to probe the opposite convention.
    """
    k = 2*np.pi*freq_hz/C0
    ka = k*radius
    N = int(np.ceil(ka + 4.05*ka**(1/3) + 12))
    n = np.arange(-N, N+1)
    if pol == 'TM':
        B = sign * 1j * k * (ETA0/Zs)
    else:
        B = sign * 1j * k * (Zs/ETA0)
    num = k*sp.jvp(n, ka) - B*sp.jv(n, ka)
    den = k*sp.h2vp(n, ka) - B*sp.hankel2(n, ka)
    a_n = -num/den
    amp = np.sum(a_n * (-1.0)**n)
    return (4.0/k)*abs(amp)**2

def circle_pairs(radius, n_seg, cw=True):
    th = np.linspace(0.0, 2.0*np.pi, n_seg+1)
    if cw:
        th = th[::-1]
    xs = radius*np.cos(th); ys = radius*np.sin(th)
    return [{"x1": float(xs[i]), "y1": float(ys[i]), "x2": float(xs[i+1]), "y2": float(ys[i+1])}
            for i in range(n_seg)]

radius, freq, nseg = 0.5, 0.3, 128
Zs = complex(100.0, 50.0)   # ohms
snap = {"segments": [{"name": "ibc", "properties": ["2", "1", "1", "0", "0"],
                      "point_pairs": circle_pairs(radius, nseg, cw=True)}],
        "ibcs": [["1", "constant", str(Zs.real), str(Zs.imag), "0", "0"]],
        "dielectrics": []}

for pol in ("TM", "TE"):
    res = R.solve_monostatic_rcs_2d(geometry_snapshot=snap, frequencies_ghz=[freq],
                                    elevations_deg=[0.0], polarization=pol,
                                    geometry_units="meters")
    sig = res["samples"][0]["rcs_linear"]
    rp = sigma_ibc_cylinder(radius, freq*1e9, pol, Zs, +1)
    rm = sigma_ibc_cylinder(radius, freq*1e9, pol, Zs, -1)
    print(f"{pol}: solver {sig:.6f}   analytic(+) {rp:.6f} rel {abs(sig-rp)/rp:.3e}   "
          f"analytic(-) {rm:.6f} rel {abs(sig-rm)/rm:.3e}")
