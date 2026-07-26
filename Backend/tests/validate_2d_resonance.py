"""Interior-resonance behavior of the 2D SLP paths + the cfie_alpha warning.

Findings pinned here (July 2026, from a user report of cfie_alpha having no
effect on a large PEC square):

1. cfie_alpha is inert on every monostatic path — now a loud warning.
2. That is SAFE for far-field accuracy: with the indirect SLP ansatz, a
   resonant null density sigma_0 has S sigma_0 = 0 on the contour, hence
   S sigma_0 == 0 everywhere outside (exterior uniqueness) — the null space
   does not radiate.  The gates verify <= 0.02 dB vs the Mie series AT the
   discrete interior resonance while the conditioning spikes ~40x.
3. A Robin-style CFIE (injecting a Burton-Miller coupling into the Robin
   alpha) was tried and REJECTED: for the SLP ansatz it imposes a second,
   physically false boundary condition and shifts the RCS ~9.5 dB
   everywhere.  Do not re-attempt without the Brakhage-Werner combined-
   source ansatz.
"""
import math
import sys
import warnings

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from geometry_io import parse_geometry, build_geometry_snapshot
from rcs_solver import (solve_monostatic_rcs_2d, MaterialLibrary,
                        _build_panels, _build_coupled_panel_info,
                        _build_linear_mesh_interface_aware,
                        _build_linear_coupled_infos,
                        _assemble_robin_bie_system)
import mie_reference as MR

warnings.filterwarnings("ignore")
ok = True


def gate(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {label} {detail}")


A_R = 0.05
C0 = 299_792_458.0
th = np.linspace(0, 2 * math.pi, 161)[::-1]
pts = [(A_R * math.cos(t), A_R * math.sin(t)) for t in th]
lines = "\n".join(f"{p[0]!r} {p[1]!r} {q[0]!r} {q[1]!r}"
                  for p, q in zip(pts[:-1], pts[1:]))
geo = ("Title: c\nSegment: c 2\nproperties: 2 0 0 0 0\n" + lines +
       "\nIBCS_Resistances:\nDielectrics:\n")
t_, s_, i_, d_ = parse_geometry(geo)
SNAP = build_geometry_snapshot(t_, s_, i_, d_)
MATS = MaterialLibrary.from_entries([], [], base_dir=".")
KA_RES = 2.4050          # discrete interior Dirichlet resonance (j_{0,1})


def cond_at(ka, pol):
    f_ghz = ka * C0 / (2 * math.pi * A_R) / 1e9
    k0 = 2 * math.pi * f_ghz * 1e9 / C0
    panels = _build_panels(SNAP, 1.0, C0 / (f_ghz * 1e9))
    infos_p = _build_coupled_panel_info(panels, MATS, f_ghz, pol, k0)
    mesh, _ = _build_linear_mesh_interface_aware(panels, infos_p)
    infos = _build_linear_coupled_infos(mesh, MATS, f_ghz, pol, k0)
    A, _, _ = _assemble_robin_bie_system(mesh, infos, pol, k0)
    return np.linalg.cond(A)


for pol in ("TM", "TE"):
    c_res = cond_at(KA_RES, pol)
    c_off = cond_at(2.2, pol)
    gate(f"{pol}: conditioning spikes at the interior resonance",
         c_res > 10.0 * c_off, f"(cond {c_off:.0f} -> {c_res:.0f})")
    for ka in (2.2, KA_RES, 3.8317):
        f_ghz = ka * C0 / (2 * math.pi * A_R) / 1e9
        r = solve_monostatic_rcs_2d(SNAP, [f_ghz], [15.0], pol,
                                    geometry_units="meters")
        ref = MR.sigma_pec_cylinder(A_R, f_ghz * 1e9, pol)
        err = abs(10 * math.log10(r["samples"][0]["rcs_linear"] / ref))
        gate(f"{pol} ka={ka:.4f}: far field immune (<= 0.02 dB vs Mie)",
             err < 0.02, f"(err {err:.4f}, residual "
             f"{r['samples'][0]['linear_residual']:.1e})")

# the dead-knob warning
f_ghz = 2.2 * C0 / (2 * math.pi * A_R) / 1e9
r = solve_monostatic_rcs_2d(SNAP, [f_ghz], [15.0], "TM",
                            geometry_units="meters", cfie_alpha=0.5)
warns = r["metadata"].get("warnings", []) or []
gate("cfie_alpha > 0 emits the no-effect warning",
     any("cfie_alpha" in str(w) and "NO EFFECT" in str(w) for w in warns))
r0 = solve_monostatic_rcs_2d(SNAP, [f_ghz], [15.0], "TM",
                             geometry_units="meters", cfie_alpha=0.0)
gate("cfie_alpha still does not change monostatic results",
     r["samples"][0]["rcs_linear"] == r0["samples"][0]["rcs_linear"])

print("ALL 2D RESONANCE GATES PASS" if ok else "2D RESONANCE GATES FAILED")
