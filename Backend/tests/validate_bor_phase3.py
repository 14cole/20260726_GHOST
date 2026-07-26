"""Phase-3 gate battery: PMCHWT dielectrics and coated PEC for the BoR solver.

Gate (BOR_SOLVER_PLAN.md): dielectric sphere and coated PEC sphere vs Mie,
lossless + lossy, both pols, <= 0.1 dB.  Plus degenerate-limit anchors that
detect sign errors the direct comparisons could miss.
"""
import math
import sys
import warnings

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from bor_solver import (solve_bor_dielectric, solve_bor_coated_pec,
                        sphere_generatrix)
import mie_sphere as M

warnings.filterwarnings("ignore", category=RuntimeWarning)
ok = True


def gate(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {label} {detail}")


a = 0.1
ANGLES = [0.0, 45.0, 90.0]


def worst_db(out, ref):
    return max(abs(10 * math.log10(s / ref))
               for s in out["sigma_vv"] + out["sigma_hh"])


# 1. Homogeneous dielectric spheres vs Mie (lossless, lossy, magnetic,
#    high contrast), 3 aspects x both pols.
for label, ka, eps, mu, n in (
    ("lossless eps=2.56", 1.5, 2.56, 1.0, 48),
    ("lossy eps=2.5-1.2j", 2.0, 2.5 - 1.2j, 1.0, 48),
    ("magnetic eps=1.8-0.3j mu=1.6-0.9j", 2.0, 1.8 - 0.3j, 1.6 - 0.9j, 48),
    ("high contrast eps=9", 1.5, 9.0, 1.0, 72),
):
    f = ka * M.C0 / (2 * math.pi * a)
    ref = M.sigma_dielectric_sphere(a, eps, mu, f)
    out = solve_bor_dielectric(sphere_generatrix(a, n), f, ANGLES, eps, mu)
    w = worst_db(out, ref)
    gate(f"dielectric sphere {label} <= 0.1 dB", w < 0.1, f"(worst {w:.3f})")

# 2. eps = mu = 1 body scatters (almost) nothing: null vs same-size PEC.
f = 2.0 * M.C0 / (2 * math.pi * a)
pec = M.sigma_pec_sphere(a, f)
out = solve_bor_dielectric(sphere_generatrix(a, 48), f, [0.0, 60.0], 1.0, 1.0)
null = max(10 * math.log10(s / pec) for s in out["sigma_vv"] + out["sigma_hh"])
gate("eps=1 body null <= -30 dB rel PEC", null < -30.0, f"(depth {null:.0f} dB)")

# 3. Coated PEC spheres vs Mie (lossy, thin, magnetic RAM, lossless).
for label, b, eps, mu, no, nc in (
    ("lossy eps=3-0.5j b=0.06", 0.06, 3.0 - 0.5j, 1.0, 48, 32),
    ("thin coating b=0.09 eps=2-0.5j", 0.09, 2.0 - 0.5j, 1.0, 48, 44),
    ("magnetic RAM b=0.07 eps=1.8-0.3j mu=1.6-0.9j", 0.07, 1.8 - 0.3j, 1.6 - 0.9j, 48, 36),
    ("lossless eps=4 b=0.06", 0.06, 4.0, 1.0, 64, 40),
):
    ref = M.sigma_coated_pec_sphere(b, a, eps, mu, f)
    out = solve_bor_coated_pec(sphere_generatrix(a, no), sphere_generatrix(b, nc),
                               f, ANGLES, eps, mu)
    w = worst_db(out, ref)
    gate(f"coated PEC sphere {label} <= 0.1 dB", w < 0.1, f"(worst {w:.3f})")

# 4. Degenerate coating eps=1: equals the bare PEC sphere of the CORE radius
#    (exercises every cross operator with an "air layer").
b = 0.07
ref = M.sigma_pec_sphere(b, f)
out = solve_bor_coated_pec(sphere_generatrix(a, 48), sphere_generatrix(b, 36),
                           f, [0.0, 60.0], 1.0, 1.0)
w = worst_db(out, ref)
gate("eps=1 coating == bare PEC core <= 0.05 dB", w < 0.05, f"(worst {w:.3f})")

# 5. Mirror symmetry (sphere maps to itself under z -> -z): theta vs 180-theta.
eps = 2.5 - 1.2j
o1 = solve_bor_dielectric(sphere_generatrix(a, 48), f, [35.0], eps)
o2 = solve_bor_dielectric(sphere_generatrix(a, 48), f, [145.0], eps)
d = max(abs(10 * math.log10(o1["sigma_vv"][0] / o2["sigma_vv"][0])),
        abs(10 * math.log10(o1["sigma_hh"][0] / o2["sigma_hh"][0])))
gate("dielectric mirror symmetry", d < 1e-6, f"(max {d:.2e} dB)")

print("ALL PHASE-3 GATES PASS" if ok else "PHASE-3 GATES FAILED")
