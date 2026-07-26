"""Phase-2 gate battery: CFIE + IBC for the BoR solver."""
import math
import sys

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from bor_solver import solve_bor, BorPecSolver, sphere_generatrix
from bor_kernels import ETA0
import mie_sphere as M

ok = True

def gate(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {label} {detail}")

a = 0.1

# 1. CFIE / MFIE sphere accuracy off-resonance.
for form, tol in (("cfie", 0.1), ("mfie", 0.15)):
    worst = 0.0
    for ka, n in ((1.0, 32), (3.0, 48)):
        f = ka * M.C0 / (2 * math.pi * a)
        ref = M.sigma_pec_sphere(a, f)
        out = solve_bor(sphere_generatrix(a, n), f, [0.0, 40.0, 90.0], formulation=form)
        worst = max(worst, max(abs(10 * math.log10(s / ref))
                               for s in out["sigma_vv"] + out["sigma_hh"]))
    gate(f"{form} sphere ka=1,3 <= {tol} dB", worst < tol, f"(worst {worst:.3f})")

# 2. Interior resonance: EFIE cond spikes at cavity ka, CFIE flat.
conds_e, conds_c = [], []
for ka in (2.72, 2.746, 2.77):
    f = ka * M.C0 / (2 * math.pi * a)
    s = BorPecSolver(sphere_generatrix(a, 48), f)
    Ze = s.assemble_mode(1, 3)
    Zc = 0.5 * Ze + 0.5 * ETA0 * s.assemble_mfie_mode(1, 3)
    mask = s.basis_mask(1)
    conds_e.append(np.linalg.cond(Ze[np.ix_(mask, mask)]))
    conds_c.append(np.linalg.cond(Zc[np.ix_(mask, mask)]))
gate("EFIE conditioning spikes at cavity ka",
     conds_e[1] > 1.5 * max(conds_e[0], conds_e[2]),
     f"(cond {conds_e[0]:.0f} -> {conds_e[1]:.0f} -> {conds_e[2]:.0f})")
gate("CFIE conditioning flat through resonance",
     max(conds_c) < 300, f"(max {max(conds_c):.0f})")

# 3. IBC sphere vs impedance-Mie (incl. the M-radiation far field).
f = 2.0 * M.C0 / (2 * math.pi * a)
worst = 0.0
for zs in (100 + 50j, 150 - 80j, 188.365, 300 + 150j):
    refz = M.sigma_impedance_sphere(a, f, zs)
    out = solve_bor(sphere_generatrix(a, 48), f, [0.0, 60.0], zs=zs)
    worst = max(worst, max(abs(10 * math.log10(s / refz))
                           for s in out["sigma_vv"] + out["sigma_hh"]))
gate("IBC sphere 4 impedances <= 0.05 dB", worst < 0.05, f"(worst {worst:.3f})")

# 4. Weston's theorem: Zs = eta0 -> deep null.
pec = M.sigma_pec_sphere(a, f)
out = solve_bor(sphere_generatrix(a, 48), f, [0.0, 45.0], zs=ETA0)
depth = max(10 * math.log10(s / pec) for s in out["sigma_vv"] + out["sigma_hh"])
gate("Weston null <= -50 dB rel PEC", depth < -50.0, f"(depth {depth:.0f} dB)")

# 5. Zs = 0 equals PEC.
out = solve_bor(sphere_generatrix(a, 48), f, [30.0], zs=0.0)
d = abs(10 * math.log10(out["sigma_vv"][0] / M.sigma_pec_sphere(a, f)))
gate("Zs=0 == PEC", d < 0.05, f"(diff {d:.3f} dB)")

# 6. Scalar vs per-element array identical.
o1 = solve_bor(sphere_generatrix(a, 48), f, [30.0], zs=150 - 80j)
o2 = solve_bor(sphere_generatrix(a, 48), f, [30.0], zs=np.full(48, 150 - 80j))
gate("scalar == per-element array",
     o1["sigma_vv"][0] == o2["sigma_vv"][0])

# 7. Tapered IBC mirror symmetry: taper A->B at theta == taper B->A at 180-theta
# (reflection z -> -z maps the sphere to itself and reverses the generatrix).
ne = 48
t = (np.arange(ne) + 0.5) / ne
zs_ab = (50.0 + 350.0 * t).astype(complex)          # nose(+z) 50 -> tail 400
zs_ba = zs_ab[::-1].copy()
oa = solve_bor(sphere_generatrix(a, ne), f, [35.0], zs=zs_ab)
ob = solve_bor(sphere_generatrix(a, ne), f, [145.0], zs=zs_ba)
d = abs(10 * math.log10(oa["sigma_vv"][0] / ob["sigma_vv"][0]))
d2 = abs(10 * math.log10(oa["sigma_hh"][0] / ob["sigma_hh"][0]))
gate("tapered IBC mirror symmetry", max(d, d2) < 1e-6,
     f"(VV {d:.2e} dB, HH {d2:.2e} dB)")

print("ALL PHASE-2 GATES PASS" if ok else "PHASE-2 GATES FAILED")
