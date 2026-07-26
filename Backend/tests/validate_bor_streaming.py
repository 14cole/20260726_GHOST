"""Phase-7b gate battery: streaming far-block assembly.

The streamed path uses the same xi grid, the same FFT, and the same
Galerkin points as the table path — so its first gate is bit-level
equivalence.  The capability gate then runs a CFIE sphere at a size whose
table build would thrash memory (the ka=20/N_t=800 demo — 48 GB of tables
vs 2 GB of streamed single-precision blocks, 0.0019 dB vs Mie — is
recorded in BOR_SOLVER_PLAN.md; the battery uses ka=10 to stay fast).
"""
import math
import sys
import warnings

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from bor_solver import solve_bor, sphere_generatrix, estimate_bor_table_gb
from bor_streaming import estimate_streaming_gb
import mie_sphere as M

warnings.filterwarnings("ignore", category=RuntimeWarning)
ok = True


def gate(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {label} {detail}")


a = 0.1

# 1. Bit-level equivalence with the table path, all three formulations.
f = 2.0 * M.C0 / (2 * math.pi * a)
pts = sphere_generatrix(a, 48)
thetas = [0.0, 45.0, 90.0, 135.0, 180.0]
for label, kw in (
    ("EFIE", dict(formulation="efie")),
    ("CFIE", dict(formulation="cfie")),
    ("IBC", dict(formulation="efie", zs=150 - 80j)),
):
    ot = solve_bor(pts, f, thetas, assembly="tables", **kw)
    os_ = solve_bor(pts, f, thetas, assembly="streaming", **kw)
    d = max(abs(np.array(os_["sigma_vv"] + os_["sigma_hh"]) /
                np.array(ot["sigma_vv"] + ot["sigma_hh"]) - 1))
    gate(f"streaming == tables ({label})", d < 1e-12, f"(rel {d:.1e})")

# 2. Capability: CFIE sphere at ka=10 / 400 elements — the table path's
#    construction transient thrashes here; streaming solves it in ~1 min.
ka = 10.0
fk = ka * M.C0 / (2 * math.pi * a)
ref = M.sigma_pec_sphere(a, fk)
out = solve_bor(sphere_generatrix(a, 400), fk, [0.0, 60.0, 120.0],
                formulation="cfie", workers=8)
w = max(abs(10 * math.log10(s / ref)) for s in out["sigma_vv"] + out["sigma_hh"])
gate("streaming CFIE ka=10 sphere <= 0.01 dB", w < 0.01, f"(worst {w:.4f})")
gate("large case auto-selected streaming", out["assembly"] == "streaming",
     f"({out['assembly']}/{out['table_precision']})")
gate("residual clean", out["linear_residual"] < 1e-10,
     f"({out['linear_residual']:.1e})")

# 3. Small cases stay on the (validated) table path under auto.
out = solve_bor(pts, f, [30.0], formulation="cfie")
gate("small case auto-selects tables", out["assembly"] == "tables")

# 4. Native sampling kernel (phase 7c): report availability; when present,
#    the equivalence gates above already ran through it.  Also check the
#    NumPy fallback agrees with the native path.
import bor_streaming as BSTR
if BSTR._NATIVE is not None:
    from bor_solver import BorPecSolver
    s1 = BorPecSolver(pts, f)
    s1.enable_streaming(6, efie=True, mfie=True)
    saved = BSTR._NATIVE
    BSTR._NATIVE = None
    s2 = BorPecSolver(pts, f)
    s2.enable_streaming(6, efie=True, mfie=True)
    BSTR._NATIVE = saved
    d = max(np.max(np.abs(s1._stream.Z - s2._stream.Z)) /
            np.max(np.abs(s2._stream.Z)),
            np.max(np.abs(s1._stream.K - s2._stream.K)) /
            np.max(np.abs(s2._stream.K)))
    gate("native kernel == NumPy fallback", d < 1e-12, f"(rel {d:.1e})")
else:
    gate("native kernel not present (NumPy fallback in use)", True,
         "(compile bor_stream_kernel.c for the speedup)")

# 5. Phase-7d mode-block re-sweeps: a tiny stream budget forces multiple
#    sampling sweeps over aligned mode ranges — results must stay
#    bit-identical to the tables, including under threaded mode waves.
ot = solve_bor(pts, f, thetas, assembly="tables", formulation="cfie")
os_ = solve_bor(pts, f, thetas, assembly="streaming",
                stream_budget_gb=0.002, formulation="cfie")
d = max(abs(np.array(os_["sigma_vv"] + os_["sigma_hh"]) /
            np.array(ot["sigma_vv"] + ot["sigma_hh"]) - 1))
gate("mode-blocked streaming == tables", d < 1e-12,
     f"(block {os_['stream_mode_block']}, sweeps {os_['stream_sweeps']}, rel {d:.1e})")
gate("mode blocking actually re-swept", os_["stream_sweeps"] >= 2)
os_ = solve_bor(pts, f, thetas, assembly="streaming",
                stream_budget_gb=0.002, workers=4, formulation="cfie")
d = max(abs(np.array(os_["sigma_vv"] + os_["sigma_hh"]) /
            np.array(ot["sigma_vv"] + ot["sigma_hh"]) - 1))
gate("mode-blocked + threaded waves == tables", d < 1e-12,
     f"(block {os_['stream_mode_block']}, sweeps {os_['stream_sweeps']}, rel {d:.1e})")

# 6. Memory estimates: streamed nodal blocks ~16x below the tables.
et = estimate_bor_table_gb(800, 32, "cfie")
es = estimate_streaming_gb(800, 32, "cfie")
gate("streaming estimate ~12-16x below tables", 10.0 < et / es < 20.0,
     f"(tables {et:.0f} GB, streaming {es:.1f} GB)")

print("ALL STREAMING GATES PASS" if ok else "STREAMING GATES FAILED")
