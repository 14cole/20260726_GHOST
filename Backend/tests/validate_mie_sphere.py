"""Phase-0 gates: anchor battery for mie_sphere.py (all independent of textbook coefficient tables)."""
import math
import sys

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mie_sphere as M

C0 = M.C0
ok = True

def check(label, val, ref, rtol, note=""):
    global ok
    rel = abs(val - ref) / max(abs(ref), 1e-300)
    good = rel <= rtol
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {label}: {val:.6e} vs {ref:.6e}  rel={rel:.2e} (tol {rtol:.0e}) {note}")

def rayleigh(k, a, eps, mu):
    return 4*math.pi * k**4 * a**6 * abs((eps-1)/(eps+2) - (mu-1)/(mu+2))**2

# ── 1. Rayleigh dipole limits (ka = 0.01; next correction O(ka^2) ~ 1e-4) ──
a, f = 0.001, 0.01e9 * (0.01 / (2*math.pi*0.01e9*0.001/C0))  # set ka = 0.01
k = 2*math.pi*f/C0
assert abs(k*a - 0.01) < 1e-12
check("PEC Rayleigh (9 pi k^4 a^6)", M.sigma_pec_sphere(a, f), 9*math.pi*k**4*a**6, 5e-4)
check("diel eps=3 Rayleigh", M.sigma_dielectric_sphere(a, 3.0, 1.0, f), rayleigh(k, a, 3.0, 1.0), 5e-4)
check("magnetic mu=3 Rayleigh", M.sigma_dielectric_sphere(a, 1.0, 3.0, f), rayleigh(k, a, 1.0, 3.0), 5e-4)
check("eps=3,mu=2 Rayleigh (pins A/B relative sign)",
      M.sigma_dielectric_sphere(a, 3.0, 2.0, f), rayleigh(k, a, 3.0, 2.0), 5e-4)

# ── 2. Geometric-optics limit: sigma/(pi a^2) -> 1 ──
for ka, tol in ((50.0, 0.03), (300.0, 0.006)):
    a2 = 0.5
    f2 = ka * C0 / (2*math.pi*a2)
    got = M.sigma_pec_sphere(a2, f2) / (math.pi * a2**2)
    check(f"PEC GO limit ka={ka:g}: sigma/pi a^2", got, 1.0, tol)

# ── 3. Mie resonance region sanity: max sigma/pi a^2 in [0.8,1.3] ~ 3.66 ──
a3 = 0.1
kas = np.linspace(0.8, 1.3, 251)
vals = [M.sigma_pec_sphere(a3, ka*C0/(2*math.pi*a3)) / (math.pi*a3**2) for ka in kas]
peak = max(vals)
good = 3.6 <= peak <= 3.7
ok &= good
print(f"{'PASS' if good else 'FAIL'}  PEC resonance peak sigma/pi a^2 = {peak:.4f} (expect ~3.66) at ka={kas[int(np.argmax(vals))]:.3f}")

# ── 4. Optical theorem ──
a4, f4 = 0.1, 3e9
sca, ext = M.cross_sections_dielectric_sphere(a4, 4.0, 1.0, f4)
check("lossless: sigma_ext == sigma_sca", ext, sca, 1e-10)
sca_l, ext_l = M.cross_sections_dielectric_sphere(a4, 4.0-1.0j, 1.0, f4)
good = ext_l > sca_l > 0
ok &= good
print(f"{'PASS' if good else 'FAIL'}  lossy: sigma_ext ({ext_l:.4e}) > sigma_sca ({sca_l:.4e})")

# ── 5. Impedance sphere limits ──
zs0 = M.sigma_impedance_sphere(a4, f4, 0.0)
pec = M.sigma_pec_sphere(a4, f4)
check("Zs=0 == PEC", zs0, pec, 1e-12)
zmatch = M.sigma_impedance_sphere(a4, f4, M.ETA0)
good = zmatch < 1e-8 * pec
ok &= good
print(f"{'PASS' if good else 'FAIL'}  Weston's theorem: Zs=eta0 backscatter == 0 "
      f"(got {zmatch:.2e} m^2 vs PEC {pec:.2e})")
# and a mid-impedance value must sit strictly between: 0 < sigma(0.5 eta0) < PEC
zmid = M.sigma_impedance_sphere(a4, f4, 0.5 * M.ETA0)
good = 0 < zmid < pec
ok &= good
print(f"{'PASS' if good else 'FAIL'}  Zs=eta0/2 between 0 and PEC: {10*math.log10(zmid/pec):.1f} dB rel PEC")

# ── 6. Coated-sphere degenerate limits ──
coat_air = M.sigma_coated_pec_sphere(0.08, 0.1, 1.0, 1.0, f4)
pec_core = M.sigma_pec_sphere(0.08, f4)
check("coat eps=1 == bare PEC(core)", coat_air, pec_core, 1e-10)
thin = M.sigma_coated_pec_sphere(0.1 - 1e-9, 0.1, 3.0, 1.0, f4)
check("thickness->0 == bare PEC", thin, M.sigma_pec_sphere(0.1, f4), 1e-5)
tiny_core = M.sigma_coated_pec_sphere(1e-6, 0.1, 3.0-0.5j, 1.2, f4)
homog = M.sigma_dielectric_sphere(0.1, 3.0-0.5j, 1.2, f4)
check("core->0 == homogeneous dielectric sphere", tiny_core, homog, 1e-6)

# ── 7. Bistatic consistency: theta=180 equals monostatic; VV==HH at back ──
svv, shh = M.sigma_bistatic_pec_sphere(a4, f4, 180.0)
check("bistatic 180 (VV) == monostatic", svv, pec, 1e-10)
check("bistatic 180 (HH) == monostatic", shh, pec, 1e-10)

# ── 8. Canonical published value: PEC sphere ka=2pi a/lambda=1, sigma/pi a^2 ──
# The Mie series at ka = 1.0 gives sigma/(pi a^2) ~= 3.65 (classic curve).
got = M.sigma_pec_sphere(a3, 1.0*C0/(2*math.pi*a3)) / (math.pi*a3**2)
good = 3.5 <= got <= 3.8
ok &= good
print(f"{'PASS' if good else 'FAIL'}  PEC ka=1: sigma/pi a^2 = {got:.4f} (classic ~3.65)")

print("ALL GATES PASS" if ok else "GATES FAILED")
