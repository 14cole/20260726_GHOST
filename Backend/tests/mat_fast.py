"""Fast coarse-mesh sign checks: IBC cylinder TM/TE + TYPE 4 impedance backing."""
import math, sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matlib import circle_pairs, snapshot, sigma_impedance_cylinder, db, run
import mie_reference as mie

freq_ghz = 0.15   # ka ~ 1.57, coarser problem
freq_hz = freq_ghz * 1e9
a = 0.5
b = 0.6
NSEG = 40  # panels: 40*n

def pairs(r):
    return circle_pairs(r, nseg=NSEG)

print("--- fast PEC baseline ---", flush=True)
seg = [{"name": "cyl", "seg_type": "2", "properties": ["2", "2", "0", "0", "0"],
        "point_pairs": pairs(a)}]
for pol in ("TM", "TE"):
    got, form = run(snapshot(seg), pol, freq_ghz)
    ref = mie.sigma_pec_cylinder(a, freq_hz, pol)
    print(f"PEC {pol}: solver {db(got):8.3f}  mie {db(ref):8.3f}  diff {db(got)-db(ref):+7.3f}  [{form}]", flush=True)

print("--- fast IBC cylinder Z=188.4 and Z=376.7 ---", flush=True)
for Z in (188.365, 376.73):
    ib = [["1", "constant", str(Z), "0", "0", "0"]]
    seg = [{"name": "cyl", "seg_type": "2", "properties": ["2", "2", "1", "0", "0"],
            "point_pairs": pairs(a)}]
    for pol in ("TM", "TE"):
        got, form = run(snapshot(seg, ibcs=ib), pol, freq_ghz)
        ref = sigma_impedance_cylinder(a, freq_hz, Z, pol)
        print(f"IBC Z={Z:7.1f} {pol}: solver {db(got):8.3f}  mie {db(ref):8.3f}  diff {db(got)-db(ref):+7.3f}  [{form}]", flush=True)

print("--- fast TYPE4 impedance backing under air coating ---", flush=True)
for Z in (188.365,):
    dl = [["1", "1.0", "0", "1.0", "0"]]
    ib = [["1", "constant", str(Z), "0", "0", "0"]]
    segs = [
        {"name": "outer", "seg_type": "3", "properties": ["3", "2", "0", "1", "0"],
         "point_pairs": pairs(b)},
        {"name": "inner", "seg_type": "4", "properties": ["4", "2", "1", "1", "0"],
         "point_pairs": pairs(a)},
    ]
    for pol in ("TM", "TE"):
        got, form = run(snapshot(segs, ibcs=ib, diels=dl), pol, freq_ghz)
        ref = sigma_impedance_cylinder(a, freq_hz, Z, pol)
        print(f"T4 Z={Z:7.1f} {pol}: solver {db(got):8.3f}  mie {db(ref):8.3f}  diff {db(got)-db(ref):+7.3f}  [{form}]", flush=True)

print("--- fast coated PEC (TYPE3+TYPE4, zero IBC) ---", flush=True)
eps = complex(2.56, 0.0)
dl = [["1", str(eps.real), str(eps.imag), "1", "0"]]
segs = [
    {"name": "outer", "seg_type": "3", "properties": ["3", "2", "0", "1", "0"],
     "point_pairs": pairs(b)},
    {"name": "inner", "seg_type": "4", "properties": ["4", "2", "0", "1", "0"],
     "point_pairs": pairs(a)},
]
for pol in ("TM", "TE"):
    got, form = run(snapshot(segs, diels=dl), pol, freq_ghz)
    ref = mie.sigma_coated_pec_cylinder(a, b, eps, 1.0, freq_hz, pol)
    print(f"coat {pol}: solver {db(got):8.3f}  mie {db(ref):8.3f}  diff {db(got)-db(ref):+7.3f}  [{form}]", flush=True)

print("--- fast lossy dielectric cylinder eps=2-0.5j (TYPE 3) ---", flush=True)
eps = complex(2.0, -0.5)
dl = [["1", str(eps.real), str(eps.imag), "1", "0"]]
seg = [{"name": "cyl", "seg_type": "3", "properties": ["3", "2", "0", "1", "0"],
        "point_pairs": pairs(a)}]
for pol in ("TM", "TE"):
    got, form = run(snapshot(seg, diels=dl), pol, freq_ghz)
    ref = mie.sigma_dielectric_cylinder(a, eps, 1.0, freq_hz, pol)
    print(f"lossy {pol}: solver {db(got):8.3f}  mie {db(ref):8.3f}  diff {db(got)-db(ref):+7.3f}  [{form}]", flush=True)
print("DONE", flush=True)
