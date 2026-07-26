#!/usr/bin/env python3
"""
Body-shadowing (STL geometric occlusion) gate for occluder.py.

Self-contained: a synthetic blocker plate + known points, plus one integration
through expand_perimeter (a door shadowed vs unshadowed by the blocker).

  T0  binary STL round-trips through read_stl.
  T1  a point directly behind the blocker (look into it) is SHADOWED.
  T2  looking the other way (blocker behind the point) is VISIBLE.
  T3  a point off the blocker's footprint is VISIBLE.
  T4  a point in FRONT of the blocker (nearer the radar) is VISIBLE.
  T5  integration: a door whose line to the radar passes through the blocker
      is masked to ~0; move the blocker aside and it returns in full.

Run from tests/:  python3 validate_occluder.py
"""

import math
import os
import struct
import sys

import numpy as np

sys.path.insert(0, "..")

from occluder import Occluder, read_stl                                  # noqa: E402
from line_expand import SeamCoefficients, expand_perimeter, dbsm         # noqa: E402
import math as _m                                                       # noqa: E402

_fails = []


def gate(label, ok, note=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {note}")
    if not ok:
        _fails.append(label)


def write_binary_stl(path, tris):
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", len(tris)))
        for t in tris:
            n = np.cross(t[1] - t[0], t[2] - t[0])
            n = n / (np.linalg.norm(n) + 1e-30)
            fh.write(struct.pack("<3f", *n))
            for v in t:
                fh.write(struct.pack("<3f", *v))
            fh.write(struct.pack("<H", 0))


print("=" * 74)
print("Body-shadowing (STL geometric occlusion) gate")
print("=" * 74)

# a square blocker plate at x = 0.10, spanning y,z in [-0.05, 0.05], two triangles
P, H = 0.10, 0.05
plate = np.array([[[P, -H, -H], [P, H, -H], [P, H, H]],
                  [[P, -H, -H], [P, H, H], [P, -H, H]]], dtype=float)

print("\nT0 STL round-trip")
write_binary_stl("_occ_test.stl", plate)
back = read_stl("_occ_test.stl")
gate("T0 binary STL round-trips", back.shape == plate.shape and np.allclose(back, plate, atol=1e-6),
     f"({back.shape[0]} triangles)")
os.remove("_occ_test.stl")

occ = Occluder(plate)
print("\nT1-T4 point/blocker geometry")
gate("T1 point behind blocker (look toward it) is SHADOWED",
     not bool(occ.visible(np.array([[0., 0., 0.]]), np.array([1., 0., 0.]))[0]))
gate("T2 look away from blocker -> VISIBLE",
     bool(occ.visible(np.array([[0., 0., 0.]]), np.array([-1., 0., 0.]))[0]))
gate("T3 point off the blocker footprint -> VISIBLE",
     bool(occ.visible(np.array([[0., 0.12, 0.]]), np.array([1., 0., 0.]))[0]))
gate("T4 point in front of blocker (nearer radar) -> VISIBLE",
     bool(occ.visible(np.array([[0.2, 0., 0.]]), np.array([1., 0., 0.]))[0]))

print("\nT5 integration through expand_perimeter (door shadowed vs not)")
FREQ = 6.0
phi = np.arange(0.0, 180.1, 5.0)
coef = SeamCoefficients(FREQ, phi, np.full(len(phi), 0.01 + 0j), np.full(len(phi), 0.01 + 0j))
# a small door loop at x = 0.06, facing +x, y,z within [-0.02, 0.02]
s = 0.02
loop = [(0.06, -s, -s), (0.06, s, -s), (0.06, s, s), (0.06, -s, s), (0.06, -s, -s)]
per = np.array([[loop[i], loop[i + 1]] for i in range(len(loop) - 1)])
nfn = lambda p: np.tile([1.0, 0.0, 0.0], (len(np.atleast_2d(p)), 1))
look = np.array([[1.0, 0.0, 0.0]])                          # broadside to the door

F_open = expand_perimeter(per, coef, nfn, look)             # no blocker
# blocker COVERING the door footprint (x=0.10, spans the door in y,z)
occ_block = Occluder(plate)
F_shad = expand_perimeter(per, coef, nfn, look, occluder=occ_block)
# blocker moved ASIDE (y in [0.2,0.3]) -> does not cover the door
aside = plate.copy(); aside[:, :, 1] += 0.25
F_clear = expand_perimeter(per, coef, nfn, look, occluder=Occluder(aside))

open_lvl = abs(F_open["F_vv"][0])
gate("T5a door shadowed by blocker is masked to ~0",
     abs(F_shad["F_vv"][0]) < 1e-3 * open_lvl,
     f"(shadowed {abs(F_shad['F_vv'][0]):.2e} vs open {open_lvl:.2e})")
gate("T5b blocker moved aside -> door returns unchanged",
     abs(F_clear["F_vv"][0] - F_open["F_vv"][0]) < 1e-9 * open_lvl,
     f"(rel diff {abs(F_clear['F_vv'][0] - F_open['F_vv'][0]) / open_lvl:.1e})")
sig_shad = 4 * _m.pi * abs(F_shad["F_vv"][0]) ** 2
gate("T5c shadowed feature reads -200 dBsm (clean sentinel)",
     abs(float(dbsm(sig_shad)) - (-200.0)) < 1e-6, f"({float(dbsm(sig_shad)):.1f} dBsm)")

print(f"\n{'=' * 74}")
print("ALL GATES PASSED" if not _fails else f"{len(_fails)} FAILED")
for f in _fails:
    print(f"   FAILED: {f}")
print("=" * 74)
sys.exit(1 if _fails else 0)
