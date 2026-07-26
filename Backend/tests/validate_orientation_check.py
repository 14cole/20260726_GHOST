"""Validate the hard-error orientation/consistency preflight."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")

from validation_2d_helpers import make_circle_segment
from rcs_solver import solve_monostatic_rcs_2d, validate_geometry_snapshot_for_solver
import mie_reference as mie


def expect(label, snap, should_raise, needle=""):
    try:
        validate_geometry_snapshot_for_solver(snap, base_dir=".")
        raised, msg = False, ""
    except ValueError as e:
        raised, msg = True, str(e)
    ok = raised == should_raise and (needle in msg if should_raise and needle else True)
    print(f"{'PASS' if ok else 'FAIL'}  {label}: raised={raised}" + (f"  [{msg.splitlines()[1][:100]}...]" if raised else ""))
    return ok


def seg_from_points(name, seg_type, pts, ibc="0", pos="0", neg="0"):
    pairs = [{"x1": pts[i][0], "y1": pts[i][1], "x2": pts[i+1][0], "y2": pts[i+1][1]}
             for i in range(len(pts) - 1)]
    return {"name": name, "seg_type": str(seg_type),
            "properties": [str(seg_type), "0", ibc, pos, neg], "point_pairs": pairs}


ok = True

# 1. Correct CW PEC circle -> no error, solves, matches Mie.
snap = {"title": "p", "segments": [make_circle_segment("c", 2, 0.05, 64, cw=True)],
        "ibcs": [], "dielectrics": []}
ok &= expect("CW PEC circle (correct)", snap, False)

# 2. CCW PEC circle -> hard error.
snap = {"title": "p", "segments": [make_circle_segment("c", 2, 0.05, 64, cw=False)],
        "ibcs": [], "dielectrics": []}
ok &= expect("CCW PEC circle (wrong)", snap, True, "requires CW")

# 3. Hollow shell: CCW void (correct) passes; CW void errors.
def shell(inner_cw):
    return {"title": "s", "segments": [
        make_circle_segment("outer", 3, 0.06, 64, pos_mat="1", cw=True),
        make_circle_segment("void", 3, 0.04, 64, pos_mat="1", cw=inner_cw)],
        "ibcs": [], "dielectrics": [["1", "2.5", "0", "1", "0"]]}
ok &= expect("shell with CCW void (correct)", shell(False), False)
ok &= expect("shell with CW void (wrong)", shell(True), True, "nested void")

# 4. Coated PEC: inner TYPE 4 CW (correct) passes; CCW errors.
def coated(inner_cw):
    return {"title": "c", "segments": [
        make_circle_segment("outer", 3, 0.06, 64, pos_mat="1", cw=True),
        make_circle_segment("inner", 4, 0.04, 64, pos_mat="1", cw=inner_cw)],
        "ibcs": [], "dielectrics": [["1", "3", "0", "1", "0"]]}
ok &= expect("coated PEC inner CW (correct)", coated(True), False)
ok &= expect("coated PEC inner CCW (wrong)", coated(False), True, "TYPE 4")

# 5. Air-side continuity: PEC square from two open chains.
# Correct: both drawn so travel is CW around the square, head-to-tail.
top = [(0, 1), (1, 1), (1, 0)]          # left->right along top, down right side
bot = [(1, 0), (0, 0), (0, 1)]          # right->left along bottom, up left side
snap = {"title": "sq", "segments": [
    seg_from_points("upper", 2, top), seg_from_points("lower", 2, bot)],
    "ibcs": [], "dielectrics": []}
ok &= expect("2-chain CW square (correct)", snap, False)

# Reversed second chain: meets end-to-end -> air sides disagree.
bot_rev = list(reversed(bot))
snap = {"title": "sq", "segments": [
    seg_from_points("upper", 2, top), seg_from_points("lower", 2, bot_rev)],
    "ibcs": [], "dielectrics": []}
ok &= expect("2-chain square, one reversed (wrong)", snap, True, "opposite sides")

# Mixed types: air/PEC chain + air/dielectric chain (the user's example).
snap = {"title": "sq", "segments": [
    seg_from_points("pecwall", 2, top),
    seg_from_points("dielwall", 3, bot_rev, pos="1")],
    "ibcs": [], "dielectrics": [["1", "4", "0", "1", "0"]]}
ok &= expect("T2+T3 walls, air sides disagree (wrong)", snap, True, "air side")

# 6. Both chains consistently reversed (CCW loop): each junction is
# head-to-tail, but the whole loop is inside-out -> stitched-loop check.
snap = {"title": "sq", "segments": [
    seg_from_points("upper", 2, list(reversed(top))),
    seg_from_points("lower", 2, list(reversed(bot)))],
    "ibcs": [], "dielectrics": []}
ok &= expect("2-chain CCW square, consistent but inside-out (wrong)", snap, True, "loop")

# 7. Open strip / free-floating sheet: never flagged.
snap = {"title": "strip", "segments": [
    seg_from_points("plate", 2, [(-0.05, 0), (0.05, 0)])], "ibcs": [], "dielectrics": []}
ok &= expect("open PEC strip", snap, False)
snap = {"title": "card", "segments": [
    make_circle_segment("sheet", 1, 0.05, 64, ibc="1", cw=False)],
    "ibcs": [["1", "constant", "150", "0", "0", "0"]], "dielectrics": []}
ok &= expect("closed TYPE 1 sheet drawn CCW (orientation irrelevant)", snap, False)

# 8. End-to-end solve still green after preflight wiring.
snap = {"title": "p", "segments": [make_circle_segment("c", 2, 0.05, 128, cw=True)],
        "ibcs": [], "dielectrics": []}
out = solve_monostatic_rcs_2d(snap, [3.0], [0.0], "TE", geometry_units="meters")
got = out["samples"][0]["rcs_db"]
ref = 10 * math.log10(mie.sigma_pec_cylinder(0.05, 3e9, "TE"))
end_ok = abs(got - ref) < 0.05
ok &= end_ok
print(f"PASS  end-to-end TE PEC: {got:.3f} dB vs Mie {ref:.3f} dB" if end_ok
      else f"FAIL  end-to-end TE PEC: {got:.3f} vs {ref:.3f}")

print("ALL GATES PASSED" if ok else "GATES FAILED")
sys.exit(0 if ok else 1)
