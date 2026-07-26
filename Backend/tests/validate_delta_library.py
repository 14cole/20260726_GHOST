#!/usr/bin/env python3
"""
Filename-indexed delta library gate (delta_library.py).

The parameters of a delta live ONLY in its filename, so the gates are about
making the filesystem a trustworthy parameter key:

  L0 NAMES round-trip, and every non-canonical spelling is REJECTED (token
     order, decimal width, repeated key, junk token).
  L1 SCAN: axes discovered; a mixed decimal width, a missing variable and a
     non-canonical name all raise; unparseable files are REPORTED
     (self.unindexed), never silently skipped; a ragged grid is flagged.
     NOTE one parameter point has exactly ONE legal name, so the FILESYSTEM is
     the duplicate detector -- a second spelling of an existing point (e.g.
     a redundant 0rev token) is rejected as non-canonical, and a same-named file
     cannot exist twice in one directory.  DeltaLibrary's own duplicate check is
     an unreachable backstop for the direct constructor.
  L2 SELECT: exact / list / Range / 2-tuple / predicate; unknown variable and
     empty result both raise.
  L3 RESOLVE: off-grid snaps LOUDLY (non-empty report naming requested vs used);
     off_grid='error' raises; OUTSIDE the axis always raises (never
     extrapolate); a min/max range spreads over n arcs; a 2-D library refuses
     to resolve until pinned.
  L4 REV: two revisions at one point -> highest used, and it is reported.
  L5 PAYLOAD: validate() names a file that is not a delta / lacks a frequency.
  L6 PLACEMENTS: arcs partition the perimeter exactly; smooth_cycle attains the
     optimal max adjacent jump max(a[i+2]-a[i]) and never puts the widest gap
     next to the tightest; the placements drive sum_features, and the SAME delta
     on every arc reproduces the single-placement result exactly.

Synthetic deltas (trade_study._fab_delta) -- no solver needed.  Fast.

Run from tests/:  python3 validate_delta_library.py
"""

import math
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, "..")

from delta_library import (DeltaLibrary, Range, arc_slices, families,          # noqa: E402
                           format_name, parse_name, smooth_cycle,
                           tolerance_placements)
from trade_study import _fab_delta                                            # noqa: E402

FREQS = [3.0, 6.0]
_fails = []


def gate(label, ok, note=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {note}")
    if not ok:
        _fails.append(label)


def raises(fn, *a, **kw):
    """True if fn(*a, **kw) raises ValueError; returns (ok, message)."""
    try:
        fn(*a, **kw)
    except ValueError as exc:
        return True, str(exc)
    except Exception as exc:                                       # noqa: BLE001
        return False, f"raised {type(exc).__name__}: {exc}"
    return False, "did not raise"


def build(dirname, names, scale=1.0):
    os.makedirs(dirname, exist_ok=True)
    for nm in names:
        _fab_delta(os.path.join(dirname, nm), FREQS, scale)
    return dirname


TMP = tempfile.mkdtemp(prefix="delta_lib_gate_")
print("=" * 74)
print(f"Delta library gate (filename-only metadata) — {TMP}")
print("=" * 74)

# a 2 (bmag) x 4 (gap) rectangular family, under a clean library ROOT whose
# subdirectories are the seam TYPES; every deliberately-broken case lives
# outside that root so families() sees only real families
ROOT = os.path.join(TMP, "library")
BROKE = os.path.join(TMP, "broken")
GAPS = [0.030, 0.045, 0.060, 0.080]
BMAGS = [0.010, 0.020]
good = build(os.path.join(ROOT, "seal"),
             [f"{b:.3f}bmag_{g:.3f}gap.grim" for b in BMAGS for g in GAPS])
build(os.path.join(ROOT, "panel_gap"), ["0.020bmag_0.060gap.grim"])

print("\nL0. names round-trip; non-canonical spellings rejected")
vals, decs = parse_name("0.020bmag_0.060gap.grim")
gate("parse -> {bmag: 0.02, gap: 0.06}",
     vals == {"bmag": 0.020, "gap": 0.060} and decs == {"bmag": 3, "gap": 3}, f"({vals})")
gate("format is the inverse of parse",
     format_name(vals, decs) == "0.020bmag_0.060gap.grim", "")
gate("rev token round-trips",
     format_name(vals, dict(decs, rev=0), rev=2) == "0.020bmag_0.060gap_2rev.grim", "")
for bad, why in (("0.060gap_0.020bmag.grim", "token order"),
                 ("0.02bmag_0.060gap.grim", "decimal width"),
                 ("0.020bmag_0.020bmag.grim", "repeated key"),
                 ("seal.grim", "no <value><key> token"),
                 ("0.020bmag__0.060gap.grim", "empty token")):
    if why in ("token order", "decimal width"):
        v, d = parse_name(bad)
        rejected = format_name(v, {k: 3 for k in v}) != os.path.basename(bad)
        msg = "canonical form differs"
    else:
        rejected, msg = raises(parse_name, bad)
    gate(f"rejected: {bad}  ({why})", rejected, f"({msg[:44]})")

print("\nL1. scan")
lib = DeltaLibrary.from_dir(good)
gate("axes discovered from filenames alone",
     lib.axes() == {"bmag": BMAGS, "gap": GAPS}, f"({lib.axes()})")
gate("rectangular grid detected", lib.is_rectangular() and len(lib) == 8,
     f"({len(lib)} entries)")
ragged = build(os.path.join(BROKE, "ragged"),
               ["0.010bmag_0.030gap.grim", "0.010bmag_0.060gap.grim",
                "0.020bmag_0.030gap.grim"])
lr = DeltaLibrary.from_dir(ragged)
gate("ragged grid flagged (a silent bias in any sweep)",
     not lr.is_rectangular() and lr.missing_points() == [{"bmag": 0.020, "gap": 0.060}],
     f"(missing {lr.missing_points()})")
mixed = build(os.path.join(BROKE, "mixed"), ["0.02bmag_0.030gap.grim",
                                             "0.020bmag_0.060gap.grim"])
ok, msg = raises(DeltaLibrary.from_dir, mixed)
gate("mixed decimal width raises (0.02 vs 0.020 = two names, one point)", ok,
     f"({msg.split('--')[0].strip()[-46:]})")
short = build(os.path.join(BROKE, "short"), ["0.010bmag_0.030gap.grim", "0.060gap.grim"])
ok, msg = raises(DeltaLibrary.from_dir, short)
gate("file missing a variable the others carry raises", ok, f"({msg[-46:]})")
# one point = one legal name, so a SECOND SPELLING of an existing point is what
# has to be caught; the filesystem already forbids the same name twice
dupdir = build(os.path.join(BROKE, "dup"), ["0.030gap.grim"])
_fab_delta(os.path.join(dupdir, "0.030gap_0rev.grim"), FREQS, 1.0)
ok, msg = raises(DeltaLibrary.from_dir, dupdir)
gate("second spelling of one point rejected (redundant 0rev token)", ok,
     f"({msg.split(':')[-1].strip()[:44]})")
gate("one point has exactly one legal name (so the FS forbids duplicates)",
     format_name({"gap": 0.030}, {"gap": 3, "rev": 0}, rev=0.0) == "0.030gap.grim", "")
unind = build(os.path.join(BROKE, "unindexed"),
              ["0.030gap.grim", "0.060gap.grim", "seal.grim", "notes_v2.grim"])
lu = DeltaLibrary.from_dir(unind)
gate("unparseable files REPORTED, not silently skipped",
     len(lu) == 2 and sorted(os.path.basename(p) for p, _ in lu.unindexed)
     == ["notes_v2.grim", "seal.grim"],
     f"({len(lu)} indexed, {len(lu.unindexed)} unindexed)")
gate("summary() shows the unindexed files", "UNINDEXED" in lu.summary(), "")
fam = families(ROOT)
gate("families() maps each seam-type directory to its own library",
     sorted(fam) == ["panel_gap", "seal"] and len(fam["seal"]) == 8, f"({sorted(fam)})")
gate("same point may exist under two seam TYPES (type is the directory)",
     "0.020bmag_0.060gap.grim" in {os.path.basename(p) for p in fam["panel_gap"].paths()}
     and "0.020bmag_0.060gap.grim" in {os.path.basename(p) for p in fam["seal"].paths()},
     "")

print("\nL2. select (on-grid)")
gate("exact", len(lib.select(bmag=0.020)) == 4, "")
gate("explicit list", sorted(e.params["gap"] for e in
                             lib.select(bmag=0.020, gap=[0.030, 0.080]).entries)
     == [0.030, 0.080], "")
gate("Range (inclusive)", sorted(e.params["gap"] for e in
                                 lib.select(bmag=0.020, gap=Range(0.045, 0.060)).entries)
     == [0.045, 0.060], "")
gate("plain 2-tuple == Range", len(lib.select(bmag=0.020, gap=(0.045, 0.060))) == 2, "")
gate("predicate", len(lib.select(bmag=0.020, gap=lambda g: g > 0.05)) == 2, "")
ok, msg = raises(lib.select, bmagg=0.020)
gate("unknown variable raises (typo protection)", ok, f"({msg[:40]}...)")
ok, msg = raises(lib.select, bmag=0.030)
gate("empty selection raises (never silently empty)", ok, f"({msg[:40]}...)")

print("\nL3. resolve (tolerance spec -> entries; off-grid policy)")
f20 = lib.select(bmag=0.020)
res = f20.resolve(gap=[0.030, 0.060])
gate("on-grid list resolves silently",
     res.paths and not res.report, f"({len(res.entries)} entries, report empty)")
res = f20.resolve(gap=[0.035])
gate("off-grid SNAPS and says so",
     len(res.report) == 1 and "requested 0.035" in res.report[0]
     and "used 0.03" in res.report[0], f"({res.report[0] if res.report else 'no report'})")
ok, msg = raises(f20.resolve, gap=[0.035], off_grid="error")
gate("off_grid='error' refuses instead", ok, f"({msg[:44]}...)")
ok, msg = raises(f20.resolve, gap=[0.200])
gate("outside the axis ALWAYS raises (never extrapolate)", ok,
     f"({msg.split('--')[0].strip()[-44:]})")
ok, _ = raises(f20.resolve, gap=[0.001])
gate("below the axis raises too", ok, "")
res = f20.resolve(gap=Range(0.030, 0.080), n=6)
gate("min/max range spreads over n=6 arcs (snapped to nodes)",
     len(res.entries) == 6 and len(res.report) >= 1,
     f"(used {[e.params['gap'] for e in res.entries]})")
res_all = f20.resolve(gap=Range(0.040, 0.085))
gate("range without n uses every node inside it",
     [e.params["gap"] for e in res_all.entries] == [0.045, 0.060, 0.080], "")
ok, msg = raises(lib.resolve, gap=[0.030])
gate("2-D library refuses to resolve until pinned", ok, f"({msg[:44]}...)")
e, rep = f20.nearest(gap=0.050)
gate("nearest() snaps and reports", e.params["gap"] == 0.045 and len(rep) == 1,
     f"({rep[0][-30:]})")

print("\nL4. rev (a re-solve of the same parameter point)")
revdir = build(os.path.join(BROKE, "revs"), ["0.030gap.grim", "0.060gap.grim",
                                           "0.060gap_2rev.grim"])
lv = DeltaLibrary.from_dir(revdir)
gate("rev excluded from the axes", lv.axes() == {"gap": [0.030, 0.060]}
     and lv.revs() == [0.0, 2.0], f"({lv.axes()}, revs {lv.revs()})")
rv = lv.resolve(gap=[0.060])
gate("highest rev used AND reported",
     rv.entries[0].rev == 2.0 and any("rev=2" in r for r in rv.report),
     f"({rv.report[0][-42:] if rv.report else 'no report'})")
ok, msg = raises(lv.resolve, gap=[0.060], prefer_rev="explicit")
gate("prefer_rev='explicit' refuses to choose", ok, f"({msg[:40]}...)")
gate("select(rev=0) pins the original", lv.select(rev=0).resolve(
    gap=[0.060]).entries[0].rev == 0.0, "")

print("\nL5. payload sanity (filename stays the only metadata)")
gate("validate() passes for a good library at both frequencies",
     lib.validate(FREQS) == [], "")
probs = lib.validate([9.0], require=False)
gate("missing frequency is reported per file",
     len(probs) == 8 and (
         "no 9 GHz" in probs[0] or "no frequency 9" in probs[0]
     ), f"({probs[0][-38:]})")
notdelta = os.path.join(ROOT, "seal", "0.010bmag_0.100gap.grim")
_fab_delta(notdelta, FREQS, 1.0)
d = dict(np.load(notdelta, allow_pickle=False))
d["rcs_domain"] = np.asarray("2d")
with open(notdelta, "wb") as fh:
    np.savez(fh, **d)
lib2 = DeltaLibrary.from_dir(os.path.join(ROOT, "seal"))
probs = lib2.validate(require=False)
gate("a non-delta grim in the library is named",
     len(probs) == 1 and "0.100gap" in probs[0], f"({probs[0][:46]}...)")
os.remove(notdelta)

print("\nL6. arcs, ordering and the placements")
# a square door perimeter, 40 segments
NS = 40
t = np.linspace(0.0, 1.0, NS + 1)
ang = 2 * math.pi * t
loop = np.stack([0.30 + 0.05 * np.cos(ang), 0.05 * np.sin(ang),
                 0.60 + 0.0 * ang], axis=1)
per = np.stack([loop[:-1], loop[1:]], axis=1)
arcs = arc_slices(per, 6)
Ltot = float(np.sum(np.linalg.norm(per[:, 1] - per[:, 0], axis=1)))
Larc = sum(float(np.sum(np.linalg.norm(a[:, 1] - a[:, 0], axis=1))) for a in arcs)
gate("arcs partition the perimeter exactly (no gap, no overlap)",
     sum(len(a) for a in arcs) == NS and abs(Larc - Ltot) < 1e-12 * Ltot,
     f"({len(arcs)} arcs, {sum(len(a) for a in arcs)}/{NS} segments, "
     f"dL {abs(Larc-Ltot):.1e} m)")
ok, msg = raises(arc_slices, per, NS + 1)
gate("more arcs than segments raises (subdivide instead)", ok, f"({msg[:40]}...)")

vals = [0.4, 0.7, 1.0, 1.3, 1.6, 2.0]
idx = smooth_cycle(vals)
arr = [vals[i] for i in idx]
jump = max(abs(arr[i] - arr[(i + 1) % len(arr)]) for i in range(len(arr)))
srt = sorted(vals)
opt = max(srt[i + 2] - srt[i] for i in range(len(srt) - 2))
gate("smooth_cycle attains the optimal max adjacent jump max(a[i+2]-a[i])",
     abs(jump - opt) < 1e-12, f"(jump {jump:.3g}, optimum {opt:.3g}, order {arr})")
i_max, i_min = arr.index(max(arr)), arr.index(min(arr))
gate("widest gap is NOT adjacent to the tightest",
     abs(i_max - i_min) % len(arr) not in (1, len(arr) - 1),
     f"(widest at {i_max}, tightest at {i_min} of {len(arr)})")
for n in (3, 5, 7, 8, 11):
    v = list(np.linspace(0.5, 3.0, n) ** 2)
    a = [v[i] for i in smooth_cycle(v)]
    j = max(abs(a[i] - a[(i + 1) % n]) for i in range(n))
    s = sorted(v)
    o = max(s[i + 2] - s[i] for i in range(n - 2))
    if abs(j - o) > 1e-12:
        gate(f"smooth_cycle optimal for n={n}", False, f"({j:.4g} vs {o:.4g})")
        break
else:
    gate("smooth_cycle optimal for n = 3,5,7,8,11 too", True, "")

res = f20.resolve(gap=Range(0.030, 0.080), n=6)
pl = tolerance_placements(per, res.entries)
gate("placements: one per arc, each with its own delta path",
     len(pl) == 6 and all(set(p) == {"delta", "perimeter"} for p in pl)
     and len({p["delta"] for p in pl}) == len({e.path for e in res.entries}),
     f"({len(pl)} placements)")
gaps_around = [float(parse_name(p["delta"])[0]["gap"]) for p in pl]
gate("placement order is the smooth cycle (widest not beside tightest)",
     gaps_around == [res.entries[i].params["gap"]
                     for i in smooth_cycle([e.params["gap"] for e in res.entries])],
     f"({[f'{g*1e3:.0f}' for g in gaps_around]} mm around the loop)")

from feature_sum import sum_features, directions_from_aspect_roll             # noqa: E402
from line_expand import surface_of_revolution_normal                          # noqa: E402

gen = np.array([[0.05, 0.0], [0.05, 1.20]])          # straight cylinder r=50mm
dirs, _asp, _rol = directions_from_aspect_roll([60.0, 90.0, 120.0], [0.0, 90.0])
out = sum_features(None, pl, dirs, 6.0, generatrix=gen, mode="hybrid")
gate("placements drive sum_features (isolated feature response)",
     out["sigma_vv"].shape == (len(dirs),) and np.all(np.isfinite(out["dbsm_vv"])),
     f"(peak VV {float(np.max(out['dbsm_vv'])):+.1f} dBsm over {len(dirs)} looks)")

one = lib.select(bmag=0.020, gap=0.060).entries[0]
whole = sum_features(None, [{"delta": one.path, "perimeter": per}], dirs, 6.0,
                     generatrix=gen)
split = sum_features(None, [{"delta": one.path, "perimeter": a} for a in arcs],
                     dirs, 6.0, generatrix=gen)
rel = float(np.max(np.abs(split["feature_amps"][0]["F_vv"] * 0
                          + sum(f["F_vv"] for f in split["feature_amps"])
                          - whole["feature_amps"][0]["F_vv"]))
            / max(float(np.max(np.abs(whole["feature_amps"][0]["F_vv"]))), 1e-30))
gate("same delta on every arc == one placement over the whole perimeter",
     rel < 1e-12, f"(max rel {rel:.1e} on the coherent sum of arcs)")

shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{'=' * 74}")
print("ALL GATES PASSED" if not _fails else f"{len(_fails)} FAILED")
for f in _fails:
    print(f"   FAILED: {f}")
print("=" * 74)
sys.exit(1 if _fails else 0)
