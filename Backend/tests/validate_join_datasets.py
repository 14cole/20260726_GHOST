#!/usr/bin/env python3
"""
Production filename grammar + join/pair gate (grim_naming.py).

The convention:

    SEAL-00-01_0.010gap_OPN.grim              featured (OPN)
    SEAL-00-01_0.010gap_FRD.grim              clean    (FRD)
    SEAL-00-01_0.010gap.grim                  the delta (no marker)
    HH_2.000GHz_SEAL-00-01_0.010gap_OPN.grim  as it comes off the solver

  J0 NAMES: parse/format round-trip; study id vs parameter tokens; the role
     marker; the solver prefix.  Bad names raise where they are read.
  J1 JOIN: N single-(pol, frequency) files -> ONE file per variation, and the
     amplitude of every cell survives EXACTLY.  A ragged grid, a duplicate cell,
     a different angle sweep and mixed units all RAISE instead of guessing.
  J2 POL ALIASES: the filename may say TM/TE while the file says HH/VV -- both
     join, and the original primaries are preserved.
  J3 PAIR: OPN matches FRD by base name; a lone one is reported, not dropped;
     two files of one role raise.
  J4 END TO END: join -> pair -> make_delta_grim gives a delta whose name is the
     base, and the library indexes it WITHOUT renaming (study id preserved).
  J5 STUDY IS CATEGORICAL: it is not an axis, select() pins it, and resolve()
     refuses to span two studies.

Synthetic grims, no solver.  Fast.  Run from tests/:  python3 validate_join_datasets.py
"""

import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, "..")

from grim_naming import (canon_pol, format_base, group_solver_files,           # noqa: E402
                         join_grims, pair_variants, parse_base,
                         parse_solver_name, parse_variation, variation_name)
from delta_library import DeltaLibrary, Range, format_name, parse_study        # noqa: E402
from feature_sum import (                                                       # noqa: E402
    PHYSICAL_2D_AMPLITUDE_CONVENTION,
    PHYSICAL_2D_FIELD_DOMAIN,
    PHYSICAL_2D_PHASE_REFERENCE,
    _load_grim,
    make_delta_grim,
)

C0 = 299_792_458.0
ANG = np.arange(0.0, 180.1, 10.0)
_fails = []


def gate(label, ok, note=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {note}")
    if not ok:
        _fails.append(label)


def raises(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except ValueError as exc:
        return True, str(exc)
    except Exception as exc:                                       # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return False, "did not raise"


def unit(path, pol_file, freq, seed):
    """One solver file: a single polarization at a single frequency."""
    rng = np.random.default_rng(seed)
    amp = (rng.normal(size=(len(ANG), 1, 1, 1))
           + 1j * rng.normal(size=(len(ANG), 1, 1, 1)))
    k = 2 * math.pi * freq * 1e9 / C0
    np.savez(open(path, "wb"),
             azimuths=ANG, elevations=np.array([0.0]),
             frequencies=np.asarray([freq], float),
             polarizations=np.asarray([pol_file], str),
             polarization_alias_primary=("TE" if pol_file == "VV" else "TM"),
             polarization_aliases_json=json.dumps([pol_file]),
             rcs_power=(np.abs(amp) ** 2 / (4 * k)).astype(np.float32),
             rcs_phase=np.angle(amp).astype(np.float32),
             rcs_domain="power_phase", power_domain="linear_rcs", source_path="",
             history="synthetic unit",
             units=json.dumps({"azimuth": "deg", "elevation": "deg",
                               "frequency": "GHz", "rcs_log_unit": "dBke",
                               "rcs_linear_quantity": "sigma_2d"}),
             phase_reference=PHYSICAL_2D_PHASE_REFERENCE,
             amplitude_convention=PHYSICAL_2D_AMPLITUDE_CONVENTION,
             raw_complex_amplitude_preserved=True,
             rcs_amp_real=amp.real.astype(np.float64),
             rcs_amp_imag=amp.imag.astype(np.float64),
             complex_field_domain=PHYSICAL_2D_FIELD_DOMAIN)
    # Read back what the file holds, so "survives exactly" tests the join.
    return path, _load_grim(path)["_amp"][:, :, 0, 0]


TMP = tempfile.mkdtemp(prefix="join_gate_")
print("=" * 74)
print(f"Filename grammar + join/pair gate — {TMP}")
print("=" * 74)

print("\nJ0. names")
r = parse_solver_name("HH_2.000GHz_SEAL-00-01_0.010gap_OPN.grim")
gate("solver name parses", (r["pol"], r["freq_ghz"], r["base"], r["role"])
     == ("HH", 2.0, "SEAL-00-01_0.010gap", "OPN"), f"({r['variation']})")
gate("role marker read", parse_variation("SEAL-00-01_0.010gap_FRD.grim")
     == ("SEAL-00-01_0.010gap", "FRD"), "")
gate("no marker == a delta", parse_variation("SEAL-00-01_0.010gap.grim")
     == ("SEAL-00-01_0.010gap", None), "")
study, params, decs = parse_base("SEAL-00-01_0.002bmag_0.010gap")
gate("study id and parameters separated",
     study == "SEAL-00-01" and params == {"bmag": 0.002, "gap": 0.010}
     and decs == {"bmag": 3, "gap": 3}, f"({study}, {params})")
gate("format_base is the inverse",
     format_base(study, params, decs) == "SEAL-00-01_0.002bmag_0.010gap", "")
gate("variation_name adds the marker",
     variation_name("SEAL-00-01_0.010gap", "OPN") == "SEAL-00-01_0.010gap_OPN.grim"
     and variation_name("SEAL-00-01_0.010gap") == "SEAL-00-01_0.010gap.grim", "")
gate("TM/TE and HH/VV canonicalise together",
     canon_pol("TM") == canon_pol("HH") == "HH"
     and canon_pol("TE") == canon_pol("VV") == "VV", "")
for bad, why in (("SEAL-00-01_0.010gap.grim", "no POL_FREQ prefix"),
                 ("XX_2.000GHz_SEAL-00-01_0.010gap.grim", "unknown pol")):
    ok, msg = raises(parse_solver_name, bad)
    gate(f"rejected as a solver name: {why}", ok, f"({msg[:40]})")
ok, msg = raises(parse_base, "SEAL-00-01_widegap")
gate("a non-<value><key> parameter chunk raises", ok, f"({msg[-42:]})")
ok, msg = raises(parse_base, "0.010gap_0.020gap")
gate("a name with no study id raises here", ok, f"({msg[:44]})")

print("\nJ1. join: many (pol, frequency) files -> one per variation")
src = os.path.join(TMP, "src")
os.makedirs(src, exist_ok=True)
truth = {}
seed = 0
for role in ("OPN", "FRD"):
    for gap in (0.006, 0.010):
        for pol in ("VV", "HH"):
            for f in (2.0, 4.0):
                seed += 1
                nm = f"{pol}_{f:.3f}GHz_SEAL-00-01_{gap:.3f}gap_{role}.grim"
                _p, a = unit(os.path.join(src, nm), pol, f, seed)
                truth[(role, gap, pol, f)] = a
groups, unparsed = group_solver_files(
    [os.path.join(src, n) for n in sorted(os.listdir(src))])
gate("grouped by variation", len(groups) == 4 and not unparsed,
     f"({len(groups)} variations, {len(unparsed)} unparsed)")
joined = {}
for variation, recs in sorted(groups.items()):
    out = join_grims([r["path"] for r in recs], os.path.join(TMP, variation))
    joined[variation] = out
g = _load_grim(joined["SEAL-00-01_0.010gap_OPN"])
pols = [str(p) for p in np.asarray(g["polarizations"]).ravel()]
freqs = np.asarray(g["frequencies"], float)
gate("joined file carries both axes",
     g["_amp"].shape == (len(ANG), 1, 2, 2) and pols == ["VV", "HH"]
     and list(freqs) == [2.0, 4.0], f"(shape {g['_amp'].shape}, pols {pols})")
worst = 0.0
for jp, p in enumerate(pols):
    for kf, f in enumerate(freqs):
        a = truth[("OPN", 0.010, p, float(f))]
        worst = max(worst, float(np.max(np.abs(g["_amp"][:, :, kf, jp] - a))))
gate("every cell's amplitude survives EXACTLY", worst == 0.0,
     f"(max |diff| {worst:.1e})")
gate("the 2-D convention is untouched",
     json.loads(str(g["units"]))["rcs_linear_quantity"] == "sigma_2d", "")
# refusals
ragged = [r["path"] for r in groups["SEAL-00-01_0.006gap_OPN"]][:3]
ok, msg = raises(join_grims, ragged, os.path.join(TMP, "ragged"))
gate("ragged (pol x frequency) grid raises", ok and "incomplete" in msg,
     f"({msg[:44]}...)")
dup = [ragged[0], ragged[0]]
ok, msg = raises(join_grims, dup, os.path.join(TMP, "dup"))
gate("duplicate (pol, frequency) raises", ok, f"({msg[:44]}...)")
odd = os.path.join(src, "VV_2.000GHz_ODD-00-01_0.010gap_OPN.grim")
rng = np.random.default_rng(99)
with np.load(ragged[0], allow_pickle=False) as z:
    d = {k: z[k] for k in z.files}
d["azimuths"] = ANG[:-1]
d["rcs_power"] = d["rcs_power"][:-1]
d["rcs_phase"] = d["rcs_phase"][:-1]
d["rcs_amp_real"] = d["rcs_amp_real"][:-1]
d["rcs_amp_imag"] = d["rcs_amp_imag"][:-1]
np.savez(open(odd, "wb"), **d)
ok, msg = raises(join_grims, [ragged[0], odd], os.path.join(TMP, "mixang"))
gate("different angle sweep raises", ok and "angle sweep" in msg, f"({msg[:40]}...)")
conv = os.path.join(src, "HH_4.000GHz_CONV-00-01_0.010gap_OPN.grim")
with np.load(ragged[1], allow_pickle=False) as z:
    d = {k: z[k] for k in z.files}
d["amplitude_convention"] = np.asarray("incompatible synthetic convention")
np.savez(open(conv, "wb"), **d)
ok, msg = raises(
    join_grims, [ragged[0], conv], os.path.join(TMP, "mixconvention"))
gate("different coherent-field convention raises",
     ok and "convention differs" in msg, f"({msg[:46]}...)")

print("\nJ2. the filename pol may differ from the in-file label")
tm = os.path.join(src, "TM_2.000GHz_SEAL-00-02_0.010gap_OPN.grim")
te = os.path.join(src, "TE_2.000GHz_SEAL-00-02_0.010gap_OPN.grim")
unit(tm, "HH", 2.0, 501)          # TM in the name, HH inside -- as the solver writes
unit(te, "VV", 2.0, 502)
out = join_grims([tm, te], os.path.join(TMP, "aliascase"))
ga = _load_grim(out)
gate("TM/TE-named files join by their canonical pol",
     [str(p) for p in np.asarray(ga["polarizations"]).ravel()] == ["VV", "HH"], "")
gate("the original primaries are preserved",
     set(str(ga["polarization_alias_primary"]).split(",")) == {"TM", "TE"},
     f"({str(ga['polarization_alias_primary'])})")

print("\nJ3. pair OPN with FRD")
pairs, unmatched = pair_variants(list(joined.values()))
gate("pairs found by base name", len(pairs) == 2 and not unmatched,
     f"({[p['base'] for p in pairs]})")
gate("the delta name drops the marker",
     all(p["delta_name"] == p["base"] + ".grim" for p in pairs), "")
lone = [joined["SEAL-00-01_0.010gap_OPN"]]
pairs1, un1 = pair_variants(lone)
gate("a lone OPN is REPORTED, not dropped",
     not pairs1 and len(un1) == 1 and "FRD" in un1[0]["reason"],
     f"({un1[0]['reason']})")
twice = os.path.join(TMP, "SEAL-00-01_0.010gap_OPN.grim")
shutil.copyfile(joined["SEAL-00-01_0.010gap_OPN"], twice + ".dup.grim")
ok, msg = raises(pair_variants, [twice, os.path.join(TMP, "x", "SEAL-00-01_0.010gap_OPN.grim")])
gate("two files of one role raise", ok, f"({msg[:44]}...)")

print("\nJ4. end to end: join -> pair -> delta -> library")
libdir = os.path.join(TMP, "library")
os.makedirs(libdir, exist_ok=True)
for p in pairs:
    make_delta_grim(p["clean"], p["featured"], os.path.join(libdir, p["delta_name"]))
lib = DeltaLibrary.from_dir(libdir)
gate("the delta keeps the production name",
     sorted(os.path.basename(e.path) for e in lib.entries)
     == ["SEAL-00-01_0.006gap.grim", "SEAL-00-01_0.010gap.grim"],
     f"({[os.path.basename(e.path) for e in lib.entries]})")
gate("the library indexes it without renaming (study preserved)",
     all(e.study == "SEAL-00-01" for e in lib.entries)
     and lib.axes() == {"gap": [0.006, 0.010]}, f"({lib.axes()}, {lib.studies()})")
gate("and it is a delta on the 2-D convention",
     lib.validate([2.0, 4.0]) == [], "")

print("\nJ5. the study id is categorical, not an axis")
gate("study excluded from the axes", "study" not in lib.axes(), "")
two = os.path.join(TMP, "twostudies")
os.makedirs(two, exist_ok=True)
for e in lib.entries:
    shutil.copyfile(e.path, os.path.join(two, os.path.basename(e.path)))
shutil.copyfile(lib.entries[0].path,
                os.path.join(two, "SEAL-00-02_0.006gap.grim"))
lib2 = DeltaLibrary.from_dir(two)
gate("two studies coexist in one folder", lib2.studies() == ["SEAL-00-01", "SEAL-00-02"],
     f"({lib2.studies()})")
ok, msg = raises(lib2.resolve, gap=[0.006])
gate("resolve REFUSES to span studies", ok and "studies" in msg, f"({msg[:44]}...)")
gate("select(study=...) pins one",
     len(lib2.select(study="SEAL-00-01")) == 2
     and lib2.select(study="SEAL-00-01").resolve(gap=[0.006]).entries[0].study
     == "SEAL-00-01", "")

shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{'=' * 74}")
print("ALL GATES PASSED" if not _fails else f"{len(_fails)} FAILED")
for f in _fails:
    print(f"   FAILED: {f}")
print("=" * 74)
sys.exit(1 if _fails else 0)
