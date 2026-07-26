#!/usr/bin/env python3
"""
Interoperability gate with the GRIM_Revised_2 viewer tool (grim_dataset.RcsGrid).

Both projects use .grim (npz).  These gates pin the parts that could silently
diverge as either tool changes:

  X0 RcsGrid.load reads all three flavours this repo writes (2-D cut, delta,
     3-D radar export); power is identical and phase agrees to float32.
  X1 THE CONVENTION TABLE -- and there are only TWO data types, 2-D and 3-D.
     rcs_power is the physical quantity in both tools, but rcs_amp is the
     solver's FIELD amplitude, so sqrt(power)/|amp| is
       1/(2 sqrt(k))  for 2-D (sigma_2d = |A|^2/(4k), dBke; per frequency),
       sqrt(4 pi)     for 3-D (sigma = 4 pi |F|^2, dBsm).
     A DELTA IS 2-D, not a third type: rcs_domain='delta' is an orthogonal axis
     (the samples are a difference), and it must not change the units.
     grim_compat.amp_scale must predict both from the file's TAGS alone, and must
     still honour legacy deltas tagged power_domain='delta_amp_sq'.
  X2 ROUND TRIP.  RcsGrid.load -> .save -> our loader keeps the complex
     amplitude bit-exactly AND keeps the domain tags: a delta must still read
     back as rcs_domain='delta', because sum_features ROUTES on that tag (a
     delta misread as a full-object coefficient is a silent physics error).
  X3 A DERIVED grid (cropped) must NOT write its now-stale amplitude, and
     grim_compat.from_grid can attach a correct one.
  X4 field_amplitude returns exactly this repo's amplitude.
  X5 load_pattern_any feeds point_scatterer_amplitude: placing a pattern read
     through RcsGrid == placing the .grim directly.
  X6 a magnitude-only pattern (no phase) RAISES rather than assuming zero phase.

SKIPS cleanly (exit 0) when GRIM_Revised_2 is not present -- nothing else in this
repo imports it.

Run from tests/:  python3 validate_grim_compat.py
"""

import json
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, "..")

try:
    import grim_compat as gc                                        # noqa: E402
    RcsGrid = gc.rcsgrid_class()
except ImportError as exc:
    print("=" * 74)
    print("GRIM_Revised_2 interoperability gate — SKIPPED")
    print(f"  {exc}")
    print("=" * 74)
    sys.exit(0)

from feature_sum import (_load_grim, make_delta_grim,                # noqa: E402
                         PHYSICAL_2D_PHASE_REFERENCE,
                         PHYSICAL_2D_AMPLITUDE_CONVENTION,
                         PHYSICAL_2D_FIELD_DOMAIN,
                         PHYSICAL_3D_AMPLITUDE_CONVENTION,
                         point_pattern_convention_metadata,
                         point_scatterer_amplitude, tag_as_delta,
                         load_seam_from_grim, directions_from_aspect_roll)

C0 = 299_792_458.0
FREQS = [3.0, 6.0]
_fails = []


def raises(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except ValueError as exc:
        return True, str(exc)
    except Exception as exc:                                       # noqa: BLE001
        return False, f"raised {type(exc).__name__}: {exc}"
    return False, "did not raise"


def gate(label, ok, note=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {note}")
    if not ok:
        _fails.append(label)


def fab_2d(path, freqs=FREQS, scale=1.0, sigma2d=True):
    """A 2-D solver-style cut: rcs_power = sigma_2d = |A|^2/(4k), amp = A."""
    phi = np.arange(0.0, 180.1, 10.0)
    rng = np.random.default_rng(3)
    amp = (scale * (rng.normal(size=(len(phi), 1, len(freqs), 2))
                    + 1j * rng.normal(size=(len(phi), 1, len(freqs), 2))))
    pw = np.empty_like(np.abs(amp))
    for kf, f in enumerate(freqs):
        k = 2 * math.pi * f * 1e9 / C0
        pw[:, :, kf, :] = (np.abs(amp[:, :, kf, :]) ** 2 / (4 * k) if sigma2d
                           else np.abs(amp[:, :, kf, :]) ** 2)
    units = {"azimuth": "deg", "elevation": "deg", "frequency": "GHz",
             "rcs_log_unit": "dBke",
             "rcs_linear_quantity": "sigma_2d" if sigma2d else "amp_sq"}
    with open(path, "wb") as fh:
        np.savez(fh, azimuths=phi, elevations=np.array([0.0]),
                 frequencies=np.asarray(freqs, float),
                 polarizations=np.asarray(["HH", "VV"], dtype=str),
                 polarization_alias_primary="TM,TE",
                 polarization_aliases_json=json.dumps(["TM", "TE"]),
                 rcs_power=pw.astype(np.float32),
                 rcs_phase=np.angle(amp).astype(np.float32),
                 rcs_domain="power_phase", power_domain="linear_rcs",
                 source_path="", history="gate 2-D cut", units=json.dumps(units),
                 phase_reference=PHYSICAL_2D_PHASE_REFERENCE,
                 raw_complex_amplitude_preserved=True,
                 amplitude_convention=PHYSICAL_2D_AMPLITUDE_CONVENTION,
                 rcs_amp_real=amp.real.astype(np.float64),
                 rcs_amp_imag=amp.imag.astype(np.float64),
                 complex_field_domain=PHYSICAL_2D_FIELD_DOMAIN)
    return path


def fab_3d(path, freqs=FREQS, with_phase=True, rcs_domain="power_phase"):
    """A 3-D export: rcs_power = sigma = 4 pi |F|^2, amp = F, pols VV/HH/VH."""
    az = np.arange(0.0, 360.1, 30.0)
    el = np.arange(0.0, 90.1, 30.0)
    rng = np.random.default_rng(7)
    sh = (len(az), len(el), len(freqs), 3)
    amp = 0.02 * (rng.normal(size=sh) + 1j * rng.normal(size=sh))
    # The grid includes both 0 and 360 degrees, which are the same physical
    # direction; keep the periodic seam complex-field continuous.
    amp[-1, ...] = amp[0, ...]
    units = {"azimuth": "deg", "elevation": "deg", "frequency": "GHz",
             "rcs_log_unit": "dBsm", "rcs_linear_quantity": "sigma_3d"}
    phase = (np.angle(amp) if with_phase
             else np.full(sh, np.nan)).astype(np.float32)
    payload = dict(azimuths=az, elevations=el,
                   frequencies=np.asarray(freqs, float),
                   polarizations=np.asarray(["VV", "HH", "VH"], dtype=str),
                   polarization_alias_primary="VV",
                   polarization_aliases_json=json.dumps(["VV", "HH", "VH"]),
                   rcs_power=(4 * math.pi * np.abs(amp) ** 2).astype(np.float32),
                   rcs_phase=phase, rcs_domain=rcs_domain,
                   power_domain="linear_rcs", source_path="",
                   history="gate 3-D export", units=json.dumps(units),
                   phase_reference="gate", raw_complex_amplitude_preserved=True,
                   amplitude_convention=PHYSICAL_3D_AMPLITUDE_CONVENTION,
                   complex_field_domain="gate")
    if rcs_domain == "delta":
        payload.update(point_pattern_convention_metadata())
    if with_phase:
        payload["rcs_amp_real"] = amp.real.astype(np.float64)
        payload["rcs_amp_imag"] = amp.imag.astype(np.float64)
    with open(path, "wb") as fh:
        np.savez(fh, **payload)
    return path


TMP = tempfile.mkdtemp(prefix="grim_compat_gate_")
print("=" * 74)
print("GRIM_Revised_2 interoperability gate")
print(f"  viewer tool: {os.path.dirname(RcsGrid.__module__ and gc._grim_revised_dir())}"
      f"/{os.path.basename(gc._grim_revised_dir())}")
print("=" * 74)

cut = fab_2d(os.path.join(TMP, "cut.grim"))
clean = fab_2d(os.path.join(TMP, "clean.grim"), scale=0.5)
delta = make_delta_grim([clean], [cut], os.path.join(TMP, "delta.grim"))
radar = fab_3d(os.path.join(TMP, "radar.grim"))
CASES = [("2-D cut", cut), ("delta", delta), ("3-D export", radar)]

print("\nX0. RcsGrid reads what this repo writes")
for label, p in CASES:
    d = _load_grim(p)
    g = RcsGrid.load(p)
    pw_same = np.array_equal(np.asarray(d["rcs_power"], np.float32),
                             np.asarray(g.rcs_power, np.float32))
    dphi = np.angle(np.asarray(g.rcs)) - np.angle(d["_amp"])
    dphi = np.abs((dphi + np.pi) % (2 * np.pi) - np.pi)
    m = np.abs(d["_amp"]) > 1e-9 * np.abs(d["_amp"]).max()
    gate(f"{label}: loads, power identical, phase to float32",
         pw_same and float(np.max(dphi[m])) < 1e-5,
         f"(max |dphi| {float(np.max(dphi[m])):.1e} rad)")

print("\nX1. the convention table (predicted from tags, not measured)")
for label, p in CASES:
    d = _load_grim(p)
    pw = np.asarray(d["rcs_power"], float)
    amp = d["_amp"]
    fr = np.asarray(d["frequencies"], float)
    s = gc.amp_scale(p)
    ok = True
    detail = []
    for kf, f in enumerate(fr):
        a, w = amp[:, :, kf, :], pw[:, :, kf, :]
        m = np.abs(a) > 1e-9 * np.abs(amp).max()
        meas = float(np.mean(np.sqrt(w[m]) / np.abs(a[m])))
        pred = float(s[0] if s.size == 1 else s[kf])
        ok &= abs(meas / pred - 1.0) < 1e-3
        detail.append(f"{f:g}GHz {meas:.5f}")
    gate(f"{label}: amp_scale matches the file", ok, "(" + ", ".join(detail) + ")")
k6 = 2 * math.pi * 6.0e9 / C0
gate("2-D scale == 1/(2 sqrt(k))",
     abs(float(gc.amp_scale(cut)[1]) - 1 / (2 * math.sqrt(k6))) < 1e-12, "")
gate("3-D scale == sqrt(4 pi)",
     abs(float(gc.amp_scale(radar)[0]) - math.sqrt(4 * math.pi)) < 1e-12, "")
gate("a delta is 2-D, NOT a third type (same scale as any 2-D cut)",
     np.allclose(gc.amp_scale(delta), gc.amp_scale(cut)),
     f"(delta {gc.amp_scale(delta)} vs cut {gc.amp_scale(cut)})")
gate("delta keeps rcs_domain='delta' (the orthogonal routing axis)",
     str(_load_grim(delta)["rcs_domain"]) == "delta"
     and str(_load_grim(delta)["power_domain"]) == "linear_rcs", "")
# a delta must plot as dBke correctly: 10log10(k * sigma_2d), no 4k offset
dd = _load_grim(delta)
kk = 2 * math.pi * np.asarray(dd["frequencies"], float) * 1e9 / C0
expect = np.abs(dd["_amp"]) ** 2 / (4 * kk)[None, None, :, None]
gate("delta rcs_power == |dA|^2/(4k) (so a dBke plot is right)",
     np.allclose(np.asarray(dd["rcs_power"], float), expect, rtol=1e-6),
     f"(was |dA|^2, which read {10*math.log10(4*float(kk[-1])):+.1f} dB high at "
     f"{float(np.asarray(dd['frequencies'],float)[-1]):g} GHz)")
# legacy files still on disk must keep working
legacy = os.path.join(TMP, "legacy_delta.grim")
with np.load(delta, allow_pickle=False) as z:
    dl = {k: z[k] for k in z.files}
dl["rcs_power"] = (np.abs(dd["_amp"]) ** 2).astype(np.float32)
dl["power_domain"] = "delta_amp_sq"
with open(legacy, "wb") as fh:
    np.savez(fh, **dl)
gate("legacy delta (power_domain='delta_amp_sq') still scales as 1",
     float(gc.amp_scale(legacy)[0]) == 1.0, "")
gate("legacy delta amplitude still recovers exactly",
     np.array_equal(gc.field_amplitude(RcsGrid.load(legacy), legacy),
                    _load_grim(legacy)["_amp"]), "")
ok, msg = raises(load_seam_from_grim, legacy, FREQS[0])
gate("production seam loading rejects legacy unnormalized delta power",
     ok and "power_domain='linear_rcs'" in msg, f"({msg[:64]})")

print("\nX2. round trip through the viewer tool keeps amplitude AND tags")
for label, p in CASES:
    out = RcsGrid.load(p).save(os.path.join(TMP, f"rt_{label.replace(' ', '_')}"))
    d0, d1 = _load_grim(p), _load_grim(out)
    same = all(np.array_equal(d0[k], d1[k]) for k in
               ("rcs_amp_real", "rcs_amp_imag", "rcs_power", "rcs_phase",
                "azimuths", "elevations", "frequencies", "polarizations"))
    gate(f"{label}: amplitude survives bit-exactly", same, "")
    gate(f"{label}: rcs_domain preserved ({str(d0['rcs_domain'])!r})",
         str(d1["rcs_domain"]) == str(d0["rcs_domain"])
         and str(d1["power_domain"]) == str(d0["power_domain"]), "")

print("\nX3. a DERIVED grid must not write a stale amplitude")
g = RcsGrid.load(radar)
crop = g.axis_crop(frequency_range=(5.0, 7.0))
out = crop.save(os.path.join(TMP, "cropped"))
with np.load(out, allow_pickle=False) as z:
    keys = set(z.files)
gate("cropped grid drops rcs_amp instead of writing the old shape",
     "rcs_amp_real" not in keys,
     f"(shape {crop.rcs_power.shape} vs original {np.asarray(g.rcs_power).shape})")
amp3 = _load_grim(radar)["_amp"]
kf = [i for i, f in enumerate(np.asarray(g.frequencies, float)) if 5.0 <= f <= 7.0]
out2 = gc.from_grid(crop, os.path.join(TMP, "cropped_amp"),
                    amp=amp3[:, :, kf, :], history="gate")
gate("from_grid(amp=...) makes it readable again",
     np.array_equal(_load_grim(out2)["_amp"].astype(np.complex64),
                    amp3[:, :, kf, :].astype(np.complex64)), "")

print("\nX3b. a delta built IN THE VIEWER is usable (the real interop path)")
# exactly what a user does: load the two 2-D solves, coherent_subtract, save.
# The derived grid has no rcs_amp_* (they are not part of RcsGrid's model), so the
# pipeline has to rebuild the amplitude from power + phase.
sub = RcsGrid.load(cut).coherent_subtract(RcsGrid.load(clean))
gui_delta = gc.from_grid(sub, os.path.join(TMP, "gui_delta"))
with np.load(gui_delta, allow_pickle=False) as z:
    gkeys = set(z.files)
gate("the viewer's derived file has no rcs_amp_* (nothing was stripped)",
     "rcs_amp_real" not in gkeys, f"(has {sorted(k for k in gkeys if 'rcs' in k)})")
gd = _load_grim(gui_delta)
gate("the pipeline rebuilds the amplitude from power+phase",
     gd.get("_amp_from_power_phase", False)
     and np.max(np.abs(gd["_amp"])) > 0, "")
expected_grim = _load_grim(delta)
# make_delta_grim writes the canonical TE/TM sort (VV, HH), whereas the viewer
# preserves its input order (HH, VV).  Compare physical channels, not storage
# column order.
expected_index = {
    str(pol): index
    for index, pol in enumerate(expected_grim["polarizations"])
}
expect = expected_grim["_amp"][
    ..., [expected_index[str(pol)] for pol in gd["polarizations"]]
]
rel = float(np.max(np.abs(gd["_amp"] - expect)) / np.max(np.abs(expect)))
gate("and it equals the true difference (two float32 round trips)",
     rel < 1e-4, f"(max rel {rel:.1e})")
ok, msg = raises(load_seam_from_grim, gui_delta, FREQS[0])
gate("rcs_domain is still required -- not assumed", ok,
     f"({msg.splitlines()[0][-46:]})")
ok, msg = raises(tag_as_delta, gui_delta)
gate("tag_as_delta refuses to invent missing phase conventions", ok,
     f"({msg.splitlines()[0][:46]})")
tag_as_delta(gui_delta, source_2d_grim=cut)
sc = load_seam_from_grim(gui_delta, FREQS[0])
gate("tag_as_delta with a verified source makes it loadable as a seam",
     np.max(np.abs(sc.dA_tm)) > 0 or np.max(np.abs(sc.dA_te)) > 0, "")

print("\nX4. field_amplitude recovers this repo's amplitude")
for label, p in CASES:
    a = gc.field_amplitude(RcsGrid.load(p), p)
    gate(f"{label}: exact", np.array_equal(a, _load_grim(p)["_amp"]), "")
# and from power+phase alone (amplitude stripped), to within float32
strip = os.path.join(TMP, "stripped.grim")
with np.load(radar, allow_pickle=False) as z:
    d = {k: z[k] for k in z.files if not k.startswith("rcs_amp_")}
d["raw_complex_amplitude_preserved"] = False
with open(strip, "wb") as fh:
    np.savez(fh, **d)
a = gc.field_amplitude(RcsGrid.load(strip), {"rcs_domain": "power_phase",
                                             "units": json.dumps(
                                                 {"rcs_linear_quantity": "sigma_3d"}),
                                             "frequencies": FREQS})
rel = float(np.max(np.abs(a - amp3)) / np.max(np.abs(amp3)))
gate("3-D export: recovered from power+phase alone", rel < 1e-6,
     f"(max rel {rel:.1e})")

print("\nX5. load_pattern_any -> point_scatterer_amplitude")
dirs, _a, _r = directions_from_aspect_roll([60.0, 90.0, 120.0], [0.0, 90.0])
kw = dict(location=(0.05, 0.0, -0.02), aperture_normal=(1.0, 0.0, 0.0),
          roll_ref=(0.0, 0.0, 1.0))
pattern_delta = fab_3d(os.path.join(TMP, "pattern_delta.grim"),
                       rcs_domain="delta")
direct = point_scatterer_amplitude(pattern_delta, directions=dirs,
                                   frequency_ghz=6.0, **kw)
viaany = point_scatterer_amplitude(gc.load_pattern_any(pattern_delta), directions=dirs,
                                   frequency_ghz=6.0, **kw)
scale = max(float(np.max(np.abs(direct[c]))) for c in ("F_vv", "F_hh", "F_vh"))
worst = max(float(np.max(np.abs(direct[c] - viaany[c])))
            for c in ("F_vv", "F_hh", "F_vh")) / max(scale, 1e-30)
gate("placing a pattern read through RcsGrid == placing the .grim directly",
     worst < 1e-12, f"(max rel diff {worst:.1e} -- round-off, not a difference)")
gate("feature_sum routes a non-.grim pattern through grim_compat",
     "load_pattern_any" in open("../feature_sum.py").read(), "")

print("\nX6. a magnitude-only pattern is refused, not guessed")
nophase = fab_3d(os.path.join(TMP, "nophase_pre.grim"), with_phase=False,
                 rcs_domain="delta")

print("\nX7. the body solve is a .grim like everything else")
from feature_sum import save_body_grim, load_body_grim                       # noqa: E402
rng = np.random.default_rng(11)
th = np.arange(0.0, 180.1, 10.0)
bodies = {f: {"theta_deg": th,
              "amp_vv": rng.normal(size=len(th)) + 1j * rng.normal(size=len(th)),
              "amp_hh": rng.normal(size=len(th)) + 1j * rng.normal(size=len(th))}
          for f in FREQS}
bp = save_body_grim(bodies, os.path.join(TMP, "body"), history="gate")
back = load_body_grim(bp)
worst = max(float(np.max(np.abs(np.asarray(back[f][ch]) - bodies[f][ch])
                         / np.max(np.abs(bodies[f][ch]))))
            for f in FREQS for ch in ("amp_vv", "amp_hh"))
gate("save_body_grim -> load_body_grim round-trips the amplitude",
     worst < 1e-14 and all(np.array_equal(back[f]["theta_deg"], th) for f in FREQS),
     f"(max rel {worst:.1e}, float64 raw-field storage)")
bg = _load_grim(bp)
gate("it is the 3-D convention (sigma = 4 pi |F|^2, dBsm)",
     abs(float(gc.amp_scale(bp)[0]) - math.sqrt(4 * math.pi)) < 1e-12
     and json.loads(str(bg["units"]))["rcs_log_unit"] == "dBsm", "")
gate("the aspect axis is TAGGED, so it cannot be misread as radar azimuth",
     "aspect" in json.loads(str(bg["units"])).get("azimuth_meaning", "").lower()
     and "aspect" in str(bg["history"]).lower(),
     f"({json.loads(str(bg['units']))['azimuth_meaning'][:44]}...)")
gate("the viewer tool reads it", RcsGrid.load(bp).rcs_power.shape
     == (len(th), 1, len(FREQS), 2), "")
b1 = save_body_grim(
    bodies[FREQS[0]], os.path.join(TMP, "body_single"),
    frequency_ghz=FREQS[0])
gate("a single-frequency solve works too", len(load_body_grim(b1)) == 1, "")
try:
    save_body_grim(
        bodies[FREQS[0]], os.path.join(TMP, "body_single_unknown_frequency"))
    gate("a single-frequency solve cannot invent its frequency", False,
         "(did not raise)")
except ValueError as exc:
    gate("a single-frequency solve cannot invent its frequency",
         "frequency_ghz" in str(exc), f"({str(exc)[:44]}...)")
try:
    bad = {3.0: bodies[3.0], 6.0: dict(bodies[6.0], theta_deg=th[:-1])}
    save_body_grim(bad, os.path.join(TMP, "bad"))
    gate("mismatched aspect sweeps raise", False, "(did not raise)")
except ValueError as exc:
    gate("mismatched aspect sweeps raise", "aspect sweep" in str(exc),
         f"({str(exc)[:44]}...)")
nophase = fab_3d(os.path.join(TMP, "nophase.grim"), with_phase=False,
                 rcs_domain="delta")
try:
    gc.load_pattern_any(nophase)
    gate("phase-less pattern raises", False, "(it did not)")
except ValueError as exc:
    gate("phase-less pattern raises", "no phase" in str(exc),
         f"({str(exc)[:52]}...)")

import shutil
shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{'=' * 74}")
print("ALL GATES PASSED" if not _fails else f"{len(_fails)} FAILED")
for f in _fails:
    print(f"   FAILED: {f}")
print("=" * 74)
sys.exit(1 if _fails else 0)
