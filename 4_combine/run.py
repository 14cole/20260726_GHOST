#!/usr/bin/env python3
"""
STEP 4 -- COMBINE EVERY COMPONENT WITH THE BODY
===============================================

WHAT IT DOES
  Reads what steps 2 and 3a/3b/3c already produced.  NOTHING IS RE-SOLVED --
  this is seconds, and you can re-run it with a different MODE or a different
  subset of components as often as you like.

      BODY_DIR/body.grim        the body (aspect x frequency), step 2a or 2b
      <component dirs>/*.grim   each component ALONE (az x el x freq x pol),
                                from 3a_doors, 3b_add_wing, 3c_add_cavity

  Every populated component directory must also have the completed
  provenance_manifest.json written by its step.  Its exact .grim inventory and
  byte hashes are verified before discovery, so a removed/renamed design cannot
  survive as an unrecorded stale component.

  For the line-expanded components, separate placement and summation are
  numerically equivalent within the isolated, single-bounce line-expansion
  model.  That does not make the reduced model an exact coupled-Maxwell solve;
  mutual coupling and multiple scattering with the 3-D body remain outside it.

  The grid, the frequencies and the polarizations are taken FROM the component
  files, so step 4 has nothing to keep in sync with step 3.

NOT EVERY COMPONENT MAY INTERFERE WITH EVERY OTHER
  Each component file carries a combine_role tag saying whether its PHASE can
  be trusted against the others (see Backend/components.py):

    coherent   doors, cavities, a bare wing.  Summed as complex amplitudes.
    power      a separately exported PO-level estimate, such as a wing-root
               corner, whose internal double-bounce phase is not tracked.  Its
               POWER appears only in combination_estimate_power.

  Step 3b writes the wing and its phase-unknown corner estimate separately, so
  no fictitious internal cross term is baked into either file.  Every component
  must carry an explicit role and pass strict sigma_3d/dBsm,
  origin/time-sign, amplitude-normalization, phase, and
  rcs_power=4*pi*|F|^2 checks. Missing or unknown semantics are refused rather
  than guessed.

  rcs_amp holds the coherent field (body + coherent components), and rcs_power
  ALWAYS equals 4*pi*|rcs_amp|^2.  The selected MODE estimate, including any
  power-role terms, is stored separately as combination_estimate_power.

OUTPUTS
  Combined/body_azel.grim    the bare body on the components' own az/el grid --
                             the reference to difference against
  Combined/<OUT_NAME>.grim   everything together  <- THE DELIVERABLE
  Combined/provenance_manifest.json
                             atomic exact-input/output commit marker

MODE   how the separately labelled ENGINEERING ESTIMATE is formed
  "hybrid"    components interfere with each other, then power-add to
              the body.  The component-to-component phase depends only on their
              separation and is trustworthy; the body-to-component phase leans
              on the calibration and is the fragile one, so it is not used.
  "coherent"  (default) body and coherent components phase-summed; phase-unknown
              components are still power-added only in the estimate.
  "envelope"  powers added, no interference anywhere.  Most conservative.

  MODE never changes the primary rcs_amp/rcs_power pair; it changes only
  combination_estimate_power.

KNOBS (below)
  MODE, COMPONENT_DIRS, ONLY, SKIP, OUT_NAME, BODY_DIR

    python3 run.py
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(os.path.dirname(HERE), "Backend")
sys.path.insert(0, BACKEND)

# ─── KNOBS ──────────────────────────────────────────────────────────────────
MODE = "coherent"         # estimate: "hybrid" | "coherent" | "envelope"

COMPONENT_DIRS = [os.path.join("..", "3a_doors", "Output"),
                  os.path.join("..", "3b_add_wing", "Output"),
                  os.path.join("..", "3c_add_cavity", "Output")]
ONLY = []                 # [] = every component found; else a list of names
SKIP = []                 # names to leave out -- what did that one buy me?

OUT_NAME = "vehicle"
BODY_DIR = os.path.join("..", "2b_solve_body_hpc", "Body")
                              # HPC production default; use 2a.../Body locally
# ────────────────────────────────────────────────────────────────────────────

from frame import AXIS_AZ_DEG, AXIS_EL_DEG, ROLL_DEG                  # noqa: E402
from components import (COMPONENT_AMPLITUDE_CONVENTION,              # noqa: E402
                        combine_component_fields,
                        validate_component_schema)
from feature_sum import (export_radar_grim, load_body_grim,           # noqa: E402
                         radar_grid_aspects, _load_grim,
                         verify_body_artifact_bundle)
from line_expand import dbsm                                          # noqa: E402
from workflow_provenance import verify_component_output_manifest     # noqa: E402
from workflow_provenance import (backend_source_fingerprint,          # noqa: E402
                                 runtime_environment_fingerprint,
                                 sha256_file, stable_json_fingerprint,
                                 write_artifact_in_progress,
                                 write_artifact_manifest)

OUTPUT_SCHEMA = "ghost.workflow.combined-output-provenance.v1"


def _here(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(HERE, p))


def amps(path):
    g = _load_grim(path)
    try:
        role = validate_component_schema(g, os.path.basename(path))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return (g["rcs_amp_real"] + 1j * g["rcs_amp_imag"],
            np.asarray(g["azimuths"], float), np.asarray(g["elevations"], float),
            np.asarray(g["frequencies"], float),
            [str(p) for p in np.asarray(g["polarizations"]).ravel()],
            role)


def _store_combined_fields(payload, total_amp, estimate_power, pols, mode):
    """Store one truthful complex-field/power pair without a linear floor."""
    total = np.asarray(total_amp, dtype=complex)
    estimate = np.asarray(estimate_power, dtype=float)
    if estimate.shape != total.shape:
        raise ValueError(
            f"combination estimate shape {estimate.shape} != field shape "
            f"{total.shape}.")
    if not np.all(np.isfinite(estimate)) or np.any(estimate < 0.0):
        raise ValueError(
            "combination_estimate_power must be finite and non-negative.")

    # Preserve coherent fields in float64: small component differences and
    # deep interference nulls must not be rounded away before later reuse.
    # Derive primary power from the exact field samples that are
    # written, so even serialization roundoff cannot make the two arrays
    # describe different physical fields.  Exact nulls remain exact zeros.
    real = total.real.astype(np.float64)
    imag = total.imag.astype(np.float64)
    stored_power = 4.0 * np.pi * (
        real.astype(float) ** 2 + imag.astype(float) ** 2)
    stored_amp = real.astype(float) + 1j * imag.astype(float)
    payload["rcs_power"] = stored_power.astype(np.float32)
    payload["combination_estimate_power"] = estimate.astype(np.float32)
    payload["combination_estimate_mode"] = np.asarray(mode)
    payload["combination_estimate_semantics"] = np.asarray(
        "engineering/statistical estimate; not represented by rcs_amp")
    payload["rcs_phase"] = np.angle(stored_amp).astype(np.float32)
    payload["rcs_amp_real"] = real
    payload["rcs_amp_imag"] = imag
    payload["polarizations"] = np.asarray(pols, dtype=str)
    payload["combine_role"] = np.asarray("coherent")
    payload["amplitude_convention"] = np.asarray(
        COMPONENT_AMPLITUDE_CONVENTION)
    return payload


def _verified_component_inventories():
    """Require a completed exact manifest before trusting any step-3 folder."""

    inventories = []
    for directory in COMPONENT_DIRS:
        output_dir = _here(directory)
        if not os.path.isdir(output_dir):
            continue
        has_grims = bool(glob.glob(os.path.join(output_dir, "*.grim")))
        has_manifest = os.path.exists(
            os.path.join(output_dir, "provenance_manifest.json")
        )
        if not has_grims and not has_manifest:
            continue
        try:
            payload = verify_component_output_manifest(output_dir)
        except ValueError as exc:
            raise SystemExit(
                f"uncommitted or changed component bundle {output_dir}: {exc}"
            ) from exc
        inventories.append((output_dir, payload))
    return inventories


def discover(inventories):
    """Components from verified manifest inventories, with combine roles."""
    found = []
    for output_dir, manifest in inventories:
        for filename in sorted(manifest["expected_outputs"]):
            p = os.path.join(output_dir, filename)
            name = os.path.splitext(os.path.basename(p))[0]
            if name == OUT_NAME:
                continue
            if ONLY and name not in ONLY:
                continue
            if name in SKIP:
                print(f"        SKIP  {name}")
                continue
            found.append((
                name,
                p,
                os.path.basename(output_dir.rstrip(os.sep)),
            ))
    return found


def _combined_provenance_payload(
    body_dir, inventories, expected_outputs
):
    body_manifests = [
        os.path.join(body_dir, name)
        for name in ("cache_manifest.json", "collection_manifest.json")
        if os.path.isfile(os.path.join(body_dir, name))
    ]
    if len(body_manifests) != 1:
        raise ValueError(
            "the body directory must contain exactly one local/HPC commit "
            "manifest."
        )
    root = os.path.dirname(HERE)
    return {
        "schema": OUTPUT_SCHEMA,
        "mode": str(MODE),
        "out_name": str(OUT_NAME),
        "only": sorted(str(value) for value in ONLY),
        "skip": sorted(str(value) for value in SKIP),
        "body_manifest_name": os.path.basename(body_manifests[0]),
        "body_manifest_sha256": sha256_file(body_manifests[0]),
        "component_manifests_sha256": {
            os.path.relpath(output_dir, HERE):
                sha256_file(
                    os.path.join(output_dir, "provenance_manifest.json")
                )
            for output_dir, _payload in inventories
        },
        "axis_az_deg": float(AXIS_AZ_DEG),
        "axis_el_deg": float(AXIS_EL_DEG),
        "roll_deg": float(ROLL_DEG),
        "workflow_source_sha256": backend_source_fingerprint(
            BACKEND,
            {
                "4_combine/run.py": os.path.abspath(__file__),
                "grid.py": os.path.join(root, "grid.py"),
            },
        ),
        "runtime_environment_sha256":
            runtime_environment_fingerprint(),
        "expected_outputs": sorted(expected_outputs),
    }


def main():
    if (
        not isinstance(OUT_NAME, str)
        or not OUT_NAME
        or OUT_NAME in (".", "..", "body_azel", "provenance_manifest")
        or "/" in OUT_NAME
        or "\\" in OUT_NAME
    ):
        raise SystemExit(
            "OUT_NAME must be a safe filename stem distinct from body_azel "
            "and provenance_manifest."
        )
    body_dir = _here(BODY_DIR)
    body_grim = os.path.join(body_dir, "body.grim")
    if not os.path.exists(body_grim):
        raise SystemExit(f"no {body_grim} -- run 2a_solve_body_local (or 2b) "
                         f"first.")
    try:
        verify_body_artifact_bundle(body_dir)
    except ValueError as exc:
        raise SystemExit(f"uncommitted or changed body bundle: {exc}") from exc
    if MODE not in ("hybrid", "coherent", "envelope"):
        raise SystemExit(f"MODE must be hybrid | coherent | envelope, got "
                         f"{MODE!r}.")

    inventories = _verified_component_inventories()
    comps = discover(inventories)
    if not comps:
        raise SystemExit(
            "no component *.grim found in " +
            ", ".join(os.path.relpath(_here(d), HERE) for d in COMPONENT_DIRS) +
            "\n  Run 3a_doors (and 3b/3c if you have them) first.")

    combined_dir = os.path.join(HERE, "Combined")
    os.makedirs(combined_dir, exist_ok=True)
    expected_outputs = ["body_azel.grim", f"{OUT_NAME}.grim"]
    stale = sorted(
        name for name in os.listdir(combined_dir)
        if name.lower().endswith(".grim") and name not in expected_outputs
    )
    if stale:
        raise SystemExit(
            "Combined/ contains stale .grim output(s) outside this run's "
            f"exact inventory: {stale}. Move/archive them before combining."
        )
    try:
        provenance_payload = _combined_provenance_payload(
            body_dir, inventories, expected_outputs
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    provenance_sha256 = stable_json_fingerprint(provenance_payload)
    write_artifact_in_progress(
        combined_dir,
        OUTPUT_SCHEMA,
        expected_outputs,
        dict(provenance_payload, run_sha256=provenance_sha256),
        manifest_name="provenance_manifest.json",
    )

    # the grid comes from the components, so nothing has to be kept in sync
    a0, az, el, fr, pols, _r0 = amps(comps[0][1])
    coherent, power_terms, listing = [], [], []
    for name, path, folder in comps:
        a, az2, el2, fr2, p2, role = amps(path)
        if not (np.array_equal(az, az2) and np.array_equal(el, el2)
                and np.array_equal(fr, fr2) and pols == p2):
            raise SystemExit(
                f"{name} is on a different grid from {comps[0][0]}.\n"
                f"  Every component must share one grid -- it is declared once "
                f"in ../grid.py,\n  so re-run whichever 3x step is stale.  A "
                f"component from an older grid is\n  REFUSED here, never "
                f"silently resampled.")
        (coherent if role == "coherent" else power_terms).append(a)
        listing.append((name, folder, role))

    print(f"STEP 4  {len(comps)} component(s) on az {len(az)} x el {len(el)} x "
          f"freq {len(fr)} x pol {pols}, mode={MODE}")
    for name, folder, role in listing:
        print(f"        {role:<9} {name:<20} ({folder})")
    if power_terms:
        print(f"        -> {len(power_terms)} power-role component(s) add POWER, "
              f"they are not phase-summed")

    # Coarse interpolation of a complex BoR field can move nulls by tens of dB
    # and rotate phase by >90 degrees.  The body solve is therefore required to
    # contain every aspect this exact component grid maps to.
    body_results = load_body_grim(body_grim)
    required_aspects = radar_grid_aspects(
        az, el, AXIS_AZ_DEG, AXIS_EL_DEG)
    for f in fr:
        matches = [v for k, v in body_results.items()
                   if abs(float(k) - float(f)) < 1e-6]
        if not matches:
            raise SystemExit(
                f"Body/body.grim has no {float(f):g} GHz body result.")
        have_aspects = np.asarray(matches[0]["theta_deg"], dtype=float)
        missing = [
            float(q) for q in required_aspects
            if not np.any(np.isclose(have_aspects, q, rtol=0.0, atol=1e-9))
        ]
        if missing:
            raise SystemExit(
                "Body/body.grim is too coarse for this component grid: "
                f"{len(missing)} exact BoR aspect(s) are missing at "
                f"{float(f):g} GHz (first {missing[:5]}).\n"
                "  Re-run step 2a/2b with ASPECT_STEP_DEG = None. "
                "Production body fields are never coarsely interpolated.")

    # the body on the components' own grid, same fixed attitude -- reuses the
    # gated exporter rather than re-deriving the aspect mapping and V/H rotation
    prof = np.loadtxt(os.path.join(body_dir, "body_profile.csv"),
                      delimiter=",", skiprows=1)
    body_path = export_radar_grim(
        os.path.join(HERE, "Combined", "body_azel"),
        bor_result=body_results, placements=[], generatrix=prof,
        frequencies_ghz=fr, azimuths_deg=az, elevations_deg=el,
        axis_az_deg=AXIS_AZ_DEG, axis_el_deg=AXIS_EL_DEG, roll_deg=ROLL_DEG,
        history="step 4 body on the components' az/el grid")
    b_all, _az, _el, _fr, b_pols, _r = amps(body_path)
    body = b_all[:, :, :, [b_pols.index(p) for p in pols]]
    print("        wrote Combined/body_azel.grim   peak "
          + "  ".join(f"{p} {dbsm(4*np.pi*np.max(np.abs(body[:,:,:,i]))**2):+.1f}"
                      for i, p in enumerate(pols)) + " dBsm")

    total_amp, coherent_power, estimate_power = combine_component_fields(
        body, coherent, power_terms, mode=MODE)

    with np.load(body_path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    _store_combined_fields(d, total_amp, estimate_power, pols, MODE)
    d["history"] = (
        f"step 4 mode={MODE} body + {[n for n, _f, _r in listing]} | "
        f"rcs_amp is the coherent total of the body and the {len(coherent)} "
        f"coherent component(s); rcs_power=4*pi*|rcs_amp|^2; "
        f"combination_estimate_power uses mode={MODE} and power-adds "
        f"{len(power_terms)} phase-unknown component(s)")
    out = os.path.join(HERE, "Combined", f"{OUT_NAME}.grim")
    from grim_io import _save_grim_npz
    _save_grim_npz(d, out)

    # Recheck every committed input bundle and the exact output schemas before
    # publishing the final commit marker.
    try:
        verify_body_artifact_bundle(body_dir)
        current_inventories = _verified_component_inventories()
        current_payload = _combined_provenance_payload(
            body_dir, current_inventories, expected_outputs
        )
        amps(body_path)
        amps(out)
    except (OSError, ValueError, SystemExit) as exc:
        raise SystemExit(
            "an input or output became invalid while step 4 was running. "
            "Combined files were preserved but no complete manifest was "
            f"written: {exc}"
        ) from exc
    if stable_json_fingerprint(current_payload) != provenance_sha256:
        raise SystemExit(
            "the body, components, grid, selection, workflow source, or "
            "numerical runtime changed while step 4 was running. Combined "
            "files were preserved but no complete manifest was written."
        )
    write_artifact_manifest(
        combined_dir,
        OUTPUT_SCHEMA,
        expected_outputs,
        dict(provenance_payload, run_sha256=provenance_sha256),
        manifest_name="provenance_manifest.json",
    )

    print(f"        wrote Combined/{OUT_NAME}.grim")
    print("        committed Combined/provenance_manifest.json")
    tot = np.asarray(d["rcs_power"], float)
    est = np.asarray(d["combination_estimate_power"], float)
    for i, p in enumerate(pols):
        bp = 4 * np.pi * np.abs(body[:, :, :, i]) ** 2
        b, t = dbsm(np.max(bp)), dbsm(np.max(tot[:, :, :, i]))
        e = dbsm(np.max(est[:, :, :, i]))
        # only where the body HAS a return: a ratio against a null or against
        # the -200 dBsm floor is arithmetic, not information (a bare
        # axisymmetric body has no cross-pol at all in the el = 0 rows)
        live = bp > np.max(bp) * 1e-4
        if np.any(live):
            lift = 10 * np.log10(np.max(tot[:, :, :, i][live] / bp[live]))
            note = (f"biggest lift {lift:+.1f} dB (where the body is within "
                    f"40 dB of peak)")
        else:
            note = ("the body has no return in this channel -- the components "
                    "ARE the signal")
        print(f"        {p}: body {b:+7.1f}   coherent {t:+7.1f}   "
              f"{MODE} estimate {e:+7.1f} dBsm   {note}")
    print(f"\nDONE    Combined/{OUT_NAME}.grim is the deliverable.")
    print("        Use SKIP = ['<name>'] to see what any one component bought.")


if __name__ == "__main__":
    main()
