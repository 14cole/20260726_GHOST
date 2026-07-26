"""Focused regressions for coherent component and compact-pattern schemas."""

import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)

import feature_sum as fs
from components import (
    COMPONENT_AMPLITUDE_CONVENTION,
    COMPONENT_COMPLEX_FIELD_DOMAIN,
    COMPONENT_PHASE_REFERENCE,
    validate_component_schema,
)

_spec = importlib.util.spec_from_file_location(
    "combine_step", os.path.join(ROOT, "4_combine", "run.py"))
combine_step = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(combine_step)


def _component_payload(amp=None, *, role="coherent", tagged=True):
    if amp is None:
        amp = np.zeros((1, 1, 1, 3), dtype=np.complex64)
    amp = np.asarray(amp, dtype=np.complex64)
    power = 4.0 * math.pi * (
        amp.real.astype(float) ** 2 + amp.imag.astype(float) ** 2)
    payload = {
        "azimuths": np.array([0.0]),
        "elevations": np.array([0.0]),
        "frequencies": np.array([3.0]),
        "polarizations": np.asarray(["VV", "HH", "VH"]),
        "rcs_amp_real": amp.real.astype(np.float32),
        "rcs_amp_imag": amp.imag.astype(np.float32),
        "rcs_phase": np.angle(amp).astype(np.float32),
        "rcs_power": power.astype(np.float32),
        "rcs_domain": "power_phase",
        "power_domain": "linear_rcs",
        "units": json.dumps({
            "azimuth": "deg", "elevation": "deg", "frequency": "GHz",
            "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
        }),
        "phase_reference": COMPONENT_PHASE_REFERENCE,
        "amplitude_convention": COMPONENT_AMPLITUDE_CONVENTION,
        "complex_field_domain": COMPONENT_COMPLEX_FIELD_DOMAIN,
        "raw_complex_amplitude_preserved": True,
    }
    if tagged:
        payload["combine_role"] = role
    return payload


def _write_2d(
        path, *, phase=fs.PHYSICAL_2D_PHASE_REFERENCE,
        amplitude_convention=fs.PHYSICAL_2D_AMPLITUDE_CONVENTION,
        domain=fs.PHYSICAL_2D_FIELD_DOMAIN, value=1.0,
        polarizations=("HH", "VV")):
    polarizations = tuple(polarizations)
    amp = np.full(
        (3, 1, 1, len(polarizations)),
        complex(value),
        dtype=np.complex128,
    )
    k = 2.0 * math.pi * 3.0e9 / fs.C0
    with open(path, "wb") as fh:
        np.savez(
            fh,
            azimuths=np.array([0.0, 90.0, 180.0]),
            elevations=np.array([0.0]),
            frequencies=np.array([3.0]),
            polarizations=np.asarray(polarizations),
            rcs_amp_real=amp.real.astype(np.float64),
            rcs_amp_imag=amp.imag.astype(np.float64),
            rcs_phase=np.angle(amp).astype(np.float32),
            rcs_power=(np.abs(amp) ** 2 / (4.0 * k)).astype(np.float32),
            units=json.dumps({
                "azimuth": "deg", "elevation": "deg", "frequency": "GHz",
                "rcs_log_unit": "dBke",
                "rcs_linear_quantity": "sigma_2d",
            }),
            phase_reference=phase,
            amplitude_convention=amplitude_convention,
            complex_field_domain=domain,
            raw_complex_amplitude_preserved=True,
            rcs_domain="power_phase",
            power_domain="linear_rcs",
        )
    return path


def _point_pattern(az=None, el=None):
    az = np.asarray(
        [0.0, 120.0, 240.0, 360.0] if az is None else az, dtype=float)
    el = np.asarray([0.0, 45.0, 90.0] if el is None else el, dtype=float)
    amp = np.ones((len(az), len(el), 1, 3), dtype=complex)
    return {
        "azimuths": az,
        "elevations": el,
        "frequencies": [3.0],
        "polarizations": ["VV", "HH", "VH"],
        "amp": amp,
        **fs.point_pattern_convention_metadata(),
    }


class TestStep4ComponentSchema(unittest.TestCase):
    def test_valid_tagged_component_passes_and_untagged_is_rejected(self):
        self.assertEqual(
            validate_component_schema(_component_payload()), "coherent")
        with self.assertRaisesRegex(ValueError, "missing combine_role"):
            validate_component_schema(_component_payload(tagged=False))

    def test_unknown_role_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown combine_role"):
            validate_component_schema(
                _component_payload(role="mystery"))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "unknown.grim")
            with open(path, "wb") as fh:
                np.savez(fh, **_component_payload(role="mystery"))
            with self.assertRaisesRegex(SystemExit, "unknown combine_role"):
                combine_step.amps(path)

    def test_units_and_field_conventions_are_enforced(self):
        for key, bad in (
            ("phase_reference", "different origin"),
            ("amplitude_convention", "sqrt(sigma)"),
            ("complex_field_domain", "unknown field"),
        ):
            payload = _component_payload()
            payload[key] = bad
            with self.subTest(key=key), self.assertRaisesRegex(
                    ValueError, key):
                validate_component_schema(payload)
        payload = _component_payload()
        payload["units"] = json.dumps({
            "rcs_log_unit": "dBke", "rcs_linear_quantity": "sigma_2d"})
        with self.assertRaisesRegex(ValueError, "sigma_3d"):
            validate_component_schema(payload)

    def test_linear_floor_at_exact_field_null_is_rejected(self):
        payload = _component_payload()
        payload["rcs_power"][:] = 1.0e-20
        with self.assertRaisesRegex(ValueError, r"4\*pi"):
            validate_component_schema(payload)

    def test_overflowed_expected_power_is_rejected(self):
        payload = _component_payload()
        payload["rcs_amp_real"] = np.full(
            (1, 1, 1, 3), 1.0e308, dtype=np.float64
        )
        payload["rcs_amp_imag"] = np.zeros(
            (1, 1, 1, 3), dtype=np.float64
        )
        payload["rcs_phase"] = np.zeros(
            (1, 1, 1, 3), dtype=np.float32
        )
        payload["rcs_power"] = np.ones(
            (1, 1, 1, 3), dtype=np.float32
        )
        with self.assertRaisesRegex(ValueError, "too large"):
            validate_component_schema(payload)

    def test_step4_output_keeps_exact_zero_and_truthful_power_pair(self):
        payload = _component_payload()
        total = np.zeros((1, 1, 1, 3), dtype=complex)
        estimate = np.zeros_like(total.real)
        got = combine_step._store_combined_fields(
            payload, total, estimate, ["VV", "HH", "VH"], "coherent")
        self.assertTrue(np.array_equal(got["rcs_power"], np.zeros(total.shape)))
        self.assertTrue(np.array_equal(
            got["combination_estimate_power"], np.zeros(total.shape)))
        self.assertEqual(validate_component_schema(got), "coherent")

    def test_feature_export_keeps_physical_nulls_as_zero(self):
        generatrix = np.array(
            [[0.0, 0.1], [0.03, 0.0], [0.0, -0.1]])
        with tempfile.TemporaryDirectory() as td:
            path = fs.export_radar_grim(
                os.path.join(td, "null"), bor_result=None, placements=[],
                generatrix=generatrix, frequencies_ghz=[3.0],
                azimuths_deg=[0.0], elevations_deg=[0.0])
            grim = fs._load_grim(path)
        self.assertTrue(np.array_equal(
            grim["rcs_power"], np.zeros((1, 1, 1, 3))))
        self.assertEqual(validate_component_schema(grim), "coherent")


class TestDeltaConventionSafety(unittest.TestCase):
    def test_small_delta_survives_large_field_subtraction(self):
        with tempfile.TemporaryDirectory() as td:
            clean = _write_2d(
                os.path.join(td, "clean.grim"), value=1.0)
            featured = _write_2d(
                os.path.join(td, "featured.grim"), value=1.0 + 1.0e-10)
            out = fs.make_delta_grim(
                clean, featured, os.path.join(td, "small_delta.grim"))
            with np.load(out, allow_pickle=False) as data:
                self.assertEqual(
                    data["rcs_amp_real"].dtype, np.dtype(np.float64))
                np.testing.assert_allclose(
                    data["rcs_amp_real"], 1.0e-10, rtol=1.0e-6, atol=0.0)

    def test_delta_metadata_says_featured_minus_clean(self):
        with tempfile.TemporaryDirectory() as td:
            clean = _write_2d(os.path.join(td, "clean.grim"), value=1.0)
            featured = _write_2d(
                os.path.join(td, "featured.grim"), value=1.5)
            out = fs.make_delta_grim(
                clean, featured, os.path.join(td, "delta.grim"))
            with np.load(out, allow_pickle=False) as data:
                self.assertEqual(
                    str(data["complex_field_domain"]),
                    "featured_minus_clean_far_field_amplitude_delta")
                self.assertEqual(
                    str(data["amplitude_convention"]),
                    fs.PHYSICAL_2D_AMPLITUDE_CONVENTION,
                )
                amp = (data["rcs_amp_real"].astype(float)
                       + 1j * data["rcs_amp_imag"].astype(float))
                np.testing.assert_allclose(amp, 0.5)

    def test_phase_or_amplitude_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            clean = _write_2d(os.path.join(td, "clean.grim"))
            bad_phase = _write_2d(
                os.path.join(td, "phase.grim"), phase="other origin")
            with self.assertRaisesRegex(ValueError, "phase_reference"):
                fs.make_delta_grim(
                    clean, bad_phase, os.path.join(td, "phase_delta.grim"))
            bad_amp = _write_2d(
                os.path.join(td, "amp.grim"),
                amplitude_convention="A=-j*B")
            with self.assertRaisesRegex(ValueError, "amplitude_convention"):
                fs.make_delta_grim(
                    clean, bad_amp, os.path.join(td, "amp_delta.grim"))

    def test_body_and_seam_roles_cannot_be_interchanged(self):
        with tempfile.TemporaryDirectory() as td:
            clean = _write_2d(os.path.join(td, "clean.grim"))
            featured = _write_2d(
                os.path.join(td, "featured.grim"), value=1.5)
            seam = fs.make_delta_grim(
                clean, featured, os.path.join(td, "seam.grim"))
            body = fs.save_body_grim(
                {
                    3.0: {
                        "theta_deg": [0.0, 180.0],
                        "amp_vv": [1.0 + 0.0j, 1.0 + 0.0j],
                        "amp_hh": [2.0 + 0.0j, 2.0 + 0.0j],
                    }
                },
                os.path.join(td, "body.grim"),
            )

            with self.assertRaisesRegex(ValueError, "rcs_domain.*delta"):
                fs.load_seam_from_grim(body, 3.0)
            with self.assertRaisesRegex(ValueError, "rcs_domain='power_phase'"):
                fs.load_body_grim(seam)

    def test_declared_delta_placement_uses_strict_seam_loader(self):
        with tempfile.TemporaryDirectory() as td:
            clean = _write_2d(os.path.join(td, "clean.grim"))
            featured = _write_2d(
                os.path.join(td, "featured.grim"), value=1.5)
            delta = fs.make_delta_grim(
                clean, featured, os.path.join(td, "delta.grim"))
            with mock.patch.object(
                fs,
                "load_seam_from_grim",
                side_effect=RuntimeError("strict seam loader sentinel"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "strict seam loader sentinel"
                ):
                    fs.sum_features(
                        None,
                        [{
                            "delta": delta,
                            "kind": "delta",
                            "perimeter": np.asarray(
                                [[[0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0]]]
                            ),
                            "normal": [1.0, 0.0, 0.0],
                        }],
                        np.asarray([[1.0, 0.0, 0.0]]),
                        3.0,
                    )

    def test_incomplete_or_mismatched_delta_channels_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            clean_both = _write_2d(os.path.join(td, "clean_both.grim"))
            featured_both = _write_2d(
                os.path.join(td, "featured_both.grim"), value=1.5)
            clean_hh = _write_2d(
                os.path.join(td, "clean_hh.grim"),
                polarizations=("HH",),
            )
            featured_hh = _write_2d(
                os.path.join(td, "featured_hh.grim"),
                value=1.5,
                polarizations=("HH",),
            )

            with self.assertRaisesRegex(ValueError, "complete TM/TE"):
                fs.make_delta_grim(
                    clean_hh, featured_hh,
                    os.path.join(td, "incomplete.grim"))
            with self.assertRaisesRegex(ValueError, "polarization sets differ"):
                fs.make_delta_grim(
                    clean_both, featured_hh,
                    os.path.join(td, "mismatch.grim"))
            with self.assertRaisesRegex(ValueError, "complete TM/TE"):
                fs.load_coefficients_from_grim(clean_hh, 3.0)

            delta = fs.make_delta_grim(
                clean_both, featured_both,
                os.path.join(td, "complete.grim"))
            with np.load(delta, allow_pickle=False) as data:
                payload = {key: data[key] for key in data.files}
            for key in (
                    "rcs_power", "rcs_phase",
                    "rcs_amp_real", "rcs_amp_imag"):
                payload[key] = payload[key][..., :1]
            payload["polarizations"] = np.asarray(["HH"])
            incomplete_delta = os.path.join(td, "one_channel_delta.grim")
            with open(incomplete_delta, "wb") as stream:
                np.savez(stream, **payload)
            with self.assertRaisesRegex(ValueError, "complete TM/TE"):
                fs.load_seam_from_grim(incomplete_delta, 3.0)

    def test_tm_te_and_hh_vv_alias_sets_remain_compatible(self):
        with tempfile.TemporaryDirectory() as td:
            clean = _write_2d(
                os.path.join(td, "clean.grim"),
                polarizations=("TM", "TE"),
            )
            featured = _write_2d(
                os.path.join(td, "featured.grim"),
                polarizations=("HH", "VV"),
                value=1.5,
            )
            delta = fs.make_delta_grim(
                clean, featured, os.path.join(td, "alias_delta.grim"))
            coeff = fs.load_seam_from_grim(delta, 3.0)
            np.testing.assert_allclose(coeff.dA_tm, 0.5)
            np.testing.assert_allclose(coeff.dA_te, 0.5)

    def test_role_loaders_enforce_normalization_reference_and_cut_axis(self):
        with tempfile.TemporaryDirectory() as td:
            clean = _write_2d(os.path.join(td, "clean.grim"))
            featured = _write_2d(
                os.path.join(td, "featured.grim"), value=1.5)
            delta = fs.make_delta_grim(
                clean, featured, os.path.join(td, "delta.grim"))
            body = fs.save_body_grim(
                {
                    3.0: {
                        "theta_deg": [0.0, 180.0],
                        "amp_vv": [1.0 + 0.0j, 1.0 + 0.0j],
                        "amp_hh": [2.0 + 0.0j, 2.0 + 0.0j],
                    }
                },
                os.path.join(td, "body.grim"),
            )

            def rewrite(source, name, mutate):
                with np.load(source, allow_pickle=False) as data:
                    payload = {key: data[key] for key in data.files}
                mutate(payload)
                target = os.path.join(td, name)
                with open(target, "wb") as stream:
                    np.savez(stream, **payload)
                return target

            bad_delta_units = rewrite(
                delta,
                "bad_delta_units.grim",
                lambda payload: payload.__setitem__(
                    "units",
                    json.dumps({
                        "azimuth": "deg",
                        "elevation": "deg",
                        "frequency": "GHz",
                        "rcs_log_unit": "dBsm",
                        "rcs_linear_quantity": "sigma_2d",
                    }),
                ),
            )
            with self.assertRaisesRegex(ValueError, "sigma_2d/dBke"):
                fs.load_seam_from_grim(bad_delta_units, 3.0)

            bad_delta_reference = rewrite(
                delta,
                "bad_delta_reference.grim",
                lambda payload: payload.__setitem__(
                    "phase_reference", "unknown coupon origin"),
            )
            with self.assertRaisesRegex(ValueError, "phase_reference"):
                fs.load_seam_from_grim(bad_delta_reference, 3.0)

            def duplicate_elevation(payload):
                payload["elevations"] = np.asarray([0.0, 1.0])
                for key in (
                        "rcs_power", "rcs_phase",
                        "rcs_amp_real", "rcs_amp_imag"):
                    payload[key] = np.repeat(payload[key], 2, axis=1)

            bad_delta_elevation = rewrite(
                delta, "bad_delta_elevation.grim", duplicate_elevation)
            with self.assertRaisesRegex(ValueError, "singleton elevation"):
                fs.load_seam_from_grim(bad_delta_elevation, 3.0)

            bad_body_domain = rewrite(
                body,
                "bad_body_domain.grim",
                lambda payload: payload.__setitem__(
                    "complex_field_domain",
                    "generic_3d_far_field_amplitude",
                ),
            )
            with self.assertRaisesRegex(ValueError, "complex_field_domain"):
                fs.load_body_grim(bad_body_domain)

            bad_body_elevation = rewrite(
                body, "bad_body_elevation.grim", duplicate_elevation)
            with self.assertRaisesRegex(ValueError, "singleton elevation"):
                fs.load_body_grim(bad_body_elevation)

    def test_grim_loader_rejects_unphysical_or_unknown_fields(self):
        with tempfile.TemporaryDirectory() as td:
            base = _write_2d(os.path.join(td, "base.grim"))

            def corrupt(name, key, mutate):
                with np.load(base, allow_pickle=False) as data:
                    payload = {k: data[k] for k in data.files}
                payload[key] = mutate(payload.get(key))
                path = os.path.join(td, name)
                with open(path, "wb") as fh:
                    np.savez(fh, **payload)
                return path

            negative = corrupt(
                "negative.grim", "rcs_power",
                lambda value: -np.ones_like(value))
            with self.assertRaisesRegex(ValueError, "negative"):
                fs._load_grim(negative)

            nonfinite = corrupt(
                "nonfinite.grim", "rcs_amp_real",
                lambda value: np.full_like(value, np.nan))
            with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                fs._load_grim(nonfinite)

            mismatch = corrupt(
                "mismatch.grim", "rcs_power",
                lambda value: np.asarray(value) * 2.0)
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                fs._load_grim(mismatch)

            bad_phase = corrupt(
                "phase_mismatch.grim", "rcs_phase",
                lambda value: np.asarray(value) + 0.25)
            with self.assertRaisesRegex(ValueError, "rcs_phase is inconsistent"):
                fs._load_grim(bad_phase)

            unknown_units = corrupt(
                "unknown_units.grim", "units",
                lambda _value: json.dumps({
                    "rcs_linear_quantity": "mystery",
                    "rcs_log_unit": "dB?",
                }))
            with self.assertRaisesRegex(ValueError, "sigma_2d.*sigma_3d"):
                fs._load_grim(unknown_units)


class TestPointPatternSupport(unittest.TestCase):
    def test_explicit_convention_is_required(self):
        pattern = _point_pattern()
        del pattern["phase_reference"]
        with self.assertRaisesRegex(ValueError, "phase_reference"):
            fs._load_pattern(pattern)

    def test_partial_azimuth_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete 360-degree"):
            fs._load_pattern(_point_pattern(az=[0.0, 90.0, 180.0]))

    def test_lit_elevation_outside_support_is_rejected(self):
        pattern = _point_pattern(el=[0.0, 30.0])
        look = fs._direction(0.0, 45.0)[None, :]
        with self.assertRaisesRegex(ValueError, "elevation support"):
            fs.point_scatterer_amplitude(
                pattern, location=(0.0, 0.0, 0.0),
                aperture_normal=(0.0, 0.0, 1.0),
                roll_ref=(1.0, 0.0, 0.0),
                directions=look, frequency_ghz=3.0)


class TestBodyInterpolationSafety(unittest.TestCase):
    def test_out_of_support_query_is_not_clamped(self):
        body = {
            "theta_deg": [10.0, 20.0],
            "amp_vv": [1.0 + 0.0j, 2.0 + 0.0j],
        }
        with self.assertRaisesRegex(ValueError, "outside stored support"):
            fs._bor_amp_interp(body, "amp_vv", [9.0])
        # Tiny floating-point endpoint drift is the only accepted clipping.
        self.assertEqual(
            fs._bor_amp_interp(body, "amp_vv", 10.0 - 5e-10),
            1.0 + 0.0j)

    def test_malformed_or_nonfinite_body_tables_are_rejected(self):
        cases = [
            ({"theta_deg": [[0.0, 1.0]], "amp_vv": [1.0, 2.0]},
             "one-dimensional"),
            ({"theta_deg": [0.0, 1.0], "amp_vv": [1.0]},
             "matching arrays"),
            ({"theta_deg": [0.0, 0.0], "amp_vv": [1.0, 2.0]},
             "unique"),
            ({"theta_deg": [0.0, np.nan], "amp_vv": [1.0, 2.0]},
             "NaN or infinite"),
            ({"theta_deg": [0.0, 1.0],
              "amp_vv": [1.0, complex(np.inf, 0.0)]},
             "NaN or infinite"),
        ]
        for body, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                    ValueError, message):
                fs._bor_amp_interp(body, "amp_vv", [0.5])
        body = {"theta_deg": [0.0, 1.0], "amp_vv": [1.0, 2.0]}
        with self.assertRaisesRegex(ValueError, "queries must be finite"):
            fs._bor_amp_interp(body, "amp_vv", [np.nan])


class TestCornerInputSafety(unittest.TestCase):
    def _call(self, **overrides):
        kwargs = {
            "fold": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            "n_wing": (1.0, 0.0, 0.0),
            "n_body": (0.0, 1.0, 0.0),
            "face_width": 0.1,
            "directions": np.array([[1.0, 1.0, 0.0]]),
            "frequency_ghz": 3.0,
        }
        kwargs.update(overrides)
        return fs.corner_amplitude(**kwargs)

    def test_invalid_fold_normals_and_scalars_are_rejected(self):
        cases = [
            ({"fold": [0.0, 1.0, 2.0]}, "fold must have shape"),
            ({"fold": [[0.0, 0.0, 0.0], [np.nan, 0.0, 1.0]]},
             "NaN or infinite"),
            ({"n_wing": (0.0, 0.0, 0.0)}, "n_wing must be nonzero"),
            ({"n_body": (np.nan, 1.0, 0.0)}, "n_body must be a finite"),
            ({"face_width": 0.0}, "face_width"),
            ({"frequency_ghz": np.inf}, "frequency_ghz"),
            ({"retro_halfwidth_deg": 0.0}, "retro_halfwidth_deg"),
            ({"internal_phase_deg": np.nan}, "internal_phase_deg"),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                    ValueError, message):
                self._call(**overrides)

    def test_invalid_directions_are_rejected_before_normalization(self):
        for directions in (
                np.array([[0.0, 0.0, 0.0]]),
                np.array([[np.nan, 0.0, 1.0]]),
                np.array([1.0, 2.0])):
            with self.subTest(shape=np.asarray(directions).shape), \
                    self.assertRaisesRegex(ValueError, "directions"):
                self._call(directions=directions)


if __name__ == "__main__":
    unittest.main()
