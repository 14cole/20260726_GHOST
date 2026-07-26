"""Focused fail-closed and metadata-preservation tests for GRIM export."""

import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grim_io


def _sample(
    angle=0.0,
    frequency=1.0,
    power=None,
    amp_real=1.0,
    amp_imag=0.0,
    residual=1.0e-9,
):
    if power is None:
        k0 = 2.0 * math.pi * frequency * 1.0e9 / grim_io.C0
        power = (amp_real * amp_real + amp_imag * amp_imag) / (4.0 * k0)
    return {
        "frequency_ghz": frequency,
        "theta_inc_deg": angle,
        "theta_scat_deg": angle,
        "rcs_linear": power,
        "rcs_db": -300.0 if power == 0.0 else 10.0 * math.log10(power),
        "rcs_amp_real": amp_real,
        "rcs_amp_imag": amp_imag,
        "rcs_amp_phase_deg": math.degrees(math.atan2(amp_imag, amp_real)),
        "linear_residual": residual,
    }


class TestFailClosedSampleExport(unittest.TestCase):
    def test_raw_complex_field_keeps_float64_subtraction_precision(self):
        samples = [
            _sample(angle=0.0, amp_real=1.0),
            _sample(angle=1.0, amp_real=1.0 + 1.0e-10),
        ]
        payload = grim_io._build_grid_for_samples(samples, "TM")
        self.assertEqual(payload["rcs_amp_real"].dtype, np.dtype(np.float64))
        self.assertAlmostEqual(
            float(payload["rcs_amp_real"][1, 0, 0, 0]
                  - payload["rcs_amp_real"][0, 0, 0, 0]),
            1.0e-10,
            delta=1.0e-15,
        )

    def test_nonfinite_negative_and_missing_sample_values_are_rejected(self):
        cases = [
            ("negative power", {"rcs_linear": -1.0}),
            ("NaN power", {"rcs_linear": math.nan}),
            ("infinite power", {"rcs_linear": math.inf}),
            ("NaN real amplitude", {"rcs_amp_real": math.nan}),
            ("infinite imaginary amplitude", {"rcs_amp_imag": -math.inf}),
            ("NaN angle", {"theta_scat_deg": math.nan}),
            ("NaN frequency", {"frequency_ghz": math.nan}),
            ("zero frequency", {"frequency_ghz": 0.0}),
        ]
        for label, update in cases:
            with self.subTest(label=label):
                row = _sample()
                row.update(update)
                with self.assertRaises(ValueError):
                    grim_io._build_grid_for_samples([row], "TM")

        missing = _sample()
        del missing["rcs_amp_real"]
        with self.assertRaisesRegex(ValueError, "missing required field"):
            grim_io._build_grid_for_samples([missing], "TM")

    def test_sparse_cross_product_is_rejected_as_incomplete(self):
        samples = [
            _sample(angle=0.0, frequency=1.0),
            _sample(angle=90.0, frequency=2.0),
        ]
        with self.assertRaisesRegex(ValueError, "complete rectangular"):
            grim_io._build_grid_for_samples(samples, "TM")

    def test_save_revalidates_shape_and_finiteness_before_opening_destination(self):
        payload = grim_io._build_grid_for_samples([_sample()], "TM")
        payload["rcs_power"] = np.full((1, 1, 1, 1), math.nan)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "existing.grim")
            marker = b"existing file must survive validation failure"
            with open(path, "wb") as stream:
                stream.write(marker)
            with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                grim_io._save_grim_npz(payload, path)
            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), marker)

        payload = grim_io._build_grid_for_samples([_sample()], "TM")
        payload["rcs_phase"] = np.zeros((1, 1, 2, 1))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "not rectangular/complete"):
                grim_io._save_grim_npz(payload, os.path.join(tmp, "bad_shape"))

    def test_central_writer_promotes_coherent_fields_to_float64(self):
        payload = grim_io._build_grid_for_samples([_sample()], "TM")
        payload["rcs_amp_real"] = payload["rcs_amp_real"].astype(
            np.float32
        )
        payload["rcs_amp_imag"] = payload["rcs_amp_imag"].astype(
            np.float32
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = grim_io._save_grim_npz(
                payload, os.path.join(tmp, "promoted")
            )
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(
                    saved["rcs_amp_real"].dtype, np.dtype(np.float64)
                )
                self.assertEqual(
                    saved["rcs_amp_imag"].dtype, np.dtype(np.float64)
                )

    def test_overflowed_expected_power_cannot_bypass_consistency_gate(self):
        payload = grim_io._build_grid_for_samples([_sample()], "TM")
        payload["rcs_amp_real"] = np.full(
            (1, 1, 1, 1), 1.0e308, dtype=np.float64
        )
        payload["rcs_amp_imag"] = np.zeros(
            (1, 1, 1, 1), dtype=np.float64
        )
        payload["rcs_phase"] = np.zeros(
            (1, 1, 1, 1), dtype=np.float32
        )
        payload["rcs_power"] = np.ones(
            (1, 1, 1, 1), dtype=np.float32
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "too large"):
                grim_io._save_grim_npz(
                    payload, os.path.join(tmp, "overflow")
                )


class TestSolverMetadataExport(unittest.TestCase):
    def test_exact_null_and_solver_audit_metadata_survive_npz_export(self):
        result = {
            "solver": "bor_mom_rcs",
            "scattering_mode": "monostatic",
            "polarization": "VV",
            "polarization_export": "VV",
            "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
            "amplitude_convention": (
                "F physical far-field amplitude; sigma_3d=4*pi*|F|^2"
            ),
            "samples": [_sample(power=0.0, amp_real=0.0, amp_imag=0.0)],
            "metadata": {
                "residual_norm_max": 2.0e-8,
                "warnings": ["material table sampled outside characterized band"],
                "quality_gate": {
                    "passed": False,
                    "violations": ["test violation"],
                },
                "per_frequency": [{
                    "frequency_ghz": 1.0,
                    "linear_residual": 2.0e-8,
                    "mode_converged": False,
                    "mode_cap": 12,
                    "mode_last_relative_increment": math.inf,
                }],
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            written = grim_io.export_result_to_grim(
                result, os.path.join(tmp, "null")
            )
            with np.load(written[0], allow_pickle=False) as saved:
                self.assertEqual(float(saved["rcs_power"][0, 0, 0, 0]), 0.0)
                self.assertEqual(float(saved["rcs_amp_real"][0, 0, 0, 0]), 0.0)
                self.assertIn("solver_metadata_json", saved.files)
                audit = json.loads(str(saved["solver_metadata_json"]))

        self.assertEqual(audit["schema"], grim_io.SOLVER_METADATA_SCHEMA)
        self.assertEqual(audit["solver"], "bor_mom_rcs")
        self.assertFalse(
            audit["metadata"]["per_frequency"][0]["mode_converged"]
        )
        self.assertEqual(
            audit["metadata"]["per_frequency"][0][
                "mode_last_relative_increment"
            ],
            {"__nonfinite_float__": "infinity"},
        )
        self.assertEqual(
            audit["metadata"]["warnings"],
            ["material table sampled outside characterized band"],
        )
        self.assertFalse(audit["metadata"]["quality_gate"]["passed"])
        self.assertEqual(
            audit["sample_diagnostics"][0]["linear_residual"], 1.0e-9
        )

    def test_metadata_json_is_stable_across_dictionary_insertion_order(self):
        base = {
            "solver": "test",
            "samples": [_sample()],
            "metadata": {"z": 1, "a": {"second": 2, "first": 1}},
        }
        reordered = {
            "metadata": {"a": {"first": 1, "second": 2}, "z": 1},
            "samples": [_sample()],
            "solver": "test",
        }
        self.assertEqual(
            grim_io._solver_metadata_json(base),
            grim_io._solver_metadata_json(reordered),
        )


class TestDbkeCsvUnits(unittest.TestCase):
    def test_sigma_2d_export_retains_absolute_dbke_semantics(self):
        result = {
            "solver": "2d_bie_mom_rcs",
            "rcs_log_unit": "dBke",
            "rcs_linear_quantity": "sigma_2d",
            "samples": [_sample(frequency=3.0, power=0.5)],
        }
        expected = grim_io.compute_dbke_from_linear(0.5, 3.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = grim_io.export_result_to_dbke_csv(
                result, os.path.join(tmp, "result")
            )
            with open(path, encoding="utf-8") as stream:
                header = stream.readline().strip().split(",")
                values = stream.readline().strip().split(",")
        self.assertIn("dbke", header)
        self.assertAlmostEqual(
            float(values[header.index("dbke")]), expected, places=10
        )

    def test_sigma_3d_is_rejected_before_destination_is_opened(self):
        result = {
            "solver": "bor_mom_rcs",
            "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
            "samples": [_sample()],
        }
        marker = "existing output must survive"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "result.csv")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(marker)
            with self.assertRaisesRegex(
                ValueError, r"only accepts 2-D sigma_2d/dBke"
            ):
                grim_io.export_result_to_dbke_csv(result, path)
            with open(path, encoding="utf-8") as stream:
                self.assertEqual(stream.read(), marker)


if __name__ == "__main__":
    unittest.main()
