"""Focused regressions for 2D material/dispatch safety fixes."""

import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rcs_solver as rcs
import grim_io


def _circle_pairs(radius: float, count: int):
    """Clockwise polygon, matching the user-facing outward-normal convention."""

    points = [
        (
            radius * math.cos(-2.0 * math.pi * idx / count),
            radius * math.sin(-2.0 * math.pi * idx / count),
        )
        for idx in range(count + 1)
    ]
    return [
        {
            "x1": points[idx][0],
            "y1": points[idx][1],
            "x2": points[idx + 1][0],
            "y2": points[idx + 1][1],
        }
        for idx in range(count)
    ]


def _segment(name, seg_type, pairs, *, ibc=0, pos_mat=0, neg_mat=0):
    return {
        "name": name,
        "seg_type": str(seg_type),
        "properties": [
            str(seg_type),
            "1",
            str(ibc),
            str(pos_mat),
            str(neg_mat),
        ],
        "point_pairs": list(pairs),
    }


def _snapshot(segments, *, ibcs=None, dielectrics=None):
    return {
        "title": "2d safety regression",
        "segments": list(segments),
        "ibcs": list(ibcs or []),
        "dielectrics": list(dielectrics or []),
    }


def _complex_amplitude(sample):
    return complex(sample["rcs_amp_real"], sample["rcs_amp_imag"])


class TestMaterialPreflight(unittest.TestCase):
    def test_material_tables_reject_bad_rows_duplicates_and_extrapolation(self):
        bad_rows = {
            "short": "1.0 10.0\n",
            "extra": "1.0 10.0 0.0 99.0\n",
            "nonnumeric": "1.0 resistance 0.0\n",
            "duplicate": "1.0 10.0 0.0\n1.0 20.0 0.0\n",
            "nonpositive": "0.0 10.0 0.0\n",
        }
        with tempfile.TemporaryDirectory() as tmp:
            for label, contents in bad_rows.items():
                with self.subTest(label=label):
                    path = os.path.join(tmp, f"mat_{label}")
                    with open(path, "w", encoding="utf-8") as stream:
                        stream.write(contents)
                    with self.assertRaises(ValueError):
                        rcs._load_impedance_table(path)

            valid_path = os.path.join(tmp, "mat_valid")
            with open(valid_path, "w", encoding="utf-8") as stream:
                stream.write("2.0 20.0 0.0\n1.0 10.0 0.0\n")
            table = rcs._load_impedance_table(valid_path)
            self.assertAlmostEqual(table.sample(1.5).real, 15.0)
            with self.assertRaisesRegex(ValueError, "outside"):
                table.sample(0.5)
            with self.assertRaisesRegex(ValueError, "outside"):
                table.sample(2.5)

    def test_malformed_or_missing_geometry_fields_are_not_defaulted(self):
        base = _segment(
            "body",
            2,
            [{"x1": 0.0, "y1": 0.0, "x2": 0.02, "y2": 0.0}],
        )
        malformed_coordinate = _snapshot([dict(
            base,
            point_pairs=[
                {"x1": "not-a-number", "y1": 0.0, "x2": 0.02, "y2": 0.0}
            ],
        )])
        missing_coordinate = _snapshot([dict(
            base,
            point_pairs=[{"x1": 0.0, "y1": 0.0, "x2": 0.02}],
        )])
        malformed_n_seg = dict(base)
        malformed_n_seg["properties"] = ["2", "automatic-ish", "0", "0", "0"]
        malformed_n = _snapshot([malformed_n_seg])
        for snap, message in (
            (malformed_coordinate, "finite numeric value"),
            (missing_coordinate, "missing coordinate field"),
            (malformed_n, "N must be an integer"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    rcs.validate_geometry_snapshot_for_solver(
                        snap, base_dir=".", meters_scale=1.0
                    )

    def test_segment_header_and_properties_type_must_match(self):
        segment = _segment(
            "ambiguous_body",
            2,
            [{"x1": 0.0, "y1": 0.0, "x2": 0.02, "y2": 0.0}],
        )
        segment["properties"][0] = "1"
        with self.assertRaisesRegex(
                ValueError,
                r"declares TYPE 2 in its header but TYPE 1 in properties\[0\]"):
            rcs.validate_geometry_snapshot_for_solver(
                _snapshot([segment]), base_dir=".", meters_scale=1.0
            )

    def test_negative_segment_material_flags_are_rejected_even_when_unused(self):
        pairs = [{"x1": 0.0, "y1": 0.0, "x2": 0.02, "y2": 0.0}]
        cases = (
            ("IBC", _segment("negative_ibc", 2, pairs, ibc=-1)),
            ("pos_mat", _segment("negative_pos", 2, pairs, pos_mat=-1)),
            ("neg_mat", _segment("negative_neg", 2, pairs, neg_mat=-1)),
        )
        for field_name, segment in cases:
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(
                        ValueError, rf"{field_name} flag must be non-negative"):
                    rcs.validate_geometry_snapshot_for_solver(
                        _snapshot([segment]), base_dir=".", meters_scale=1.0
                    )

    def test_disconnected_primitives_within_one_segment_are_rejected(self):
        segment = _segment(
            "broken_chain",
            2,
            [
                {"x1": 0.0, "y1": 0.0, "x2": 0.01, "y2": 0.0},
                {"x1": 0.02, "y1": 0.0, "x2": 0.03, "y2": 0.0},
            ],
        )
        with self.assertRaisesRegex(
                ValueError,
                r"disconnected primitive chain.*must chain head-to-tail"):
            rcs.validate_geometry_snapshot_for_solver(
                _snapshot([segment]), base_dir=".", meters_scale=1.0
            )

    def test_ibc_on_type3_transmission_interface_is_rejected(self):
        snap = _snapshot(
            [_segment("air_dielectric", 3, _circle_pairs(0.03, 12), ibc=1, pos_mat=1)],
            ibcs=[["1", "constant", "100", "0", "0", "0"]],
            dielectrics=[["1", "4", "0", "1", "0"]],
        )
        with self.assertRaisesRegex(ValueError, r"TYPE 3.*IBC flag.*not implemented"):
            rcs.validate_geometry_snapshot_for_solver(snap, base_dir=".", meters_scale=1.0)

    def test_ibc_on_type5_transmission_interface_is_rejected(self):
        snap = _snapshot(
            [_segment("dielectric_dielectric", 5, _circle_pairs(0.03, 12), ibc=1, pos_mat=1, neg_mat=2)],
            ibcs=[["1", "constant", "100", "0", "0", "0"]],
            dielectrics=[
                ["1", "4", "0", "1", "0"],
                ["2", "2", "0", "1", "0"],
            ],
        )
        with self.assertRaisesRegex(ValueError, r"TYPE 5.*IBC flag.*not implemented"):
            rcs.validate_geometry_snapshot_for_solver(snap, base_dir=".", meters_scale=1.0)

    def test_ibc_remains_accepted_on_supported_types(self):
        ibcs = [["1", "constant", "100", "0", "0", "0"]]
        dielectrics = [["1", "4", "0", "1", "0"]]
        cases = [
            _snapshot(
                [_segment("sheet", 1, [{"x1": 0, "y1": 0, "x2": 0.02, "y2": 0}], ibc=1)],
                ibcs=ibcs,
            ),
            _snapshot(
                [_segment("body", 2, _circle_pairs(0.03, 12), ibc=1)],
                ibcs=ibcs,
            ),
            _snapshot(
                [_segment("coating_backing", 4, _circle_pairs(0.03, 12), ibc=1, pos_mat=1)],
                ibcs=ibcs,
                dielectrics=dielectrics,
            ),
        ]
        for snap in cases:
            with self.subTest(seg_type=snap["segments"][0]["seg_type"]):
                rcs.validate_geometry_snapshot_for_solver(snap, base_dir=".", meters_scale=1.0)


class TestTeType2Topology(unittest.TestCase):
    def setUp(self):
        self.open_strip = _snapshot(
            [
                _segment(
                    "open_pec_strip",
                    2,
                    [{"x1": -0.025, "y1": 0.0, "x2": 0.025, "y2": 0.0}],
                )
            ]
        )

    def test_open_type2_te_is_rejected_by_public_solvers(self):
        calls = [
            lambda: rcs.solve_monostatic_rcs_2d(
                self.open_strip, [1.0], [90.0], "TE", geometry_units="meters"
            ),
            lambda: rcs.solve_bistatic_rcs_2d(
                self.open_strip, [1.0], [90.0], [90.0], "TE", geometry_units="meters"
            ),
            lambda: rcs.compute_surface_currents(
                self.open_strip, 1.0, 90.0, "TE", geometry_units="meters"
            ),
        ]
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, r"Open TYPE 2.*not supported.*TE"):
                    call()

    def test_open_type2_tm_remains_supported(self):
        result = rcs.solve_monostatic_rcs_2d(
            self.open_strip, [1.0], [90.0], "TM", geometry_units="meters"
        )
        self.assertTrue(math.isfinite(result["samples"][0]["rcs_db"]))

    def test_closed_and_stitched_type2_te_remain_supported(self):
        pairs = _circle_pairs(0.03, 16)
        closed = _snapshot([_segment("closed", 2, pairs)])
        stitched = _snapshot(
            [
                _segment(f"quarter_{idx}", 2, pairs[4 * idx : 4 * (idx + 1)])
                for idx in range(4)
            ]
        )
        for snap in (closed, stitched):
            with self.subTest(segment_count=len(snap["segments"])):
                result = rcs.solve_monostatic_rcs_2d(
                    snap, [1.0], [0.0], "TE", geometry_units="meters"
                )
                self.assertTrue(math.isfinite(result["samples"][0]["rcs_db"]))


class TestPublicQuantitySemantics(unittest.TestCase):
    def test_boundary_density_export_is_not_labeled_as_physical_current(self):
        snap = _snapshot(
            [_segment("closed_pec", 2, _circle_pairs(0.02, 12))]
        )
        result = rcs.compute_boundary_densities(
            snap, 1.0, 0.0, "TM", geometry_units="meters"
        )
        self.assertEqual(result["quantity"], "boundary_integral_layer_density")
        self.assertFalse(result["is_physical_surface_current"])
        self.assertIn("not", result["interpretation"])

    def test_stored_2d_amplitude_convention_is_explicit(self):
        snap = _snapshot(
            [_segment("open_pec_strip", 2, [
                {"x1": -0.02, "y1": 0.0, "x2": 0.02, "y2": 0.0}
            ])]
        )
        result = rcs.solve_monostatic_rcs_2d(
            snap, [1.0], [90.0], "TM", geometry_units="meters"
        )
        self.assertEqual(
            result["amplitude_convention"],
            "A_physical_asymptotic = +j * B_stored",
        )
        payload = grim_io._build_grid_for_samples(
            result["samples"], "TM", rcs_linear_quantity="sigma_2d"
        )
        self.assertIn("A=j*B", payload["phase_reference"])
        self.assertEqual(
            payload["complex_field_domain"],
            "2d_layer_potential_bare_integral_amplitude_B",
        )


class TestBistaticTaperAndQualityMetadata(unittest.TestCase):
    def test_te_tapered_ibc_bistatic_diagonal_matches_monostatic(self):
        snap = _snapshot(
            [_segment("tapered_body", 2, _circle_pairs(0.04, 24), ibc=1)],
            ibcs=[["1", "linear", "20", "5", "500", "80"]],
        )
        mono = rcs.solve_monostatic_rcs_2d(
            snap, [1.0], [37.0], "TE", geometry_units="meters"
        )
        bistatic = rcs.solve_bistatic_rcs_2d(
            snap, [1.0], [37.0], [37.0], "TE", geometry_units="meters"
        )
        a_mono = _complex_amplitude(mono["samples"][0])
        a_bistatic = _complex_amplitude(bistatic["samples"][0])
        relative_error = abs(a_bistatic - a_mono) / max(abs(a_mono), 1.0e-15)
        self.assertLess(relative_error, 1.0e-11)

    def test_quality_gate_infers_uncomputed_condition_placeholder(self):
        gate = rcs.evaluate_quality_gate(
            {
                "residual_norm_max": 1.0e-10,
                "condition_est_max": float("nan"),
                "warnings": [],
            }
        )
        self.assertTrue(gate["passed"])
        self.assertFalse(gate["values"]["condition_est_computed"])

    def test_quality_gate_infers_legacy_finite_condition(self):
        gate = rcs.evaluate_quality_gate(
            {
                "residual_norm_max": 1.0e-10,
                "condition_est_max": 2.0e6,
                "warnings": [],
            }
        )
        self.assertFalse(gate["passed"])
        self.assertTrue(gate["values"]["condition_est_computed"])

    def test_solver_computes_requested_condition_estimate(self):
        snap = _snapshot([_segment("closed", 2, _circle_pairs(0.03, 12))])
        result = rcs.solve_monostatic_rcs_2d(
            snap,
            [1.0],
            [0.0],
            "TE",
            geometry_units="meters",
            compute_condition_number=True,
            strict_quality_gate=True,
        )
        metadata = result["metadata"]
        self.assertTrue(metadata["condition_est_computed"])
        self.assertTrue(math.isfinite(metadata["condition_est_max"]))
        self.assertGreater(metadata["condition_est_max"], 0.0)
        self.assertTrue(metadata["quality_gate"]["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
