"""Regressions for passive-material branches and 2D RCS null handling."""

import math
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rcs_solver as rcs
from geometry_io import parse_geometry


class TestCausalMaterialBranch(unittest.TestCase):
    def test_exact_lossless_double_negative_uses_negative_index(self):
        n = rcs._causal_medium_index(-4.0 + 0.0j, -1.0 + 0.0j)
        self.assertAlmostEqual(n.real, -2.0, places=14)
        self.assertAlmostEqual(n.imag, 0.0, places=14)
        self.assertGreater(rcs._medium_eta(-4.0 + 0.0j, -1.0 + 0.0j).real, 0.0)

    def test_double_negative_branch_is_continuous_with_limiting_loss(self):
        n_lossless = rcs._causal_medium_index(-4.0 + 0.0j, -1.0 + 0.0j)
        n_lossy = rcs._causal_medium_index(-4.0 - 1.0e-9j, -1.0 - 1.0e-9j)
        self.assertLess(n_lossy.real, 0.0)
        self.assertLess(n_lossy.imag, 0.0)
        self.assertLess(abs(n_lossy - n_lossless), 2.0e-9)

    def test_ordinary_lossless_medium_keeps_positive_index(self):
        n = rcs._causal_medium_index(4.0 + 0.0j, 1.0 + 0.0j)
        self.assertAlmostEqual(n.real, 2.0, places=14)
        self.assertAlmostEqual(n.imag, 0.0, places=14)


class TestMaterialInputSafety(unittest.TestCase):
    def test_explicit_enz_is_rejected_not_replaced_with_air(self):
        with self.assertRaisesRegex(ValueError, r"near-ENZ.*not be replaced"):
            rcs.MaterialLibrary.from_entries(
                [], [["1", "0", "0", "1", "0"]], base_dir="."
            )

    def test_explicit_mnz_is_rejected_not_replaced_with_air(self):
        with self.assertRaisesRegex(ValueError, r"near-MNZ.*not be replaced"):
            rcs.MaterialLibrary.from_entries(
                [], [["1", "2", "0", "0", "0"]], base_dir="."
            )

    def test_supported_small_enz_value_is_preserved(self):
        lib = rcs.MaterialLibrary.from_entries(
            [], [["1", "1e-9", "0", "1", "0"]], base_dir="."
        )
        eps, mu = lib.get_medium(1, 1.0)
        self.assertEqual(eps, 1.0e-9 + 0.0j)
        self.assertEqual(mu, 1.0 + 0.0j)

    def test_invalid_inline_material_token_is_not_defaulted(self):
        with self.assertRaisesRegex(ValueError, r"epsilon real part.*finite numeric"):
            rcs.MaterialLibrary.from_entries(
                [], [["1", "not-a-number", "0", "1", "0"]], base_dir="."
            )

    def test_missing_inline_fields_are_not_defaulted_to_air(self):
        for row in (["1"], ["1", "4"], ["1", "4", "0", "1", ""]):
            with self.subTest(row=row):
                with self.assertRaisesRegex(
                        ValueError, r"inline material requires exactly"):
                    rcs.MaterialLibrary.from_entries([], [row], base_dir=".")

    def test_duplicate_inline_flags_are_rejected(self):
        with self.assertRaisesRegex(
                ValueError, r"Duplicate dielectric material flag 1"):
            rcs.MaterialLibrary.from_entries(
                [], [["1", "2", "0", "1", "0"],
                     ["1", "3", "0", "1", "0"]], base_dir=".")
        with self.assertRaisesRegex(ValueError, r"Duplicate IBC material flag 1"):
            rcs.MaterialLibrary.from_entries(
                [["1", "constant", "20", "0", "0", "0"],
                 ["1", "constant", "30", "0", "0", "0"]],
                [], base_dir=".")

    def test_direct_material_rows_require_positive_integer_flags(self):
        bad_flags = ("not-a-flag", "0", "-1", "1.5")
        for flag in bad_flags:
            with self.subTest(kind="IBC", flag=flag):
                with self.assertRaisesRegex(
                        ValueError,
                        r"definition flag must be (?:an integer|a positive integer)"):
                    rcs.MaterialLibrary.from_entries(
                        [[flag, "constant", "20", "0", "0", "0"]],
                        [],
                        base_dir=".",
                    )
            with self.subTest(kind="dielectric", flag=flag):
                with self.assertRaisesRegex(
                        ValueError,
                        r"definition flag must be (?:an integer|a positive integer)"):
                    rcs.MaterialLibrary.from_entries(
                        [],
                        [[flag, "2", "0", "1", "0"]],
                        base_dir=".",
                    )

    def test_geo_material_rows_require_positive_integer_flags(self):
        prefix = (
            "Title: bad material flag\n"
            "Segment: body 2\n"
            "properties: 2 1 0 0 0\n"
            "0 0 1 0\n"
        )
        for flag in ("not-a-flag", "0", "-1"):
            with self.subTest(kind="IBC", flag=flag):
                text = (
                    prefix
                    + "IBCS_Resistances:\n"
                    + f"{flag} constant 20 0 0 0\n"
                    + "Dielectrics:\n"
                )
                with self.assertRaisesRegex(
                        ValueError,
                        r"IBC row must start with a positive integer flag"):
                    parse_geometry(text)
            with self.subTest(kind="dielectric", flag=flag):
                text = (
                    prefix
                    + "IBCS_Resistances:\n"
                    + "Dielectrics:\n"
                    + f"{flag} 2 0 1 0\n"
                )
                with self.assertRaisesRegex(
                        ValueError,
                        r"Dielectric row must start with a positive integer flag"):
                    parse_geometry(text)

    def test_geo_segment_requires_exactly_one_properties_line(self):
        missing = (
            "Title: missing properties\n"
            "Segment: body 2\n"
            "0 0 1 0\n"
            "IBCS_Resistances:\n"
            "Dielectrics:\n"
        )
        duplicate = (
            "Title: duplicate properties\n"
            "Segment: body 2\n"
            "properties: 2 1 0 0 0\n"
            "properties: 2 2 0 0 0\n"
            "0 0 1 0\n"
            "IBCS_Resistances:\n"
            "Dielectrics:\n"
        )
        with self.assertRaisesRegex(
                ValueError, r"missing its required properties line"):
            parse_geometry(missing)
        with self.assertRaisesRegex(
                ValueError, r"more than one properties line"):
            parse_geometry(duplicate)

    def test_programmatic_snapshot_may_use_header_type_without_properties(self):
        snapshot = {
            "title": "programmatic snapshot",
            "segments": [{
                "name": "open_tm_body",
                "seg_type": "2",
                "properties": [],
                "point_pairs": [{
                    "x1": 0.0, "y1": 0.0, "x2": 0.02, "y2": 0.0,
                }],
            }],
            "ibcs": [],
            "dielectrics": [],
        }
        report = rcs.validate_geometry_snapshot_for_solver(
            snapshot, base_dir=".", meters_scale=1.0
        )
        self.assertEqual(report["segment_count"], 1)

    def test_material_sidecar_never_falls_back_to_process_cwd(self):
        with tempfile.TemporaryDirectory() as geometry_dir, \
                tempfile.TemporaryDirectory() as unrelated_cwd:
            with open(os.path.join(unrelated_cwd, "mat.51"), "w") as handle:
                handle.write("1 2 0 1 0\n")
            with mock.patch.object(
                    rcs.os, "getcwd", return_value=unrelated_cwd):
                with self.assertRaisesRegex(
                        FileNotFoundError, r"declared material directory"):
                    rcs.MaterialLibrary.from_entries(
                        [], [["51"]], base_dir=geometry_dir)

    def test_file_backed_snapshot_inherits_its_own_material_directory(self):
        with tempfile.TemporaryDirectory() as geometry_dir, \
                tempfile.TemporaryDirectory() as explicit_dir:
            snapshot = {
                "source_path": os.path.join(geometry_dir, "model.geo")
            }
            self.assertEqual(
                rcs._material_base_dir_for_snapshot(snapshot, None),
                os.path.abspath(geometry_dir),
            )
            self.assertEqual(
                rcs._material_base_dir_for_snapshot(snapshot, explicit_dir),
                os.path.abspath(explicit_dir),
            )

    def test_only_pathless_snapshot_uses_documented_cwd_default(self):
        with tempfile.TemporaryDirectory() as programmatic_dir:
            with mock.patch.object(
                    rcs.os, "getcwd", return_value=programmatic_dir):
                self.assertEqual(
                    rcs._material_base_dir_for_snapshot({}, None),
                    os.path.abspath(programmatic_dir),
                )

    def test_undefined_positive_material_flag_is_not_air(self):
        lib = rcs.MaterialLibrary({}, {})
        with self.assertRaisesRegex(ValueError, r"Undefined dielectric flag 7"):
            lib.get_medium(7, 1.0)
        with self.assertRaisesRegex(ValueError, r"Undefined IBC flag 7"):
            lib.get_impedance(7, 1.0)

    def test_internal_type1_virtual_region_remains_air(self):
        lib = rcs.MaterialLibrary({}, {})
        self.assertEqual(
            rcs._region_medium(lib, rcs.VIRTUAL_SHEET_REGION_START, 1.0),
            (1.0 + 0.0j, 1.0 + 0.0j),
        )

    def test_negative_ibc_resistance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"negative resistance.*not supported"):
            rcs.MaterialLibrary.from_entries(
                [["1", "constant", "-1", "20", "0", "0"]],
                [],
                base_dir=".",
            )

    def test_negative_resistance_in_table_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mat.51")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write("1.0 -0.5 20.0\n")
            with self.assertRaisesRegex(ValueError, r"negative resistance.*not supported"):
                rcs.MaterialLibrary.from_entries([["51"]], [], base_dir=tmp)

    def test_gain_sign_epsilon_and_mu_are_rejected(self):
        cases = [
            ["1", "2", "0.01", "1", "0"],
            ["1", "2", "0", "1", "0.01"],
        ]
        for row in cases:
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, r"gain-sign.*not supported"):
                    rcs.MaterialLibrary.from_entries([], [row], base_dir=".")

    def test_passive_loss_sign_is_accepted(self):
        lib = rcs.MaterialLibrary.from_entries(
            [],
            [["1", "-2", "-0.1", "-1", "-0.05"]],
            base_dir=".",
        )
        eps, mu = lib.get_medium(1, 1.0)
        self.assertLess(eps.imag, 0.0)
        self.assertLess(mu.imag, 0.0)
        self.assertLess(rcs._causal_medium_index(eps, mu).imag, 0.0)


class TestRcsNullPreservation(unittest.TestCase):
    def test_zero_and_deep_nulls_remain_zero_or_deep_in_linear_units(self):
        k = 2.0
        amplitudes = np.asarray([0.0 + 0.0j, 1.0e-10 + 0.0j])
        sigma = rcs._rcs_sigma_from_amp(amplitudes, k)
        self.assertEqual(sigma[0], 0.0)
        self.assertAlmostEqual(
            sigma[1],
            rcs.RCS_NORM_NUMERATOR * 1.0e-20 / k,
            delta=1.0e-36,
        )
        self.assertLess(sigma[1], rcs.EPS)

    def test_db_floor_does_not_mutate_linear_values(self):
        sigma = np.asarray([0.0, 1.0e-20, 1.0])
        original = sigma.copy()
        db = rcs._rcs_db_from_sigma(sigma)
        np.testing.assert_array_equal(sigma, original)
        self.assertEqual(db[0], 10.0 * math.log10(rcs.RCS_DB_FLOOR_LINEAR))
        self.assertEqual(db[1], 10.0 * math.log10(rcs.RCS_DB_FLOOR_LINEAR))
        self.assertEqual(db[2], 0.0)


if __name__ == "__main__":
    unittest.main()
