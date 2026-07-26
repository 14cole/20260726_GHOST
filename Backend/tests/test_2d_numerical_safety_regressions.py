"""Regressions for 2D FMM convergence and material-wavelength meshing."""

import copy
import math
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rcs_solver as rcs


def _line_snapshot(eps_real: float, n_value: int):
    return {
        "segments": [
            {
                "name": "dielectric_line",
                "seg_type": "3",
                "properties": ["3", str(n_value), "0", "1", "0"],
                "point_pairs": [
                    {"x1": 0.0, "y1": 0.0, "x2": 0.3, "y2": 0.0},
                ],
            }
        ],
        "ibcs": [],
        "dielectrics": [["1", str(eps_real), "0", "1", "0"]],
    }


def _clockwise_circle_snapshot(eps_real: float, count: int = 8):
    points = [
        (
            0.03 * math.cos(-2.0 * math.pi * idx / count),
            0.03 * math.sin(-2.0 * math.pi * idx / count),
        )
        for idx in range(count + 1)
    ]
    return {
        "segments": [
            {
                "name": "dielectric_circle",
                "seg_type": "3",
                "properties": ["3", "0", "0", "1", "0"],
                "point_pairs": [
                    {
                        "x1": points[idx][0],
                        "y1": points[idx][1],
                        "x2": points[idx + 1][0],
                        "y2": points[idx + 1][1],
                    }
                    for idx in range(count)
                ],
            }
        ],
        "ibcs": [],
        "dielectrics": [["1", str(eps_real), "0", "1", "0"]],
    }


class TestFmmGmresSafety(unittest.TestCase):
    def setUp(self):
        if rcs._SCIPY_SPARSE_LINALG is None:
            self.skipTest("SciPy sparse linear algebra is unavailable")
        self.operator = rcs._SCIPY_SPARSE_LINALG.LinearOperator(
            (2, 2),
            matvec=lambda vector: np.asarray(vector, dtype=np.complex128),
            dtype=np.complex128,
        )
        self.rhs = np.asarray([[1.0 + 0.0j], [2.0 + 0.0j]])

    def test_positive_nonconvergence_status_aborts_without_field(self):
        with mock.patch.object(
            rcs,
            "_gmres_compat",
            return_value=(np.zeros(2, dtype=np.complex128), 7),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                r"multi-region indirect.*did not converge.*No unconverged field or RCS",
            ):
                rcs._solve_fmm_gmres_columns(
                    self.operator,
                    self.rhs,
                    np.asarray([37.0]),
                    formulation="multi-region indirect",
                    restart=80,
                    maxiter=500,
                    rtol=1.0e-10,
                )

    def test_breakdown_status_aborts_with_actionable_dense_option(self):
        with mock.patch.object(
            rcs,
            "_gmres_compat",
            return_value=(np.zeros(2, dtype=np.complex128), -1),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                r"TE Robin MFIE.*breakdown/illegal-input.*solver_method='auto'",
            ):
                rcs._solve_fmm_gmres_columns(
                    self.operator,
                    self.rhs,
                    np.asarray([0.0]),
                    formulation="TE Robin MFIE",
                    restart=50,
                    maxiter=300,
                    rtol=1.0e-10,
                )

    def test_converged_columns_are_returned(self):
        expected = np.asarray([1.0 + 0.0j, 2.0 + 0.0j])
        with mock.patch.object(
            rcs, "_gmres_compat", return_value=(expected.copy(), 0)
        ):
            actual = rcs._solve_fmm_gmres_columns(
                self.operator,
                self.rhs,
                np.asarray([15.0]),
                formulation="TE Robin MFIE",
                restart=50,
                maxiter=300,
                rtol=1.0e-10,
            )
        np.testing.assert_array_equal(actual[:, 0], expected)

    def test_nonfinite_iterate_is_rejected_even_with_success_status(self):
        candidate = np.asarray([np.nan + 0.0j, 0.0 + 0.0j])
        with mock.patch.object(
            rcs, "_gmres_compat", return_value=(candidate, 0)
        ):
            with self.assertRaisesRegex(RuntimeError, r"non-finite field values"):
                rcs._solve_fmm_gmres_columns(
                    self.operator,
                    self.rhs,
                    np.asarray([0.0]),
                    formulation="TE Robin MFIE",
                    restart=50,
                    maxiter=300,
                    rtol=1.0e-10,
                )

    def test_te_robin_fmm_converged_path_matches_dense(self):
        try:
            __import__("fmm_helmholtz_2d")
        except ImportError:
            self.skipTest("2D FMM backend is unavailable")
        snapshot = _clockwise_circle_snapshot(1.0, count=16)
        snapshot["segments"][0]["name"] = "pec_circle"
        snapshot["segments"][0]["seg_type"] = "2"
        snapshot["segments"][0]["properties"] = ["2", "1", "0", "0", "0"]
        snapshot["dielectrics"] = []
        dense = rcs.solve_monostatic_rcs_2d(
            snapshot, [1.0], [0.0], "TE",
            geometry_units="meters", solver_method="auto",
        )
        fmm = rcs.solve_monostatic_rcs_2d(
            snapshot, [1.0], [0.0], "TE",
            geometry_units="meters", solver_method="fmm",
        )
        sigma_dense = dense["samples"][0]["rcs_linear"]
        sigma_fmm = fmm["samples"][0]["rcs_linear"]
        self.assertLess(abs(sigma_fmm - sigma_dense) / sigma_dense, 1.0e-8)
        self.assertLess(fmm["samples"][0]["linear_residual"], 1.0e-8)

    def test_multi_region_fmm_converged_path_matches_dense(self):
        try:
            __import__("fmm_helmholtz_2d")
        except ImportError:
            self.skipTest("2D FMM backend is unavailable")
        snapshot = _clockwise_circle_snapshot(4.0, count=8)
        inner = copy.deepcopy(snapshot["segments"][0])
        inner["name"] = "pec_core"
        inner["seg_type"] = "4"
        inner["properties"] = ["4", "0", "0", "1", "0"]
        for pair in inner["point_pairs"]:
            for key in ("x1", "y1", "x2", "y2"):
                pair[key] *= 0.5
        snapshot["segments"].append(inner)
        dense = rcs.solve_monostatic_rcs_2d(
            snapshot, [1.0], [0.0], "TM",
            geometry_units="meters", solver_method="auto",
        )
        fmm = rcs.solve_monostatic_rcs_2d(
            snapshot, [1.0], [0.0], "TM",
            geometry_units="meters", solver_method="fmm",
        )
        sigma_dense = dense["samples"][0]["rcs_linear"]
        sigma_fmm = fmm["samples"][0]["rcs_linear"]
        self.assertLess(abs(sigma_fmm - sigma_dense) / sigma_dense, 1.0e-8)
        self.assertLess(fmm["samples"][0]["linear_residual"], 1.0e-8)


class TestMaterialWavelengthMeshing(unittest.TestCase):
    def _library(self, snapshot):
        return rcs.MaterialLibrary.from_entries(
            snapshot["ibcs"], snapshot["dielectrics"], base_dir="."
        )

    def test_eps9_auto_mesh_resolves_three_times_shorter_wavelength(self):
        air = _line_snapshot(1.0, 0)
        eps9 = _line_snapshot(9.0, 0)
        lam_air, index_air, _ = rcs._mesh_wavelength_for_snapshot(
            air, self._library(air), 1.0
        )
        lam_eps9, index_eps9, flags = rcs._mesh_wavelength_for_snapshot(
            eps9, self._library(eps9), 1.0
        )
        self.assertAlmostEqual(index_air, 1.0)
        self.assertAlmostEqual(index_eps9, 3.0)
        self.assertAlmostEqual(lam_air / lam_eps9, 3.0)
        self.assertEqual(flags, [1])

        panels_air = rcs._build_panels(air, 1.0, lam_air)
        panels_eps9 = rcs._build_panels(eps9, 1.0, lam_eps9)
        self.assertGreater(len(panels_eps9), len(panels_air))
        self.assertGreaterEqual(len(panels_eps9), 2.8 * len(panels_air))

    def test_negative_n_uses_material_wavelength(self):
        air = _line_snapshot(1.0, -10)
        eps9 = _line_snapshot(9.0, -10)
        lam_air, _, _ = rcs._mesh_wavelength_for_snapshot(
            air, self._library(air), 1.0
        )
        lam_eps9, _, _ = rcs._mesh_wavelength_for_snapshot(
            eps9, self._library(eps9), 1.0
        )
        panels_air = rcs._build_panels(air, 1.0, lam_air)
        panels_eps9 = rcs._build_panels(eps9, 1.0, lam_eps9)
        self.assertGreater(len(panels_eps9), len(panels_air))
        self.assertGreaterEqual(len(panels_eps9), 2.8 * len(panels_air))

    def test_explicit_positive_n_is_preserved(self):
        eps9 = _line_snapshot(9.0, 7)
        wavelength, _, _ = rcs._mesh_wavelength_for_snapshot(
            eps9, self._library(eps9), 1.0
        )
        self.assertEqual(len(rcs._build_panels(eps9, 1.0, wavelength)), 7)

    def test_unused_high_index_material_does_not_over_refine(self):
        snapshot = _line_snapshot(1.0, 0)
        snapshot["dielectrics"].append(["2", "100", "0", "1", "0"])
        wavelength, max_index, flags = rcs._mesh_wavelength_for_snapshot(
            snapshot, self._library(snapshot), 1.0
        )
        self.assertAlmostEqual(wavelength, rcs.C0 / 1.0e9)
        self.assertAlmostEqual(max_index, 1.0)
        self.assertEqual(flags, [1])

    def test_dispersive_index_is_resampled_at_each_mesh_frequency(self):
        snapshot = _line_snapshot(1.0, 0)
        table = rcs.MediumTable(
            freqs_ghz=np.asarray([1.0, 2.0]),
            eps_values=np.asarray([1.0 + 0.0j, 9.0 + 0.0j]),
            mu_values=np.asarray([1.0 + 0.0j, 1.0 + 0.0j]),
        )
        library = rcs.MaterialLibrary({}, {1: table})
        wavelength_1, index_1, _ = rcs._mesh_wavelength_for_snapshot(
            snapshot, library, 1.0
        )
        wavelength_2, index_2, _ = rcs._mesh_wavelength_for_snapshot(
            snapshot, library, 2.0
        )
        self.assertAlmostEqual(index_1, 1.0)
        self.assertAlmostEqual(index_2, 3.0)
        self.assertAlmostEqual(wavelength_1, rcs.C0 / 1.0e9)
        self.assertAlmostEqual(wavelength_2, rcs.C0 / 6.0e9)

    def test_stage1_metadata_records_controlling_material_scale(self):
        snapshot = _clockwise_circle_snapshot(9.0)
        prepared = rcs.prepare_linear_galerkin_system(
            snapshot,
            frequency_ghz=1.0,
            polarization="TM",
            geometry_units="meters",
        )
        metadata = prepared["metadata"]
        self.assertAlmostEqual(metadata["mesh_max_refractive_index"], 3.0)
        self.assertAlmostEqual(
            metadata["mesh_wavelength_m"], rcs.C0 / (3.0e9)
        )
        self.assertEqual(metadata["mesh_material_flags"], [1])


if __name__ == "__main__":
    unittest.main()
