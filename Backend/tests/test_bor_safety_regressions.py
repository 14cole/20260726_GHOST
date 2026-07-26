"""Focused regressions for BoR material, formulation, and modal safety."""

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bor_dispatch
import bor_kernels
import bor_solver
import mie_sphere


def _sphere_pairs(radius: float = 0.03, count: int = 8):
    """North-to-south right-half profile for an outward-normal BoR."""

    angles = [math.pi * idx / count for idx in range(count + 1)]
    points = [
        (radius * math.sin(angle), radius * math.cos(angle))
        for angle in angles
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


def _segment(name, seg_type, *, ibc=0, pos_mat=0, neg_mat=0):
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
        "point_pairs": _sphere_pairs(),
    }


def _snapshot(segments, *, ibcs=None, dielectrics=None):
    return {
        "title": "BoR safety regression",
        "segments": list(segments),
        "ibcs": list(ibcs or []),
        "dielectrics": list(dielectrics or []),
    }


class TestBorDispatchPreflight(unittest.TestCase):
    def test_shared_preflight_receives_material_base_dir_and_unit_scale(self):
        snapshot = _snapshot([_segment("body", 2)])
        with tempfile.TemporaryDirectory() as material_dir:
            with mock.patch.object(
                bor_dispatch,
                "validate_geometry_snapshot_for_solver",
                side_effect=RuntimeError("preflight sentinel"),
            ) as validator:
                with self.assertRaisesRegex(RuntimeError, "preflight sentinel"):
                    bor_dispatch.solve_monostatic_rcs_bor(
                        snapshot,
                        [1.0],
                        [0.0],
                        "VV",
                        geometry_units="inches",
                        material_base_dir=material_dir,
                    )
        validator.assert_called_once_with(
            snapshot,
            base_dir=material_dir,
            meters_scale=0.0254,
        )

    def test_file_backed_snapshot_supplies_sidecar_directory(self):
        snapshot = _snapshot([_segment("body", 2)])
        with tempfile.TemporaryDirectory() as geometry_dir:
            snapshot["source_path"] = os.path.join(
                geometry_dir, "body.geo"
            )
            with mock.patch.object(
                bor_dispatch,
                "validate_geometry_snapshot_for_solver",
                side_effect=RuntimeError("preflight sentinel"),
            ) as validator:
                with self.assertRaisesRegex(
                    RuntimeError, "preflight sentinel"
                ):
                    bor_dispatch.solve_monostatic_rcs_bor(
                        snapshot,
                        [1.0],
                        [0.0],
                        "VV",
                        geometry_units="meters",
                    )
        validator.assert_called_once_with(
            snapshot,
            base_dir=os.path.abspath(geometry_dir),
            meters_scale=1.0,
        )

    def test_undefined_type2_ibc_is_rejected_before_solve(self):
        snapshot = _snapshot([_segment("impedance_body", 2, ibc=7)])
        with self.assertRaisesRegex(ValueError, r"undefined IBC flag 7"):
            bor_dispatch.solve_monostatic_rcs_bor(
                snapshot, [1.0], [0.0], "VV", geometry_units="meters"
            )

    def test_ibc_on_unimplemented_bor_interface_types_is_rejected(self):
        ibcs = [["1", "constant", "100", "0", "0", "0"]]
        dielectrics = [
            ["1", "4", "-0.1", "1", "0"],
            ["2", "2", "-0.1", "1", "0"],
        ]
        cases = [
            _segment("outer", 3, ibc=1, pos_mat=1),
            _segment("core", 4, ibc=1, pos_mat=1),
            _segment("layer", 5, ibc=1, pos_mat=1, neg_mat=2),
        ]
        for segment in cases:
            with self.subTest(seg_type=segment["seg_type"]):
                snapshot = _snapshot(
                    [segment], ibcs=ibcs, dielectrics=dielectrics
                )
                with self.assertRaisesRegex(
                    ValueError,
                    rf"BoR TYPE {segment['seg_type']}.*IBC flag 1.*not implemented",
                ):
                    bor_dispatch.solve_monostatic_rcs_bor(
                        snapshot,
                        [1.0],
                        [0.0],
                        "VV",
                        geometry_units="meters",
                    )

    def test_supported_type2_ibc_passes_bor_specific_guard(self):
        snapshot = _snapshot(
            [_segment("impedance_body", 2, ibc=1)],
            ibcs=[["1", "constant", "100", "0", "0", "0"]],
        )
        bor_dispatch._reject_unsupported_bor_ibc_interfaces(snapshot)

    def test_exact_rcs_null_is_preserved_in_linear_output(self):
        snapshot = _snapshot([_segment("body", 2)])
        solved = {
            "sigma_vv": [0.0],
            "sigma_hh": [0.0],
            "amp_vv": [0.0j],
            "amp_hh": [0.0j],
            "modes_used": 2,
            "mode_cap": 4,
            "mode_converged": True,
            "mode_quiet_count": 2,
            "mode_last_relative_increment": 0.0,
            "n_unknowns": 8,
            "linear_residual": 0.0,
            "condition_est_computed": True,
            "max_cond": 10.0,
            "warnings": [],
        }
        with mock.patch.object(bor_dispatch, "solve_bor", return_value=solved):
            result = bor_dispatch.solve_monostatic_rcs_bor(
                snapshot, [1.0], [0.0], "VV", geometry_units="meters"
            )
        self.assertEqual(result["samples"][0]["rcs_linear"], 0.0)
        self.assertTrue(math.isfinite(result["samples"][0]["rcs_db"]))

    def test_low_mesh_reference_cannot_underresolve_higher_solve_frequency(self):
        snapshot = _snapshot([_segment("body", 2)])
        solved = {
            "sigma_vv": [0.0],
            "sigma_hh": [0.0],
            "amp_vv": [0.0j],
            "amp_hh": [0.0j],
            "modes_used": 2,
            "mode_cap": 4,
            "mode_converged": True,
            "mode_quiet_count": 2,
            "mode_last_relative_increment": 0.0,
            "n_unknowns": 8,
            "linear_residual": 0.0,
            "condition_est_computed": True,
            "max_cond": 10.0,
            "warnings": [],
        }
        with mock.patch.object(bor_dispatch, "solve_bor", return_value=solved):
            result = bor_dispatch.solve_monostatic_rcs_bor(
                snapshot,
                [1.0, 4.0],
                [0.0],
                "VV",
                geometry_units="meters",
                mesh_reference_ghz=0.5,
            )
        expected = bor_kernels.C0 / 4.0e9
        self.assertAlmostEqual(
            result["metadata"]["mesh_wavelength_m"], expected, places=15
        )
        self.assertEqual(
            result["metadata"]["mesh_control_frequencies_ghz"],
            [0.5, 1.0, 4.0],
        )

    def test_conductor_controls_are_forwarded_and_quality_is_attested(self):
        snapshot = _snapshot([_segment("body", 2)])
        amps = [
            math.sqrt(1.0 / (4.0 * math.pi)),
            math.sqrt(2.0 / (4.0 * math.pi)),
        ]
        solved = {
            "sigma_vv": [1.0, 2.0],
            "sigma_hh": [1.0, 2.0],
            "amp_vv": [complex(value) for value in amps],
            "amp_hh": [complex(value) for value in amps],
            "modes_used": 2,
            "mode_cap": 4,
            "mode_converged": True,
            "mode_quiet_count": 2,
            "mode_last_relative_increment": 0.0,
            "n_unknowns": 8,
            "linear_residual": 1.0e-10,
            "condition_est_computed": True,
            "max_cond": 10.0,
            "assembly": "streaming",
            "table_precision": "single",
            "stream_mode_block": 4,
            "stream_sweeps": 1,
            "warnings": [],
        }
        with mock.patch.object(
            bor_dispatch, "solve_bor", return_value=solved
        ) as solve:
            result = bor_dispatch.solve_monostatic_rcs_bor(
                snapshot,
                [1.0],
                [0.0, 90.0],
                "VV",
                geometry_units="meters",
                assembly="streaming",
                table_precision="single",
                stream_budget_gb=0.25,
            )

        kwargs = solve.call_args.kwargs
        self.assertEqual(kwargs["assembly"], "streaming")
        self.assertEqual(kwargs["table_precision"], "single")
        self.assertEqual(kwargs["stream_budget_gb"], 0.25)
        metadata = result["metadata"]
        self.assertEqual(metadata["frequency_count"], 1)
        self.assertEqual(metadata["aspect_count"], 2)
        self.assertEqual(metadata["output_aspect_count"], 2)
        self.assertTrue(metadata["quality_gate"]["passed"])
        self.assertTrue(metadata["far_table_controls_applicable"])
        self.assertEqual(
            metadata["per_frequency"][0]["assembly"], "streaming"
        )
        self.assertEqual(
            metadata["per_frequency"][0]["table_precision"], "single"
        )
        self.assertEqual(
            metadata["quality_gate"]["thresholds"]["residual_norm_max"],
            bor_dispatch.BOR_LINEAR_RESIDUAL_MAX,
        )

    def test_dispatch_quality_gate_rejects_bad_low_level_result(self):
        snapshot = _snapshot([_segment("body", 2)])
        amp = complex(math.sqrt(1.0 / (4.0 * math.pi)))
        solved = {
            "sigma_vv": [1.0],
            "sigma_hh": [1.0],
            "amp_vv": [amp],
            "amp_hh": [amp],
            "modes_used": 2,
            "mode_cap": 4,
            "mode_converged": True,
            "mode_quiet_count": 2,
            "mode_last_relative_increment": 0.0,
            "n_unknowns": 8,
            "linear_residual": 1.0e-3,
            "condition_est_computed": True,
            "max_cond": 10.0,
            "warnings": [],
        }
        with mock.patch.object(
            bor_dispatch, "solve_bor", return_value=solved
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                r"BoR quality gate failed: residual_norm_max=.*exceeds",
            ):
                bor_dispatch.solve_monostatic_rcs_bor(
                    snapshot,
                    [1.0],
                    [0.0],
                    "VV",
                    geometry_units="meters",
                )

    def test_dispatch_quality_gate_rejects_missing_residual_telemetry(self):
        snapshot = _snapshot([_segment("body", 2)])
        amp = complex(math.sqrt(1.0 / (4.0 * math.pi)))
        solved = {
            "sigma_vv": [1.0],
            "sigma_hh": [1.0],
            "amp_vv": [amp],
            "amp_hh": [amp],
            "modes_used": 2,
            "mode_cap": 4,
            "mode_converged": True,
            "mode_quiet_count": 2,
            "mode_last_relative_increment": 0.0,
            "n_unknowns": 8,
            "warnings": [],
        }
        with mock.patch.object(
            bor_dispatch, "solve_bor", return_value=solved
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                r"residual_nonfinite_count=1 must be zero",
            ):
                bor_dispatch.solve_monostatic_rcs_bor(
                    snapshot,
                    [1.0],
                    [0.0],
                    "VV",
                    geometry_units="meters",
                )

    def test_dispatch_quality_gate_rejects_missing_condition_telemetry(self):
        snapshot = _snapshot([_segment("body", 2)])
        amp = complex(math.sqrt(1.0 / (4.0 * math.pi)))
        solved = {
            "sigma_vv": [1.0],
            "sigma_hh": [1.0],
            "amp_vv": [amp],
            "amp_hh": [amp],
            "modes_used": 2,
            "mode_cap": 4,
            "mode_converged": True,
            "mode_quiet_count": 2,
            "mode_last_relative_increment": 0.0,
            "n_unknowns": 8,
            "linear_residual": 0.0,
            "warnings": [],
        }
        with mock.patch.object(
            bor_dispatch, "solve_bor", return_value=solved
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                r"condition_missing_or_nonfinite_count=1 must be zero",
            ):
                bor_dispatch.solve_monostatic_rcs_bor(
                    snapshot,
                    [1.0],
                    [0.0],
                    "VV",
                    geometry_units="meters",
                )

    def test_dispatch_quality_gate_rejects_inconsistent_rcs_and_amplitude(self):
        snapshot = _snapshot([_segment("body", 2)])
        solved = {
            "sigma_vv": [1.0],
            "sigma_hh": [1.0],
            "amp_vv": [0.0j],
            "amp_hh": [0.0j],
            "modes_used": 2,
            "mode_cap": 4,
            "mode_converged": True,
            "mode_quiet_count": 2,
            "mode_last_relative_increment": 0.0,
            "n_unknowns": 8,
            "linear_residual": 0.0,
            "condition_est_computed": True,
            "max_cond": 10.0,
            "warnings": [],
        }
        with mock.patch.object(
            bor_dispatch, "solve_bor", return_value=solved
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                r"power_amplitude_inconsistent_count=1 must be zero",
            ):
                bor_dispatch.solve_monostatic_rcs_bor(
                    snapshot,
                    [1.0],
                    [0.0],
                    "VV",
                    geometry_units="meters",
                )

    def test_nonconductor_controls_are_rejected_instead_of_ignored(self):
        snapshot = _snapshot(
            [_segment("dielectric", 3, pos_mat=1)],
            dielectrics=[["1", "4", "0", "1", "0"]],
        )
        cases = (
            {"assembly": "streaming"},
            {"table_precision": "single"},
            {"stream_budget_gb": 0.25},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(
                    ValueError,
                    r"does not implement.*controls",
                ):
                    bor_dispatch.solve_monostatic_rcs_bor(
                        snapshot,
                        [1.0],
                        [0.0],
                        "VV",
                        geometry_units="meters",
                        **kwargs,
                    )

    def test_nonconductor_defaults_remain_valid_and_marked_not_applicable(self):
        snapshot = _snapshot(
            [_segment("dielectric", 3, pos_mat=1)],
            dielectrics=[["1", "4", "0", "1", "0"]],
        )
        amp = complex(math.sqrt(1.0 / (4.0 * math.pi)))
        solved = {
            "sigma_vv": [1.0],
            "sigma_hh": [1.0],
            "amp_vv": [amp],
            "amp_hh": [amp],
            "modes_used": 2,
            "mode_cap": 4,
            "mode_converged": True,
            "mode_quiet_count": 2,
            "mode_last_relative_increment": 0.0,
            "n_unknowns": 16,
            "linear_residual": 0.0,
            "condition_est_computed": True,
            "max_cond": 10.0,
            "warnings": [],
        }
        with mock.patch.object(
            bor_dispatch,
            "solve_bor_dielectric",
            return_value=solved,
        ):
            result = bor_dispatch.solve_monostatic_rcs_bor(
                snapshot,
                [1.0],
                [0.0],
                "VV",
                geometry_units="meters",
            )
        self.assertTrue(result["metadata"]["quality_gate"]["passed"])
        self.assertFalse(
            result["metadata"]["far_table_controls_applicable"]
        )

    def test_hpc_runner_forwards_its_stream_budget(self):
        backend_dir = Path(bor_dispatch.__file__).resolve().parent
        source = (
            backend_dir / "run_hpc_bor_monostatic.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "stream_budget_gb=STREAM_BUDGET_GB",
            source,
        )


class TestBorFormulationAndModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points = bor_solver.sphere_generatrix(0.03, 8)

    def test_unknown_formulation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"Unsupported BoR formulation"):
            bor_solver.solve_bor(
                self.points, 1.0e9, [40.0], formulation="not-a-formulation"
            )

    def test_non_efie_impedance_is_rejected(self):
        for formulation in ("cfie", "mfie"):
            with self.subTest(formulation=formulation):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{formulation.upper()} with nonzero surface impedance",
                ):
                    bor_solver.solve_bor(
                        self.points,
                        1.0e9,
                        [40.0],
                        formulation=formulation,
                        zs=100.0,
                        n_modes=0,
                    )

    def test_mode_cap_nonconvergence_aborts_before_dispatch_returns_field(self):
        snapshot = _snapshot([_segment("body", 2)])
        with self.assertRaisesRegex(
            RuntimeError,
            r"mode truncation did not reach.*No RCS/amplitude result",
        ):
            bor_dispatch.solve_monostatic_rcs_bor(
                snapshot,
                [1.0],
                [40.0],
                "VV",
                geometry_units="meters",
                n_modes=0,
                mode_tol=1.0e-30,
                workers=1,
            )

    def test_large_or_nonfinite_linear_residual_aborts_before_field(self):
        def assemble(_mode):
            return np.eye(1, dtype=np.complex128), np.ones(1, dtype=bool)

        def rhs(_mode, _theta, _polarization):
            return np.ones(1, dtype=np.complex128)

        def farfield(_mode, solution, _theta, _polarization):
            return complex(solution[0])

        for bad_solution, message in (
            (np.zeros((1, 1), dtype=np.complex128), "exceeds the release limit"),
            (
                np.full((1, 1), complex(math.nan, 0.0), dtype=np.complex128),
                "non-finite linear-system solution",
            ),
        ):
            with self.subTest(message=message):
                with mock.patch.object(
                    bor_solver.np.linalg,
                    "solve",
                    return_value=bad_solution,
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        bor_solver._mode_sweep(
                            1,
                            [40.0],
                            ("VV",),
                            0,
                            1.0e-6,
                            assemble,
                            rhs,
                            farfield,
                        )

    def test_condition_monitor_is_computed_and_fails_closed(self):
        def rhs(_mode, _theta, _polarization):
            return np.ones(2, dtype=np.complex128)

        def farfield(_mode, solution, _theta, _polarization):
            return complex(solution[0])

        good, _modes, stats = bor_solver._mode_sweep(
            2,
            [40.0],
            ("VV",),
            0,
            1.0e-6,
            lambda _mode: (
                np.asarray([[2.0, 0.0], [0.0, 1.0]],
                           dtype=np.complex128),
                np.ones(2, dtype=bool),
            ),
            rhs,
            farfield,
            monitor_cond=True,
        )
        self.assertTrue(np.all(np.isfinite(good)))
        self.assertTrue(stats["condition_est_computed"])
        self.assertLessEqual(stats["max_cond"], 2.01)

        with self.assertRaisesRegex(
            RuntimeError, r"condition .*exceeds the release limit"
        ):
            bor_solver._mode_sweep(
                2,
                [40.0],
                ("VV",),
                0,
                1.0e-6,
                lambda _mode: (
                    np.asarray(
                        [[1.0, 0.0], [0.0, 1.0e-13]],
                        dtype=np.complex128,
                    ),
                    np.ones(2, dtype=bool),
                ),
                rhs,
                farfield,
                monitor_cond=True,
            )


class TestDirectGeneratrixSafety(unittest.TestCase):
    def test_direct_pmchwt_requires_closed_or_individually_valid_surfaces(self):
        open_shell = np.asarray([[0.1, 0.1], [0.1, -0.1]])
        with self.assertRaisesRegex(ValueError, r"closed BoR"):
            bor_solver.solve_bor_dielectric(
                open_shell, 1.0e9, [40.0], 2.0, n_modes=0
            )
        with self.assertRaisesRegex(ValueError, r"closed BoR"):
            bor_solver.solve_bor_coated_pec(
                bor_solver.sphere_generatrix(0.12, 8),
                open_shell,
                1.0e9,
                [40.0],
                2.0,
                n_modes=0,
            )

    def test_direct_material_and_ibc_inputs_are_passive_and_nonsingular(self):
        points = bor_solver.sphere_generatrix(0.03, 8)
        for eps, message in (
            (0.0, "ENZ"),
            (2.0 + 0.1j, "passive media"),
            (complex(math.nan, 0.0), "finite"),
        ):
            with self.subTest(eps=eps):
                with self.assertRaisesRegex(ValueError, message):
                    bor_solver.solve_bor_dielectric(
                        points, 1.0e9, [40.0], eps, n_modes=0
                    )
        with self.assertRaisesRegex(ValueError, r"negative resistance"):
            bor_solver.solve_bor(
                points,
                1.0e9,
                [40.0],
                formulation="efie",
                zs=-1.0,
                n_modes=0,
            )
        with self.assertRaisesRegex(
                RuntimeError, r"Closed-body lossless/reactive IBC"):
            bor_solver.solve_bor(
                points,
                1.0e9,
                [40.0],
                formulation="efie",
                zs=50.0j,
                n_modes=0,
            )

    def test_partial_coating_rejects_effectively_reactive_bare_ibc(self):
        radius = 0.03
        thickness = 0.005
        theta_edge = 0.5 * math.pi

        def arc(rad, start, stop, count):
            theta = np.linspace(start, stop, count + 1)
            return np.column_stack([
                rad * np.sin(theta),
                rad * np.cos(theta),
            ])

        covered = arc(radius, 0.0, theta_edge, 4)
        bare = arc(radius, theta_edge, math.pi, 4)
        outer = arc(radius + thickness, 0.0, theta_edge, 4)
        edge = np.linspace(outer[-1], covered[-1], 3)
        interface = np.vstack([outer, edge[1:]])

        with self.assertRaisesRegex(
            RuntimeError,
            r"partial-coated body with lossless/reactive bare IBC",
        ):
            bor_solver.solve_bor_partial_coating(
                interface,
                covered,
                [bare],
                1.0e9,
                [40.0],
                2.0 - 0.1j,
                bare_zs=[50.0j],
                n_modes=0,
            )

    def test_malformed_and_nonfinite_coordinate_arrays_are_rejected(self):
        cases = [
            ([[0.0, 1.0, 2.0]], r"finite \(N, 2\)"),
            ([[0.0, 1.0], [0.1, math.nan]], r"all be finite"),
            ([[0.0 + 1.0j, 1.0], [0.1, 0.0]], r"must be real"),
        ]
        for points, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    bor_solver.solve_bor(points, 1.0e9, [0.0])

    def test_negative_radius_and_zero_length_elements_are_rejected(self):
        cases = [
            (
                [[0.0, 1.0], [-0.1, 0.0], [0.0, -1.0]],
                r"rho coordinates must be >= 0",
            ),
            (
                [[0.0, 1.0], [0.1, 0.0], [0.1, 0.0], [0.0, -1.0]],
                r"zero or near-zero length",
            ),
        ]
        for points, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    bor_solver.solve_bor(points, 1.0e9, [0.0])

    def test_reversed_closed_body_is_rejected(self):
        points = bor_solver.sphere_generatrix(0.03, 8)[::-1]
        with self.assertRaisesRegex(ValueError, r"from its \+z axis.*-z axis"):
            bor_solver.solve_bor(points, 1.0e9, [0.0])

    def test_nonadjacent_self_intersection_is_rejected(self):
        bow_tie = np.asarray(
            [
                [0.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [0.0, -1.0],
            ]
        )
        with self.assertRaisesRegex(ValueError, r"self-intersection/overlap"):
            bor_solver.solve_bor(bow_tie, 1.0e9, [0.0])

    def test_cfie_requires_axis_closed_body_but_efie_keeps_open_shell(self):
        open_shell = np.asarray([[0.1, 0.1], [0.1, -0.1]])
        validated = bor_solver._validate_solve_bor_generatrix(
            open_shell, "efie"
        )
        np.testing.assert_array_equal(validated, open_shell)
        with self.assertRaisesRegex(ValueError, r"CFIE requires a closed BoR"):
            bor_solver.solve_bor(
                open_shell, 1.0e9, [40.0], formulation="cfie"
            )

    @staticmethod
    def _reentrant_profile(gap):
        return np.asarray(
            [
                [0.0, 2.0],
                [1.0, 2.0],
                [1.0, 0.0],
                [0.1, 0.0],
                [0.1, 2.0 - gap],
                [0.0, 2.0 - gap],
            ],
            dtype=float,
        )

    def test_true_far_gap_includes_distant_fold_walls(self):
        solver = bor_solver.BorPecSolver(
            self._reentrant_profile(0.01), 1.0e6
        )
        self.assertAlmostEqual(solver._far_gap(), 0.01, places=12)
        self.assertGreater(
            abs(solver._far_gap_pair[1] - solver._far_gap_pair[0]), 2
        )

    def test_close_fold_fails_closed_before_table_or_streaming_assembly(self):
        points = self._reentrant_profile(1.0e-4)
        for assembly in ("tables", "streaming"):
            with self.subTest(assembly=assembly):
                with self.assertRaisesRegex(
                    ValueError,
                    r"far-quadrature preflight.*requires .*samples.*cap",
                ):
                    bor_solver.solve_bor(
                        points,
                        1.0e6,
                        [0.0],
                        n_modes=0,
                        assembly=assembly,
                    )


class TestFarKernelResolutionCap(unittest.TestCase):
    def test_fft_resolution_requirement_cannot_be_silently_capped(self):
        with self.assertRaisesRegex(
            ValueError, r"requires .* samples but the safety cap is 128"
        ):
            bor_kernels.n_xi_for_pairs(
                1.0,
                rho_max=1.0,
                m_max=1,
                d_min=1.0e-4,
                cap=128,
            )


class TestCausalNegativeIndexBranch(unittest.TestCase):
    def test_passive_double_negative_branch_decays_and_has_forward_impedance(self):
        eps_r = -2.0 - 0.1j
        mu_r = -1.5 - 0.1j
        index, impedance = bor_solver._causal_medium(eps_r, mu_r)
        reference_index = mie_sphere._causal_index(eps_r, mu_r)

        self.assertLess(index.real, 0.0)
        self.assertLess(index.imag, 0.0)
        self.assertGreater(impedance.real, 0.0)
        self.assertAlmostEqual(index.real, reference_index.real, places=14)
        self.assertAlmostEqual(index.imag, reference_index.imag, places=14)

    def test_lossless_double_negative_branch_uses_negative_index(self):
        index, impedance = bor_solver._causal_medium(-2.0, -1.5)
        self.assertLess(index.real, 0.0)
        self.assertEqual(index.imag, 0.0)
        self.assertGreater(impedance.real, 0.0)


if __name__ == "__main__":
    unittest.main()
