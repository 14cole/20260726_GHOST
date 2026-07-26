"""Focused regressions for fail-closed 2-D solve quality gates."""

import ast
import math
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rcs_solver as rcs


def _pec_circle_snapshot(radius: float = 0.03, count: int = 8):
    points = [
        (
            radius * math.cos(-2.0 * math.pi * idx / count),
            radius * math.sin(-2.0 * math.pi * idx / count),
        )
        for idx in range(count + 1)
    ]
    return {
        "title": "quality-gate PEC circle",
        "segments": [
            {
                "name": "pec_circle",
                "seg_type": "2",
                "properties": ["2", "1", "0", "0", "0"],
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
        "dielectrics": [],
    }


class TestQualityResidualAggregation(unittest.TestCase):
    def test_summary_preserves_finite_max_and_counts_nonfinite_values(self):
        maximum, mean, nonfinite = rcs._summarize_residuals(
            [3.0e-5, float("nan"), 2.0e-5, float("inf")]
        )
        self.assertEqual(maximum, 3.0e-5)
        self.assertEqual(mean, 2.5e-5)
        self.assertEqual(nonfinite, 2)

        gate = rcs.evaluate_quality_gate(
            {
                "residual_norm_max": maximum,
                "residual_nonfinite_count": nonfinite,
                "warnings": [],
            }
        )
        self.assertFalse(gate["passed"])
        self.assertIn("residual_nonfinite_count=2", gate["reason"])

    def test_missing_or_nonfinite_residual_fails_closed(self):
        for metadata in (
            {"warnings": []},
            {"residual_norm_max": float("nan"), "warnings": []},
            {"residual_norm_max": float("inf"), "warnings": []},
        ):
            with self.subTest(metadata=metadata):
                self.assertFalse(rcs.evaluate_quality_gate(metadata)["passed"])

    def test_release_default_rejects_residual_above_one_part_per_million(self):
        gate = rcs.evaluate_quality_gate(
            {
                "residual_norm_max": 9.0e-6,
                "residual_nonfinite_count": 0,
                "warnings": [],
            }
        )
        self.assertFalse(gate["passed"])
        self.assertEqual(
            gate["thresholds"]["residual_norm_max"], 1.0e-6
        )


class TestBistaticQualityGate(unittest.TestCase):
    def test_bistatic_reports_aggregate_gate_and_honors_thresholds(self):
        result = rcs.solve_bistatic_rcs_2d(
            _pec_circle_snapshot(),
            [1.0],
            [17.0],
            [17.0, 53.0],
            "TM",
            geometry_units="meters",
            quality_thresholds={
                "residual_norm_max": 1.0e-6,
                "warnings_max": 10,
            },
            strict_quality_gate=True,
        )
        metadata = result["metadata"]
        sample_max = max(sample["linear_residual"] for sample in result["samples"])
        self.assertAlmostEqual(metadata["residual_norm_max"], sample_max)
        self.assertEqual(metadata["residual_nonfinite_count"], 0)
        self.assertEqual(metadata["warning_count"], len(metadata["warnings"]))
        self.assertTrue(metadata["quality_gate"]["passed"])
        self.assertEqual(
            metadata["quality_gate"]["thresholds"]["residual_norm_max"], 1.0e-6
        )

    def test_bistatic_computes_condition_when_requested(self):
        result = rcs.solve_bistatic_rcs_2d(
            _pec_circle_snapshot(),
            [1.0],
            [17.0],
            [17.0, 53.0],
            "TM",
            geometry_units="meters",
            compute_condition_number=True,
        )
        metadata = result["metadata"]
        self.assertTrue(metadata["condition_est_computed"])
        self.assertTrue(math.isfinite(metadata["condition_est_max"]))
        self.assertEqual(metadata["solver_method"], "dense_lu")

    def test_bistatic_strict_mode_rejects_nonfinite_residual_count(self):
        with mock.patch.object(
            rcs,
            "_summarize_residuals",
            return_value=(1.0e-12, 1.0e-12, 1),
        ):
            with self.assertRaisesRegex(
                ValueError, r"Quality gate failed:.*residual_nonfinite_count=1"
            ):
                rcs.solve_bistatic_rcs_2d(
                    _pec_circle_snapshot(),
                    [1.0],
                    [0.0],
                    [0.0],
                    "TM",
                    geometry_units="meters",
                    strict_quality_gate=True,
                )

    def test_unimplemented_or_unknown_solver_methods_fail_closed(self):
        for method, message in (
            ("gmres", "not implemented"),
            ("made-up", "Unsupported 2-D solver_method"),
        ):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, message):
                    rcs.solve_monostatic_rcs_2d(
                        _pec_circle_snapshot(),
                        [1.0],
                        [0.0],
                        "TM",
                        geometry_units="meters",
                        solver_method=method,
                    )

        with self.assertRaisesRegex(ValueError, "not implemented by the bistatic"):
            rcs.solve_bistatic_rcs_2d(
                _pec_circle_snapshot(),
                [1.0],
                [0.0],
                [0.0],
                "TM",
                geometry_units="meters",
                solver_method="fmm",
            )


class TestProductionRunnerStrictness(unittest.TestCase):
    def test_production_runners_request_strict_quality_and_condition_number(self):
        backend_dir = Path(rcs.__file__).resolve().parent
        project_dir = backend_dir.parent
        paths = (
            backend_dir / "run_local_monostatic.py",
            backend_dir / "run_hpc_monostatic.py",
            project_dir / "1a_solve_2d_local" / "run.py",
            project_dir / "3b_add_wing" / "run.py",
        )
        for path in paths:
            filename = str(path.relative_to(project_dir))
            with self.subTest(filename=filename):
                tree = ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=filename,
                )
                solver_calls = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "solve_monostatic_rcs_2d"
                ]
                self.assertTrue(solver_calls)
                for call in solver_calls:
                    strict = next(
                        (
                            keyword.value
                            for keyword in call.keywords
                            if keyword.arg == "strict_quality_gate"
                        ),
                        None,
                    )
                    self.assertIsInstance(strict, ast.Constant)
                    self.assertIs(strict.value, True)
                    condition = next(
                        (
                            keyword.value
                            for keyword in call.keywords
                            if keyword.arg == "compute_condition_number"
                        ),
                        None,
                    )
                    self.assertIsInstance(condition, ast.Constant)
                    self.assertIs(condition.value, True)

    def test_gui_defaults_to_strict_and_bistatic_forwards_gate(self):
        backend_dir = Path(rcs.__file__).resolve().parent
        source = (backend_dir / "solver_tab.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "self.chk_strict_quality.setChecked(True)", source
        )
        self.assertNotIn("strict_quality_gate=False", source)

        tree = ast.parse(source, filename="solver_tab.py")
        bistatic_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "solve_bistatic_rcs_2d"
        ]
        self.assertTrue(bistatic_calls)
        for call in bistatic_calls:
            keywords = {keyword.arg for keyword in call.keywords}
            self.assertIn("quality_thresholds", keywords)
            self.assertIn("strict_quality_gate", keywords)
            self.assertIn("compute_condition_number", keywords)


if __name__ == "__main__":
    unittest.main(verbosity=2)
