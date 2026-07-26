"""Focused regressions for solver-tab physical units and run identity."""

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "ghost-matplotlib-test-cache"),
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import solver_tab


def _row(freq, incidence, observation, linear=1.0):
    return {
        "frequency_ghz": float(freq),
        "theta_inc_deg": float(incidence),
        "theta_scat_deg": float(observation),
        "rcs_linear": float(linear),
    }


class TestSolverTabPhysicalDisplay(unittest.TestCase):
    def test_2d_uses_absolute_dbke_while_bor_uses_dbsm(self):
        row = _row(3.0, 0.0, 0.0, linear=0.5)
        result_2d = {
            "solver": "2d_bie_mom_rcs",
            "scattering_mode": "monostatic",
        }
        expected_dbke = 10.0 * math.log10(
            (2.0 * math.pi * 3.0e9 / 299_792_458.0) * 0.5
        )
        self.assertAlmostEqual(
            solver_tab._display_db_value(result_2d, row),
            expected_dbke,
            places=12,
        )

        result_bor = {
            "solver": "bor_mom_rcs",
            "scattering_mode": "monostatic",
            "rcs_linear_quantity": "sigma_3d",
        }
        self.assertAlmostEqual(
            solver_tab._display_db_value(result_bor, row),
            10.0 * math.log10(0.5),
            places=12,
        )

    def test_bistatic_plot_groups_do_not_merge_incidence_sweeps(self):
        result = {
            "solver": "2d_bie_mom_rcs",
            "scattering_mode": "bistatic",
            "samples": [
                _row(1.0, 10.0, 0.0),
                _row(1.0, 10.0, 90.0),
                _row(1.0, 20.0, 0.0),
                _row(1.0, 20.0, 90.0),
            ],
        }
        groups = solver_tab._result_plot_groups(result)
        self.assertEqual(set(groups), {(1.0, 10.0), (1.0, 20.0)})
        self.assertEqual([len(rows) for rows in groups.values()], [2, 2])

        counts = solver_tab._result_sample_counts(result)
        self.assertEqual(counts["frequency_count"], 1)
        self.assertEqual(counts["incidence_count"], 2)
        self.assertEqual(counts["observation_count"], 2)
        self.assertIn("2 incidence angle(s)", solver_tab._result_summary(result))
        history = solver_tab._result_history(
            result,
            units="meters",
            polarization="HH",
        )
        self.assertIn("result_kind=2d_bistatic", history)
        self.assertIn("inc_count=2", history)
        self.assertIn("obs_count=2", history)

    def test_bor_counts_and_history_identify_bor(self):
        result = {
            "solver": "bor_mom_rcs",
            "scattering_mode": "monostatic",
            "samples": [
                _row(1.0, 0.0, 0.0),
                _row(1.0, 90.0, 90.0),
                _row(2.0, 0.0, 0.0),
                _row(2.0, 90.0, 90.0),
            ],
        }
        self.assertEqual(
            solver_tab._result_sample_counts(result)["aspect_count"], 2
        )
        self.assertIn("BoR monostatic RCS", solver_tab._result_summary(result))
        history = solver_tab._result_history(
            result,
            units="inches",
            polarization="VV",
            manual_export=True,
        )
        self.assertIn("result_kind=bor", history)
        self.assertIn("aspect_count=2", history)
        self.assertIn("manual_export=1", history)

    def test_absent_or_malformed_quality_gate_never_displays_pass(self):
        for metadata in (
            {},
            {"quality_gate": {}},
            {"quality_gate": {"passed": "true"}},
            {"quality_gate": {"passed": False}},
        ):
            with self.subTest(metadata=metadata):
                suffix = solver_tab._quality_gate_suffix(metadata)
                self.assertNotIn("PASS", suffix)
        self.assertEqual(
            solver_tab._quality_gate_suffix(
                {"quality_gate": {"passed": True}}
            ),
            " Quality gate: PASS.",
        )

    def test_gui_does_not_offer_dead_gmres_or_claim_fmm_fallback(self):
        source = (
            Path(solver_tab.__file__).resolve().read_text(encoding="utf-8")
        )
        self.assertNotIn('userData="gmres"', source)
        self.assertNotIn("Iterative (GMRES)", source)
        self.assertIn("Unsupported FMM requests reject the solve", source)
        self.assertIn('"Incidence (deg)"', source)
        self.assertIn('"Observation (deg)"', source)

    def test_bistatic_mesh_check_uses_two_bistatic_conditioned_solves(self):
        worker = solver_tab._SolveWorker(
            snapshot={"mesh": "base"},
            source_path="shape.geo",
            base_dir="",
            frequencies=[3.0],
            elevations=[10.0],
            pol="TM",
            units="meters",
            quality_thresholds={"max_relative_residual": 1.0e-6},
            strict_quality_gate=True,
            mesh_convergence=True,
            mesh_fine_factor=2.0,
            mesh_rms_limit_db=1.0,
            mesh_max_abs_limit_db=2.0,
            strict_mesh_convergence=True,
            scattering_mode="bistatic",
            observation_angles=[20.0, 30.0],
        )
        fake_result = {
            "samples": [
                {
                    **_row(3.0, 10.0, 20.0),
                    "rcs_db": 0.0,
                },
                {
                    **_row(3.0, 10.0, 30.0),
                    "rcs_db": 0.0,
                },
            ],
            "metadata": {},
        }
        finished = []
        errors = []
        worker.finished.connect(lambda result, path: finished.append(result))
        worker.error.connect(errors.append)

        with (
            mock.patch.object(
                solver_tab,
                "solve_bistatic_rcs_2d",
                side_effect=[fake_result, fake_result],
            ) as bistatic,
            mock.patch.object(
                solver_tab, "solve_monostatic_rcs_2d"
            ) as monostatic,
            mock.patch.object(
                solver_tab,
                "scale_snapshot_panel_density",
                return_value={"mesh": "fine"},
            ),
        ):
            worker.run()

        self.assertEqual(errors, [])
        self.assertEqual(len(finished), 1)
        self.assertEqual(bistatic.call_count, 2)
        monostatic.assert_not_called()
        self.assertEqual(
            [call.kwargs["geometry_snapshot"] for call in bistatic.call_args_list],
            [{"mesh": "base"}, {"mesh": "fine"}],
        )
        for call in bistatic.call_args_list:
            self.assertEqual(call.kwargs["incidence_angles_deg"], [10.0])
            self.assertEqual(
                call.kwargs["observation_angles_deg"], [20.0, 30.0]
            )
            self.assertIs(call.kwargs["strict_quality_gate"], True)
            self.assertIs(call.kwargs["compute_condition_number"], True)

    def test_nonmesh_bistatic_is_conditioned_and_never_falls_through(self):
        worker = solver_tab._SolveWorker(
            snapshot={"mesh": "base"},
            source_path="shape.geo",
            base_dir="",
            frequencies=[3.0],
            elevations=[10.0],
            pol="TE",
            units="meters",
            quality_thresholds={},
            strict_quality_gate=True,
            mesh_convergence=False,
            mesh_fine_factor=2.0,
            mesh_rms_limit_db=1.0,
            mesh_max_abs_limit_db=2.0,
            strict_mesh_convergence=False,
            scattering_mode="bistatic",
            observation_angles=[20.0],
        )
        with (
            mock.patch.object(
                solver_tab, "solve_bistatic_rcs_2d", return_value={}
            ) as bistatic,
            mock.patch.object(
                solver_tab, "solve_monostatic_rcs_2d"
            ) as monostatic,
        ):
            worker._run_2d(worker.snapshot, worker._on_progress)
        monostatic.assert_not_called()
        self.assertIs(
            bistatic.call_args.kwargs["compute_condition_number"], True
        )

        worker.observation_angles = []
        with (
            mock.patch.object(solver_tab, "solve_bistatic_rcs_2d") as bistatic,
            mock.patch.object(
                solver_tab, "solve_monostatic_rcs_2d"
            ) as monostatic,
        ):
            with self.assertRaisesRegex(ValueError, "observation angle"):
                worker._run_2d(worker.snapshot, worker._on_progress)
        bistatic.assert_not_called()
        monostatic.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
