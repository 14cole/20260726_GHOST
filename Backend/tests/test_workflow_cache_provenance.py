"""Focused fail-closed checks for the local workflow cache manifests."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def _load_runner(relative_path: str, module_name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


R1A = _load_runner("1a_solve_2d_local/run.py", "_workflow_run_1a")
R1BS = _load_runner("1b_solve_2d_hpc/submit.py", "_workflow_submit_1b")
R1B = _load_runner("1b_solve_2d_hpc/collect.py", "_workflow_collect_1b")
R1C = _load_runner("1c_build_deltas/run.py", "_workflow_run_1c")
R2A = _load_runner("2a_solve_body_local/run.py", "_workflow_run_2a")
R2BS = _load_runner("2b_solve_body_hpc/submit.py", "_workflow_submit_2b")
R3A = _load_runner("3a_doors/run.py", "_workflow_run_3a")
R3B = _load_runner("3b_add_wing/run.py", "_workflow_run_3b")
R3C = _load_runner("3c_add_cavity/run.py", "_workflow_run_3c")
R4 = _load_runner("4_combine/run.py", "_workflow_run_4")
R5 = _load_runner("5_rank_designs/run.py", "_workflow_run_5")

import workflow_provenance as PROVENANCE
import run_hpc_bor_monostatic as HPC_BOR
import run_hpc_monostatic as HPC_2D


class WorkflowCacheProvenanceTests(unittest.TestCase):
    def test_hpc_drivers_generate_frozen_manifest_and_slurm_worker(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            frd_2d = root / "two_d" / "FRD"
            opn_2d = root / "two_d" / "OPN"
            frd_bor = root / "bor" / "FRD"
            opn_bor = root / "bor" / "OPN"
            for path in (frd_2d, opn_2d, frd_bor, opn_bor):
                path.mkdir(parents=True)
            shutil.copy2(
                ROOT
                / "1a_solve_2d_local"
                / "geometries"
                / "FRD"
                / "SEAL-00-01_0.006gap.geo",
                frd_2d / "coupon.geo",
            )
            shutil.copy2(
                ROOT / "2b_solve_body_hpc" / "body.geo",
                frd_bor / "body.geo",
            )

            with mock.patch.multiple(
                HPC_2D,
                FRD_DIR=str(frd_2d),
                OPN_DIR=str(opn_2d),
                OUTPUT_DIR=str(root / "runs_2d"),
                FREQUENCIES_GHZ=[3.0],
                AZIMUTHS_DEG=[0.0, 90.0],
                POLARIZATIONS=["TM", "TE"],
                GEOMETRY_UNITS="meters",
                N_NODES=1,
                N_JOBS=1,
                SUBMIT=False,
            ):
                HPC_2D.submit()
            run_2d = sorted((root / "runs_2d").glob("run_*"))[-1]
            manifest_2d = json.loads(
                (run_2d / "manifest.json").read_text(encoding="utf-8")
            )
            PROVENANCE.sha256_file(run_2d / "driver_configured.py")
            from hpc_common import require_hpc_run_provenance
            require_hpc_run_provenance(
                manifest_2d, "ghost.hpc.2d-run.v1"
            )
            slurm_2d = (run_2d / "submit_job0.slurm").read_text(
                encoding="utf-8"
            )
            self.assertIn("set -euo pipefail", slurm_2d)
            self.assertIn("--worker", slurm_2d)

            with mock.patch.multiple(
                HPC_BOR,
                FRD_DIR=str(frd_bor),
                OPN_DIR=str(opn_bor),
                OUTPUT_DIR=str(root / "runs_bor"),
                FREQUENCIES_GHZ=[3.0],
                ASPECTS_DEG=[0.0, 90.0, 180.0],
                POLARIZATIONS=["VV", "HH"],
                GEOMETRY_UNITS="meters",
                N_NODES=1,
                N_JOBS=1,
                AZEL_ENABLE=False,
                SUBMIT=False,
            ):
                HPC_BOR.submit()
            run_bor = sorted((root / "runs_bor").glob("run_*"))[-1]
            manifest_bor = json.loads(
                (run_bor / "manifest.json").read_text(encoding="utf-8")
            )
            require_hpc_run_provenance(
                manifest_bor, "ghost.hpc.bor-run.v1"
            )
            slurm_bor = (run_bor / "submit_job0.slurm").read_text(
                encoding="utf-8"
            )
            self.assertIn("set -euo pipefail", slurm_bor)
            self.assertIn("--worker", slurm_bor)

    def test_numbered_hpc_wrappers_generate_configured_production_runs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            step_1b = root / "1b"
            frd = step_1b / "geometries" / "FRD"
            opn = step_1b / "geometries" / "OPN"
            frd.mkdir(parents=True)
            opn.mkdir(parents=True)
            shutil.copy2(
                ROOT
                / "1a_solve_2d_local"
                / "geometries"
                / "FRD"
                / "SEAL-00-01_0.006gap.geo",
                frd / "coupon.geo",
            )
            shutil.copy2(
                ROOT
                / "1a_solve_2d_local"
                / "geometries"
                / "OPN"
                / "SEAL-00-01_0.006gap.geo",
                opn / "coupon.geo",
            )
            old_argv = sys.argv
            try:
                with mock.patch.multiple(
                    R1BS,
                    HERE=str(step_1b),
                    FREQUENCIES_GHZ=[3.0],
                    ANGLES_DEG=[0.0, 90.0],
                    POLARIZATIONS=["TM", "TE"],
                    N_NODES=1,
                    N_JOBS=1,
                    SUBMIT=False,
                ):
                    R1BS.main()
            finally:
                sys.argv = old_argv
            run_1b = Path(
                (step_1b / "submitted.txt").read_text(
                    encoding="utf-8"
                ).strip()
            )
            manifest_1b = json.loads(
                (run_1b / "manifest.json").read_text(encoding="utf-8")
            )
            from hpc_common import require_hpc_run_provenance
            require_hpc_run_provenance(
                manifest_1b, "ghost.hpc.2d-run.v1"
            )

            step_2b = root / "2b"
            step_2b.mkdir()
            shutil.copy2(
                ROOT / "2b_solve_body_hpc" / "body.geo",
                step_2b / "body.geo",
            )
            old_argv = sys.argv
            try:
                with mock.patch.multiple(
                    R2BS,
                    HERE=str(step_2b),
                    FREQUENCIES_GHZ=[3.0],
                    ASPECT_STEP_DEG=None,
                    POLARIZATIONS=["VV", "HH"],
                    N_NODES=1,
                    N_JOBS=1,
                    WORKERS_PER_UNIT=1,
                    SUBMIT=False,
                ):
                    R2BS.main()
            finally:
                sys.argv = old_argv
            run_2b = Path(
                (step_2b / "submitted.txt").read_text(
                    encoding="utf-8"
                ).strip()
            )
            manifest_2b = json.loads(
                (run_2b / "manifest.json").read_text(encoding="utf-8")
            )
            require_hpc_run_provenance(
                manifest_2b, "ghost.hpc.bor-run.v1"
            )

    def test_numbered_hpc_submitters_require_complete_physical_channels(self):
        old_1b_pols = R1BS.POLARIZATIONS
        old_1b_freqs = R1BS.FREQUENCIES_GHZ
        old_2b_pols = R2BS.POLARIZATIONS
        try:
            R1BS.POLARIZATIONS = ["TM"]
            with self.assertRaisesRegex(SystemExit, r"exactly TM and TE"):
                R1BS._validate_config()

            R1BS.POLARIZATIONS = ["TM", "TE"]
            R1BS.FREQUENCIES_GHZ = [1.0001, 1.0002]
            with self.assertRaisesRegex(
                SystemExit, r"output-name precision"
            ):
                R1BS._validate_config()

            R2BS.POLARIZATIONS = ["VV"]
            with self.assertRaisesRegex(SystemExit, r"exactly VV and HH"):
                R2BS._validate_config()
        finally:
            R1BS.POLARIZATIONS = old_1b_pols
            R1BS.FREQUENCIES_GHZ = old_1b_freqs
            R2BS.POLARIZATIONS = old_2b_pols

    def test_1a_rejects_legacy_cache_then_accepts_exact_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            result = out / "one.grim"
            result.write_bytes(b"solver-result")
            old_force = R1A.FORCE
            R1A.FORCE = False
            try:
                with self.assertRaises(SystemExit):
                    R1A._cache_is_reusable(td, ["one.grim"], "run-hash")

                manifest = {
                    "schema": R1A.CACHE_SCHEMA,
                    "status": "complete",
                    "run_sha256": "run-hash",
                    "expected_outputs": ["one.grim"],
                    "output_sha256": {
                        "one.grim": R1A._sha256_file(result),
                    },
                }
                (out / "cache_manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8")
                self.assertTrue(
                    R1A._cache_is_reusable(td, ["one.grim"], "run-hash"))

                result.write_bytes(b"tampered")
                with self.assertRaises(SystemExit):
                    R1A._cache_is_reusable(td, ["one.grim"], "run-hash")
            finally:
                R1A.FORCE = old_force

    def test_delta_builder_refuses_but_does_not_delete_stale_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            stale = Path(td) / "old-study.grim"
            stale.write_bytes(b"user-artifact")
            with self.assertRaises(SystemExit):
                R1C._refuse_unexpected_grims(
                    td, {"current-study.grim"}, "test/")
            self.assertEqual(stale.read_bytes(), b"user-artifact")

    def test_delta_builder_accepts_only_committed_2d_solver_role(self):
        with tempfile.TemporaryDirectory() as td:
            result = Path(td) / "unit.grim"
            result.write_bytes(b"field")
            PROVENANCE.write_artifact_manifest(
                td,
                "ghost.workflow.body-local-cache.v1",
                ["unit.grim"],
            )
            with self.assertRaisesRegex(
                SystemExit, r"wrong artifact role/schema"
            ):
                R1C._verified_input_state(td, [str(result)])

    def test_body_cache_requires_both_exact_output_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            body = out / "body.grim"
            profile = out / "body_profile.csv"
            body.write_bytes(b"body")
            profile.write_bytes(b"profile")
            manifest = {
                "schema": R2A.CACHE_SCHEMA,
                "status": "complete",
                "run_sha256": "body-run",
                "expected_outputs": ["body.grim", "body_profile.csv"],
                "output_sha256": {
                    "body.grim": R2A._sha256_file(body),
                    "body_profile.csv": R2A._sha256_file(profile),
                },
            }
            (out / "cache_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            old_force = R2A.FORCE
            R2A.FORCE = False
            try:
                self.assertTrue(R2A._cache_is_reusable(td, "body-run"))
                profile.write_bytes(b"edited-profile")
                with self.assertRaises(SystemExit):
                    R2A._cache_is_reusable(td, "body-run")
            finally:
                R2A.FORCE = old_force

    def test_wing_stale_output_check_is_non_destructive(self):
        with tempfile.TemporaryDirectory() as td:
            expected = Path(td) / "wing_TM.grim"
            stale = Path(td) / "wing_old.grim"
            expected.write_bytes(b"expected")
            stale.write_bytes(b"stale")
            with self.assertRaises(SystemExit):
                R3B._refuse_unexpected_grims(
                    td, {"wing_TM.grim"}, "test/", pattern="wing_*.grim")
            self.assertTrue(expected.exists())
            self.assertTrue(stale.exists())

    def test_long_run_signature_guards_fail_closed(self):
        before = {"input_sha256": {"one": "a"}, "source": "stable"}
        after = {"input_sha256": {"one": "b"}, "source": "stable"}
        with self.assertRaises(SystemExit):
            R1A._require_unchanged_signature(before, after, "input")
        with self.assertRaises(SystemExit):
            R1C._require_unchanged_signature(before, after)
        with self.assertRaises(SystemExit):
            R2A._require_unchanged_signature(before, after)

    def test_door_and_cavity_stale_checks_preserve_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            expected = Path(td) / "current.grim"
            stale = Path(td) / "old.grim"
            expected.write_bytes(b"current")
            stale.write_bytes(b"old")
            with self.assertRaises(SystemExit):
                R3A._refuse_unexpected_grims(td, ["current.grim"])
            with self.assertRaises(SystemExit):
                R3C._refuse_unexpected_grims(td, ["current.grim"])
            self.assertEqual(stale.read_bytes(), b"old")

    def test_step4_discovers_only_from_verified_exact_manifests(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "Output"
            output.mkdir()
            component = output / "door.grim"
            component.write_bytes(b"component")
            PROVENANCE.write_artifact_manifest(
                str(output),
                "ghost.workflow.doors-output-provenance.v1",
                ["door.grim"],
                {"run_sha256": "a" * 64},
                manifest_name="provenance_manifest.json",
            )
            old_dirs = R4.COMPONENT_DIRS
            old_only = R4.ONLY
            old_skip = R4.SKIP
            try:
                R4.COMPONENT_DIRS = [str(output)]
                R4.ONLY = []
                R4.SKIP = []
                inventories = R4._verified_component_inventories()
                found = R4.discover(inventories)
                self.assertEqual(
                    [(name, Path(path).name) for name, path, _folder in found],
                    [("door", "door.grim")],
                )

                (output / "removed_design.grim").write_bytes(b"stale")
                with self.assertRaises(SystemExit):
                    R4._verified_component_inventories()
            finally:
                R4.COMPONENT_DIRS = old_dirs
                R4.ONLY = old_only
                R4.SKIP = old_skip

    def test_step4_and_step5_provenance_payloads_bind_committed_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            body_dir = root / "Body"
            component_dir = root / "Output"
            body_dir.mkdir()
            component_dir.mkdir()
            (body_dir / "body.grim").write_bytes(b"body")
            (body_dir / "body_profile.csv").write_bytes(b"profile")
            PROVENANCE.write_artifact_manifest(
                str(body_dir),
                "ghost.workflow.body-local-cache.v1",
                ["body.grim", "body_profile.csv"],
            )
            component = component_dir / "door.grim"
            component.write_bytes(b"door")
            PROVENANCE.write_artifact_manifest(
                str(component_dir),
                "ghost.workflow.doors-output-provenance.v1",
                ["door.grim"],
                manifest_name="provenance_manifest.json",
            )
            component_manifest = json.loads(
                (component_dir / "provenance_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            combined = R4._combined_provenance_payload(
                str(body_dir),
                [(str(component_dir), component_manifest)],
                ["body_azel.grim", "vehicle.grim"],
            )
            self.assertEqual(
                combined["schema"],
                "ghost.workflow.combined-output-provenance.v1",
            )

            dataset = root / "delta.grim"
            track = root / "track.txt"
            dataset.write_bytes(b"delta")
            track.write_bytes(b"track")

            class Library:
                @staticmethod
                def paths():
                    return [str(dataset)]

            old_dataset_dir = R5.DATASETS_DIR
            R5.DATASETS_DIR = str(root)
            try:
                trade = R5._trade_provenance_payload(
                    Library(), str(track), str(body_dir)
                )
            finally:
                R5.DATASETS_DIR = old_dataset_dir
            self.assertEqual(
                trade["schema"],
                "ghost.workflow.trade-study-provenance.v1",
            )
            self.assertEqual(
                set(trade["datasets_sha256"]), {"delta.grim"}
            )

    def test_hpc_collector_semantically_rechecks_staged_pairs(self):
        with tempfile.TemporaryDirectory() as td:
            workflow = Path(td) / "step"
            run_dir = Path(td) / "run"
            source_results = run_dir / "results"
            workflow.mkdir()
            source_results.mkdir(parents=True)
            output = source_results / "TM_1.000GHz_coupon_FFD.grim"
            sidecar = Path(str(output) + ".provenance.json")
            output.write_bytes(b"field")
            sidecar.write_text("{}", encoding="utf-8")
            manifest = {
                "schema": "ghost.hpc.2d-run.v1",
                "run_id": "run_fixture",
                "solver_source_sha256": "a" * 64,
                "runtime_environment_sha256": "b" * 64,
                "solver_config": {"geometry_units": "meters"},
                "n_units": 1,
                "units": [],
            }
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            status = {
                "run_dir": run_dir,
                "manifest": manifest,
                "n_done": 1,
                "n_units": 1,
                "pending": 0,
            }
            attestation_roots = []

            def check_attestations(root, _manifest):
                root = Path(root)
                attestation_roots.append(root)
                self.assertTrue(
                    (root / "results" / output.name).is_file()
                )
                self.assertTrue(
                    (root / "results" / sidecar.name).is_file()
                )

            old_here = R1B.HERE
            old_argv = sys.argv
            R1B.HERE = str(workflow)
            sys.argv = ["collect.py", str(run_dir)]
            try:
                with mock.patch.object(
                    R1B, "run_status", return_value=status
                ), mock.patch.object(
                    R1B, "require_hpc_run_provenance"
                ), mock.patch.object(
                    R1B,
                    "require_hpc_output_attestations",
                    side_effect=check_attestations,
                ), mock.patch.object(
                    R1B, "group_solver_files", return_value=({}, [])
                ):
                    R1B.main()
            finally:
                R1B.HERE = old_here
                sys.argv = old_argv
            self.assertEqual(attestation_roots[0], run_dir)
            self.assertNotEqual(attestation_roots[1], run_dir)
            self.assertEqual(
                attestation_roots[1].name.startswith(".collect-stage-"),
                True,
            )
            self.assertTrue(
                (workflow / "results" / "collection_manifest.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
