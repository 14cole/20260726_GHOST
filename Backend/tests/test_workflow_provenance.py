"""Regressions for solver-workflow source and runtime fingerprints."""

import json
import os
import sys
import tempfile
import unittest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

import workflow_provenance as provenance
from feature_sum import geometry_input_fingerprint
from hpc_common import (
    require_hpc_output_attestations,
    require_hpc_run_provenance,
)
import run_local_monostatic


class TestWorkflowProvenance(unittest.TestCase):
    def test_backend_source_paths_include_native_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            names = ("solver.py", "kernel.c", "kernel.so", "notes.txt")
            for name in names:
                with open(
                    os.path.join(folder, name), "wb"
                ) as stream:
                    stream.write(name.encode("ascii"))
            got = {
                os.path.basename(path)
                for path in provenance.backend_source_paths(folder)
            }
        self.assertEqual(got, {"solver.py", "kernel.c", "kernel.so"})

    def test_runtime_environment_fingerprint_is_stable_sha256(self):
        first = provenance.runtime_environment_fingerprint()
        second = provenance.runtime_environment_fingerprint()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertTrue(all(char in "0123456789abcdef" for char in first))

    def test_output_attestation_detects_modified_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            output = os.path.join(folder, "unit.grim")
            with open(output, "wb") as stream:
                stream.write(b"original")
            expected = {"run_id": "run_fixture", "frequency_ghz": 1.0}
            provenance.write_output_attestation(output, expected)
            provenance.verify_output_attestation(output, expected)
            with open(output, "wb") as stream:
                stream.write(b"modified")
            with self.assertRaisesRegex(
                ValueError, r"bytes differ"
            ):
                provenance.verify_output_attestation(output, expected)

    def test_artifact_manifest_is_an_exact_bundle_commit_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            output = os.path.join(folder, "unit.grim")
            sidecar = output + ".provenance.json"
            with open(output, "wb") as stream:
                stream.write(b"field")
            with open(sidecar, "wb") as stream:
                stream.write(b"attestation")
            provenance.write_artifact_manifest(
                folder,
                "ghost.hpc.2d-collection.v1",
                ["unit.grim", "unit.grim.provenance.json"],
                {"run_id": "run_fixture"},
            )
            payload = provenance.verify_artifact_manifest(
                folder,
                ["unit.grim", "unit.grim.provenance.json"],
                exact_grim_set=True,
                expected_schema="ghost.hpc.2d-collection.v1",
            )
            self.assertEqual(payload["status"], "complete")

            with open(sidecar, "ab") as stream:
                stream.write(b"-tampered")
            with self.assertRaisesRegex(ValueError, r"differs"):
                provenance.verify_artifact_manifest(
                    folder,
                    ["unit.grim"],
                    exact_grim_set=True,
                )

    def test_in_progress_marker_invalidates_an_old_complete_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            output = os.path.join(folder, "unit.grim")
            with open(output, "wb") as stream:
                stream.write(b"old field")
            provenance.write_artifact_manifest(
                folder,
                "ghost.workflow.fixture.v1",
                ["unit.grim"],
            )
            provenance.write_artifact_in_progress(
                folder,
                "ghost.workflow.fixture.v1",
                ["unit.grim"],
            )
            with self.assertRaisesRegex(
                ValueError, r"not a completed"
            ):
                provenance.verify_artifact_manifest(
                    folder, ["unit.grim"]
                )
            with open(output, "rb") as stream:
                self.assertEqual(stream.read(), b"old field")

    def test_component_manifest_verifies_existing_wing_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            output_dir = os.path.join(folder, "Output")
            os.mkdir(output_dir)
            wing = os.path.join(output_dir, "fin.grim")
            summary = os.path.join(folder, "wing_dbsm.csv")
            with open(wing, "wb") as stream:
                stream.write(b"wing field")
            with open(summary, "wb") as stream:
                stream.write(b"diagnostic")
            manifest = {
                "schema": "ghost.workflow.wing-output-provenance.v1",
                "status": "complete",
                "expected_outputs": ["fin.grim"],
                "output_sha256": {
                    "fin.grim": provenance.sha256_file(wing),
                    "../wing_dbsm.csv": provenance.sha256_file(summary),
                },
            }
            with open(
                os.path.join(output_dir, "provenance_manifest.json"),
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump(manifest, stream)
            provenance.verify_component_output_manifest(output_dir)

            with open(wing, "ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, r"committed bytes"):
                provenance.verify_component_output_manifest(output_dir)

    def test_artifact_manifest_refuses_uncommitted_and_ambiguous_bundles(self):
        with tempfile.TemporaryDirectory() as folder:
            output = os.path.join(folder, "unit.grim")
            with open(output, "wb") as stream:
                stream.write(b"field")
            with self.assertRaisesRegex(ValueError, r"exactly one"):
                provenance.verify_artifact_manifest(
                    folder, ["unit.grim"]
                )

            provenance.write_artifact_manifest(
                folder,
                "ghost.hpc.2d-collection.v1",
                ["unit.grim"],
            )
            with open(
                os.path.join(folder, "cache_manifest.json"),
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump(
                    {
                        "schema": "ghost.local.cache.v1",
                        "status": "complete",
                    },
                    stream,
                )
            with self.assertRaisesRegex(ValueError, r"exactly one"):
                provenance.verify_artifact_manifest(
                    folder, ["unit.grim"]
                )

    def test_artifact_manifest_refuses_path_aliases_and_extra_grims(self):
        with tempfile.TemporaryDirectory() as folder:
            output = os.path.join(folder, "unit.grim")
            with open(output, "wb") as stream:
                stream.write(b"field")
            with self.assertRaisesRegex(
                ValueError, r"canonical relative path"
            ):
                provenance.write_artifact_manifest(
                    folder,
                    "ghost.hpc.2d-collection.v1",
                    ["nested/../unit.grim"],
                )

            provenance.write_artifact_manifest(
                folder,
                "ghost.hpc.2d-collection.v1",
                ["unit.grim"],
            )
            nested = os.path.join(folder, "nested")
            os.mkdir(nested)
            with open(
                os.path.join(nested, "unexpected.GRIM"), "wb"
            ) as stream:
                stream.write(b"uncommitted")
            with self.assertRaisesRegex(ValueError, r"\.grim set differs"):
                provenance.verify_artifact_manifest(
                    folder, ["unit.grim"], exact_grim_set=True
                )

    def test_artifact_manifest_malformed_json_shape_is_controlled_refusal(self):
        with tempfile.TemporaryDirectory() as folder:
            with open(
                os.path.join(folder, "collection_manifest.json"),
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump(["not", "an", "object"], stream)
            with self.assertRaisesRegex(ValueError, r"manifest object"):
                provenance.verify_artifact_manifest(folder, [])

    def test_hpc_collection_rechecks_frozen_material_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            geometry = os.path.join(folder, "body.geo")
            table = os.path.join(folder, "mat.51")
            with open(geometry, "w", encoding="utf-8") as stream:
                stream.write("Title: provenance fixture\n")
            with open(table, "w", encoding="utf-8") as stream:
                stream.write("1 2 0\n")
            input_hash = geometry_input_fingerprint(geometry, "meters")
            manifest = {
                "schema": "ghost.hpc.2d-run.v1",
                "solver_source_sha256": "a" * 64,
                "runtime_environment_sha256": "b" * 64,
                "solver_config": {"geometry_units": "meters"},
                "n_units": 1,
                "units": [{
                    "geometry": geometry,
                    "geometry_stem": "body",
                    "geometry_input_sha256": input_hash,
                    "polarization": "TM",
                    "frequency_ghz": 1.0,
                    "azimuths_deg": [0.0, 90.0],
                }],
            }
            require_hpc_run_provenance(
                manifest, "ghost.hpc.2d-run.v1"
            )
            with open(table, "w", encoding="utf-8") as stream:
                stream.write("1 3 0\n")
            with self.assertRaisesRegex(
                ValueError, r"geometry/material input changed"
            ):
                require_hpc_run_provenance(
                    manifest, "ghost.hpc.2d-run.v1"
                )

    def test_hpc_output_attestation_requires_full_solve_spec(self):
        with tempfile.TemporaryDirectory() as folder:
            results = os.path.join(folder, "results")
            os.mkdir(results)
            unit = {
                "geometry": os.path.join(folder, "coupon.geo"),
                "geometry_stem": "coupon",
                "geometry_input_sha256": "c" * 64,
                "polarization": "TM",
                "frequency_ghz": 3.0,
                "azimuths_deg": [0.0, 45.0, 90.0],
            }
            manifest = {
                "schema": "ghost.hpc.2d-run.v1",
                "run_id": "run_fixture",
                "solver_source_sha256": "a" * 64,
                "runtime_environment_sha256": "b" * 64,
                "solver_config": {
                    "geometry_units": "meters",
                    "solver_method": "direct",
                },
                "n_units": 1,
                "units": [unit],
            }
            output = os.path.join(
                results, "TM_3.000GHz_coupon.grim"
            )
            with open(output, "wb") as stream:
                stream.write(b"attested field")
            complete = {
                "run_id": manifest["run_id"],
                "solver_source_sha256":
                    manifest["solver_source_sha256"],
                "runtime_environment_sha256":
                    manifest["runtime_environment_sha256"],
                "geometry_input_sha256":
                    unit["geometry_input_sha256"],
                "run_solve_spec_sha256":
                    provenance.manifest_solve_spec_fingerprint(manifest),
                "unit_solve_spec_sha256":
                    provenance.unit_solve_spec_fingerprint(unit),
                "solver_config_sha256":
                    provenance.stable_json_fingerprint(
                        manifest["solver_config"]
                    ),
                "angular_grid_kind": "azimuths_deg",
                "angular_grid_deg": [0.0, 45.0, 90.0],
                "polarization": "TM",
                "frequency_ghz": 3.0,
            }

            new_fields = (
                "run_solve_spec_sha256",
                "unit_solve_spec_sha256",
                "solver_config_sha256",
                "angular_grid_kind",
                "angular_grid_deg",
            )
            for missing in new_fields:
                legacy = dict(complete)
                legacy.pop(missing)
                provenance.write_output_attestation(output, legacy)
                with self.subTest(missing=missing):
                    with self.assertRaisesRegex(
                        ValueError, rf"match {missing}"
                    ):
                        require_hpc_output_attestations(folder, manifest)

            provenance.write_output_attestation(output, complete)
            require_hpc_output_attestations(folder, manifest)

            changed = json.loads(json.dumps(manifest))
            changed["solver_config"]["solver_method"] = "auto"
            with self.assertRaisesRegex(
                ValueError, r"run_solve_spec_sha256"
            ):
                require_hpc_output_attestations(folder, changed)

    def test_hpc_output_inventory_refuses_collisions_and_stale_sidecars(self):
        with tempfile.TemporaryDirectory() as folder:
            results = os.path.join(folder, "results")
            os.mkdir(results)
            base = {
                "geometry_stem": "coupon",
                "geometry_input_sha256": "c" * 64,
                "polarization": "TM",
                "azimuths_deg": [0.0],
            }
            first = dict(base, frequency_ghz=1.0001)
            second = dict(base, frequency_ghz=1.0002)
            manifest = {
                "schema": "ghost.hpc.2d-run.v1",
                "run_id": "run_fixture",
                "solver_source_sha256": "a" * 64,
                "runtime_environment_sha256": "b" * 64,
                "solver_config": {"geometry_units": "meters"},
                "n_units": 2,
                "units": [first, second],
            }
            with self.assertRaisesRegex(ValueError, r"colliding"):
                require_hpc_output_attestations(folder, manifest)

            manifest["units"] = [first]
            manifest["n_units"] = 1
            output = os.path.join(
                results, "TM_1.000GHz_coupon.grim"
            )
            with open(output, "wb") as stream:
                stream.write(b"field")
            expected = {
                "run_id": "run_fixture",
                "solver_source_sha256": "a" * 64,
                "runtime_environment_sha256": "b" * 64,
                "geometry_input_sha256": "c" * 64,
                "run_solve_spec_sha256":
                    provenance.manifest_solve_spec_fingerprint(manifest),
                "unit_solve_spec_sha256":
                    provenance.unit_solve_spec_fingerprint(first),
                "solver_config_sha256":
                    provenance.stable_json_fingerprint(
                        manifest["solver_config"]
                    ),
                "angular_grid_kind": "azimuths_deg",
                "angular_grid_deg": [0.0],
                "polarization": "TM",
                "frequency_ghz": 1.0001,
            }
            provenance.write_output_attestation(output, expected)
            with open(
                os.path.join(
                    results, "stale.grim.provenance.json"
                ),
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump({}, stream)
            with self.assertRaisesRegex(
                ValueError, r"exact attestation set"
            ):
                require_hpc_output_attestations(folder, manifest)

    def test_local_runner_reuses_only_exact_attested_output(self):
        with tempfile.TemporaryDirectory() as folder:
            geometry = os.path.join(folder, "coupon.geo")
            with open(geometry, "w", encoding="utf-8") as stream:
                stream.write("Title: attested local fixture\n")
            run_dir = os.path.join(folder, "run_fixture")
            results_dir = os.path.join(run_dir, "results")
            os.makedirs(results_dir)
            unit = {
                "geometry": geometry,
                "geometry_stem": "coupon",
                "geometry_input_sha256": geometry_input_fingerprint(
                    geometry, run_local_monostatic.GEOMETRY_UNITS
                ),
                "polarization": "TM",
                "frequency_ghz": 1.0,
                "azimuths_deg": [0.0],
            }
            manifest = {
                "run_id": "run_fixture",
                "solver_source_sha256":
                    run_local_monostatic._solver_source_fingerprint(),
                "runtime_environment_sha256":
                    provenance.runtime_environment_fingerprint(),
            }
            with open(
                os.path.join(run_dir, "manifest.json"),
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump(manifest, stream)
            output = os.path.join(
                results_dir, "TM_1.000GHz_coupon.grim"
            )
            with open(output, "wb") as stream:
                stream.write(b"attested")
            expected = run_local_monostatic._unit_attestation_fields(
                manifest, unit
            )
            provenance.write_output_attestation(output, expected)
            status, got = run_local_monostatic._solve_and_export(
                unit, {}, folder, results_dir
            )
            self.assertEqual(status, "skipped")
            self.assertEqual(got, output)
            with open(output, "wb") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, r"bytes differ"):
                run_local_monostatic._solve_and_export(
                    unit, {}, folder, results_dir
                )

    def test_unit_attestation_binds_angles_solver_config_and_manifest(self):
        unit = {
            "geometry": "/frozen/coupon.geo",
            "geometry_stem": "coupon",
            "geometry_input_sha256": "c" * 64,
            "polarization": "TM",
            "frequency_ghz": 3.0,
            "azimuths_deg": [0.0, 90.0],
        }
        manifest = {
            "run_id": "run_fixture",
            "solver_source_sha256": "a" * 64,
            "runtime_environment_sha256": "b" * 64,
            "solver_config": {
                "geometry_units": "meters",
                "solver_method": "direct",
            },
            "units": [unit],
        }
        reference = run_local_monostatic._unit_attestation_fields(
            manifest, unit
        )

        changed_unit = dict(unit)
        changed_unit["azimuths_deg"] = [0.0, 45.0, 90.0]
        changed_manifest = dict(manifest)
        changed_manifest["units"] = [changed_unit]
        changed_angles = run_local_monostatic._unit_attestation_fields(
            changed_manifest, changed_unit
        )
        self.assertNotEqual(
            reference["unit_solve_spec_sha256"],
            changed_angles["unit_solve_spec_sha256"],
        )
        self.assertNotEqual(
            reference["run_solve_spec_sha256"],
            changed_angles["run_solve_spec_sha256"],
        )

        changed_config_manifest = dict(manifest)
        changed_config_manifest["solver_config"] = {
            "geometry_units": "meters",
            "solver_method": "auto",
        }
        changed_config = run_local_monostatic._unit_attestation_fields(
            changed_config_manifest, unit
        )
        self.assertNotEqual(
            reference["solver_config_sha256"],
            changed_config["solver_config_sha256"],
        )
        self.assertNotEqual(
            reference["run_solve_spec_sha256"],
            changed_config["run_solve_spec_sha256"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
