"""Focused regressions for production body/feature safety fixes."""

import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

import feature_sum as fs
import line_expand as le
from components import combine_component_fields
from occluder import Occluder


def _seg(name, seg_type, p0, p1, *, n="-20", ibc="0", pos="0", neg="0"):
    return {
        "name": name,
        "seg_type": str(seg_type),
        "properties": [str(seg_type), str(n), str(ibc), str(pos), str(neg)],
        "point_pairs": [{
            "x1": float(p0[0]), "y1": float(p0[1]),
            "x2": float(p1[0]), "y2": float(p1[1]),
        }],
    }


class TestOuterProfile(unittest.TestCase):
    def test_partial_coating_keeps_type2_and_type3_exterior(self):
        snap = {
            "segments": [
                _seg("coated_outer", 3, (0.0, 1.0), (1.0, 0.0), pos="1"),
                _seg("bare_outer", 2, (1.0, 0.0), (0.0, -1.0)),
                # Covered core is not air-facing and must stay out.
                _seg("covered_core", 4, (0.0, 0.8), (0.8, 0.0), pos="1"),
            ],
            "ibcs": [],
            "dielectrics": [["1", "2.5", "0", "1", "0"]],
        }
        got = fs.outer_generatrix(snap, "meters")
        np.testing.assert_allclose(
            got, np.array([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]]))

    def test_disconnected_air_facing_profile_is_rejected(self):
        snap = {
            "segments": [
                _seg("a", 3, (0.0, 1.0), (0.5, 0.5), pos="1"),
                _seg("b", 2, (0.8, 0.2), (0.0, -1.0)),
            ]
        }
        with self.assertRaisesRegex(ValueError, "do not form one directed|disconnected"):
            fs.outer_generatrix(snap, "meters")

    def test_unknown_profile_units_are_rejected(self):
        snap = {"segments": [_seg("body", 2, (0.0, 1.0), (0.0, -1.0))]}
        with self.assertRaisesRegex(ValueError, "Unsupported geometry units"):
            fs.outer_generatrix(snap, "metres-ish")

    def test_skin_distance_samples_chord_interior_not_only_vertices(self):
        # A 60-degree chord has both endpoints exactly on a unit cylinder but
        # bows inward by 1-cos(30 deg) at its midpoint.
        a = np.deg2rad(30.0)
        chord = np.array([[
            [np.cos(a), -np.sin(a), 0.0],
            [np.cos(a), +np.sin(a), 0.0],
        ]])
        cylinder = np.array([[1.0, -1.0], [1.0, 1.0]])
        got = le.perimeter_surface_deviation(chord, cylinder)
        self.assertAlmostEqual(got, 1.0 - np.cos(a), places=8)


class TestBodyDispatchAndAspectGrid(unittest.TestCase):
    def test_bare_pec_body_uses_dispatch_for_both_polarizations(self):
        snap = {
            "title": "bare",
            "segments": [
                _seg("top", 2, (0.0, 0.1), (0.03, 0.0)),
                _seg("bottom", 2, (0.03, 0.0), (0.0, -0.1)),
            ],
            "ibcs": [],
            "dielectrics": [],
        }
        calls = []

        def fake_dispatch(_snap, freqs, aspects, pol, **_kwargs):
            calls.append(pol)
            scale = 1.0 if pol == "VV" else 2.0
            return {
                "samples": [{
                    "theta_inc_deg": float(th),
                    "rcs_amp_real": scale,
                    "rcs_amp_imag": 0.0,
                } for th in aspects]
            }

        with mock.patch("bor_dispatch.solve_monostatic_rcs_bor",
                        side_effect=fake_dispatch):
            bodies, _ = fs.solve_vehicle_body(
                snap, [3.0], [0.0, 37.0], geometry_units="meters")
        self.assertEqual(calls, ["VV", "HH"])
        self.assertEqual(bodies[3.0]["theta_deg"], [0.0, 37.0])
        self.assertEqual(bodies[3.0]["amp_vv"], [1.0 + 0j, 1.0 + 0j])
        self.assertEqual(bodies[3.0]["amp_hh"], [2.0 + 0j, 2.0 + 0j])

    def test_file_body_snapshot_keeps_source_path_for_material_sidecars(self):
        snap = {
            "title": "file-backed",
            "segments": [],
            "ibcs": [],
            "dielectrics": [],
        }
        seen = []

        def fake_dispatch(got, _freqs, aspects, _pol, **_kwargs):
            seen.append(dict(got))
            return {
                "samples": [{
                    "theta_inc_deg": float(theta),
                    "rcs_amp_real": 1.0,
                    "rcs_amp_imag": 0.0,
                } for theta in aspects]
            }

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "body.geo")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write("test fixture")
            with mock.patch(
                    "geometry_io.parse_geometry",
                    return_value=("parsed",)), \
                    mock.patch(
                        "geometry_io.build_geometry_snapshot",
                        return_value=snap), \
                    mock.patch.object(
                        fs, "outer_generatrix",
                        return_value=np.asarray([[0.0, 0.0]])), \
                    mock.patch(
                        "bor_dispatch.solve_monostatic_rcs_bor",
                        side_effect=fake_dispatch):
                fs.solve_vehicle_body(
                    path, [1.0], [0.0], geometry_units="meters"
                )

        self.assertEqual(len(seen), 2)
        self.assertTrue(all(
            item["source_path"] == os.path.abspath(path) for item in seen
        ))

    def test_required_aspects_include_every_radar_grid_look(self):
        az = np.array([0.0, 37.0, 180.0, 271.0])
        el = np.array([-33.0, 0.0, 19.0])
        required = fs.radar_grid_aspects(az, el, axis_az_deg=12.0,
                                         axis_el_deg=7.0)
        _, axis = fs._attitude(12.0, 7.0, 0.0)
        for a in az:
            for e in el:
                q = fs._aspect_of(fs._direction(a, e)[None, :], axis)[0]
                self.assertTrue(np.any(np.isclose(required, q, atol=1e-9,
                                                  rtol=0.0)))

    def test_body_cache_fingerprint_covers_units_and_material_tables(self):
        with tempfile.TemporaryDirectory() as td:
            geo = os.path.join(td, "body.geo")
            table = os.path.join(td, "mat.51")
            with open(geo, "w") as fh:
                fh.write("Title: cache gate\n")
            with open(table, "w") as fh:
                fh.write("1 2 3\n")
            h0 = fs.geometry_input_fingerprint(geo, "meters")
            self.assertNotEqual(
                h0, fs.geometry_input_fingerprint(geo, "inches"))
            with open(table, "w") as fh:
                fh.write("1 2 4\n")
            self.assertNotEqual(
                h0, fs.geometry_input_fingerprint(geo, "meters"))


class TestRadarPolarizationAtAxis(unittest.TestCase):
    def test_axial_feature_jones_matrix_tracks_vehicle_roll(self):
        # Mock an anisotropic feature in the exact vehicle basis used by the
        # exporter.  A 45-degree vehicle roll must rotate diag(1,0) into equal
        # co-pol terms with a nonzero cross-pol term, even at a nose-on look.
        def anisotropic(*_args, **_kwargs):
            return {
                "amp_vv": np.array([1.0 + 0j]),
                "amp_hh": np.array([0.0 + 0j]),
                "amp_vh": np.array([0.0 + 0j]),
            }

        gen = np.array([[0.0, 0.1], [0.03, 0.0], [0.0, -0.1]])
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(fs, "sum_features", side_effect=anisotropic):
            p0 = fs.export_radar_grim(
                os.path.join(td, "roll0"), bor_result=None, placements=[],
                generatrix=gen, frequencies_ghz=[3.0],
                azimuths_deg=[0.0], elevations_deg=[0.0],
                axis_az_deg=0.0, axis_el_deg=0.0, roll_deg=0.0)
            p45 = fs.export_radar_grim(
                os.path.join(td, "roll45"), bor_result=None, placements=[],
                generatrix=gen, frequencies_ghz=[3.0],
                azimuths_deg=[0.0], elevations_deg=[0.0],
                axis_az_deg=0.0, axis_el_deg=0.0, roll_deg=45.0)
            with np.load(p0, allow_pickle=False) as z:
                a0 = z["rcs_amp_real"] + 1j * z["rcs_amp_imag"]
            with np.load(p45, allow_pickle=False) as z:
                a45 = z["rcs_amp_real"] + 1j * z["rcs_amp_imag"]

        np.testing.assert_allclose(np.abs(a0[0, 0, 0]), [1.0, 0.0, 0.0],
                                   atol=1e-12)
        np.testing.assert_allclose(np.abs(a45[0, 0, 0]), [0.5, 0.5, 0.5],
                                   atol=2e-7)


class TestComplexFieldSemantics(unittest.TestCase):
    def test_seam_coefficients_never_turn_missing_support_into_zero_field(self):
        coeff = le.SeamCoefficients(
            3.0,
            np.array([45.0, 90.0, 135.0]),
            np.ones(3, dtype=complex),
            np.ones(3, dtype=complex),
        )
        with self.assertRaisesRegex(ValueError, "outside characterized support"):
            coeff.sample(np.array([30.0]))
        with self.assertRaisesRegex(ValueError, "unique"):
            le.SeamCoefficients(
                3.0,
                np.array([45.0, 45.0, 90.0]),
                np.ones(3, dtype=complex),
                np.ones(3, dtype=complex),
            )

    def test_perimeter_input_rejects_nonfinite_and_zero_length_segments(self):
        coeff = le.SeamCoefficients(
            3.0,
            np.array([0.0, 90.0, 180.0]),
            np.ones(3, dtype=complex),
            np.ones(3, dtype=complex),
        )
        normal = lambda points: np.tile([1.0, 0.0, 0.0], (len(points), 1))
        cases = (
            np.array([[[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]]),
            np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        )
        for perimeter in cases:
            with self.subTest(perimeter=perimeter):
                with self.assertRaises(ValueError):
                    le.expand_perimeter(
                        perimeter,
                        coeff,
                        normal,
                        np.array([[1.0, 0.0, 0.0]]),
                    )

    def test_default_combination_is_coherent_and_power_matches_field(self):
        body = {"F_vv": np.array([1.0 + 0.0j])}
        feature = {"F_vv": np.array([-0.25 + 0.5j])}
        got = le.combine(body, [feature])
        expected_amp = np.array([0.75 + 0.5j])
        np.testing.assert_allclose(got["amp_vv"], expected_amp)
        np.testing.assert_allclose(
            got["sigma_vv"], 4.0 * np.pi * np.abs(expected_amp) ** 2)
        self.assertEqual(got["mode"], "coherent")

    def test_phase_unknown_component_never_enters_stored_field(self):
        body = np.array([0.0 + 0.0j])
        wing = np.array([1.0 + 0.0j])
        # If this arbitrary phase were coherently summed, it would cancel the
        # wing completely.  It must affect only the estimate.
        corner = np.array([-1.0 + 0.0j])
        amp, power, estimate = combine_component_fields(
            body, [wing], [corner], mode="coherent")
        np.testing.assert_allclose(amp, wing)
        np.testing.assert_allclose(power, [4.0 * np.pi])
        np.testing.assert_allclose(estimate, [8.0 * np.pi])

    def test_body_frame_export_separates_estimate_from_field_power(self):
        def synthetic(*_args, **_kwargs):
            # One direction: coherent F=1, selected in-memory estimate=8*pi.
            return {
                "amp_vv": np.array([1.0 + 0.0j]),
                "amp_hh": np.array([0.0 + 0.0j]),
                "amp_vh": np.array([0.0 + 0.0j]),
                "sigma_vv": np.array([8.0 * np.pi]),
                "sigma_hh": np.array([0.0]),
                "sigma_vh": np.array([0.0]),
            }

        gen = np.array([[0.0, 0.1], [0.03, 0.0], [0.0, -0.1]])
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(fs, "sum_features", side_effect=synthetic):
            paths = fs.export_signature_grim(
                os.path.join(td, "sig"), bor_result=None, placements=[],
                generatrix=gen, frequencies_ghz=[3.0],
                aspects_deg=[90.0], rolls_deg=[0.0], mode="hybrid")
            with np.load(paths[0], allow_pickle=False) as z:
                self.assertAlmostEqual(
                    float(z["rcs_power"][0, 0, 0, 0]), 4.0 * np.pi,
                    places=5)
                self.assertAlmostEqual(
                    float(z["combination_estimate_power"][0, 0, 0, 0]),
                    8.0 * np.pi, places=5)
                self.assertEqual(str(z["combination_estimate_mode"]),
                                 "hybrid")

    def test_corner_field_keeps_pi_phase_flip_across_fold_null(self):
        # Center the fold at the origin so placement contributes no phase.
        # The scalar aperture field must be signed sinc, not |sinc|.
        freq = 1.0
        k = 2.0 * np.pi * freq * 1e9 / fs.C0
        xvals = np.array([0.5 * np.pi, 1.5 * np.pi])
        df = xvals / k
        bhat = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        dirs = (np.sqrt(1.0 - df ** 2)[:, None] * bhat
                + df[:, None] * np.array([0.0, 0.0, 1.0]))
        got = fs.corner_amplitude(
            np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]]),
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.1, dirs, freq)

        ev, eh = fs._pol_unit_vectors(dirs)
        scalars = []
        for i, d in enumerate(dirs):
            phat = np.array([0.0, 0.0, 1.0]) - d[2] * d
            phat /= np.linalg.norm(phat)
            qhat = np.cross(d, phat)
            rot = np.array([[phat @ ev[i], phat @ eh[i]],
                            [qhat @ ev[i], qhat @ eh[i]]])
            jones = rot.T @ np.diag([1.0, -1.0]) @ rot
            matrix = np.array([[got["F_vv"][i], got["F_vh"][i]],
                               [got["F_vh"][i], got["F_hh"][i]]])
            scalars.append(np.vdot(jones, matrix) / np.vdot(jones, jones))
        self.assertGreater(scalars[0].real, 0.0)
        self.assertLess(scalars[1].real, 0.0)
        np.testing.assert_allclose(np.imag(scalars), 0.0, atol=1e-12)


class TestOccluderSafety(unittest.TestCase):
    def test_unknown_stl_units_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported STL units"):
            Occluder.from_stl("unused.stl", units="furlongs")

    def test_negative_bias_is_rejected(self):
        tri = np.array([[[0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]]])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            Occluder(tri, bias=-1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
