"""Focused analytic and fail-closed tests for ``bor_az_el_grid``."""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bor_dispatch import bor_az_el_grid


def _result(pol, theta, amplitude, frequency=2.0):
    theta_values = list(theta)
    if np.isscalar(amplitude):
        amplitudes = [complex(amplitude)] * len(theta_values)
    else:
        amplitudes = [complex(value) for value in amplitude]
    return {
        "polarization": pol,
        "samples": [
            {
                "frequency_ghz": float(frequency),
                "theta_inc_deg": angle,
                "rcs_amp_real": value.real,
                "rcs_amp_imag": value.imag,
            }
            for angle, value in zip(theta_values, amplitudes)
        ],
    }


class TestAnalyticJonesRotation(unittest.TestCase):
    def setUp(self):
        self.theta = [0.0, 90.0, 180.0]
        self.fv = 2.0 + 1.0j
        self.fh = -1.0 + 0.5j
        self.vv = _result("VV", self.theta, self.fv)
        self.hh = _result("HH", self.theta, self.fh)

    def test_horizontal_axis_waterline_quarter_turn_swaps_vv_hh(self):
        grid = bor_az_el_grid(
            self.vv,
            self.hh,
            [90.0],
            [0.0],
            axis_az_deg=0.0,
            axis_el_deg=0.0,
        )
        self.assertAlmostEqual(grid["amp"]["VV"][0, 0, 0], self.fh)
        self.assertAlmostEqual(grid["amp"]["HH"][0, 0, 0], self.fv)
        self.assertAlmostEqual(grid["amp"]["VH"][0, 0, 0], 0.0j)

    def test_vertical_looks_use_requested_e_phi_basis(self):
        grid = bor_az_el_grid(
            self.vv,
            self.hh,
            [45.0],
            [-90.0, 90.0],
            axis_az_deg=0.0,
            axis_el_deg=0.0,
        )
        expected_co = 0.5 * (self.fv + self.fh)
        self.assertAlmostEqual(grid["amp"]["VV"][0, 0, 0], expected_co)
        self.assertAlmostEqual(grid["amp"]["HH"][0, 0, 0], expected_co)
        self.assertAlmostEqual(
            grid["amp"]["VH"][0, 0, 0], 0.5 * (self.fv - self.fh)
        )
        self.assertAlmostEqual(grid["amp"]["VV"][0, 1, 0], expected_co)
        self.assertAlmostEqual(grid["amp"]["HH"][0, 1, 0], expected_co)
        self.assertAlmostEqual(
            grid["amp"]["VH"][0, 1, 0], 0.5 * (self.fh - self.fv)
        )

        input_power = abs(self.fv) ** 2 + abs(self.fh) ** 2
        for elevation_index in (0, 1):
            output_power = (
                abs(grid["amp"]["VV"][0, elevation_index, 0]) ** 2
                + abs(grid["amp"]["HH"][0, elevation_index, 0]) ** 2
                + 2.0 * abs(grid["amp"]["VH"][0, elevation_index, 0]) ** 2
            )
            self.assertAlmostEqual(output_power, input_power)

    def test_axis_aligned_looks_use_radar_basis_and_enforce_isotropy(self):
        nose = 3.0 - 2.0j
        tail = -0.5 + 0.75j
        vv = _result("VV", self.theta, [nose, self.fv, tail])
        hh = _result("HH", self.theta, [nose, self.fh, tail])
        grid = bor_az_el_grid(
            vv,
            hh,
            [0.0, 180.0],
            [0.0],
            axis_az_deg=0.0,
            axis_el_deg=0.0,
        )
        for index, expected in enumerate((nose, tail)):
            self.assertAlmostEqual(grid["amp"]["VV"][index, 0, 0], expected)
            self.assertAlmostEqual(grid["amp"]["HH"][index, 0, 0], expected)
            self.assertEqual(grid["amp"]["VH"][index, 0, 0], 0.0j)

        anisotropic_hh = _result(
            "HH", self.theta, [2.0 * nose, self.fh, tail]
        )
        with self.assertRaisesRegex(ValueError, "violate BoR isotropy"):
            bor_az_el_grid(
                vv,
                anisotropic_hh,
                [0.0],
                [0.0],
                axis_az_deg=0.0,
                axis_el_deg=0.0,
            )


class TestAspectGridValidation(unittest.TestCase):
    def setUp(self):
        self.theta = [0.0, 90.0, 180.0]
        self.vv = _result("VV", self.theta, 1.0 + 0.25j)
        self.hh = _result("HH", self.theta, 0.5 - 0.75j)

    def test_vv_hh_aspect_grids_must_match_exactly(self):
        hh_shifted = _result(
            "HH", [0.0, 90.0 + 1.0e-12, 180.0], 0.5 - 0.75j
        )
        with self.assertRaisesRegex(ValueError, "match exactly"):
            bor_az_el_grid(self.vv, hh_shifted, [45.0], [0.0])

    def test_aspect_grids_must_be_finite_unique_and_strictly_ordered(self):
        bad_grids = (
            [0.0, float("nan"), 180.0],
            [0.0, 90.0, 90.0],
            [0.0, 120.0, 90.0],
        )
        for theta in bad_grids:
            with self.subTest(theta=theta):
                vv = _result("VV", theta, 1.0)
                hh = _result("HH", theta, 1.0)
                with self.assertRaises(ValueError):
                    bor_az_el_grid(vv, hh, [45.0], [0.0])

    def test_frequency_and_amplitude_data_must_be_finite_and_matching(self):
        with self.assertRaisesRegex(ValueError, "different frequencies"):
            bor_az_el_grid(
                self.vv,
                _result("HH", self.theta, 1.0, frequency=3.0),
                [45.0],
                [0.0],
            )
        bad_vv = _result("VV", self.theta, 1.0)
        bad_vv["samples"][1]["rcs_amp_real"] = float("inf")
        with self.assertRaisesRegex(ValueError, "non-finite complex amplitude"):
            bor_az_el_grid(bad_vv, self.hh, [45.0], [0.0])

    def test_query_must_be_inside_every_aspect_grid(self):
        vv = _result("VV", [10.0, 90.0, 170.0], 1.0)
        hh = _result("HH", [10.0, 90.0, 170.0], 1.0)
        with self.assertRaisesRegex(ValueError, "endpoint clamping is not permitted"):
            bor_az_el_grid(
                vv,
                hh,
                [0.0],
                [0.0],
                axis_az_deg=0.0,
                axis_el_deg=0.0,
            )


class TestAzElAxisValidation(unittest.TestCase):
    def setUp(self):
        theta = [0.0, 90.0, 180.0]
        self.vv = _result("VV", theta, 1.0)
        self.hh = _result("HH", theta, 1.0)

    def test_azimuth_and_elevation_axes_are_validated(self):
        bad_cases = (
            ([], [0.0]),
            ([[0.0, 10.0]], [0.0]),
            ([0.0, float("nan")], [0.0]),
            ([10.0, 0.0], [0.0]),
            ([0.0, 360.0], [0.0]),
            ([0.0], []),
            ([0.0], [0.0, 0.0]),
            ([0.0], [91.0]),
        )
        for azimuth, elevation in bad_cases:
            with self.subTest(azimuth=azimuth, elevation=elevation):
                with self.assertRaises(ValueError):
                    bor_az_el_grid(self.vv, self.hh, azimuth, elevation)

    def test_axis_angles_are_finite_scalars_with_physical_elevation(self):
        for axis_az, axis_el in (
            (float("inf"), 0.0),
            (0.0, float("nan")),
            (0.0, 90.0001),
            ([0.0, 1.0], 0.0),
        ):
            with self.subTest(axis_az=axis_az, axis_el=axis_el):
                with self.assertRaises(ValueError):
                    bor_az_el_grid(
                        self.vv,
                        self.hh,
                        [0.0],
                        [0.0],
                        axis_az_deg=axis_az,
                        axis_el_deg=axis_el,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
