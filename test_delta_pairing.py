from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grim_naming import pair_variants


class DeltaPairingTests(unittest.TestCase):
    def test_one_clean_subset_is_reused_by_multiple_featured_cases(self) -> None:
        paths = [
            "SEAL-00-00_0.050bmag_FRD.grim",
            "SEAL-00-00_0.050bmag_0.010het_0.020crv_OPN.grim",
            "SEAL-00-00_0.050bmag_0.015het_0.025crv_OPN.grim",
            "SEAL-00-00_0.050bmag_0.020het_0.020crv_OPN.grim",
        ]
        pairs, unmatched = pair_variants(paths)
        self.assertEqual(len(pairs), 3)
        self.assertFalse(unmatched)
        self.assertEqual(
            {pair["clean_base"] for pair in pairs},
            {"SEAL-00-00_0.050bmag"},
        )
        self.assertEqual(
            {pair["delta_name"] for pair in pairs},
            {
                "SEAL-00-00_0.050bmag_0.010het_0.020crv.grim",
                "SEAL-00-00_0.050bmag_0.015het_0.025crv.grim",
                "SEAL-00-00_0.050bmag_0.020het_0.020crv.grim",
            },
        )

    def test_most_specific_compatible_clean_baseline_wins(self) -> None:
        pairs, unmatched = pair_variants([
            "SEAL-00-00_0.050bmag_FRD.grim",
            "SEAL-00-00_0.050bmag_0.010het_FRD.grim",
            "SEAL-00-00_0.050bmag_0.010het_0.020crv_OPN.grim",
        ])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(
            pairs[0]["clean_base"],
            "SEAL-00-00_0.050bmag_0.010het",
        )
        self.assertEqual(len(unmatched), 1)
        self.assertIn("not used", unmatched[0]["reason"])

    def test_equally_specific_clean_baselines_are_ambiguous(self) -> None:
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            pair_variants([
                "SEAL-00-00_0.050bmag_0.010het_FRD.grim",
                "SEAL-00-00_0.050bmag_0.020crv_FRD.grim",
                "SEAL-00-00_0.050bmag_0.010het_0.020crv_OPN.grim",
            ])

    def test_different_shared_parameter_value_is_not_compatible(self) -> None:
        pairs, unmatched = pair_variants([
            "SEAL-00-00_0.040bmag_FRD.grim",
            "SEAL-00-00_0.050bmag_0.010het_OPN.grim",
        ])
        self.assertFalse(pairs)
        self.assertEqual(len(unmatched), 2)


if __name__ == "__main__":
    unittest.main()
