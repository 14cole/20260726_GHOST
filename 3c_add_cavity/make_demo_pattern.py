#!/usr/bin/env python3
"""Normalize the shipped fabricated cavity delta to the repository 3-D schema.

This is a deterministic maintenance helper, not a solver.  It preserves the
existing fabricated complex pattern and fixes/records its physical encoding:

    rcs_power = sigma_3d = 4*pi*|F|^2
    rcs_domain = delta
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(os.path.dirname(HERE), "Backend")
sys.path.insert(0, BACKEND)

from feature_sum import (_load_pattern,                    # noqa: E402
                         point_pattern_convention_metadata)

PATH = os.path.join(HERE, "cavity_pattern.grim")


def main():
    with np.load(PATH, allow_pickle=False) as z:
        payload = {key: z[key] for key in z.files}
    amp = (np.asarray(payload["rcs_amp_real"], dtype=float)
           + 1j * np.asarray(payload["rcs_amp_imag"], dtype=float))
    payload["rcs_power"] = (
        4.0 * math.pi * np.abs(amp) ** 2).astype(np.float32)
    payload["rcs_phase"] = np.angle(amp).astype(np.float32)
    for key, value in point_pattern_convention_metadata().items():
        payload[key] = np.asarray(value)
    payload["power_domain"] = np.asarray("linear_rcs")
    payload["units"] = np.asarray(json.dumps({
        "azimuth": "deg",
        "elevation": "deg",
        "frequency": "GHz",
        "rcs_log_unit": "dBsm",
        "rcs_linear_quantity": "sigma_3d",
    }))
    payload["history"] = np.asarray(
        "fabricated cavity differential demo; normalized by "
        "make_demo_pattern.py")
    payload["raw_complex_amplitude_preserved"] = np.asarray(True)
    with open(PATH, "wb") as fh:
        np.savez(fh, **payload)
    _load_pattern(PATH)
    print(PATH)


if __name__ == "__main__":
    main()
