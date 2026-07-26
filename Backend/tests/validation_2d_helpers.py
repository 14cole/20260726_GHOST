"""Side-effect-free helpers shared by executable 2-D validation scripts."""

import math

import numpy as np

from rcs_solver import solve_monostatic_rcs_2d


def make_circle_segment(name, seg_type, radius, n_prim, ibc="0",
                        pos_mat="0", neg_mat="0", cw=True,
                        cx=0.0, cy=0.0):
    """Return a polygonal circle without importing an executable test suite."""
    points = []
    for index in range(n_prim + 1):
        angle = 2.0 * math.pi * index / n_prim
        if cw:
            angle = -angle
        points.append((
            cx + radius * math.cos(angle),
            cy + radius * math.sin(angle),
        ))
    pairs = [
        {"x1": points[index][0], "y1": points[index][1],
         "x2": points[index + 1][0], "y2": points[index + 1][1]}
        for index in range(n_prim)
    ]
    return {
        "name": name,
        "seg_type": str(seg_type),
        "properties": [
            str(seg_type), "0", str(ibc), str(pos_mat), str(neg_mat)
        ],
        "point_pairs": pairs,
    }


def check_mie_case(label, snapshot, polarization, frequency_ghz,
                   sigma_reference_m, *, max_error_db=0.20,
                   max_isotropy_spread_db=0.05):
    """Run one two-angle monostatic case, print diagnostics, and return bool."""
    try:
        result = solve_monostatic_rcs_2d(
            geometry_snapshot=snapshot,
            frequencies_ghz=[frequency_ghz],
            elevations_deg=[0.0, 37.0],
            polarization=polarization,
            geometry_units="meters",
        )
    except Exception as exc:  # noqa: BLE001 - validator reports solver failures
        print(f"{label:42s} {polarization}  FAIL: "
              f"{type(exc).__name__}: {exc}")
        return False

    samples = result["samples"]
    sigma0 = float(samples[0]["rcs_linear"])
    sigma1 = float(samples[1]["rcs_linear"])
    if not (np.isfinite(sigma0) and np.isfinite(sigma1)
            and sigma0 > 0.0 and sigma1 > 0.0):
        print(f"{label:42s} {polarization}  FAIL: nonpositive/nonfinite RCS")
        return False
    db0 = 10.0 * math.log10(sigma0)
    db1 = 10.0 * math.log10(sigma1)
    reference_db = 10.0 * math.log10(float(sigma_reference_m))
    error = db0 - reference_db
    spread = abs(db0 - db1)
    formulation = result["metadata"]["formulation"]
    passed = (abs(error) <= max_error_db
              and spread <= max_isotropy_spread_db)
    print(
        f"{label:42s} {polarization}  {'PASS' if passed else 'FAIL'}: "
        f"solver={db0:8.3f} dB ref={reference_db:8.3f} dB "
        f"err={error:+7.3f} dB angle-spread={spread:.4f} dB "
        f"[{formulation}]"
    )
    return passed
