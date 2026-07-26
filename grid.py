#!/usr/bin/env python3
"""
THE OUTPUT GRID -- stated once, used by every component step.

Steps 3a, 3b and 3c each write ONE COMPONENT ALONE onto this grid, and step 4
sums them.  Summing complex amplitudes only means anything if every component
was evaluated at the SAME look directions and frequencies, so the grid is
declared here instead of three times over.  Step 4 still refuses a mismatch --
but that is a backstop, not the mechanism.

Change it here and re-run every 3x step.  A component left over from an older
grid will be REFUSED by step 4, not silently resampled.

FREQUENCIES_GHZ must be a subset of what the body was solved at (step 2a/2b);
the body cannot be interpolated onto a frequency it never saw.
"""

import numpy as np

FREQUENCIES_GHZ = [3.0, 6.0]
AZIMUTHS_DEG = np.arange(0.0, 360.1, 10.0)
# A right-angle wing root peaks on the BISECTOR of the two faces, 45 deg off
# each -- so a narrow elevation band around the horizon can miss the brightest
# thing on the vehicle entirely.  Keep this wide enough to contain the corners
# you have, then narrow it once you know where they are.
ELEVATIONS_DEG = np.arange(-60.0, 60.1, 15.0)

# any subset, in this order.  VH is the cross-pol channel: a bare axisymmetric
# body has none at all in the el = 0 rows, so a non-zero VH there is entirely
# the components.
POLARIZATIONS = ["VV", "HH", "VH"]
