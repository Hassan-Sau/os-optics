"""Diffraction near a focus: the one place the Engine computes focal fields and point spread functions.

* ``objective`` -- the ideal aplanatic objective, its reference sphere and a window between it and the focus;
* ``quadrature`` -- samplings of the reference sphere (polar Gauss-Legendre rule, Cartesian grid of direction sines);
* ``debye`` -- the vector Debye integral by direct summation and by matrix Fourier transform;
* ``bessel`` -- the same integral for a rotationally symmetric pupil, reduced to Bessel integrals;
* ``focus`` -- a pupil's focal field through these evaluations, its powers and plane integrals;
* ``metrics`` -- peak, widths, encircled energy and Zernike polynomials.

Results-side analyses (``results.focal_field``) are built on it; no run's result depends on it.
"""

from .focus import BESSEL, DIRECT, GRID, Focus, PupilMode, Sampling
from .objective import Objective, Window

__all__ = ["BESSEL", "DIRECT", "Focus", "GRID", "Objective", "PupilMode", "Sampling", "Window"]
