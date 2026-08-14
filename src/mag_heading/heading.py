import numpy as np


def compute_heading(cx, cy, cz):
    """(theta, phi) in radians from compass vector (cx, cy, cz).

    theta: angle between B and the z-axis (polar), in [0, pi].
    phi: angle between the x-y projection of B and the x-axis (azimuth), in (-pi, pi].
    Matches the MagArrow heading definition in the vendor guide appendix (Figure 17).
    """
    cx = np.atleast_1d(np.asarray(cx, dtype=float))
    cy = np.atleast_1d(np.asarray(cy, dtype=float))
    cz = np.atleast_1d(np.asarray(cz, dtype=float))

    magnitude = np.sqrt(cx**2 + cy**2 + cz**2)
    if np.any(magnitude == 0):
        raise ValueError("compass vector magnitude is zero; cannot compute heading")

    theta = np.arccos(np.clip(cz / magnitude, -1.0, 1.0))
    phi = np.arctan2(cy, cx)
    return theta, phi


def unit_vector(theta, phi):
    """Cartesian unit-sphere embedding of (theta, phi).

    Used instead of raw (theta, phi) for interpolation/nearest-neighbor distance,
    since the embedding is free of the phi wraparound (0/2pi) and pole (theta=0/pi)
    discontinuities that would otherwise distort angular distances.
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=-1)
