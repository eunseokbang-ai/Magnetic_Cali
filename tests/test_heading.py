import numpy as np
import pytest

from mag_heading.heading import compute_heading, unit_vector


def test_theta_on_axes():
    theta, _ = compute_heading(0, 0, 1)
    assert theta[0] == pytest.approx(0.0)

    theta, _ = compute_heading(0, 0, -1)
    assert theta[0] == pytest.approx(np.pi)

    theta, _ = compute_heading(1, 0, 0)
    assert theta[0] == pytest.approx(np.pi / 2)


def test_phi_in_xy_plane():
    _, phi = compute_heading(1, 0, 0)
    assert phi[0] == pytest.approx(0.0)

    _, phi = compute_heading(0, 1, 0)
    assert phi[0] == pytest.approx(np.pi / 2)

    _, phi = compute_heading(-1, 0, 0)
    assert phi[0] == pytest.approx(np.pi)


def test_magnitude_invariance():
    theta1, phi1 = compute_heading(3, 4, 5)
    theta2, phi2 = compute_heading(30, 40, 50)
    assert theta1[0] == pytest.approx(theta2[0])
    assert phi1[0] == pytest.approx(phi2[0])


def test_zero_vector_raises():
    with pytest.raises(ValueError):
        compute_heading(0, 0, 0)


def test_unit_vector_round_trip():
    rng = np.random.default_rng(0)
    cx, cy, cz = rng.normal(size=(3, 50))
    theta, phi = compute_heading(cx, cy, cz)
    uv = unit_vector(theta, phi)

    norm = np.sqrt(cx**2 + cy**2 + cz**2)
    expected = np.stack([cx, cy, cz], axis=-1) / norm[:, None]
    np.testing.assert_allclose(uv, expected, atol=1e-10)


def test_vectorized_matches_scalar():
    cx = np.array([1.0, 0.0, -1.0])
    cy = np.array([0.0, 1.0, 0.0])
    cz = np.array([0.0, 0.0, 0.0])
    theta, phi = compute_heading(cx, cy, cz)
    for i in range(3):
        t_i, p_i = compute_heading(cx[i], cy[i], cz[i])
        assert theta[i] == pytest.approx(t_i[0])
        assert phi[i] == pytest.approx(p_i[0])
