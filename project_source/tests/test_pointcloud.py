"""Tests for approximate k-NN point-cloud flow interpolation."""

import numpy as np

from mrsimtracks.pointcloud import PointCloudFlow


def make_flow(*, start_frame=0):
    axis = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    coordinates = np.array(
        np.meshgrid(axis, axis, axis, indexing="ij"), dtype=np.float32
    ).reshape(3, -1).T
    velocities = np.empty((len(coordinates), 3, 3), dtype=np.float32)
    velocities[:, :, 0] = (1.0, 0.0, 0.0)
    velocities[:, :, 1] = (3.0, 2.0, 0.0)
    velocities[:, :, 2] = (1.0, 0.0, 0.0)
    return PointCloudFlow(
        coordinates,
        velocities,
        frame_dt=0.1,
        k=4,
        support_radius=0.3,
        start_frame=start_frame,
    )


def test_exact_nodes_and_temporal_interpolation():
    flow = make_flow()
    points = flow.coordinates[[0, 31, 80]]
    velocity, valid, nearest = flow.sample_v(points, time=0.05)

    assert valid.all()
    assert (nearest >= 0).all()
    np.testing.assert_allclose(velocity, [[2.0, 1.0, 0.0]] * 3, atol=1e-5)


def test_start_frame_and_periodic_wrap():
    flow = make_flow(start_frame=2)
    point = flow.coordinates[[20]]
    at_start, _, _ = flow.sample_v(point, time=0.0)
    after_wrap, _, _ = flow.sample_v(point, time=0.05)

    np.testing.assert_allclose(at_start, [[1.0, 0.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(after_wrap, [[1.0, 0.0, 0.0]], atol=1e-6)


def test_far_points_are_invalid():
    flow = make_flow()
    velocity, valid, nearest = flow.sample_v([[9.0, 9.0, 9.0]], time=0.0)

    assert not valid[0]
    assert nearest[0] == -1
    np.testing.assert_array_equal(velocity, 0.0)


def test_seed_points_are_repeatable_and_moving():
    flow = make_flow()
    a = flow.seed_points(20, rng=np.random.default_rng(7))
    b = flow.seed_points(20, rng=np.random.default_rng(7))

    np.testing.assert_array_equal(a, b)
    velocity, valid, _ = flow.sample_v(a, time=0.0)
    assert valid.all()
    assert (np.linalg.norm(velocity, axis=1) > 0).all()
