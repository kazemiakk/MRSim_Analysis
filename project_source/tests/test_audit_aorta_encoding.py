import numpy as np
import pytest
from scripts.audit_aorta_encoding import trapezoid_moments


def test_trapezoid_area_and_centroid():
    area,moment=trapezoid_moments(2,1,2,1,3,20)
    assert area==pytest.approx(6)
    assert moment/area==pytest.approx(5)


def test_positive_then_negative_bipolar_moment():
    a=trapezoid_moments(2,1,0,1,0,4)
    b=trapezoid_moments(-2,1,0,1,2,4)
    np.testing.assert_allclose(a+b,[0,-4],atol=1e-14)


def test_partial_trapezoid_at_adc():
    area,moment=trapezoid_moments(2,1,2,1,0,.5)
    assert area==pytest.approx(.25)
    assert moment==pytest.approx(1/12)
