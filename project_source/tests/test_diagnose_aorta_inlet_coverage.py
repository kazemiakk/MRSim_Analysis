import numpy as np
import pytest
from scripts.diagnose_aorta_inlet_coverage import exclusion_categories,rim_margin,trilinear_stencil


def test_signed_rim_distance_for_square():
    eq=np.array([[1,0,-1],[-1,0,-1],[0,1,-1],[0,-1,-1]],dtype=float)
    np.testing.assert_allclose(rim_margin([[0,0],[.8,0],[2,0]],eq),[1,.2,-1])


def test_stencil_weights_and_boundary_guard():
    axes=(np.arange(3,dtype=float),)*3
    loc,w=trilinear_stencil(axes,[[.25,.5,1.5]])
    np.testing.assert_allclose(w.sum(axis=1),1)
    np.testing.assert_allclose((w[:,:,None]*loc).sum(axis=1),[[.25,.5,1.5]])
    with pytest.raises(ValueError,match='outside'):
        trilinear_stencil(axes,[[3,0,0]])


def test_exclusion_classes_partition_without_promoting_point_signal():
    parts=exclusion_categories([True,False,False],[True,True,False])
    np.testing.assert_array_equal(np.sum(list(parts.values()),axis=0),1)
    np.testing.assert_array_equal(parts['accepted_complete_stencil'],[True,False,False])
