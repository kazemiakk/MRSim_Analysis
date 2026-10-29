import numpy as np
from scipy.spatial import ConvexHull
from scripts.diagnose_aorta_inlet_signal import prism_counts


def test_prism_counts_excludes_wall_and_upstream():
    h=ConvexHull(np.array([[-1,-1],[1,-1],[1,1],[-1,1]]))
    points=np.array([[0,0,1],[0,0,5],[0,0,-1],[3,0,1],[0,3,5],[0,0,8]],dtype=float)
    n=np.array([0,0,1]);u=np.array([1,0,0]);v=np.array([0,1,0])
    np.testing.assert_array_equal(prism_counts(points,n,0,u,v,h.equations,np.array([0,4,8])),[1,2])


def test_empty_prism_and_translated_plane():
    h=ConvexHull(np.array([[-1,-1],[1,-1],[1,1],[-1,1]]))
    n=np.array([0,0,1]);u=np.array([1,0,0]);v=np.array([0,1,0]);edges=np.array([0,4,8])
    np.testing.assert_array_equal(prism_counts(np.empty((0,3)),n,0,u,v,h.equations,edges),[0,0])
    np.testing.assert_array_equal(prism_counts(np.array([[0,0,6],[0,0,11]]),n,5,u,v,h.equations,edges),[1,1])
