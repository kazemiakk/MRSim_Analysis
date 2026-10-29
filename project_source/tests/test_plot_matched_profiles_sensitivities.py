import numpy as np
import pytest
from scripts.plot_matched_profiles_sensitivities import weighted_stats,compact_derivatives,main
from scripts.plot_matched_profiles_sensitivities import frame
import matplotlib.pyplot as plt


def test_weighted_mean_and_percentiles_not_unweighted():
    result=weighted_stats([1.,10.,100.],[98.,1.,1.])
    assert result['mean']==pytest.approx(2.08)
    assert result['p95']==1 and result['p99']==10 and result['n']==3


def test_empty_and_missing_statistics_are_unavailable():
    assert weighted_stats([np.nan,10],[1,0])==dict(mean=None,p95=None,p99=None,weight=0.,n=0)
    with pytest.raises(ValueError):weighted_stats([1,2],[1])


def test_compact_derivatives_analytic_anisotropic_rotation():
    d=np.array([.1,.2,.3]);z,y,x=np.indices((5,6,7))*d[::-1,None,None,None]
    u=np.stack((-2*y,2*x,np.zeros_like(x)),axis=-1)
    values,mask=compact_derivatives(u,d,np.ones(u.shape[:-1],bool))
    assert mask.sum()==60
    np.testing.assert_allclose(values['vorticity_magnitude_s_inv'],4,atol=1e-12)
    np.testing.assert_allclose(values['divergence_s_inv'],0,atol=1e-12)
    np.testing.assert_allclose(values['strain_frobenius_s_inv'],0,atol=1e-12)
    np.testing.assert_allclose(values['dissipation_per_viscosity_s_inv2'],0,atol=1e-12)


def test_compact_derivatives_linear_strain_and_empty():
    z,y,x=np.indices((5,5,5),dtype=float)
    u=np.stack((x,2*y,-3*z),axis=-1)
    values,mask=compact_derivatives(u,np.ones(3),np.ones(u.shape[:-1],bool))
    np.testing.assert_allclose(values['strain_frobenius_s_inv'],np.sqrt(14))
    np.testing.assert_allclose(values['dissipation_per_viscosity_s_inv2'],28)
    empty=np.zeros_like(mask);values,mask=compact_derivatives(u,np.ones(3),empty)
    assert not mask.any() and all(len(x)==0 for x in values.values())


def test_existing_completed_target_is_preserved(tmp_path):
    with pytest.raises(FileExistsError):main(tmp_path)


def test_actual_method_and_threshold_legends_are_concise():
    fig=plt.figure();frame(fig,{'name':'phantom_1m'},'test')
    assert [t.get_text() for t in fig.legends[0].get_texts()]==['CFD','Koma']
    plt.close(fig)
    fig=plt.figure();frame(fig,{'name':'aorta_1m'},'test',thresholds=True)
    assert [t.get_text() for t in fig.legends[0].get_texts()]==['25%','50%','75%']
    plt.close(fig)
