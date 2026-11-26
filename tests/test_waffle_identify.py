import unittest
import numpy as np
from waffle_pi.identify import fit_geometry,fit_laser,fit_detector_bins,wheel_predict,fit_wheels


class Identification(unittest.TestCase):
    def test_geometry(self):
        rng=np.random.default_rng(4)
        w=rng.uniform(-3,3,(100,2))
        g=w*np.array([.032,.034])
        x=np.column_stack((w,g.sum(1)/2,(g[:,1]-g[:,0])/.29))
        result=fit_geometry(x[:60],x[60:])
        np.testing.assert_allclose(result['motor']['radius_m'],[.032,.034],atol=1e-6)
        self.assertAlmostEqual(result['motor']['track_m'],.29,places=5)
    def test_laser(self):
        x=np.column_stack((np.ones(40),np.ones(40)*1.02,np.ones(40)))
        result=fit_laser(x,x.copy())
        self.assertAlmostEqual(result['laser']['range_bias_m'],.02)
    def test_detector(self):
        x=np.zeros((40,6))
        x[:,0]=.5
        x[:,1]=1.
        x[:,4]=.04
        result=fit_detector_bins(x,x.copy(),1.)
        self.assertAlmostEqual(result['detector_bins'][0]['bias_m'][0],.04)
    def test_motor(self):
        t=np.arange(100)*.05
        a=np.column_stack((t,2*np.sign(np.sin(t*3)),2*np.sign(np.cos(t*3)),np.zeros((100,2))))
        b=np.column_stack((t,2*np.sign(np.cos(t*4)),2*np.sign(np.sin(t*4)),np.zeros((100,2))))
        p=[[.1,12.,15.,1.,0.],[.2,15.,18.,.9,0.]]
        for x in (a,b):
            x[:,3:]=wheel_predict(x,p,0,.05)
        result=fit_wheels(a,b,.05,max_delay=0.)
        self.assertLess(max(result['holdout_rmse_rad_s']),.05)
