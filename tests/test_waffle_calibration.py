import unittest
import numpy as np
from waffle_pi.calibration import fit_motor,fit_detector


class Calibration(unittest.TestCase):
    def test_motor_synthetic(self):
        t=np.arange(0,12,.05)
        cmd=.12*(np.sin(2*t)>0)
        speed=np.zeros_like(cmd)
        for i in range(1,len(t)):
            speed[i]=speed[i-1]+(1-np.exp(-.05/.2))*(cmd[i-1]-speed[i-1])
        data=np.column_stack((t,cmd,np.zeros_like(t),speed,np.zeros_like(t)))
        result=fit_motor(data)
        self.assertLess(result['holdout_rmse_rad_s'],.3)
        self.assertFalse(result['safety_validated'])
    def test_detector(self):
        result=fit_detector(np.array([[.5,1.,0.,0.,.1,0.],[.5,0.,0.,0.,0.,0.]]))
        self.assertAlmostEqual(result['recall'],.5)
        self.assertAlmostEqual(result['bins'][0]['rms_error_m'],.1/np.sqrt(2))
