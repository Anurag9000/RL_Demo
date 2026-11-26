import math
import unittest
import numpy as np
from waffle_pi.scan_geometry import ScanGeometry


class ScanGeometryTest(unittest.TestCase):
    def test_actual_ray_alignment(self):
        for start,step in ((0.,math.pi/180),(-math.pi,math.pi/180),(1.,-math.pi/180)):
            n=360
            geom=ScanGeometry.from_metadata(n,start,start+(n-1)*step,step)
            phi=np.linspace(-math.radians(geom.fov_deg)/2,math.radians(geom.fov_deg)/2,n)
            points=np.stack((np.cos(phi),np.sin(phi)),-1)
            actual=geom.detections_to_ros(points)
            raw_angles=start+np.arange(n)*step
            if step>0:
                raw_angles=raw_angles[::-1]
            expected=np.stack((np.cos(raw_angles),np.sin(raw_angles)),-1)
            np.testing.assert_allclose(actual,expected,atol=2e-6)
    def test_scan_cleanup_and_reverse(self):
        g=ScanGeometry.from_metadata(360,0.,359*math.pi/180,math.pi/180)
        x=np.ones(360)
        x[0],x[1],x[2],x[3]=2.,float('inf'),float('nan'),.01
        out=g.preprocess(x,.12,3.5)
        self.assertEqual(out[-1],2.)
        np.testing.assert_allclose(out[-4:-1],[29.99]*3)
        with self.assertRaises(ValueError):
            g.preprocess(np.full(360,np.inf),.12,3.5)
    def test_duplicate_endpoint_and_bad_metadata(self):
        g=ScanGeometry.from_metadata(361,0.,2*math.pi,math.pi/180)
        self.assertEqual(g.count,360)
        self.assertTrue(g.panoramic)
        self.assertEqual(len(g.preprocess(np.ones(361),.12,3.5)),360)
        with self.assertRaises(ValueError):
            ScanGeometry.from_metadata(360,0.,1.,math.pi/180)
