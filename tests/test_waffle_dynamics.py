import math
import unittest
from waffle_pi.dynamics import DriveLimits, cartesian_to_wheels, wheels_to_twist, command_twist, advance_wheels, integrate_pose


class TestDynamics(unittest.TestCase):
    def test_straight(self):
        limits = DriveLimits()
        v,w = command_twist((1.,0.),0.,limits)
        self.assertAlmostEqual(v,0.26)
        self.assertAlmostEqual(w,0.)
        self.assertAlmostEqual(integrate_pose((0.,0.,0.),(v,w),.25)[0],.065)

    def test_turn(self):
        v,w = command_twist((0.,1.),0.,DriveLimits())
        self.assertAlmostEqual(v,0.)
        self.assertGreater(w,0.)

    def test_bounds(self):
        limits = DriveLimits()
        for i in range(360):
            a = math.radians(i)
            wheels = cartesian_to_wheels((math.cos(a),math.sin(a)),0.,limits)
            self.assertLessEqual(max(map(abs,wheels)),limits.max_wheel+1e-10)
            v,w = wheels_to_twist(wheels,limits)
            self.assertLessEqual(abs(v),limits.max_linear+1e-10)
            self.assertLessEqual(abs(w),limits.max_angular+1e-10)

    def test_accel_and_invalid(self):
        limits = DriveLimits()
        self.assertLessEqual(max(map(abs,advance_wheels((0.,0.),(7.,-7.),.05,limits))),.600001)
        with self.assertRaises(ValueError):
            command_twist((float('nan'),0.),0.,limits)


if __name__ == '__main__':
    unittest.main()
