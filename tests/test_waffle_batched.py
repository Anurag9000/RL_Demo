import unittest
try:
    import torch
except ImportError:
    torch = None
from waffle_pi.dynamics import DriveLimits, cartesian_to_wheels, advance_wheels, wheels_to_twist, integrate_pose


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestBatched(unittest.TestCase):
    def test_scalar_parity(self):
        from waffle_pi.batched import step
        limits = DriveLimits()
        pose = torch.tensor([[0.,0.,0.],[1.,2.,2.]],dtype=torch.float64)
        wheels = torch.zeros(2,2,dtype=torch.float64)
        action = torch.tensor([[1.,0.],[-.1,.3]],dtype=torch.float64)
        for _ in range(10):
            expected = []
            for i in range(2):
                target = cartesian_to_wheels(action[i].tolist(),float(pose[i,2]),limits)
                actual = advance_wheels(wheels[i].tolist(),target,.25,limits)
                expected.append(integrate_pose(pose[i].tolist(),wheels_to_twist(actual,limits),.25))
            pose,wheels,_,_ = step(pose,wheels,action,limits,.25)
            torch.testing.assert_close(pose,torch.tensor(expected,dtype=pose.dtype))
