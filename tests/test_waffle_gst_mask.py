import unittest
import torch
from gst_updated.scripts.wrapper.crowd_nav_interface_parallel import displacement_mask


class GSTMask(unittest.TestCase):
    def test_adjacent_visibility(self):
        v=torch.tensor([[[1.,0.,1.,1.,1.],[1.,1.,1.,1.,0.]]])
        adjacent=displacement_mask(v,True)
        torch.testing.assert_close(adjacent,torch.tensor([[[1.,0.,0.,1.,1.],[1.,1.,1.,1.,0.]]]))
        torch.testing.assert_close(displacement_mask(v),torch.tensor([[[1.,1.,0.,1.,1.],[1.,0.,0.,0.,0.]]]))
    def test_all_visible_identical(self):
        v=torch.ones(3,4,5)
        torch.testing.assert_close(displacement_mask(v),displacement_mask(v,True))
