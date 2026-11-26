import unittest
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'PyTorch not installed')
class Orca(unittest.TestCase):
    def test_no_neighbors(self):
        from waffle_pi.orca_tensor import orca_velocity
        p=torch.zeros(3,1,2)
        v=torch.zeros_like(p)
        pref=torch.tensor([[[2.,0.]],[[0.,.5]],[[0.,0.]]])
        out=orca_velocity(p,v,torch.full((3,1),.3),pref,torch.ones(3,1))
        torch.testing.assert_close(out,torch.tensor([[[1.,0.]],[[0.,.5]],[[0.,0.]]]))
    def test_head_on_and_coincident(self):
        from waffle_pi.orca_tensor import orca_velocity
        p=torch.tensor([[[-1.,0.],[1.,0.]],[[0.,0.],[0.,0.]]])
        v=torch.zeros_like(p)
        pref=torch.tensor([[[1.,0.],[-1.,0.]],[[1.,0.],[-1.,0.]]])
        out=orca_velocity(p,v,torch.full((2,2),.3),pref,torch.ones(2,2))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue((out.norm(dim=-1)<=1.00001).all())
        self.assertLess(float(out[0,0,0]),1.)
        torch.testing.assert_close(out[:,0],-out[:,1])
    def test_native_rvo2_optional(self):
        try:
            import rvo2
        except ImportError:
            self.skipTest('Python-RVO2 not installed')
        from waffle_pi.orca_tensor import orca_velocity
        p=torch.tensor([[[-2.,0.],[1.,.4],[.5,-1.3]]])
        v=torch.tensor([[[.2,.1],[-.4,.2],[.2,-.4]]])
        pref=torch.tensor([[[.9,.1],[-.8,.2],[.1,.9]]])
        rad=torch.tensor([[.3,.25,.35]])
        max_speed=torch.tensor([[1.,.8,1.2]])
        sim=rvo2.PyRVOSimulator(.25,10.,3,5.,5.,.3,1.)
        for i in range(3):
            sim.addAgent(tuple(p[0,i].tolist()),10.,3,5.,5.,float(rad[0,i])+.16,
                         float(max_speed[0,i]),tuple(v[0,i].tolist()))
            sim.setAgentPrefVelocity(i,tuple(pref[0,i].tolist()))
        sim.doStep()
        native=torch.tensor([[sim.getAgentVelocity(i) for i in range(3)]])
        out=orca_velocity(p,v,rad,pref,max_speed)
        torch.testing.assert_close(out,native,atol=2e-4,rtol=2e-4)
