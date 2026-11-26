import unittest
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'PyTorch not installed')
class FastOps(unittest.TestCase):
    def test_gae(self):
        from waffle_pi.fast_ops import gae_scan
        torch.manual_seed(42)
        r=torch.randn(17,5,1)
        v=torch.randn(18,5,1)
        m=(torch.rand(18,5,1)>.2).float()
        b=(torch.rand(18,5,1)>.1).float()
        for proper in (False,True):
            a=torch.zeros_like(r[0])
            expected=torch.empty_like(r)
            for k in reversed(range(len(r))):
                a=r[k]+.99*v[k+1]*m[k+1]-v[k]+.99*.95*m[k+1]*a
                if proper:
                    a=a*b[k+1]
                expected[k]=a+v[k]
            torch.testing.assert_close(gae_scan(r,v,m,b,.99,.95,proper),expected)
    def test_gru_outputs_and_gradients(self):
        from waffle_pi.fast_ops import masked_gru
        torch.manual_seed(4)
        gru=torch.nn.GRU(7,9).double()
        x=torch.randn(5,3,2,7,dtype=torch.double,requires_grad=True)
        h=torch.randn(1,3,2,9,dtype=torch.double)
        m=(torch.rand(5,3,1)>.3).double()
        out,hidden=masked_gru(gru,x,h,m)
        reference=[]
        refh=h
        for k in range(5):
            y,refh=gru(x[k:k+1].reshape(1,6,7),(refh*m[k].reshape(1,3,1,1)).reshape(1,6,9))
            refh=refh.reshape(1,3,2,9)
            reference.append(y.reshape(1,3,2,9))
        ref=torch.cat(reference)
        torch.testing.assert_close(out,ref)
        torch.testing.assert_close(hidden,refh)
        params=(x,*gru.parameters())
        actual=torch.autograd.grad(out.square().sum(),params,retain_graph=True)
        expected=torch.autograd.grad(ref.square().sum(),params)
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b)
