import unittest
import copy
import torch


@unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
class CapturedLearner(unittest.TestCase):
    def test_warmup_rollback_and_adam_parity(self):
        from waffle_pi.cuda_learner import StaticLearner
        device=torch.device('cuda',torch.cuda.current_device())
        torch.manual_seed(4)
        a=torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.Tanh(),torch.nn.Linear(8,2)).to(device)
        b=copy.deepcopy(a)
        oa=torch.optim.Adam(a.parameters(),lr=1e-3,capturable=True)
        ob=torch.optim.Adam(b.parameters(),lr=1e-3,capturable=True)
        def loss(model,sample):
            x,y=sample
            l=(model(x)-y).square().mean()
            return l,l.detach().reshape(1)
        graph=StaticLearner(lambda sample:loss(a,sample),oa,a.parameters(),.5,device)
        for k in range(4):
            x,y=torch.randn(7,3,device=device),torch.randn(7,2,device=device)
            ob.zero_grad(set_to_none=True)
            l,stats=loss(b,(x,y))
            l.backward()
            torch.nn.utils.clip_grad_norm_(b.parameters(),.5)
            ob.step()
            torch.testing.assert_close(graph((x,y)),stats,rtol=2e-4,atol=1e-6)
            for p,q in zip(a.parameters(),b.parameters()):
                torch.testing.assert_close(p,q,rtol=2e-4,atol=2e-6)
                self.assertEqual(int(oa.state[p]['step']),k+1)
                torch.testing.assert_close(oa.state[p]['exp_avg'],ob.state[q]['exp_avg'],rtol=2e-4,atol=1e-6)
