import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from arguments import get_args
from waffle_pi.training_state import save,load


class State(unittest.TestCase):
    def test_adam_and_rng_roundtrip(self):
        args=get_args(['--no-cuda','--waffle-pi','--waffle-native-action','--waffle-live-orca'])
        device=torch.device('cpu')
        a=torch.nn.Linear(2,1)
        def agent(model):
            return SimpleNamespace(optimizer=torch.optim.Adam(model.parameters(),lr=.0001),
                                   entropy_coef=.01,clip_param=.2)
        opt=agent(a)
        opt.optimizer.zero_grad()
        a(torch.ones(2,2)).sum().backward()
        opt.optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.pt'
            save(path,a,opt,args,1,16,device)
            expected_rng=torch.rand(4)
            b=torch.nn.Linear(2,1)
            resumed=agent(b)
            self.assertEqual(load(path,b,resumed,args,device),(1,16))
            torch.testing.assert_close(torch.rand(4),expected_rng)
            for p,q in zip(a.parameters(),b.parameters()):
                torch.testing.assert_close(p,q)
                for k in opt.optimizer.state[p]:
                    torch.testing.assert_close(opt.optimizer.state[p][k],resumed.optimizer.state[q][k])
            # Same next minibatch gives the same Adam update.
            for model,learner in ((a,opt),(b,resumed)):
                learner.optimizer.zero_grad()
                model(torch.tensor([[.2,.7]])).square().mean().backward()
                learner.optimizer.step()
            for p,q in zip(a.parameters(),b.parameters()):
                torch.testing.assert_close(p,q)
            args.retune_optimizer=True
            args.lr=.0002
            args.entropy_coef=.02
            load(path,b,resumed,args,device)
            self.assertEqual(resumed.optimizer.param_groups[0]['lr'],.0002)
            self.assertEqual(resumed.entropy_coef,.02)
            self.assertEqual(float(resumed.optimizer.state[next(b.parameters())]['step']),1.)
