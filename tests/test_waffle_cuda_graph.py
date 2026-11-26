import unittest
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None or not torch.cuda.is_available(),'CUDA unavailable')
class Graph(unittest.TestCase):
    def test_replay_updates_inputs_and_weights(self):
        from waffle_pi.cuda_graph import StaticGraphCall
        model=torch.nn.Linear(3,2).cuda().eval()
        graph=StaticGraphCall(lambda d:model(d['x']),torch.device('cuda',0))
        x=torch.randn(4,3,device='cuda')
        torch.testing.assert_close(graph({'x':x}),model(x))
        with torch.no_grad():
            model.weight.add_(.25)
        y=torch.randn_like(x)
        torch.testing.assert_close(graph({'x':y}),model(y))
    def test_random_sampling_advances(self):
        from waffle_pi.cuda_graph import StaticGraphCall
        graph=StaticGraphCall(lambda x:x+torch.randn_like(x),torch.device('cuda',0))
        x=torch.zeros(64,device='cuda')
        a=graph(x).clone()
        b=graph(x).clone()
        self.assertFalse(torch.equal(a,b))
