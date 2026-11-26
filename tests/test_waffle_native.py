import tempfile
import json
import unittest
from pathlib import Path
import torch
from waffle_pi.native import native_twist,check_checkpoint,safety_parameters
from waffle_pi.dynamics import DriveLimits,wheels_to_twist
from waffle_pi.batched import native_targets,approximate_gate


class NativeAction(unittest.TestCase):
    def test_scalar_tensor(self):
        lim=DriveLimits(max_linear=.1,max_angular=.5)
        a=torch.tensor([[2.,-1.],[-1.,3.],[0.,0.]])
        targets=native_targets(a,lim)
        for z,wheels in zip(a,targets):
            v,w=wheels_to_twist(wheels.tolist(),lim)
            expected=native_twist(z.tolist(),lim)
            self.assertAlmostEqual(v,expected[0],places=6)
            self.assertAlmostEqual(w,expected[1],places=6)
    def test_gate(self):
        lim=DriveLimits(max_linear=.1,max_angular=.5)
        pose=torch.zeros(2,3)
        wheels=torch.zeros(2,2)
        target=native_targets(torch.ones(2,2),lim)
        humans=torch.tensor([[[.3,0.,.25]],[[2.,0.,.25]]])
        result,blocked=approximate_gate(pose,wheels,target,humans,2.5,lim,safety_parameters(None,.25))
        self.assertEqual(blocked.tolist(),[True,False])
        torch.testing.assert_close(result[0],torch.zeros(2))
        torch.testing.assert_close(result[1],target[1])
    def test_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            checkpoint=root/'checkpoints'/'p.pt'
            checkpoint.parent.mkdir()
            with self.assertRaises(ValueError):
                check_checkpoint(checkpoint,True)
            (root/'run_args.json').write_text(json.dumps({'waffle_native_action':True,'waffle_speed':.1}))
            check_checkpoint(checkpoint,True,{'waffle_speed':.1})
            with self.assertRaises(ValueError):
                check_checkpoint(checkpoint,False)
