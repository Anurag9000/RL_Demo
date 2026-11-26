import argparse
import io
import pickle
import tempfile
import unittest
from pathlib import Path
from waffle_pi.checkpoints import NamespaceUnpickler,load_gst_args


class CheckpointConfig(unittest.TestCase):
    def test_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'args.pickle'
            p.write_bytes(pickle.dumps(argparse.Namespace(obs_seq_len=5)))
            self.assertEqual(load_gst_args(p).obs_seq_len,5)
    def test_reject_unexpected_class(self):
        with self.assertRaises(pickle.UnpicklingError):
            NamespaceUnpickler(io.BytesIO(pickle.dumps(Path('.')))).load()

    def test_gst_scalar_metadata(self):
        import numpy as np
        import torch
        from waffle_pi.checkpoints import load_weights
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.pt'
            torch.save({'model_state_dict':{'w':torch.ones(2)},'train_loss_epoch':np.float64(.4)},p)
            checkpoint=load_weights(p,'cpu',gst_metadata=True)
            torch.testing.assert_close(checkpoint['model_state_dict']['w'],torch.ones(2))
            self.assertAlmostEqual(float(checkpoint['train_loss_epoch']),.4)
