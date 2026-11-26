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
