"""Checkpoint compatibility with modern PyTorch's restricted loading."""
import argparse
import inspect
import pickle
import torch


class NamespaceUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('argparse','Namespace'):
            return argparse.Namespace
        raise pickle.UnpicklingError('Unexpected GST configuration class: '+module+'.'+name)


def load_gst_args(path):
    with open(path,'rb') as handle:
        obj=NamespaceUnpickler(handle).load()
    if not isinstance(obj,argparse.Namespace):
        raise ValueError('Expected argparse.Namespace in released GST config')
    return obj


def load_weights(path,device,*,gst_metadata=False):
    # The published checkpoints use tensors and primitive metadata.
    # Old PyTorch lacks weights_only; only use repository-trusted weights there.
    kwargs={'weights_only':True} if 'weights_only' in inspect.signature(torch.load).parameters else {}
    if gst_metadata and kwargs and hasattr(torch.serialization,'safe_globals'):
        import numpy as np
        # Released GST training snapshots contain NumPy scalar loss metadata.
        # Scope the allowlist to this call; no unrestricted pickle fallback.
        scalar = np.core.multiarray.scalar if int(np.__version__.split('.')[0])<2 else np._core.multiarray.scalar
        allowed=[scalar,np.dtype,*[type(np.dtype(k)) for k in ('float32','float64','int32','int64','bool')]]
        # PyTorch >=2.6 supports explicit qualified-name aliases for old
        # NumPy paths. The documented NumPy 1.x stack does not need one.
        if scalar.__module__!='numpy.core.multiarray':
            version=tuple(map(int,torch.__version__.split('+')[0].split('.')[:2]))
            if version>=(2,6):
                allowed.append((scalar,'numpy.core.multiarray.scalar'))
        with torch.serialization.safe_globals(allowed):
            return torch.load(path,map_location=device,**kwargs)
    return torch.load(path,map_location=device,**kwargs)
