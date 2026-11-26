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


def load_weights(path,device):
    # The published checkpoints use tensors and primitive metadata.
    # Old PyTorch lacks weights_only; only use repository-trusted weights there.
    kwargs={'weights_only':True} if 'weights_only' in inspect.signature(torch.load).parameters else {}
    return torch.load(path,map_location=device,**kwargs)
