"""Opt-in static CUDA graph replay for inference, not optimizer updates.

Outputs refer to reused device buffers and must be consumed before the
next call. Model parameter addresses must remain stable. Default CUDA RNG
is captured by PyTorch; warmup changes the original random sequence.
"""
import torch


def map_tree(fn,obj):
    if isinstance(obj,dict):
        return {k:map_tree(fn,v) for k,v in obj.items()}
    if isinstance(obj,tuple):
        return tuple(map_tree(fn,v) for v in obj)
    if isinstance(obj,list):
        return [map_tree(fn,v) for v in obj]
    return fn(obj)


def copy_tree(dst,src):
    if isinstance(dst,dict):
        if dst.keys()!=src.keys():
            raise ValueError("CUDA graph dictionary keys changed")
        for k in dst:
            copy_tree(dst[k],src[k])
    elif isinstance(dst,(tuple,list)):
        if len(dst)!=len(src):
            raise ValueError("CUDA graph input arity changed")
        for d,s in zip(dst,src):
            copy_tree(d,s)
    else:
        if (dst.shape!=src.shape or dst.dtype!=src.dtype or dst.device!=src.device):
            raise ValueError("CUDA graph input shape, dtype or device changed")
        dst.copy_(src)


class StaticGraphCall:
    def __init__(self,fn,device,amp_bf16=False,warmup=3):
        self.fn,self.device=fn,torch.device(device)
        if self.device.type!='cuda' or not torch.cuda.is_available():
            raise ValueError("Explicit CUDA graph capture requires CUDA")
        self.amp_bf16,self.warmup=amp_bf16,warmup
        self.graph=None

    def _forward(self):
        # Rebuild dict containers because Policy.act updates hidden-state keys.
        args=map_tree(lambda x:x,self.static)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,
                                           enabled=self.amp_bf16,cache_enabled=False):
            return self.fn(*args)

    def __call__(self,*args):
        if self.graph is None:
            def clone(x):
                if not torch.is_tensor(x) or x.device.type!='cuda':
                    raise ValueError("CUDA graph inputs must be CUDA tensors")
                return x.detach().clone()
            with torch.cuda.device(self.device):
                self.static=map_tree(clone,args)
                side=torch.cuda.Stream(device=self.device)
                side.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(side):
                    for _ in range(self.warmup):
                        self._forward()
                torch.cuda.current_stream(self.device).wait_stream(side)
                torch.cuda.synchronize(self.device)
                graph=torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph,stream=side):
                        self.result=self._forward()
                except Exception as exc:
                    raise RuntimeError("CUDA graph capture failed; retry without explicit graph flag") from exc
                self.graph=graph
        else:
            copy_tree(self.static,args)
        self.graph.replay()
        return self.result
