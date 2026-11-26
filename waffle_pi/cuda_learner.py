"""Explicit fixed-shape PPO forward/backward/Adam CUDA Graph.

Experimental, single GPU only. Capture requires a fresh Adam optimizer with
capturable=True and constant hyperparameters. Warmup is rolled back in-place,
including optimizer moments and step counters. No implicit eager fallback.
"""
import torch
from .cuda_graph import map_tree,copy_tree


class StaticLearner:
    def __init__(self,loss_fn,optimizer,params,max_grad_norm,device,warmup=3):
        self.loss_fn,self.optimizer,self.params=loss_fn,optimizer,list(params)
        self.max_grad_norm,self.device,self.warmup=max_grad_norm,torch.device(device),warmup
        if (self.device.type!='cuda' or not self.params or warmup<1
                or any(p.device!=self.device for p in self.params)):
            raise ValueError("Captured learner requires parameters on one CUDA device")
        if not isinstance(optimizer,torch.optim.Adam) or any(not g.get('capturable') for g in optimizer.param_groups):
            raise ValueError("Expected capturable Adam")
        if optimizer.state:
            raise ValueError("Capture starts from a fresh optimizer")
        self.graph=None

    def _step(self):
        # Network forward mutates recurrent hidden dictionaries, so give it
        # fresh containers referencing the static input tensors.
        sample=map_tree(lambda x:x,self.static)
        loss,stats=self.loss_fn(sample)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params,self.max_grad_norm)
        self.optimizer.step()
        return stats

    @torch.no_grad()
    def _restore(self,snapshot):
        for p,saved in zip(self.params,snapshot):
            p.copy_(saved)
        # Warmup starts from empty Adam state. Keep allocations (and graph
        # pointers) but undo every warmup update and restore moment counters.
        for state in self.optimizer.state.values():
            for name,value in state.items():
                if not torch.is_tensor(value):
                    raise RuntimeError("Unsupported non-tensor Adam state: "+name)
                value.zero_()

    def __call__(self,sample):
        if self.graph is None:
            def clone(x):
                if not torch.is_tensor(x) or x.device!=self.device:
                    raise ValueError("Captured learner input device mismatch")
                return x.detach().clone()
            with torch.cuda.device(self.device):
                self.static=map_tree(clone,sample)
                snapshot=[p.detach().clone() for p in self.params]
                rng=torch.cuda.get_rng_state(self.device)
                side=torch.cuda.Stream(device=self.device)
                side.wait_stream(torch.cuda.current_stream(self.device))
                try:
                    with torch.cuda.stream(side),torch.enable_grad():
                        for _ in range(self.warmup):
                            self.optimizer.zero_grad(set_to_none=True)
                            self._step()
                        self._restore(snapshot)
                    torch.cuda.current_stream(self.device).wait_stream(side)
                    torch.cuda.synchronize(self.device)
                    torch.cuda.set_rng_state(rng,self.device)
                    self.optimizer.zero_grad(set_to_none=True)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=side),torch.enable_grad():
                        self.result=self._step()
                    # Preserve tensor addresses. Ensure capture itself cannot
                    # contribute an extra update before the first real replay.
                    self._restore(snapshot)
                    self.graph=graph
                except Exception as exc:
                    raise RuntimeError("PPO CUDA graph capture failed; restart without --cuda-graph-ppo") from exc
        else:
            copy_tree(self.static,sample)
        self.graph.replay()
        return self.result
