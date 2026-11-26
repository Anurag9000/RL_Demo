"""Optional CUDA FP32 GAE kernel, one thread per environment."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gae(R,V,M,B,O,N:tl.constexpr,T:tl.constexpr,G:tl.constexpr,L:tl.constexpr,
         PROPER:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=i<N
    a=tl.full((BLOCK,),0.,tl.float32)
    nxt=tl.load(V+T*N+i,valid,0.)
    for k in range(T-1,-1,-1):
        v=tl.load(V+k*N+i,valid,0.)
        r=tl.load(R+k*N+i,valid,0.)
        m=tl.load(M+(k+1)*N+i,valid,0.)
        a=r+G*nxt*m-v+G*L*m*a
        if PROPER:
            a=a*tl.load(B+(k+1)*N+i,valid,0.)
        tl.store(O+k*N+i,a+v,valid)
        nxt=v


def gae_triton(rewards,values,masks,bad_masks,gamma,lam,proper):
    if not rewards.is_cuda or any(t.dtype!=torch.float32 or not t.is_contiguous()
            for t in (rewards,values,masks,bad_masks)):
        raise ValueError("Triton GAE requires contiguous CUDA FP32 tensors")
    t,n,_=rewards.shape
    out=torch.empty_like(rewards)
    _gae[(triton.cdiv(n,128),)](rewards,values,masks,bad_masks,out,n,t,gamma,lam,proper,128)
    return out
