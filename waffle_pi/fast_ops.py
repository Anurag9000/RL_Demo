"""Optional fixed-shape PPO kernels preserving episode masks."""
import torch
import torch.nn.functional as F


def masked_gru(gru, x, hxs, masks):
    """Single-layer unidirectional GRU using its existing checkpoint weights.
    x [T,B,A,D], hxs [1,B,A,H], masks [T,B,1].
    Precompute all input projections, then apply per-step reset before GRU.
    """
    t,b,a,_ = x.shape
    h = hxs.reshape(b*a,-1)
    projected = F.linear(x.reshape(t,b*a,-1),gru.weight_ih_l0,gru.bias_ih_l0)
    outputs = []
    for k in range(t):
        h = (h.reshape(b,a,-1)*masks[k].reshape(b,1,1)).reshape(b*a,-1)
        ir,iz,inn = projected[k].chunk(3,-1)
        hr,hz,hn = F.linear(h,gru.weight_hh_l0,gru.bias_hh_l0).chunk(3,-1)
        r,z = torch.sigmoid(ir+hr),torch.sigmoid(iz+hz)
        n = torch.tanh(inn+r*hn)
        h = n+(h-n)*z
        outputs.append(h)
    return torch.stack(outputs).reshape(t,b,a,-1),h.reshape(1,b,a,-1)


def gae_scan(rewards,values,masks,bad_masks,gamma,lam,proper):
    """Inclusive reverse associative affine scan; matches legacy GAE masks."""
    delta = rewards+gamma*values[1:]*masks[1:]-values[:-1]
    coeff = gamma*lam*masks[1:]
    if proper:
        delta,coeff = delta*bad_masks[1:],coeff*bad_masks[1:]
    stride = 1
    while stride < rewards.shape[0]:
        new_delta = torch.cat((delta[:-stride]+coeff[:-stride]*delta[stride:],delta[-stride:]),0)
        coeff = torch.cat((coeff[:-stride]*coeff[stride:],coeff[-stride:]),0)
        delta = new_delta
        stride *= 2
    return delta+values[:-1]
