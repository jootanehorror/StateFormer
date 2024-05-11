import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from torch import nn

from einops import rearrange, repeat


class BiasNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        bias,
        log_scale,
        channel_dim: int,
        store_output_for_backprop: bool,
    ):
        assert bias.ndim == 1
        if channel_dim < 0:
            channel_dim = channel_dim + x.ndim
        ctx.store_output_for_backprop = store_output_for_backprop
        ctx.channel_dim = channel_dim
        for _ in range(channel_dim + 1, x.ndim):
            bias = bias.unsqueeze(-1)
        scales = (
            torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True) ** -0.5
        ) * log_scale.exp()
        ans = x * scales
        ctx.save_for_backward(
            ans.detach() if store_output_for_backprop else x,
            scales.detach(),
            bias.detach(),
            log_scale.detach(),
        )
        return ans

    @staticmethod
    def backward(ctx, ans_grad):
        ans_or_x, scales, bias, log_scale = ctx.saved_tensors
        if ctx.store_output_for_backprop:
            x = ans_or_x / scales
        else:
            x = ans_or_x
        x = x.detach()
        x.requires_grad = True
        bias.requires_grad = True
        log_scale.requires_grad = True
        with torch.enable_grad():
           
            scales = (
                torch.mean((x - bias) ** 2, dim=ctx.channel_dim, keepdim=True) ** -0.5
            ) * log_scale.exp()
            ans = x * scales
            ans.backward(gradient=ans_grad)
        return x.grad, bias.grad.flatten(), log_scale.grad, None, None


class BiasNorm(torch.nn.Module):


    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,  

        store_output_for_backprop: bool = False,
    ) -> None:
        super(BiasNorm, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.zeros(num_channels))

        self.store_output_for_backprop = store_output_for_backprop

    def forward(self, x):
        assert x.shape[self.channel_dim] == self.num_channels

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            channel_dim = self.channel_dim
            if channel_dim < 0:
                channel_dim += x.ndim
            bias = self.bias
            for _ in range(channel_dim + 1, x.ndim):
                bias = bias.unsqueeze(-1)
            scales = (
                torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True) ** -0.5
            ) * self.log_scale.exp()
            return x * scales

        log_scale = self.log_scale


        return BiasNormFunction.apply(
            x, self.bias, log_scale, self.channel_dim, self.store_output_for_backprop
        )
