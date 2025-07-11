import torch
from torch.distributions import MultivariateNormal as MV
from torchtyping import TensorType

from typing import List, Union, Optional

import math
import time

import gpytorch
from gpytorch.means import ConstantMean, ZeroMean, MultitaskMean
from gpytorch.kernels import RBFKernel, ScaleKernel, MultitaskKernel, GridInterpolationKernel, MaternKernel
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.models import ExactGP
from gpytorch.settings import fast_pred_var 

from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
    LMCVariationalStrategy
)
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultitaskMultivariateNormal

from torch import Tensor
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    VariationalStrategy,
    CholeskyVariationalDistribution,
    MultitaskVariationalStrategy
)

from tsflow.utils.variables import Prior


def isotropic_kernel(
    gamma: TensorType[float], t: TensorType[float, "length"], u: TensorType[float, "length"]
) -> TensorType[float, "length", "length"]:
    return gamma * torch.eye(t.size(0), device=t.device, dtype=t.dtype)

def kronecker(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: (m, n), B: (p, q)
    # 1) Add singleton dims so A can broadcast over B and vice-versa:
    A2 = rearrange(A, 'i j -> i 1 j 1')    # shape (m,1,n,1), view only
    B2 = rearrange(B, 'p q -> 1 p 1 q')    # shape (1,p,1,q), view only

    # 2) Broadcast‐multiply and collapse back to (m·p, n·q):
    return rearrange(A2 * B2, 'i p j q -> (i p) (j q)')

def radial_basis_kernel(
    gamma: TensorType[float], t: TensorType[float, "length"], u: TensorType[float, "length"]
) -> TensorType[float, "length", "length"]:
    return torch.exp(-gamma * (t - u) ** 2)


def ornstein_uhlenbeck(
    gamma: TensorType[float], t: TensorType[float, "length"], u: TensorType[float, "length"]
) -> TensorType[float, "length", "length"]:
    return torch.exp(-gamma * torch.abs(t - u))


def periodic_kernel(
    gamma: TensorType[float], t: TensorType[float, "length"], u: TensorType[float, "length"]
) -> TensorType[float, "length", "length"]:
    return torch.exp(-gamma * torch.sin(t - u) ** 2)


kernel_function_map = {
    Prior.ISO: isotropic_kernel,
    Prior.SE: radial_basis_kernel,
    Prior.OU: ornstein_uhlenbeck,
    Prior.PE: periodic_kernel,
}


def get_gp(kernel: Prior, gamma: TensorType[float], length: int, freq: int) -> TensorType[float, "length", "length"]:
    """
    Generate a Gaussian process covariance matrix using the specified kernel.

    Args:
        kernel (Prior): The type of kernel to use.
        gamma (TensorType[float]): Kernel parameter.
        length (int): The number of time points (size of the covariance matrix).
        freq (int): Frequency parameter used to determine the time scale.
    """
    t = torch.arange(length, device=gamma.device, dtype=gamma.dtype) * (torch.pi / freq)
    cov = kernel_function_map[kernel](gamma, t, t.unsqueeze(1))
    return cov


class Q0DistMultiTask(torch.nn.Module):
    def __init__(
        self,
        kernel,  # SKI-based multitask
        prediction_length: int,
        num_tasks: int = 370,
        num_latents: int = 8,
        freq: int = 24,
        gamma: float = 1.0,
        iso: float = 1e-4,
        info=None,
        context_freqs: int = 1,
        device='cuda'
    ):
        super().__init__()
        self.info = info
        self.freq = freq
        self.num_tasks = num_tasks
        self.num_latents = num_latents
        self.gamma = gamma
        self.iso = iso

        self.prediction_length = prediction_length
        self.prior_context_length = context_freqs * prediction_length
        self.condition_length = prediction_length

        self.window_length = self.condition_length + self.prediction_length

        # time grid
        self.register_buffer(
            "t", torch.arange(self.window_length, dtype=torch.float32) * (math.pi / freq)
        )

        # likelihood
        likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks).to(device)
        likelihood.noise = torch.tensor(iso, device=device)
        likelihood.raw_noise.requires_grad_(False)
        likelihood.task_noises = torch.full((num_tasks,), 1, device=device)
        likelihood.raw_task_noises.requires_grad_(False)
        self.likelihood = likelihood
        # SKI ExactGP definition
        class SKIGP(ExactGP):
            def __init__(inner_self, train_x, train_y, likelihood, num_tasks, num_latents, gamma, batch_shape):
                super().__init__(train_x, train_y, likelihood)
                self.num_tasks = num_tasks

                inner_self.mean_module = gpytorch.means.MultitaskMean(
                    gpytorch.means.ZeroMean(), num_tasks=self.num_tasks,
                )
                base_kernel = MaternKernel(nu=0.5, batch_shape=batch_shape)
                base_kernel.lengthscale = gamma
                base_kernel.raw_lengthscale.requires_grad_(False)
                
                grid_size = 2 ** math.ceil(math.log2(train_x.size(-2)))
                grid_bounds = [(train_x.min().item(), train_x.max().item())]

                ski_kernel = GridInterpolationKernel(
                    base_kernel,
                    grid_size=grid_size,
                    grid_bounds=grid_bounds,
                )
                inner_self.covar_module = MultitaskKernel(
                    ski_kernel,
                    num_tasks=self.num_tasks,
                    rank=num_latents,
                    batch_shape=batch_shape
                )

            def forward(inner_self, x):
                mean_x = inner_self.mean_module(x)
                covar_x = inner_self.covar_module(x)
                return MultitaskMultivariateNormal(mean_x, covar_x)

        # initialize the GP with dummy full window (for stable SKI grid bounds)
        dummy_x = self.t.unsqueeze(0).unsqueeze(-1)  # full time grid
        dummy_y = torch.zeros(1, self.window_length, self.num_tasks, device=device)
        # Properly instantiate the model without extra calls
        self.model = SKIGP(dummy_x, dummy_y, self.likelihood, num_tasks, num_latents, gamma, batch_shape=torch.Size([1])).to(device)

    def gp_regression(
        self,
        x: TensorType[float, "B", "C", "L"],
        prediction_length: int,
        context_points:int = 0,
    ) -> List[MultitaskMultivariateNormal]:
        """
        Condition on exactly the provided context length L and predict the next prediction_length points.
        Assumes x shape (B, C, L) where L==self.prior_context_length.
        """
        device = self.t.device
        # reshape if 2D input
        if x.ndim == 2:
            B = x.size(0) // self.num_tasks
            x = x.reshape(B, self.num_tasks, -1)

        B, C, L = x.shape
        # time tensors
        t_ctx = self.t[:context_points].unsqueeze(0).expand(B, -1).unsqueeze(-1).to(device)
        t_fut = self.t[context_points:context_points + prediction_length].unsqueeze(0).expand(B, -1).unsqueeze(-1).to(device)
       
        F = self.freq
        # seasonal parameters
        G_ctx = L // F
        rem_ctx = L % F        
        # seasonal detrending
        # x: [B, C, L]
        # reshape to [B, C, G_ctx, F] then compute loc_phase: [B, C, F]
        if rem_ctx == 0:
            x_rs = x.reshape(B, C, G_ctx, F)
        else:
            x_rs = x[:, :, :G_ctx * F].reshape(B, C, G_ctx, F)
        loc_phase = x_rs.mean(dim=2)  # [B, C, F]
        # build loc_ctx: [B, C, L]
        loc_ctx = loc_phase.repeat_interleave(G_ctx, dim=2)
        if rem_ctx > 0:
            loc_ctx = torch.cat([loc_ctx, loc_phase[:, :, :rem_ctx]], dim=2)
        # detrended context: [B, L, C]
        y_ctx = (x - loc_ctx).permute(0, 2, 1)

        # update GP train data with exactly L context observations
        self.model.set_train_data(inputs=t_ctx, targets=y_ctx[:, L-context_points:], strict=False)
        
        self.model.eval(); self.likelihood.eval()

        with gpytorch.settings.max_cholesky_size(0), \
            gpytorch.settings.max_cg_iterations(25), \
            gpytorch.settings.fast_pred_var():
            # predict next points
            f_fut = self.model(t_fut)
            y_fut = self.likelihood(f_fut)
        
        self.model.train(); self.likelihood.train()
        # re-add seasonal mean for future
        future_cycles, rem_f = divmod(prediction_length, F)
        fut_loc = loc_phase.repeat_interleave(future_cycles, dim=2) if future_cycles > 0 else torch.empty((B, C, 0), device=device)
        if rem_f > 0:
            fut_loc = torch.cat([fut_loc, loc_phase[:, :, :rem_f]], dim=2)
        # [B, pred_len, C]
        loc_fut = fut_loc.permute(0, 2, 1)

        mean_final = y_fut.mean + loc_fut
        cov_lazy = y_fut.lazy_covariance_matrix

        return MultitaskMultivariateNormal(mean_final, cov_lazy)
    

class Q0Dist(torch.nn.Module):
    def __init__(
        self,
        kernel: Prior,
        context_freqs: int,
        prediction_length: int,
        freq: int = 24,
        gamma: float = 1.0,
        iso: float = 1e-4,
        info = None,
        num_tasks = 0,
    ):
        """
        Initialize the GP distribution.

        Args:
            kernel (Prior): The kernel type (e.g. Prior.SE, Prior.OU, Prior.PE, etc.).
            context_freqs (int): The number of frequency groups in the context.
            prediction_length (int): The prediction horizon.
            freq (int, optional): Frequency parameter for time computations. Defaults to 24.
            gamma (float, optional): The kernel hyperparameter. Defaults to 1.0.
            iso (float, optional): The isotropic noise level. Defaults to 1e-4.
        """
        super().__init__()
        self.info = info
        context_length = context_freqs * prediction_length
        window_length = context_length + prediction_length
        
        # Convert gamma to a tensor for consistency.
        self.gamma = torch.tensor(gamma, dtype=torch.float64)
        self.iso = torch.tensor(iso, dtype=torch.float64)
        self.kernel = kernel
        self.freq = freq
        self.prediction_length = prediction_length

        mean = torch.zeros(window_length)
        cov = get_gp(kernel, self.gamma, window_length, freq) + self.iso * torch.eye(window_length)

        # GP reg
        t_obs = torch.cat([torch.ones(context_length), torch.zeros(prediction_length)]) == 1
        t_new = ~t_obs

        # Extract the relevant submatrices from the covariance matrix.
        K = cov[t_obs][:, t_obs]
        K_star = cov[t_obs][:, t_new]
        K_star_star = cov[t_new][:, t_new]

        # Add a small amount of noise to the diagonal to ensure invertibility.
        K += 1e-4 * torch.eye(K.size(0))
        K_inv = torch.linalg.inv(K)
        K_inv_x_K_star = K_inv @ K_star
        cov_reg = K_star_star - K_star.T @ K_inv_x_K_star

        # Register buffers
        self.register_buffer("cov_inv", torch.linalg.inv(cov).float(), persistent=False)
        self.register_buffer("K_inv_x_K_star", K_inv_x_K_star.float(), persistent=False)
        self.register_buffer("cov_reg", cov_reg.float(), persistent=False)

        self.dist = MV(loc=mean.float(), covariance_matrix=cov.float())

        if self.info:
            self.info(f"[Q0Dist] kernel={kernel}, γ={gamma}, window={context_freqs*prediction_length + prediction_length}")
        # … build cov, K_inv, etc. …
        if self.info:
            self.info(f"[Q0Dist] cov built, shape={cov.shape}")

    def _apply(self, fn):
        """
        Override _apply so that when the module is moved to a new device,
        we update our precomputed distribution (self.dist) accordingly.
        """
        super()._apply(fn)
        self.dist = MV(
            loc=self.dist.loc.to(self.cov_inv.device),
            scale_tril=self.dist.scale_tril.to(self.cov_inv.device),
        )
        return self

    def log_likelihood(self, x: TensorType[float, "batch_size", "length"]) -> TensorType[float]:
        """
        Compute the log likelihood of the data given the GP distribution.
        """
        x = x.to(self.cov_inv.device)
        return -self.dist.log_prob(x)

    def forward(self, num_samples: int) -> TensorType[float, "num_samples", "length"]:
        fj = False

        if fj == True:
            samples = self.dist.sample(torch.Size([num_samples]))
            samples = samples.view(num_samples, self.t.shape[0], self.num_tasks)
        return samples

    def gp_regression(self, x: TensorType[float], prediction_length: int) -> MV:
        """
        Perform GP regression on input data.

        Args:
            x (torch.Tensor): Input data tensor of shape [batch_size, window_length].
            prediction_length (int): Number of prediction points.

        Returns:
            MultivariateNormal: GP regression posterior.
        """
        batch_size = x.size(0)
        # Reshape x to [batch_size, num_groups, freq] for per-frequency computations.
        x_reshaped = x.reshape(batch_size, -1, self.freq)
        loc_freq = x_reshaped.mean(1, keepdims=False)

        # For periodic kernels, set location bias to zero.
        if self.kernel == Prior.PE:
            loc_freq = torch.zeros_like(loc_freq)

        # Remove the location bias.
        repeat_factor = self.K_inv_x_K_star.shape[0] // self.freq
        x_centered = x - loc_freq.repeat(1, repeat_factor)

        # Compute the regression mean.
        mean = x_centered @ self.K_inv_x_K_star
        mean = mean + loc_freq.repeat(1, prediction_length // self.freq)

        # Scale the regression covariance.
        cov = self.cov_reg
        return MV(mean, cov)
