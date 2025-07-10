import torch
from torch.distributions import MultivariateNormal as MV
from torchtyping import TensorType

from typing import List, Union, Optional

import math
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

# class Q0DistMultiTask(torch.nn.Module):
#     def __init__(
#         self,
#         kernel,  # still unused; fixed to RBF
#         prediction_length: int,
#         num_tasks: int = 370,
#         freq: int = 24,
#         gamma: float = 1.0,
#         iso: float = 1e-4,
#         info=None,
#         context_freqs: int = 1,
#         device='cuda'
#     ):
#         super().__init__()
#         self.info = info
#         self.freq = freq
#         self.num_tasks = num_tasks
#         self.gamma = torch.tensor(gamma, dtype=torch.float32)
#         self.iso = torch.tensor(iso, dtype=torch.float32)

#         self.prediction_length = prediction_length
#         self.prior_context_length = context_freqs * prediction_length
#         self.window_length = self.prior_context_length + self.prediction_length
#         # Define training time points in float32 for interpolation
#         self.register_buffer("t", torch.arange(self.window_length, dtype=torch.float32) * (torch.pi / freq))
#         self.register_buffer("t_obs", torch.cat([
#             torch.ones(self.prior_context_length), torch.zeros(self.prediction_length)
#         ]).bool())
#         self.register_buffer("t_new", ~self.t_obs)

#         # Initialize zeros for train_y in float32
#         train_x = self.t
#         train_y = torch.zeros(self.window_length, self.num_tasks, dtype=torch.float32)

#         self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.num_tasks).to(device)
#         # disable shared noise
#         self.likelihood.noise = self.iso
#         self.likelihood.raw_noise.requires_grad_(False)
#         # fix per-task noise
#         self.likelihood.task_noises = torch.full(
#             (self.num_tasks,), self.iso.item(), dtype=torch.float32, device=device
#         )

#         self.likelihood.raw_task_noises.requires_grad_(False)

#         class ExactMultitaskGP(ExactGP):
#             def __init__(self, train_x, train_y, likelihood, num_tasks):
#                 super().__init__(train_x, train_y, likelihood)
#                 self.mean_module = gpytorch.means.MultitaskMean(
#                     gpytorch.means.ZeroMean(), num_tasks=num_tasks
#                 )
#                 base_kernel = gpytorch.kernels.MaternKernel(nu=0.5).to(device)
#                 base_kernel.lengthscale = gamma
#                 base_kernel.raw_lengthscale.requires_grad = False

#                 time_dim = train_x.size(0)
#                 grid_size = 2 ** math.ceil(math.log2(time_dim))
#                 grid_kernel = GridInterpolationKernel(
#                     base_kernel,
#                     grid_size=grid_size,
#                     grid_bounds=[(train_x.min().item(), train_x.max().item())],
#                 )

#                 self.covar_module = MultitaskKernel(
#                     grid_kernel,
#                     num_tasks=num_tasks,
#                     rank=8,
#                 )

#             def forward(self, x):
#                 mean_x = self.mean_module(x)
#                 covar_x = self.covar_module(x)
#                 return MultitaskMultivariateNormal(mean_x, covar_x)

#         self.model = ExactMultitaskGP(train_x, train_y, self.likelihood, self.num_tasks).to(device)
#         # zero out low-rank factor, set diagonal to ones
#         task_covar = self.model.covar_module.task_covar_module
#         task_covar.covar_factor.data.zero_()
#         constraint = task_covar.raw_var_constraint
#         raw_var_val = constraint.inverse_transform(torch.ones_like(task_covar.raw_var))
#         task_covar.raw_var.data.copy_(raw_var_val)

#         if self.info:
#             self.info(f"[Q0DistMultiTask] kernel=RBF, γ={gamma}, window={self.window_length}")

#     def log_likelihood(self, x: TensorType[float, "B", "T*N"]) -> TensorType[float]:
#         # reshape and compute
#         return self.model(self.t).likelihood(x.reshape(-1, self.num_tasks))

#     def forward(self, num_samples: int) -> TensorType[float, "num_samples", "T*N"]:
#         pred_dist = self.likelihood(self.model(self.t))
#         return pred_dist.rsample(num_samples)

#     def gp_regression(
#         self,
#         x: TensorType[float, "B", "C", "L"],
#         prediction_length: int
#     ) -> List[MultitaskMultivariateNormal]:
#         device = self.t.device

#         if x.ndim == 2:
#             # assume original C is self.num_tasks or provided context
#             total_sequences, L_in = x.shape
#             C = self.num_tasks
#             B = total_sequences // C
#             x = x.reshape(B, C, L_in)

#         B, C, L = x.shape
#         F = self.freq
#         G_ctx = L // F
#         G_fut = prediction_length // F
#         rem = prediction_length % F

#         # full time grid (float32)
#         full_t = self.t.to(device)
#         t_ctx = full_t[len(self.t) - prediction_length - L :-prediction_length]
#         t_fut = full_t[-prediction_length:]

#         self.model.eval()
#         self.likelihood.eval()

#         results: List[MultitaskMultivariateNormal] = []
#         for b in range(B):
#             x_b = x[b].to(device).float()
#             # 1) per-frequency mean
#             x_b_reshaped = x_b.reshape(C, G_ctx, F)
#             loc_freq = x_b_reshaped.mean(dim=1)
#             loc_ctx = loc_freq.repeat_interleave(G_ctx, dim=1)
#             x_center = x_b - loc_ctx

#             # 2) train data
#             train_y = x_center.transpose(0,1)
#             self.model.set_train_data(inputs=t_ctx, targets=train_y, strict=False)

#             # 3) predict
#             f_dist = self.model(t_fut)
#             y_dist = self.likelihood(f_dist)
#             mean_all = y_dist.mean
#             cov_all = y_dist.lazy_covariance_matrix

#             # 4) re-add seasonal mean
#             parts = []
#             if G_fut>0:
#                 parts.append(loc_freq.repeat_interleave(G_fut, dim=1))
#             if rem>0:
#                 parts.append(loc_freq[:,:rem])
#             loc_fut = torch.cat(parts, dim=1)
#             offset = loc_fut.transpose(0,1).to(device)
#             mean_pred = mean_all + offset

#             # 5) package
#             mvn = MultitaskMultivariateNormal(mean_pred, cov_all)
#             results.append(mvn)
#             self.model.prediction_strategy = None
#         return results


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
        self.window_length = self.prior_context_length + self.prediction_length

        # time grid
        self.register_buffer(
            "t", torch.arange(self.window_length, dtype=torch.float32) * (math.pi / freq)
        )

        # likelihood
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.num_tasks).to(device)
        self.likelihood.noise = torch.tensor(self.iso, device=device)
        self.likelihood.raw_noise.requires_grad_(False)
        self.likelihood.task_noises = torch.full(
            (self.num_tasks,), 1, device=device)
        self.likelihood.raw_task_noises.requires_grad_(False)

        # SKI ExactGP definition
        class SKIGP(ExactGP):
            def __init__(inner_self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                inner_self.mean_module = gpytorch.means.MultitaskMean(
                    gpytorch.means.ZeroMean(), num_tasks=self.num_tasks
                )
                base_kernel = MaternKernel(nu=0.5)
                base_kernel.lengthscale = self.gamma
                base_kernel.raw_lengthscale.requires_grad_(False)
                
                grid_size = 2 ** math.ceil(math.log2(train_x.size(0)))
                grid_bounds = [(train_x.min().item(), train_x.max().item())]
                ski_kernel = GridInterpolationKernel(
                    base_kernel,
                    grid_size=grid_size,
                    grid_bounds=grid_bounds
                )
                inner_self.covar_module = MultitaskKernel(
                    ski_kernel,
                    num_tasks=self.num_tasks,
                    rank=self.num_latents
                )

            def forward(inner_self, x):
                mean_x = inner_self.mean_module(x)
                covar_x = inner_self.covar_module(x)
                return MultitaskMultivariateNormal(mean_x, covar_x)

                # initialize the GP with dummy full window (for stable SKI grid bounds)
        train_x0 = self.t.unsqueeze(-1)  # full time grid
        train_y0 = torch.zeros(self.window_length, self.num_tasks, device=device)
        # Properly instantiate the model without extra calls
        self.model = SKIGP(train_x0, train_y0, self.likelihood).to(device)
        train_x0.to(device)
        train_y0.to(device)
        self.likelihood.to(device)
        if self.info:
            self.info(
                f"[Q0DistMultiTask] SKI GP w/ analytical conditioning; "
                f"context={self.prior_context_length}, future={self.prediction_length}, tasks={self.num_tasks}"
            )

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
        
        F = self.freq
        # seasonal parameters
        G_ctx = L // F
        rem_ctx = L % F

        # time tensors
        t_ctx = self.t[L-context_points:L].unsqueeze(-1)
        t_fut = self.t[L:L + prediction_length].unsqueeze(-1)

        results: List[MultitaskMultivariateNormal] = []
        for b in range(B):
            x_b = x[b].to(device)
            # seasonal detrending on exactly L points
            x_ctx = x_b  # shape (C, L)
            # reshape into (C, G_ctx, F)
            if rem_ctx == 0:
                x_rs = x_ctx.reshape(C, G_ctx, F)
            else:
                x_rs = x_ctx[:, : G_ctx * F].reshape(C, G_ctx, F)
            # seasonal mean per task and phase: shape (C, F)
            loc_phase = x_rs.mean(dim=1)
            # reconstruct context seasonal means: tile full cycles
            loc_ctx = loc_phase.repeat_interleave(G_ctx, dim=1)  # (C, G_ctx*F)
            # append remainder
            if rem_ctx > 0:
                loc_ctx = torch.cat([loc_ctx, loc_phase[:, :rem_ctx]], dim=1)  # (C, L)
            # detrended context shape (L, C)
            y_ctx = (x_ctx - loc_ctx).transpose(0,1)

            # update GP train data with exactly L context observations
            self.model.set_train_data(inputs=t_ctx, targets=y_ctx[L-context_points:], strict=False)
            self.model.eval(); self.likelihood.eval()
            
            # predict next points
            f_fut = self.model(t_fut)
            y_fut = self.likelihood(f_fut)

            # re-add seasonal mean for future
            # loc_phase shape (C, F)
            future_cycles = prediction_length // F
            rem_f = prediction_length % F
            # tile full future cycles
            if future_cycles > 0:
                fut_loc = loc_phase.repeat_interleave(future_cycles, dim=1)
            else:
                fut_loc = torch.empty((C, 0), device=device)
            # append remainder
            if rem_f > 0:
                fut_loc = torch.cat([fut_loc, loc_phase[:, :rem_f]], dim=1)  # (C, prediction_length)
            # transpose to (prediction_length, C)
            loc_fut = fut_loc.transpose(0,1).to(device)

            mean_final = y_fut.mean + loc_fut
            results.append(MultitaskMultivariateNormal(mean_final, y_fut.lazy_covariance_matrix))
        return results



    
class Q0DistKf(torch.nn.Module):
    def __init__(
        self,
        kernel: Prior,
        context_freqs: int,
        prediction_length: int,
        freq: int = 24,
        gamma: float = 1.0,
        iso: float = 1e-4,
        info=None,
        num_tasks: int = 0,
        rank_f: int = None,         # optional low-rank for task-kernel
        dtype=torch.float32,       # allow float32 GPU
        device=None,
    ):
        super().__init__()
        self.info = info
        self.kernel = kernel
        self.freq = freq
        self.prediction_length = prediction_length
        self.iso = torch.tensor(iso, dtype=dtype)
        self.gamma = torch.tensor(gamma, dtype=dtype)
        self.num_tasks = num_tasks
        self.rank_f = rank_f or num_tasks
        self.dtype = dtype
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # move iso and gamma to device
        self.iso = self.iso.to(self.device)
        self.gamma = self.gamma.to(self.device)

        # window sizes
        self.N = context_freqs * prediction_length
        self.L = prediction_length
        window = self.N + self.L

        # --- time kernel: get first row of Toeplitz K_x and its FFT (circulant approx) ---
        Kx_full = get_gp(kernel, self.gamma, self.N, freq).to(device=self.device, dtype=self.dtype)
        kx_row = Kx_full[0].clone()                  # [N]
        eig_x_fft = torch.fft.rfft(kx_row)           # [N//2+1] complex
        self.register_buffer('kx_row', kx_row)
        self.register_buffer('eig_x_fft', eig_x_fft)

        # --- cross- and future-covariances (exact) via full prior ---
        cov_full = get_gp(kernel, self.gamma, window, freq).to(device=self.device, dtype=self.dtype)
        cov_full = cov_full + self.iso * torch.eye(window, dtype=self.dtype, device=self.device)
        t = torch.arange(window, device=self.device)
        t_obs = t < self.N
        t_new = ~t_obs
        self.register_buffer('K_star_x', cov_full[t_obs][:, t_new])
        self.register_buffer('K_star_star', cov_full[t_new][:, t_new])

        # --- placeholder prior on full window (for sampling fallback) ---
        loc_full = torch.zeros(window, dtype=self.dtype, device=self.device)
        self.prior = MV(loc=loc_full, covariance_matrix=cov_full)

        if self.info:
            self.info(f"[Q0DistKf] dtype={self.dtype}, device={self.device}, FFT-time + low-rank task, window={window}, N={self.N}, L={self.L}")

    def set_task_kernel(self, Kf: torch.Tensor):
        """
        Set or warm-start C×C task kernel; do (low-rank) eigendecomposition.
        """
        C = self.num_tasks
        dtype, device = self.dtype, self.device

        Kf = Kf.to(device=device, dtype=dtype)
        Kf = Kf + 1e-6 * torch.eye(C, dtype=dtype, device=device)
        eig_f, Uf = torch.linalg.eigh(Kf)
        r = min(self.rank_f, C)
        eig_f_r = eig_f[-r:]
        Uf_r = Uf[:, -r:]
        # store real and complex for FFT domain
        self.register_buffer('eig_f_r', eig_f_r)
        self.register_buffer('Uf_r', Uf_r)
        self.register_buffer('Uf_r_cplx', Uf_r.to(torch.complex64))

    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        """
        Marginal LL of observations under approximate posterior cov.
        x: [B*C, N] (future ignored)
        """
        Bc, _ = x.shape
        C = self.num_tasks
        N = self.N
        device, dtype = self.device, self.dtype

        x_obs = x.to(device=device, dtype=dtype)[..., :N].view(-1, C, N)
        flat = x_obs.reshape(-1, C * N)

        # build full K_x from row
        idx = torch.arange(N, device=device)
        diff = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
        Kx_full = self.kx_row[diff.long()]

        # task kernel
        KF = self.Uf_r @ torch.diag(self.eig_f_r) @ self.Uf_r.T

        Kxx = torch.kron(KF, Kx_full)
        Kxx_joint = Kxx + self.iso * torch.eye(C * N, dtype=dtype, device=device)
        mv = MV(loc=torch.zeros(C * N, dtype=dtype, device=device), covariance_matrix=Kxx_joint)
        return mv.log_prob(flat)

    def forward(self, x: torch.Tensor, num_samples: int):
        """Sample from approximate posterior at future points."""
        mv = self.gp_regression(x, self.prediction_length)
        return mv.rsample((num_samples,))

    def gp_regression(self, x: torch.Tensor, prediction_length: int) -> MV:
        """
        Approximate posterior GP using FFT for time and low-rank tasks.
        x: [B*C, N]
        returns MV(loc=[B, C*L], cov=[C*L, C*L])
        """
        Bc, _ = x.shape
        C, N, L = self.num_tasks, self.N, prediction_length
        B = Bc // C
        device, dtype = self.device, self.dtype

        # reshape & demean
        X = x.to(device=device, dtype=dtype).view(B, C, N)
        loc = X.mean(-1, keepdim=True)
        Yc = X - loc

        # FFT in time
        Y_fft = torch.fft.rfft(Yc, dim=2)
        # project tasks in FFT domain
        Y_proj = torch.einsum('ij,bjk->bik', self.Uf_r_cplx.T, Y_fft)
        # denom: [r, Nf]
        den = (self.eig_f_r.unsqueeze(1) * self.eig_x_fft.unsqueeze(0).real) + self.iso
        A_proj = Y_proj / den.unsqueeze(0)
        # back-project
        Ah_fft = torch.einsum('ij,bjk->bik', self.Uf_r_cplx, A_proj)
        # inverse FFT
        A = torch.fft.irfft(Ah_fft, n=N, dim=2)

        # predictive mean
        mean_fc = torch.einsum('bcn,nl->bcl', A, self.K_star_x) + loc
        mean = mean_fc.reshape(B, C * L)

        # posterior covariance (low-rank solve)
        # build full K_x
        idx = torch.arange(N, device=device)
        diff = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
        Kx_full = self.kx_row[diff.long()]

        KF = self.Uf_r @ torch.diag(self.eig_f_r) @ self.Uf_r.T
        Kxx_joint = torch.kron(KF, Kx_full) + self.iso * torch.eye(C * N, dtype=dtype, device=device)
        Kstarx = torch.kron(KF, self.K_star_x)
        solve_term = torch.linalg.solve(Kxx_joint, Kstarx)
        prior_star = torch.kron(KF, self.K_star_star)
        cov = prior_star - Kstarx.T.matmul(solve_term)
        
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        # eigenvalue-clamp
        w, Q = torch.linalg.eigh(cov)
        w_clamped = torch.clamp(w, min=1e-3)
        cov = (Q * w_clamped.unsqueeze(-2)) @ Q.transpose(-1, -2)
        # final jitter to guard Cholesky
        cov = cov + 1e-3 * torch.eye(cov.size(-1), device=device, dtype=dtype)
        cov = 0.5 * (cov + cov.transpose(-1, -2))

        return MV(loc=mean, covariance_matrix=cov)

    
# class Q0DistKf(torch.nn.Module):
#     def __init__(
#         self,
#         kernel: Prior,
#         context_freqs: int,
#         prediction_length: int,
#         freq: int = 24,
#         gamma: float = 1.0,
#         iso: float = 1e-4,
#         info=None,
#         num_tasks=0,
#     ):
#         super().__init__()
#         self.info = info
#         self.kernel = kernel
#         self.freq = freq
#         self.prediction_length = prediction_length
#         self.iso = torch.tensor(iso, dtype=torch.float64)
#         self.gamma = torch.tensor(gamma, dtype=torch.float64)
#         self.num_tasks = num_tasks

#         # Total window = past + future
#         context_length = context_freqs * prediction_length
#         window_length = context_length + prediction_length

#         # Full prior covariance over [0..window_length)
#         cov_full = get_gp(kernel, self.gamma, window_length, freq).double()
#         cov_full += self.iso * torch.eye(window_length, dtype=torch.float64)
#         loc_full = torch.zeros(window_length, dtype=torch.float64)

#         # Masks for observed vs new
#         t_obs = torch.arange(window_length) < context_length
#         t_new = ~t_obs

#         # Slice sub-blocks
#         K_x         = cov_full[t_obs][:, t_obs]    # [N,N]
#         K_star_x    = cov_full[t_obs][:, t_new]    # [N,L]
#         K_star_star = cov_full[t_new][:, t_new]    # [L,L]

#         # Jitter + eigendecompose K_x once
#         K_x += 1e-6 * torch.eye(K_x.size(0), dtype=torch.float64)
#         eigs_x, Ux = torch.linalg.eigh(K_x)

#         # Register buffers for input‐side
#         self.register_buffer("K_x",         K_x)
#         self.register_buffer("Ux",          Ux)
#         self.register_buffer("eigs_x",      eigs_x)
#         self.register_buffer("K_star_x",    K_star_x)
#         self.register_buffer("K_star_star", K_star_star)

#         # Prior over full window
#         self.prior = MV(loc=loc_full, covariance_matrix=cov_full)

#         if self.info:
#             self.info(f"[Q0DistKf] kernel={kernel}, γ={gamma}, window={window_length}")

#     def _apply(self, fn):
#         super()._apply(fn)
#         self.prior = MV(
#             loc=fn(self.prior.loc),
#             covariance_matrix=fn(self.prior.covariance_matrix),
#         )
#         return self

#     def log_likelihood(self, x):
#         x = x.to(self.prior.loc.device)
#         return -self.prior.log_prob(x)

#     def forward(self, num_samples):
#         return self.prior.sample((num_samples,))

#     def set_task_kernel(self, Kf: torch.Tensor):
#         """
#         Warm-start or fix a C×C task-kernel.  
#         Precompute its eigendecomposition (for mean) and the predictive
#         covariance (for reuse in gp_regression).
#         """
#         device = self.K_x.device
#         M      = self.num_tasks
#         L      = self.prediction_length
#         iso    = self.iso

#         # 1) jitter & store Kf in double
#         Kf = (
#             Kf.to(device=device, dtype=torch.float64)
#             + 1e-6 * torch.eye(M, dtype=torch.float64, device=device)
#         )
#         eigs_f, Uf = torch.linalg.eigh(Kf)  # [M], [M,M]

#         # 2) predictive covariance = Kf⊗K**  -  (Kf⊗K*x)ᵀ Σ⁻¹ (Kf⊗K*x)
#         #    BUT we’ll skip the Σ⁻¹ solve here and approximate by *prior* predictive:
#         #    this still captures cross‐series corr from Kf and time‐cov from K_**.
#         cov_star = torch.kron(Kf, self.K_star_star)   # [M*L, M*L]
#         cov_star += iso * torch.eye(M*L, dtype=torch.float64, device=device)

#         # 3) register buffers
#         self.register_buffer("Kf",       Kf)
#         self.register_buffer("Uf",       Uf)
#         self.register_buffer("eigs_f",   eigs_f)
#         self.register_buffer("cov_star", cov_star)

#     def gp_regression(self, x: torch.Tensor, prediction_length: int) -> MV:
#         """
#         x: [B*C, N]   (B batches of C tasks each, N past points per task)
#         Returns MV with batch_shape=[B], event_shape=[C*L].
#         """
#         device, dtype = self.K_x.device, self.K_x.dtype
#         x = x.to(device=device, dtype=dtype)

#         B_times_C, N = x.shape
#         C = self.num_tasks
#         B = B_times_C // C
#         L = prediction_length

#         # reshape into [B, C, N]
#         x = x.view(B, C, N)

#         # 1) de-bias per task
#         loc = x.mean(dim=2, keepdim=True)           # [B, C, 1]
#         if self.kernel == Prior.PE:
#             loc = torch.zeros_like(loc)
#         Yc  = x - loc                                # [B, C, N]

#         # 2) rotate into joint eigenbasis:
#         #    Yc_perm: [B, N, C]
#         Yc_perm = Yc.permute(0, 2, 1)
#         #    first Ux.T @ Yc_perm: [B, N, C]
#         Y1 = torch.einsum("ij,bjk->bik", self.Ux.T, Yc_perm)
#         #    then @ Uf:        [B, N, C]
#         Yt = torch.einsum("bij,jk->bik", Y1, self.Uf)

#         # 3) elementwise solve in diag:
#         #    denom: [N, C]
#         denom = (
#             self.eigs_x.unsqueeze(1) * self.eigs_f.unsqueeze(0)
#             + self.iso
#         )
#         At = Yt / denom                             # [B, N, C]

#         # 4) rotate back:
#         A1 = torch.einsum("ij,bjk->bik", self.Ux, At)   # [B, N, C]
#         A  = torch.einsum("bij,jk->bik", A1, self.Uf.T) # [B, N, C]

#         # 5) predictive mean:  f_mat[b] = A[b].T @ K_star_x  → [C, L]
#         #    Aperm: [B, C, N]
#         Aperm = A.permute(0, 2, 1)
#         #    then batch‐matmul  → [B, C, L]
#         f_mat = Aperm.matmul(self.K_star_x)

#         # 6) add bias and flatten to [B, C*L]
#         mean = (f_mat + loc).reshape(B, C * L)

#         # 7) pull in the precomputed covariance [C*L, C*L]
#         cov = self.cov_star

#         return MV(loc=mean, covariance_matrix=cov)
    

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
