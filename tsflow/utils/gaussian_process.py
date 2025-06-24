import torch
from torch.distributions import MultivariateNormal as MV
from torchtyping import TensorType

from typing import List, Union, Optional

import math
import gpytorch
from gpytorch.means import ConstantMean, ZeroMean, MultitaskMean
from gpytorch.kernels import RBFKernel, ScaleKernel, MultitaskKernel, GridInterpolationKernel
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.models import ExactGP
from gpytorch.settings import fast_pred_var 

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
        kernel,  # still unused; fixed to RBF
        prediction_length: int,
        num_tasks: int = 370,
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
        self.gamma = torch.tensor(gamma, dtype=torch.float32)
        self.iso = torch.tensor(iso, dtype=torch.float32)

        self.prediction_length = prediction_length
        self.prior_context_length = context_freqs * prediction_length
        self.window_length = self.prior_context_length + self.prediction_length
        # Define training time points in float32 for interpolation
        self.register_buffer("t", torch.arange(self.window_length, dtype=torch.float32) * (torch.pi / freq))
        self.register_buffer("t_obs", torch.cat([
            torch.ones(self.prior_context_length), torch.zeros(self.prediction_length)
        ]).bool())
        self.register_buffer("t_new", ~self.t_obs)

        # Initialize zeros for train_y in float32
        train_x = self.t
        train_y = torch.zeros(self.window_length, self.num_tasks, dtype=torch.float32)

        self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.num_tasks).to(device)
        # disable shared noise
        self.likelihood.noise = self.iso
        self.likelihood.raw_noise.requires_grad_(False)
        # fix per-task noise
        self.likelihood.task_noises = torch.full(
            (self.num_tasks,), self.iso.item(), dtype=torch.float32, device=device
        )

        self.likelihood.raw_task_noises.requires_grad_(False)

        class ExactMultitaskGP(ExactGP):
            def __init__(self, train_x, train_y, likelihood, num_tasks):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.MultitaskMean(
                    gpytorch.means.ZeroMean(), num_tasks=num_tasks
                )
                base_kernel = gpytorch.kernels.MaternKernel(nu=0.5).to(device)
                base_kernel.lengthscale = gamma
                base_kernel.raw_lengthscale.requires_grad = False

                time_dim = train_x.size(0)
                grid_size = 2 ** math.ceil(math.log2(time_dim))
                grid_kernel = GridInterpolationKernel(
                    base_kernel,
                    grid_size=grid_size,
                    grid_bounds=[(train_x.min().item(), train_x.max().item())],
                )

                self.covar_module = MultitaskKernel(
                    grid_kernel,
                    num_tasks=num_tasks,
                    rank=8,
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return MultitaskMultivariateNormal(mean_x, covar_x)

        self.model = ExactMultitaskGP(train_x, train_y, self.likelihood, self.num_tasks).to(device)
        # zero out low-rank factor, set diagonal to ones
        task_covar = self.model.covar_module.task_covar_module
        task_covar.covar_factor.data.zero_()
        constraint = task_covar.raw_var_constraint
        raw_var_val = constraint.inverse_transform(torch.ones_like(task_covar.raw_var))
        task_covar.raw_var.data.copy_(raw_var_val)

        if self.info:
            self.info(f"[Q0DistMultiTask] kernel=RBF, γ={gamma}, window={self.window_length}")

    def log_likelihood(self, x: TensorType[float, "B", "T*N"]) -> TensorType[float]:
        # reshape and compute
        return self.model(self.t).likelihood(x.reshape(-1, self.num_tasks))

    def forward(self, num_samples: int) -> TensorType[float, "num_samples", "T*N"]:
        pred_dist = self.likelihood(self.model(self.t))
        return pred_dist.rsample(num_samples)

    def gp_regression(
        self,
        x: TensorType[float, "B", "C", "L"],
        prediction_length: int
    ) -> List[MultitaskMultivariateNormal]:
        device = self.t.device

        if x.ndim == 2:
            # assume original C is self.num_tasks or provided context
            total_sequences, L_in = x.shape
            C = self.num_tasks
            B = total_sequences // C
            x = x.reshape(B, C, L_in)

        B, C, L = x.shape
        F = self.freq
        G_ctx = L // F
        G_fut = prediction_length // F
        rem = prediction_length % F

        # full time grid (float32)
        full_t = self.t.to(device)
        t_ctx = full_t[len(self.t) - prediction_length - L :-prediction_length]
        t_fut = full_t[-prediction_length:]

        self.model.eval()
        self.likelihood.eval()

        results: List[MultitaskMultivariateNormal] = []
        for b in range(B):
            x_b = x[b].to(device).float()
            # 1) per-frequency mean
            x_b_reshaped = x_b.reshape(C, G_ctx, F)
            loc_freq = x_b_reshaped.mean(dim=1)
            loc_ctx = loc_freq.repeat_interleave(G_ctx, dim=1)
            x_center = x_b - loc_ctx

            # 2) train data
            train_y = x_center.transpose(0,1)
            self.model.set_train_data(inputs=t_ctx, targets=train_y, strict=False)

            # 3) predict
            f_dist = self.model(t_fut)
            y_dist = self.likelihood(f_dist)
            mean_all = y_dist.mean
            cov_all = y_dist.lazy_covariance_matrix

            # 4) re-add seasonal mean
            parts = []
            if G_fut>0:
                parts.append(loc_freq.repeat_interleave(G_fut, dim=1))
            if rem>0:
                parts.append(loc_freq[:,:rem])
            loc_fut = torch.cat(parts, dim=1)
            offset = loc_fut.transpose(0,1).to(device)
            mean_pred = mean_all + offset

            # 5) package
            mvn = MultitaskMultivariateNormal(mean_pred, cov_all)
            results.append(mvn)
            self.model.prediction_strategy = None
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
        info = None,
        num_tasks = 0,
    ):
        super().__init__()
        self.info = info
        self.kernel = kernel
        self.freq = freq
        self.prediction_length = prediction_length
        self.iso = torch.tensor(iso, dtype=torch.float64)
        self.gamma = torch.tensor(gamma, dtype=torch.float64)
        self.num_tasks = num_tasks
        # total window = past + future
        context_length = context_freqs * prediction_length
        window_length = context_length + prediction_length

        # prior mean and full GP covariance over the whole window
        prior_loc = torch.zeros(window_length, dtype=torch.float64)
        cov_full = get_gp(kernel, self.gamma, window_length, freq).float()
        cov_full = cov_full + self.iso * torch.eye(window_length, dtype=torch.float64)

        # masks for observed vs. new
        t_obs = torch.arange(window_length) < context_length
        t_new = ~t_obs

        # slice out K_x, K_*x, and K_** (all double)
        K_x         = cov_full[t_obs][:, t_obs]
        K_star_x    = cov_full[t_obs][:, t_new]
        K_starstar = cov_full[t_new][:, t_new]

        # ensure invertibility & invert
        K_x = K_x + 1e-6 * torch.eye(K_x.size(0), dtype=torch.float64)
        K_inv_x = torch.linalg.inv(K_x)

        # pre‐compute for regression
        cov_reg = K_starstar - K_star_x.T @ (K_inv_x @ K_star_x)

        # register everything you need later
        self.register_buffer("K_x",         K_x,          persistent=False)
        self.register_buffer("K_inv_x",     K_inv_x,      persistent=False)
        self.register_buffer("K_star_x",    K_star_x,     persistent=False)
        self.register_buffer("K_star_star", K_starstar,   persistent=False)
        self.register_buffer("cov_reg",     cov_reg,      persistent=False)

        # the prior distribution over the full window
        self.dist = MV(loc=prior_loc, covariance_matrix=cov_full)

        if self.info:
            self.info(f"[Q0Dist] kernel={kernel}, γ={gamma}, window={window_length}")
            self.info(f"[Q0Dist] built K_x ({K_x.shape}), K_*x {K_star_x.shape}, K_** {K_starstar.shape}")

    def _apply(self, fn):
        """
        When the module is .to(device)-ed, move our distribution tensors too.
        """
        super()._apply(fn)
        # Move MultivariateNormal’s loc & covariance_matrix along with the buffers:
        self.dist = torch.distributions.MultivariateNormal(
            loc=fn(self.dist.loc),
            covariance_matrix=fn(self.dist.covariance_matrix),
        )
        return self

    def log_likelihood(self, x: TensorType[float, "batch_size", "length"]) -> TensorType[float]:
        """
        Compute negative log-prob under the prior GP over the full window.
        """
        # send to the same device as our dist parameters
        x = x.to(self.dist.loc.device)
        # return -log p(x)
        return -self.dist.log_prob(x)

    def forward(self, num_samples: int) -> TensorType[float, "num_samples", "length"]:
        """
        Sample full trajectories from the prior GP.
        Returns a tensor of shape [num_samples, window_length].
        """
        # Draw num_samples i.i.d. from MVN(loc, cov)
        samples = self.dist.sample((num_samples,))
        return samples

    def gp_regression(self, x: torch.Tensor, prediction_length: int):
        """
        GP regression where x is [num_tasks, context_len].
        prediction_length is L (future points per task).
        """
        # 1) Dimensions
        # bring to same device & dtype as our buffers
        device = self.K_x.device
        dtype  = self.K_x.dtype
        x = x.to(device=device, dtype=dtype)

        total_sequences, N = x.shape
        C = self.num_tasks
        B = total_sequences // C
        x = x.reshape(B, C, N)
        L = self.prediction_length

        eps  = 1e-6

        results: List[MultitaskMultivariateNormal] = []

        for batch in range(B):
            x_batch = x[batch, :, :].to(device)
            # 1) de-bias
            loc = x_batch.mean(dim=1, keepdim=True)             # [M,1]

            if self.kernel == Prior.PE:
                loc.zero_()
                
            Yc = x_batch - loc                                  # [M,N]

            # 2) task covariance with jitter
            Kf = (Yc @ self.K_inv_x @ Yc.t()) / N                        # [M,M]
            Kf = Kf + eps * torch.eye(C, device=device, dtype=dtype)

            # 3) noise diag
            D  = torch.diag(torch.full((C,), self.iso, device=device, dtype=dtype))

            # 4) build Σ = Kf⊗Kx + D⊗I_N, with jitter
            I_N   = torch.eye(N, device=device, dtype=dtype)
            Sigma = torch.kron(Kf, self.K_x) \
                + torch.kron(D, I_N)
            Sigma = Sigma + eps * torch.eye(C*L, device=device, dtype=dtype)

            # 5) solve for α = Σ⁻¹ vec(Yc)
            y_vec = Yc.t().reshape(C*N).to(dtype)
            alpha = torch.linalg.solve(Sigma, y_vec)       # [M*N]

            # 6) build cross‐covariance K_* = Kf⊗Kx_*
            K_star = torch.kron(Kf, self.K_star_x)        # [M*N, M*L]

            # 7) predictive mean
            f_mean = K_star.t() @ alpha                   # [M*L]

            # 8) predictive covariance
            cov_star = torch.kron(Kf, self.K_star_star)   # [M*L, M*L]
            cov_star = cov_star - K_star.t() @ torch.linalg.solve(Sigma, K_star)
            # symmetrize + jitter
            cov_star = 0.5 * (cov_star + cov_star.T)
            cov_star = cov_star + eps * torch.eye(C*L, device=device, dtype=dtype)

            # 9) add back loc bias into mean
            mean = (
                f_mean.view(C, L)             # [M, L]
                + loc                         # [M, L]
            ).reshape(C * L)                  # [M*L]
            results.append(MV(mean, covariance_matrix=cov_star))

        return results

    # def gp_regression(self, x: TensorType[float], prediction_length: int) -> MV:
    #     """
    #     Perform GP regression on input data.

    #     Args:
    #         x (torch.Tensor): Input data tensor of shape [batch_size, window_length].
    #         prediction_length (int): Number of prediction points.

    #     Returns:
    #         MultivariateNormal: GP regression posterior.
    #     """
    #     #assuming bs = 1
    #     num_series = x.size(0)

    #     # Reshape x to [batch_size, num_groups, freq] for per-frequency computations.
    #     x_reshaped = x.reshape(num_series, -1, self.freq)
    #     loc_freq = x_reshaped.mean(1, keepdims=False)

    #     # For periodic kernels, set location bias to zero.
    #     if self.kernel == Prior.PE:
    #         loc_freq = torch.zeros_like(loc_freq)

    #     # Remove the location bias.
    #     repeat_factor = self.K_inv_x_K_star.shape[0] // self.freq
    #     x_centered = x - loc_freq.repeat(1, repeat_factor)

    #     # # Compute the regression mean.
    #     #x: (b c) l
    #     emp_cov_hat = 1/x.size(1) * x.T @ self.K_inv_x @ x 
    #     K_f = emp_cov_hat

    #     # Feature to possibly add is learnable noise per task instead of assuming 1 like now
    #     sigma = kronecker(K_f, self.K_x) + torch.eye(K_f.size(0)) * self.iso
    #     sigma_inv = torch.linalg.inv(sigma)

    #     result_means = torch.zeros(num_series, self.prediction_length)
    #     for col in num_series:
    #         kron_k_fl_k_star_x_T = kronecker(K_f[:, col], self.K_star_x).t
    #         result_means[:, col] = kron_k_fl_k_star_x_T @ sigma_inv @ x

    #     result_means = result_means + loc_freq.repeat(1, prediction_length // self.freq)

    #     return MV(mean, cov)

    

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
