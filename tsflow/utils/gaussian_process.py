import torch
from torch.distributions import MultivariateNormal as MV
from torchtyping import TensorType

from typing import List, Union, Optional

import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel, MultitaskKernel
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.models import ExactGP
from gpytorch.settings import fast_pred_var 

from tsflow.utils.variables import Prior


def isotropic_kernel(
    gamma: TensorType[float], t: TensorType[float, "length"], u: TensorType[float, "length"]
) -> TensorType[float, "length", "length"]:
    return gamma * torch.eye(t.size(0), device=t.device, dtype=t.dtype)


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
    """
    A multitask GP with a fixed RBF time kernel and a learnable
    coregionalization matrix over `num_tasks` series.
    """
    def __init__(
        self,
        kernel: Prior,               # not used here since we fix to RBF
        context_freqs: int,
        prediction_length: int,
        num_tasks: int,
        freq: int = 24,
        gamma: float = 1.0,
        iso: float = 1e-4,
        info=None,
    ):
        super().__init__()
        self.info = info

        # how many past time‐points per series we condition on:
        self.context_length = context_freqs * prediction_length
        self.prediction_length = prediction_length
        self.num_tasks = num_tasks

        # total window (obs + new)
        self.T = self.context_length + prediction_length

        # build a 1D “time” grid [0, π/freq, 2π/freq, …]
        t = torch.arange(self.T, dtype=torch.float32) * (torch.pi / freq)
        self.register_buffer("train_x", t.unsqueeze(-1))  # shape [T,1]

        # 1) zero mean
        self.mean_module = ConstantMean()

        # 2) fixed RBF (squared‐exp) time kernel:
        #    Pyro’s RBF uses k(t,u)=exp(-½ (t-u)²/ℓ²), so to get
        #    exp(-γ (t-u)²) we set ℓ = 1/√(2γ).
        base_rbf = RBFKernel()
        ℓ = (1.0 / (2.0 * gamma)) ** 0.5
        # copy into the leaf Parameter and freeze it
        base_rbf.lengthscale = base_rbf.lengthscale.detach().fill_(ℓ)
        # base_rbf.lengthscale.requires_grad_(False)

        # wrap in a ScaleKernel so we can register diagonal iso‐noise
        time_kernel = ScaleKernel(base_rbf)
        time_kernel.outputscale = 1.0
        time_kernel.outputscale.requires_grad_(False)
        # add iso jitter on the diagonal
        time_kernel.register_buffer("noise", iso * torch.eye(self.T))

        # 3) multitask coregionalization layer (full‐rank)
        self.covar_module = MultitaskKernel(
            time_kernel,
            num_tasks=num_tasks,
            rank=num_tasks,
        )

        if self.info:
            self.info(f"[Q0MT-GP] T={self.T}, tasks={self.num_tasks}")

    def _build_prior(self) -> MultitaskMultivariateNormal:
        # mean: [T] → expand to [T, num_tasks]
        mean_t = self.mean_module(self.train_x).squeeze(-1)
        mean = mean_t.unsqueeze(-1).expand(-1, self.num_tasks)
        # covariance wraps a [T⋅num_tasks, T⋅num_tasks] internally
        covar = self.covar_module(self.train_x)
        return MultitaskMultivariateNormal(mean, covar)

    def forward(
        self,
        ids: torch.LongTensor,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Draw `num_samples` prior samples for the subset of tasks in `ids`.
        Returns: Tensor[num_samples, S, T]
        """
        ids = ids.to(self.train_x.device).flatten()
        prior = self._build_prior()                                        # → shape (T, num_tasks)
        draws = prior.sample(sample_shape=torch.Size([num_samples]))        # → (num_samples, T, num_tasks)
        draws = draws.permute(0, 2, 1)                                      # → (num_samples, num_tasks, T)
        return draws[:, ids, :]                                             # → (num_samples, S, T)

    def gp_regression(
        self,
        x: torch.Tensor,               # [S, T_obs]
        ids: torch.LongTensor,
        pred_len: int,
    ) -> MultitaskMultivariateNormal:
        """
        Exact‐GP posterior over the new block of length `self.prediction_length`.
        """
        device = x.device
        ids = ids.to(device).flatten()
        S, T_obs = x.shape
        T_new = self.prediction_length
        assert T_obs + T_new == self.T

        # split inputs
        train_x_obs = self.train_x[:T_obs, :]
        train_x_new = self.train_x[T_obs:, :]

        # build a tiny ExactGP model on the observed block
        class ExactMultitaskGP(ExactGP):
            def __init__(self, train_x, train_y, likelihood, parent):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = ConstantMean()
                # reuse the same RBF+coregionalization
                self.covar_module = MultitaskKernel(
                    parent.covar_module.base_kernel,  # the RBF
                    num_tasks=parent.num_tasks,
                    rank=parent.num_tasks,
                )
                # copy coregionalization weights
                self.covar_module.task_kernel.covar_factor.data = (
                    parent.covar_module.task_kernel.covar_factor.data.clone()
                )
                self.covar_module.task_kernel.covar_log_var.data = (
                    parent.covar_module.task_kernel.covar_log_var.data.clone()
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return MultitaskMultivariateNormal(mean_x, covar_x)

        # transpose observations to [T_obs, S]
        train_y = x.T.contiguous()
        # zero‐noise likelihood
        lik = MultitaskGaussianLikelihood(num_tasks=self.num_tasks)
        lik.noise_covar.register_buffer("noise", torch.zeros(self.num_tasks, device=device))

        model = ExactMultitaskGP(train_x_obs, train_y, lik, self)
        model.eval(); lik.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = lik(model(train_x_new))
            return posterior
        

class Q0Dist(torch.nn.Module):
    def __init__(
        self,
        kernel: Prior,
        context_freqs: int,
        prediction_length: int,
        freq: int = 24,
        gamma: float = 1.0,
        iso: float = 1e-4,
        info = None
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
        samples = self.dist.sample(torch.Size([num_samples]))
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


class Q0Linear(torch.nn.Module):
    def __init__(
        self,
        context_freqs,
        prediction_length,
        freq=24,
        **kwargs,
    ):
        super().__init__()
        context_length = context_freqs * prediction_length
        self.context_length = context_length
        self.freq = freq
        self.mean_linear = torch.nn.Sequential(
            torch.nn.Linear(context_length + 2 * freq, prediction_length),
            torch.nn.ReLU(),
            torch.nn.Linear(prediction_length, prediction_length),
            torch.nn.ReLU(),
            torch.nn.Linear(prediction_length, prediction_length),
        )
        self.cov_linear = torch.nn.Sequential(
            torch.nn.Linear(context_length + 2 * freq, prediction_length),
            torch.nn.ReLU(),
            torch.nn.Linear(prediction_length, prediction_length),
            torch.nn.ReLU(),
            torch.nn.Linear(prediction_length, prediction_length),
            torch.nn.Softplus(),
        )

    def regression(self, x, prediction_length):
        loc_freq = x.reshape(x.shape[0], -1, self.freq).mean(1, keepdims=False)
        x = x - loc_freq.repeat(1, self.context_length // self.freq)
        std_freq = (x.reshape(x.shape[0], -1, self.freq).std(1, keepdims=False)) + 1e-4
        x = x / std_freq.repeat(1, self.context_length // self.freq)
        mean = self.mean_linear(torch.cat([x, loc_freq, std_freq], 1)) + loc_freq.repeat(
            1, prediction_length // self.freq
        )
        cov = self.cov_linear(torch.cat([x, loc_freq, std_freq], 1))

        scalar = std_freq.repeat(1, prediction_length // self.freq)
        cov = cov[:, None] * torch.eye(prediction_length, device=x.device) * scalar.unsqueeze(-1)  # + 1e-3 * torch.eye(
        return mean, cov


if __name__ == "__main__":
    a = 3
    q0 = Q0Dist(kernel=Prior.SE, context_freqs=5, prediction_length=5, gamma=1.0)
    breakpoint()
    a = 5
