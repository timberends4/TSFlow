import torch
from torch.distributions import MultivariateNormal as MV
from torchtyping import TensorType

from typing import List, Union, Optional

import gpytorch
from gpytorch.means import ConstantMean, ZeroMean
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
    def __init__(
        self,
        kernel,  # still unused; fixed to RBF
        context_freqs: int,
        prediction_length: int,
        num_tasks: int = 370,
        freq: int = 24,
        gamma: float = 1.0,
        iso: float = 1e-4,
        info=None,
    ):
        super().__init__()
        self.info = info
        self.freq = freq
        self.num_tasks = 370
        self.gamma = torch.tensor(gamma, dtype=torch.float64)
        self.iso = torch.tensor(iso, dtype=torch.float64)

        self.context_length = context_freqs * prediction_length
        self.prediction_length = prediction_length
        self.window_length = self.context_length + self.prediction_length

        # Define training time points
        self.register_buffer("t", torch.arange(self.window_length, dtype=torch.float64) * (torch.pi / freq))
        self.register_buffer("t_obs", torch.cat([
            torch.ones(self.context_length), torch.zeros(self.prediction_length)
        ]).bool())
        self.register_buffer("t_new", ~self.t_obs)

        # Add explicit batch dim to x and y
        train_x = self.t
        train_y = torch.zeros(self.window_length, self.num_tasks, dtype=torch.float64)  # shape [1, T, N]

        self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.num_tasks)
        
        # Set shared noise to zero (disable shared noise)
        self.likelihood.noise = self.iso
        self.likelihood.raw_noise.requires_grad_(False)

        # # Set per-task noises to iso (fixed)
        # self.likelihood.task_noises = torch.full((num_tasks,), self.iso, dtype=torch.float64)
        # self.likelihood.raw_task_noises.requires_grad_(False)

        class ExactMultitaskGP(ExactGP):
            def __init__(self, train_x, train_y, likelihood, num_tasks):
                super().__init__(train_x, train_y, likelihood)

                self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ZeroMean(), num_tasks=num_tasks
                )
                base_kernel = RBFKernel()
                base_kernel.lengthscale = gamma
                base_kernel.raw_lengthscale.requires_grad = False

                self.covar_module = MultitaskKernel(
                    base_kernel,
                    num_tasks=num_tasks,
                    rank = 3)

            def forward(self, x):  # x: [B, T]
                mean_x = self.mean_module(x)  # [B, T]
                covar_x = self.covar_module(x)

                return MultitaskMultivariateNormal(mean_x, covar_x)

        self.info(f"NUm tasks : {self.num_tasks}")
        self.model = ExactMultitaskGP(train_x, train_y, self.likelihood, self.num_tasks)

        self.model = self.model.double()
        self.likelihood = self.likelihood.double()

        if self.info:
            self.info(f"[Q0DistMultiTask] kernel=RBF, γ={gamma}, window={self.window_length}")

    def log_likelihood(self, x: TensorType[float, "B", "T*N"]) -> TensorType[float]:
        likelihood_num = self.model(self.t).likelihood(x.reshape(-1, self.num_tasks))
        return likelihood_num

    def forward(self, num_samples: int) -> TensorType[float, "num_samples", "T*N"]:
        pred_dist = self.likelihood(self.model(self.t))
        return pred_dist.rsample(num_samples)
    
    def gp_regression(self, x: TensorType[float, "B", "C", "L"], prediction_length: int) -> List[MV]:
        batch_size, num_tasks, context_len = x.shape
        results = []

        for b in range(batch_size):
            x_b = x[b].T  # [L, N]

            # Compute group mean (loc_freq)
            loc_freq = x_b.mean(0, keepdims=True)  # [1, N]

            # Center data by subtracting loc_freq
            x_centered = x_b - loc_freq  # [L, N]

            # Build targets: center context, pad nan for future
            targets_gp = torch.cat([
                x_centered,
                torch.full((prediction_length, num_tasks), float("nan"), device=x.device, dtype=x.dtype)
            ])

            inputs_gp = self.t[:targets_gp.shape[0]]

            self.model.set_train_data(inputs=inputs_gp, targets=targets_gp, strict=False)

            # You may still choose to do quick fine-tuning
            self.model.train()
            self.likelihood.train()

            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

            for i in range(2):
                self.info("Training it {i}")
                optimizer.zero_grad()
                output = self.model(inputs_gp)
                loss = -self.likelihood(output).log_prob(targets_gp).sum()
                loss.backward()
                optimizer.step()

            # Predict
            self.model.eval()
            self.likelihood.eval()

            pred_dist = self.likelihood(self.model(self.t))

            # Add back loc_freq to the mean
            mean_rescaled = pred_dist.mean + loc_freq.repeat(pred_dist.mean.shape[0], 1)
            cov_rescaled = pred_dist.covariance_matrix

            results.append(MV(mean_rescaled, cov_rescaled))

            self.model.prediction_strategy = None

        return results

    
    # def gp_regression(self, x: TensorType[float, "B", "T"], prediction_length: int) -> MV:
    #     targets_gp = torch.cat([x.reshape(-1, self.num_tasks), torch.full((prediction_length, self.num_tasks), float("nan"), device = x.device, dtype=x.dtype)])
        
    #     self.model.set_train_data(
    #         inputs = self.t,
    #         targets = targets_gp,
    #         strict=False
    #     )

    #     pred_dist = self.likelihood(self.model(self.t))

    #     return pred_dist


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
