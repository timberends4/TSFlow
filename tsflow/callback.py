# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from itertools import chain
import matplotlib.pyplot as plt
import os 
from aim import Image
import time
from einops import rearrange

import numpy as np
import torch
from gluonts.evaluation import (
    Evaluator,
    MultivariateEvaluator,
    make_evaluation_predictions,
)
from pytorch_lightning import Callback
from tqdm.auto import tqdm

from gpytorch.mlls import ExactMarginalLogLikelihood, PredictiveLogLikelihood
from gpytorch.mlls import VariationalELBO


from tsflow.utils import Setting
from tsflow.utils.plots import plot_figures
from tsflow.utils.util import create_splitter
from tsflow.utils.util import LongScaler
from tsflow.metrics import wasserstein

import gpytorch 


class GPWarmStart(Callback):
    def __init__(self, train_loader, n_epochs: int, lr: float, info, trained_prior: bool):
        super().__init__()
        self.train_loader = train_loader
        self.n_epochs = n_epochs
        self.lr = lr
        self.info = info
        self.trained_prior = trained_prior

    def preprocess_input(self, pl_module, batch, device, context_length, prior_context_length):
        
        past = batch["past_target"].to(device)
        future = batch["future_target"].to(device)
        context_observed = batch["past_observed_values"].to(device)
        mean = batch["mean"].to(device)

        context = past[:, -context_length :]
        prior_context = past[:, -prior_context_length :]

        if isinstance(pl_module.scaler, LongScaler):
            scaled_context, loc, scale = pl_module.scaler(context, scale=mean)
        else:
            _, loc, scale = self.scaler(past, context_observed)
            scaled_context = context / scale

        scaled_prior_context = (prior_context - loc) / scale
        scaled_future = (future - loc) / scale

        x1 = torch.cat([scaled_context, scaled_future], dim=-2)
        return scaled_prior_context, scaled_context, x1, scaled_future, loc, scale


    def plot_samples_and_context(self, pl_module, trainer, batch, n_samples, plot_title):
        gp     = pl_module.q0
        device = pl_module.device

        if pl_module.trained_prior:
            gp.model.eval(); gp.likelihood.eval()
        with torch.no_grad():
            K = n_samples
            # Plot & log to Aim
            scaled_prior_context, scaled_context, x1, scaled_future, loc, scale = self.preprocess_input(pl_module, batch, device, pl_module.context_length, pl_module.prior_context_length)
            
            if pl_module.prior_name == "Q0Dist":
                dist = gp.gp_regression(rearrange(scaled_prior_context, "b l c -> (b c) l"), pl_module.prediction_length)
                fut_samples = dist.rsample(torch.Size([K]))
            
            elif pl_module.prior_name == "Q0DistMultiTask":
                dist = gp.gp_regression(rearrange(scaled_context, "b l c -> (b c) l"), pl_module.prediction_length)

                fut_samples = dist[0].rsample(torch.Size([K]))
                fut_samples = rearrange(fut_samples, "n l c -> n c l")

            loc0   = loc[0].squeeze(0).to(device)    # [C]
            scale0 = scale[0].squeeze(0).to(device)  # [C]

            # 7) pull out the context for batch0, shape [t_c, C]
            ctx_scaled = scaled_context[0]               # [t_c, C]

            # 8) make it [C, t_c], then repeat for K samples → [K, C, t_c]
            ctx_rep = ctx_scaled.permute(1,0).unsqueeze(0).repeat(K,1,1)

            # 9) concatenate [K,C,t_c] + [K,C,t_f] → [K,C,t_c+t_f], then invert scale/bias
            combined = torch.cat([ctx_rep, fut_samples], dim=-1)  # [K, C, T]
            combined = combined * scale0.unsqueeze(0).unsqueeze(-1) \
                                + loc0.unsqueeze(0).unsqueeze(-1)

            # 10) compute mean & std over the K samples → [C, T] each
            mean_full = combined.mean(dim=0).cpu().numpy()  # [C, T]
            std_full  = combined.std(dim=0).cpu().numpy()   # [C, T]

            # 11) set up time axis and which series to show
            time_axis = torch.arange(mean_full.shape[1])              # [T = t_c + t_f]
            num_series = mean_full.shape[0]                # = C
            indices = list(range(min(10, num_series)))

            # 12) plot
            fig, axes = plt.subplots(len(indices), 1,
                                    figsize=(12, 3*len(indices)),
                                    sharex=True)

            fig.suptitle(plot_title, fontsize=16)

            self.info(f"Mean full shape {mean_full.shape}")

            for i, idx in enumerate(indices):
                ax = axes[i]
                mu = mean_full[idx]       # [T]
                sd = std_full[idx]        # [T]

                # 95% & 50% CI
                ax.fill_between(time_axis, mu-1.96*sd, mu+1.96*sd, alpha=0.2)
                ax.fill_between(time_axis, mu-0.674*sd, mu+0.674*sd, alpha=0.4)

                # posterior mean
                ax.plot(time_axis, mu, alpha=0.8)

                # observed context (invert scaling)
                obs_s = ctx_scaled[:, idx].cpu().numpy()     # [t_c]
                obs = obs_s * scale0[idx].cpu().numpy() \
                    + loc0[idx].cpu().numpy()
                ax.plot(time_axis[:gp.prediction_length], obs, color="black", alpha=0.8)

                # overlay up to 10 sample paths
                for k in range(min(10, combined.shape[0])):
                    samp = combined[k, idx].cpu().numpy()
                    ax.plot(time_axis, samp, linestyle="--", alpha=0.5)

                ax.set_ylabel(f"Series {idx}")
                ax.set_ylim(0,2)

            axes[-1].set_xlabel("time")
            fig.tight_layout()

            # 13) convert to RGB and log to Aim (unchanged)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype("uint8")
            aim_img = Image(buf, format="png")
            run = trainer.logger.experiment
            step = trainer.global_step
            run.track(aim_img,
                    name=f"gp_top10_prior_samples_subplots",
                    step=step,
                    context={"subset": "figures", "time": f"time:{time.time():.3f}"}
            )
            plt.close(fig)

            self.info("Logged vertical GP prior-sample subplots")

    def estimate_empirical_covariance(self, context: torch.Tensor) -> torch.Tensor:
        B, Lc, C = context.shape
        X = context.reshape(B * Lc, C)  # [N, C]

        mean = X.mean(dim=0, keepdim=True)   # [1, C]
        X_centered = X - mean                # [N, C]

        cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)  # [C, C]
        return cov


    def compute_wasserstein_avg(self, gp, pl_module, device):
        if pl_module.trained_prior:
            gp.model.eval(); gp.likelihood.eval()
        wd = []
        wd_iso = []
        num_samples = 1000

        for data in self.train_loader:
            # — your existing preprocessing & GP sampling —
            scaled_prior_context, scaled_context, x1, scaled_future, loc, scale = \
                self.preprocess_input(pl_module, data, device, pl_module.context_length, pl_module.prior_context_length)
            
            batch_size, length, c = x1.shape

            # GP prior samples
            if pl_module.prior_name == "Q0Dist":    
                dist = gp.gp_regression(
                    rearrange(scaled_prior_context, "b l c -> (b c) l"),
                    gp.prediction_length
                )       
                fut_samples = dist.rsample(torch.Size([num_samples]))  # [K, L_f, C]
                fut_samples = rearrange(fut_samples, "n c l -> n l c")

            elif pl_module.prior_name == "Q0DistMultiTask":
                dist = gp.gp_regression(
                    rearrange(scaled_context, "b l c -> (b c) l"),
                    gp.prediction_length
                )
                fut_samples = dist[0].rsample(torch.Size([num_samples]))  # [K, L_f, C]


            # repeat context
            x0_repeated = scaled_context.repeat(num_samples, 1, 1)    # [K, L_c, C]
            # joint samples: [context | future]
            x0_samples = torch.cat([x0_repeated, fut_samples], dim=-2) # [K, L_c+L_f, C]

            # compute GP‐based WD
            wd_batch = wasserstein(x0_samples, x1)
            wd.append(wd_batch)

            # — now the isotropic baseline —
            # sample future from N(0,1) with same shape
            fut_iso = torch.randn_like(fut_samples)               # [K, L_f, C]
            x_iso_samples = torch.cat([x0_repeated, fut_iso], dim=-2) # [K, L_c+L_f, C]

            # compute isotropic WD
            wd_iso_batch = wasserstein(x_iso_samples, x1)
            wd_iso.append(wd_iso_batch)

        # average over all batches
        avg_ws     = float(np.mean(wd))
        avg_ws_iso = float(np.mean(wd_iso))
        rel_ws     = avg_ws / avg_ws_iso if avg_ws_iso != 0 else float('inf')

        self.info(f"Wasserstein distance (GP prior):      {avg_ws:.4f}")
        self.info(f"Wasserstein distance (isotropic prior): {avg_ws_iso:.4f}")
        self.info(f"Relative WD (GP / isotropic):           {rel_ws:.4f}")

        if pl_module.trained_prior:
            gp.model.train(); gp.likelihood.train()
        return avg_ws, avg_ws_iso, rel_ws


    def on_fit_start(self, trainer, pl_module):
        gp     = pl_module.q0
        device = pl_module.device

        # ---------------------------------------------------
        # WARM-START LOOP
        # ---------------------------------------------------
        batch0 = next(iter(self.train_loader))

        self.plot_samples_and_context(pl_module, trainer, batch0 ,100, "GP Prior Fixed Time Kernel Only")
        self.compute_wasserstein_avg(gp, pl_module, device)

        if self.trained_prior:
            self.info(f"Device used: {device}")
            # move GP to correct device
            gp.model      .to(device)
            gp.likelihood .to(device)

            # 1) single optimizer over all GP params
            all_params = {id(p): p for p in chain(gp.model.parameters(),
                                                gp.likelihood.parameters()) if p.requires_grad}
            
            self.info(f"All trainable params: {all_params}")
            # 1) collect model + likelihood named parameters
            named = dict(gp.model.named_parameters())
            named.update(dict(gp.likelihood.named_parameters()))

            # 2) log just the names
            param_names = list(named.keys())
            self.info(f"GP hyperparameter names:\n" + "\n".join(param_names))

            # — or if you also want shapes:
            name_shapes = [f"{n}: {tuple(t.shape)}" for n, t in named.items()]
            self.info("GP parameters (name : shape):\n" + "\n".join(name_shapes))

            optimizer = torch.optim.Adam(list(all_params.values()), lr=self.lr)

            for epoch in range(self.n_epochs):
                epoch_loss = 0.0

                for batch in self.train_loader:
                
                    scaled_prior_context, scaled_context, x1, scaled_future, loc, scale = self.preprocess_input(pl_module, batch, device, pl_module.context_length, pl_module.prior_context_length)
                    batch_size, length, c = x1.shape


                    gp.model.train(); gp.likelihood.train()
                    optimizer.zero_grad()

                    with gpytorch.settings.fast_computations(
                        covar_root_decomposition=False,
                        solves=False,
                        log_prob=False,
                    ):
                        dist = gp.gp_regression(rearrange(scaled_context, "b l c -> (b c) l"), gp.prediction_length)
                        if pl_module.prior_name == "Q0Dist":
                            loss = -dist.log_prob(rearrange(scaled_future, "b l c -> (b c) l")).mean()
                        elif pl_module.prior_name == "Q0DistMultiTask":
                            loss = -dist[0].log_prob(scaled_future).mean()

                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()

                self.info(f"[Warm-start] Epoch {epoch+1}/{self.n_epochs}  loss={epoch_loss:.3f}")

                # emp_cov = self.estimate_empirical_covariance(scaled_prior_context.cpu())
                # self.info(f"Empirical context covariance matrix:\n{emp_cov.numpy()}")

            self.plot_samples_and_context(pl_module, trainer, batch0 , 100, "Multi Task GP Prior with Fixed Time Kernel")
            self.compute_wasserstein_avg(gp, pl_module, device)

            # Freeze GP
            for name, param in gp.model.named_parameters():
                param.requires_grad_(False)
                self.info(f"Likelihood param: {name} | value: {param.data.cpu().numpy()}")

                if "covar_factor" in name:
                    # Compute full task covariance matrix
                    B_full = param @ param.T  # [num_tasks, num_tasks]
                    
                    # Compute correlation matrix
                    diag = torch.diag(B_full).sqrt()
                    denom = diag.unsqueeze(0) * diag.unsqueeze(1)
                    corr = B_full / (denom + 1e-8)  # add epsilon for numerical stability

                    # Log or print
                    self.info(f"Full task covariance matrix B:\n{B_full.cpu().numpy()}")
                    self.info(f"Task correlation matrix:\n{corr.cpu().numpy()}")

            self.info("Freezing GP likelihood parameters:")
            for name, param in gp.likelihood.named_parameters():
                param.requires_grad_(False)
                self.info(f"Likelihood param: {name} | value: {param.data.cpu().numpy()}")



class EvaluateCallback(Callback):
    def __init__(
        self,
        context_length,
        prediction_length,
        num_samples,
        model,
        datasets,
        logdir,
        eval_every=1,
        **kwargs,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.model = model
        self.datasets = datasets
        self.logdir = logdir
        self.eval_every = eval_every
        self.log_metrics = (
            {
                "CRPS",
                "ND",
                "NRMSE",
                "m_sum_CRPS",
            }
            if model.setting == Setting.MULTIVARIATE
            else {"CRPS", "ND", "NRMSE"}
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if (pl_module.current_epoch + 1) % self.eval_every == 0:
            device = pl_module.device

            pl_module.eval()
            assert pl_module.training is False
            prediction_splitter = create_splitter(
                max(
                    self.context_length + max(self.model.lags_seq),
                    self.model.prior_context_length,
                ),
                self.prediction_length,
                mode="test",
            )
            for tag, dataset in self.datasets.items():
                pl_module.num_samples = self.num_samples

                if pl_module.setting == Setting.UNIVARIATE:
                    batch_size = 1024 // self.num_samples
                    evaluator = Evaluator(num_workers=1)
                elif pl_module.setting == Setting.MULTIVARIATE:
                    batch_size = 1
                    evaluator = MultivariateEvaluator(target_agg_funcs={"sum": np.sum})

                predictor_pytorch = pl_module.get_predictor(
                    prediction_splitter,
                    batch_size=batch_size,
                    device=device,
                )
                forecast_it, ts_it = make_evaluation_predictions(
                    dataset=dataset,
                    predictor=predictor_pytorch,
                    num_samples=self.num_samples,
                )

                forecasts_pytorch = list(tqdm(forecast_it, total=len(dataset)))
                tss_pytorch = list(ts_it)
                metrics_pytorch, per_ts = evaluator(tss_pytorch, forecasts_pytorch)
                metrics_pytorch["CRPS"] = metrics_pytorch["mean_wQuantileLoss"]
                if pl_module.setting == Setting.UNIVARIATE:
                    plot_figures(
                        tss_pytorch[0:4],
                        forecasts_pytorch[0:4],
                        pl_module.context_length,
                        pl_module.prediction_length,
                        trainer,
                        set=f"{tag}",
                    )
                else:
                    metrics_pytorch["m_sum_CRPS"] = metrics_pytorch[
                        "m_sum_mean_wQuantileLoss"
                    ]
                if metrics_pytorch["CRPS"] < pl_module.best_crps and tag == "val":
                    pl_module.best_crps = metrics_pytorch["CRPS"]
                    ckpt_path = Path(self.logdir) / "best_checkpoint.ckpt"
                    torch.save(
                        pl_module.state_dict(),
                        ckpt_path,
                    )
                pl_module.log_dict(
                    {
                        f"{tag}_{metric}": metrics_pytorch[metric]
                        for metric in self.log_metrics
                    }
                )
            pl_module.train()
