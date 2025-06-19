# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from itertools import chain
import matplotlib.pyplot as plt
import os 
from aim import Image


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

class GPWarmStartApprox(Callback):
    def __init__(self, train_loader, n_epochs: int, lr: float, info):
        super().__init__()
        self.train_loader = train_loader
        self.n_epochs = n_epochs
        self.lr = lr
        self.info = info

    # def on_fit_start(self, trainer, pl_module):
    #     gp = pl_module.q0
    #     device = pl_module.device

    #     self.info(f"Device used: {device}")
    #     gp.model.to(device)
    #     gp.likelihood.to(device)

    #     # Collect all parameters
    #     all_params = {id(p): p for p in chain(gp.model.parameters(), gp.likelihood.parameters())}
    #     named = dict(gp.model.named_parameters())
    #     named.update(dict(gp.likelihood.named_parameters()))
    #     self.info(f"GP hyperparameters:{named.keys()}))")

    #     optimizer = torch.optim.Adam(list(all_params.values()), lr=self.lr)
    #     # Variational ELBO setup
    #     num_data = gp.context_length * gp.num_tasks
    #     mll_elbo = VariationalELBO(gp.likelihood, gp.model, num_data=num_data)

    #     # Prepare time grids
    #     t_full = gp.t.to(device)
    #     t_c = gp.context_length
    #     X_ctx = t_full[:t_c].unsqueeze(-1)
    #     X_fut = t_full[t_c:].unsqueeze(-1)

    #     compute_wasserstein_avg()

    #     for epoch in range(self.n_epochs):
    #         epoch_loss = 0.0
    #         for batch in self.train_loader:
    #             # Scale context and future
    #             past = batch["past_target"].to(device)
    #             obs = batch["past_observed_values"].to(device)
    #             mean = batch["mean"].to(device)
    #             fut_target = batch["future_target"].to(device)

    #             if isinstance(pl_module.scaler, LongScaler):
    #                 scaled_ctx, loc, scale = pl_module.scaler(past[:, -t_c:], scale=mean)
    #             else:
    #                 _, loc, scale = pl_module.scaler(past, obs)
    #                 scaled_ctx = past[:, -t_c:] / scale
    #             scaled_fut = (fut_target - loc) / scale

    #             # Assume batch_size = 1
    #             B, _, C = scaled_ctx.shape  # B=1
    #             Y_ctx = scaled_ctx.reshape(B*C, t_c)  # [C, t_c]
    #             Y_fut = scaled_fut.reshape(B*C, t_c)  # [C, t_f]

    #             # (A) Context ELBO
    #             gp.model.train(); gp.likelihood.train()
    #             optimizer.zero_grad()
    #             out_ctx = gp.model(X_ctx)                     # dist with batch_shape=[C], event_shape=[t_c]
    #             loss_ctx = -mll_elbo(out_ctx, Y_ctx)

    #             self.info(f"Past calculting context loss")

    #             # (B) Predictive posterior LL via fantasy model
    #             gp.model.eval(); gp.likelihood.eval()
    #             # Condition on observed context
    #             fantasy_model = gp.model.get_fantasy_model(X_ctx, Y_ctx)
    #             fantasy_model.eval()
    #             with gpytorch.settings.fast_pred_var():
    #                 f_dist = fantasy_model(X_fut)         # [C, t_f]
    #             y_dist = gp.likelihood(f_dist)            # adds noise
    #             loss_fut = -y_dist.log_prob(Y_fut).sum()

    #             # (C) Combine & step
    #             (loss_ctx + loss_fut).backward()
    #             optimizer.step()
    #             epoch_loss += (loss_ctx + loss_fut).item()

    #         self.info(f"[Warm-start] Epoch {epoch+1}/{self.n_epochs} loss={epoch_loss:.3f}")

    #     compute_wasserstein_avg()
    #     # Freeze GP
    #     for p in gp.model.parameters(): p.requires_grad_(False)
    #     for p in gp.likelihood.parameters(): p.requires_grad_(False)
    #     gp.model.eval(); gp.likelihood.eval()

        # # Plot & log to Aim
        # batch0 = next(iter(self.train_loader))
        # past = batch0["past_target"].to(device)
        # obs = batch0["past_observed_values"].to(device)
        # mean = batch0["mean"].to(device)
        # if isinstance(pl_module.scaler, LongScaler):
        #     scaled_ctx, loc, scale = pl_module.scaler(past[:, -t_c:], scale=mean)
        # else:
        #     _, loc, scale = pl_module.scaler(past, obs)
        #     scaled_ctx = past[:, -t_c:] / scale

        # B, T, C = scaled_ctx.shape
        # t_full = gp.t.to(device)

        # Y_ctx = scaled_ctx.reshape(B * C, t_c).T 
        # gp.model.set_train_data(inputs=X_ctx, targets=Y_ctx, strict=False)
        # gp.model.eval(); gp.likelihood.eval()

        # # Draw prior samples and compute stats
        # K = 100
        # with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #     f_full = gp.model(t_full)
        #     y_full = gp.likelihood(f_full)
        #     samples = y_full.rsample(torch.Size([K]))  # [K, B*C, t_c+t_f]

        # mean_full = y_full.mean.cpu().numpy()         # [B*C, t_c+t_f]
        # std_full = y_full.stddev.cpu().numpy()        # [B*C, t_c+t_f]

        #         # Plot top-10 in vertical subplots with sample paths in original scale
        # # Invert scaling: loc & scale per series
        # loc_flat = loc.reshape(-1).cpu().numpy()      # [B*C]
        # scale_flat = scale.reshape(-1).cpu().numpy()  # [B*C]

        # indices = list(range(min(10, B * C)))
        # fig, axes = plt.subplots(len(indices), 1,
        #                          figsize=(12, 3 * len(indices)),
        #                          sharex=True)
        # time_axis = t_full.cpu().numpy()  # full grid

        # for i, idx in enumerate(indices):
        #     ax = axes[i]
        #     # scaled mean/std for series idx
        #     mu_s = mean_full[:, idx]    # [t_c + t_f]
        #     sd_s = std_full[:, idx]     # [t_c + t_f]
        #     # invert to original scale
        #     mu = mu_s * scale_flat[idx] + loc_flat[idx]
        #     sd = sd_s * scale_flat[idx]

        #     # 95% CI
        #     ax.fill_between(time_axis,
        #                     mu - 1.96 * sd,
        #                     mu + 1.96 * sd,
        #                     alpha=0.2)
        #     # 50% CI
        #     ax.fill_between(time_axis,
        #                     mu - 0.674 * sd,
        #                     mu + 0.674 * sd,
        #                     alpha=0.4)
        #     # GP mean
        #     ax.plot(time_axis, mu, color='blue', alpha=0.8)
        #     # Observed context in original scale
        #     obs_s = Y_ctx[:, idx].cpu().numpy()
        #     obs_orig = obs_s * scale_flat[idx] + loc_flat[idx]
        #     ax.plot(time_axis[:t_c], obs_orig, color='black', alpha=0.8)
        #     # Overlay 10 sample paths in original scale
        #     for k in range(min(10, K)):
        #         samp_s = samples[k, :, idx].cpu().numpy()
        #         samp_orig = samp_s * scale_flat[idx] + loc_flat[idx]
        #         ax.plot(time_axis, samp_orig, alpha=0.5, linestyle='--')
        #     ax.set_ylabel(f"Series {idx}")

        # axes[-1].set_xlabel('time')
        # fig.tight_layout()
        # fig.canvas.draw()
        # buf = np.asarray(fig.canvas.buffer_rgba())
        # img = buf[:, :, :3].astype('uint8')

        # aim_img = Image(img, format='png')
        # # Log to Aim
        # run = trainer.logger.experiment
        # step = trainer.global_step
        # run.track(
        #     aim_img,
        #     name='gp_top10_prior_samples_subplots',
        #     step=step,
        #     context={'subset': 'figures'}
        # )
        
        # plt.close(fig)
        # self.info(f"Logged vertical GP prior-sample subplots")



class GPWarmStart(Callback):
    def __init__(self, train_loader, n_epochs: int, lr: float, info):
        super().__init__()
        self.train_loader = train_loader
        self.n_epochs = n_epochs
        self.lr = lr
        self.info = info

    def compute_wasserstein_avg(self, gp, pl_module, X_fut, t_c, device):
        gp.model.eval(); gp.likelihood.eval()
        wd = []
        for batch in self.train_loader:
            past = batch["past_target"].to(device)
            obs = batch["past_observed_values"].to(device)
            mean = batch["mean"].to(device)
            fut_target = batch["future_target"].to(device)

            if isinstance(pl_module.scaler, LongScaler):
                scaled_ctx, loc, scale = pl_module.scaler(past[:, -t_c:], scale=mean)
            else:
                _, loc, scale = pl_module.scaler(past, obs)
                scaled_ctx = past[:, -t_c:] / scale
            scaled_fut = (fut_target - loc) / scale

            B, _, C = scaled_ctx.shape
            Y_fut = scaled_fut.reshape(B * C, -1)  # [B*C, t_f]

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                f_dist = gp.model(X_fut)
                y_dist = gp.likelihood(f_dist)
                y_pred = y_dist.mean.T  # [B*C, t_f]

            wd_batch = wasserstein(y_pred, Y_fut)
            wd.append(wd_batch)

        avg_ws = np.mean(wd)
        self.info(f"Wasserstein distance: {avg_ws:.4f}")
        gp.model.train(); gp.likelihood.train()
        return avg_ws


    def on_fit_start(self, trainer, pl_module):
        gp     = pl_module.q0
        device = pl_module.device


        self.info(f"Device used: {device}")
        # move GP to correct device
        gp.model      .to(device)
        gp.likelihood .to(device)

        # ---------------------------------------------------
        # PRECOMPUTE EVERYTHING THAT NEVER CHANGES
        # ---------------------------------------------------

        # 1) single optimizer over all GP params
        all_params = {id(p): p for p in chain(gp.model.parameters(),
                                               gp.likelihood.parameters())}
        
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

        # 2) one MLL for the context marginal likelihood
        mll_context = ExactMarginalLogLikelihood(gp.likelihood, gp.model)

        # 3) time‐grid splits (the same for every batch/epoch)
        t_full = gp.t.to(device)                                      # [t_c + t_f]  
        t_c    = gp.context_length                                    # scalar
        X_ctx  = t_full[:t_c]                                         # [t_c]
        X_fut  = t_full[t_c:]                                         # [t_f]

        # ---------------------------------------------------
        # WARM-START LOOP
        # ---------------------------------------------------

        self.compute_wasserstein_avg(gp, pl_module, X_fut, t_c, device)

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0

            for batch in self.train_loader:
                # 4) load & scale once
                past       = batch["past_target"].to(device)           # [B, T, C]
                obs        = batch["past_observed_values"].to(device)  # [B, T, C]
                mean       = batch["mean"].to(device)                  # [B, 1, C]
                fut_target = batch["future_target"].to(device)         # [B, t_f, C]

                if isinstance(pl_module.scaler, LongScaler):
                    scaled_ctx, loc, scale = pl_module.scaler(
                        past[:, -t_c:], scale=mean
                    )
                else:
                    _, loc, scale = pl_module.scaler(past, obs)
                    scaled_ctx = past[:, -t_c:] / scale

                scaled_fut = (fut_target - loc) / scale              # [B, t_f, C]

                # 5) flatten into GP’s “batch” dimension
                B, _, C = scaled_ctx.shape[0], scaled_ctx.shape[1], scaled_ctx.shape[2]
                Y_ctx = scaled_ctx .reshape(B * C, t_c).T            # [t_c, B*C]
                Y_fut = scaled_fut .reshape(B * C, -1).T             # [t_f, B*C]

                # ---------------------------------------------------
                # (A) CONTEXT MARGINAL LL
                # ---------------------------------------------------
                gp.model.set_train_data(inputs=X_ctx, targets=Y_ctx, strict=False)
                gp.model.train(); gp.likelihood.train()
                optimizer.zero_grad()

                out_ctx   = gp.model(X_ctx)                         # [t_c, B*C]
                loss_ctx  = -mll_context(out_ctx, Y_ctx)

                # ---------------------------------------------------
                # (B) PREDICTIVE POSTERIOR LL
                # ---------------------------------------------------
                # temporarily switch to eval‐mode *only* to avoid the "must train"
                # check; gradients are still tracked
                gp.model.eval(); gp.likelihood.eval()
                with gpytorch.settings.fast_pred_var():
                    f_dist = gp.model(X_fut)                       # latent posterior
                y_dist    = gp.likelihood(f_dist)                  # adds noise

                # log_prob returns [batch_shape = B*C], event_shape = [t_f]
                loss_fut = -y_dist.log_prob(Y_fut.T).sum()

                # switch back to train mode for the next iteration
                gp.model.train(); gp.likelihood.train()

                # ---------------------------------------------------
                # (C) COMBINE & STEP
                # ---------------------------------------------------
                loss = loss_ctx + loss_fut
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            self.info(f"[Warm-start] Epoch {epoch+1}/{self.n_epochs}  loss={epoch_loss:.3f}")

        self.compute_wasserstein_avg(gp, pl_module, X_fut, t_c, device)

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

        gp.model.eval(); gp.likelihood.eval()

        # Plot & log to Aim
        batch0 = next(iter(self.train_loader))
        past = batch0["past_target"].to(device)
        obs = batch0["past_observed_values"].to(device)
        mean = batch0["mean"].to(device)
        if isinstance(pl_module.scaler, LongScaler):
            scaled_ctx, loc, scale = pl_module.scaler(past[:, -t_c:], scale=mean)
        else:
            _, loc, scale = pl_module.scaler(past, obs)
            scaled_ctx = past[:, -t_c:] / scale

        B, T, C = scaled_ctx.shape
        t_full = gp.t.to(device)
        t_ctx = t_full[:t_c]

        Y_ctx = scaled_ctx.reshape(B * C, t_c).T 
        gp.model.set_train_data(inputs=X_ctx, targets=Y_ctx, strict=False)
        gp.model.eval(); gp.likelihood.eval()

        # Draw prior samples and compute stats
        K = 100
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_full = gp.model(t_full)
            y_full = gp.likelihood(f_full)
            samples = y_full.rsample(torch.Size([K]))  # [K, B*C, t_c+t_f]

        mean_full = y_full.mean.cpu().numpy()         # [B*C, t_c+t_f]
        std_full = y_full.stddev.cpu().numpy()        # [B*C, t_c+t_f]

                # Plot top-10 in vertical subplots with sample paths in original scale
        # Invert scaling: loc & scale per series
        loc_flat = loc.reshape(-1).cpu().numpy()      # [B*C]
        scale_flat = scale.reshape(-1).cpu().numpy()  # [B*C]

        indices = list(range(min(10, B * C)))
        fig, axes = plt.subplots(len(indices), 1,
                                 figsize=(12, 3 * len(indices)),
                                 sharex=True)
        time_axis = t_full.cpu().numpy()  # full grid

        for i, idx in enumerate(indices):
            ax = axes[i]
            # scaled mean/std for series idx
            mu_s = mean_full[:, idx]    # [t_c + t_f]
            sd_s = std_full[:, idx]     # [t_c + t_f]
            # invert to original scale
            mu = mu_s * scale_flat[idx] + loc_flat[idx]
            sd = sd_s * scale_flat[idx]

            # 95% CI
            ax.fill_between(time_axis,
                            mu - 1.96 * sd,
                            mu + 1.96 * sd,
                            alpha=0.2)
            # 50% CI
            ax.fill_between(time_axis,
                            mu - 0.674 * sd,
                            mu + 0.674 * sd,
                            alpha=0.4)
            # GP mean
            ax.plot(time_axis, mu, color='blue', alpha=0.8)
            # Observed context in original scale
            obs_s = Y_ctx[:, idx].cpu().numpy()
            obs_orig = obs_s * scale_flat[idx] + loc_flat[idx]
            ax.plot(time_axis[:t_c], obs_orig, color='black', alpha=0.8)
            # Overlay 10 sample paths in original scale
            for k in range(min(10, K)):
                samp_s = samples[k, :, idx].cpu().numpy()
                samp_orig = samp_s * scale_flat[idx] + loc_flat[idx]
                ax.plot(time_axis, samp_orig, alpha=0.5, linestyle='--')
            ax.set_ylabel(f"Series {idx}")

        axes[-1].set_xlabel('time')
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        img = buf[:, :, :3].astype('uint8')

        aim_img = Image(img, format='png')
        # Log to Aim
        run = trainer.logger.experiment
        step = trainer.global_step
        run.track(
            aim_img,
            name='gp_top10_prior_samples_subplots',
            step=step,
            context={'subset': 'figures'}
        )
        
        plt.close(fig)
        self.info(f"Logged vertical GP prior-sample subplots")


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
