import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pykeops
import pytorch_lightning as pl
import torch
import yaml
from aim.pytorch_lightning import AimLogger
from aim import Text

from gluonts.dataset.loader import TrainDataLoader
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.split import OffsetSplitter
from gluonts.evaluation import Evaluator, MultivariateEvaluator, make_evaluation_predictions
from gluonts.itertools import Cached
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm.auto import tqdm
from tsflow.callback import EvaluateCallback, GPWarmStart
from tsflow.dataset import get_gts_dataset
from tsflow.model import TSFlowCond
from tsflow.utils import create_multivariate_transforms, create_transforms
from tsflow.utils.util import ConcatDataset, add_config_to_argparser, create_splitter, filter_metrics
from tsflow.utils.variables import get_season_length

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Period with BDay freq is deprecated.*"
)

# PyKeOps build folder
temp_build_folder = tempfile.mkdtemp(prefix="pykeops_build_")
pykeops.set_build_folder(temp_build_folder)
pykeops.clean_pykeops()

# Use Tensor Cores
torch.set_float32_matmul_precision('medium')


def create_model(setting, target_dim, model_params):
    model = TSFlowCond(
        setting=setting,
        target_dim=target_dim,
        context_length=model_params["context_length"],
        prediction_length=model_params["prediction_length"],
        backbone_params=model_params["backbone_params"],
        prior_params=model_params["prior_params"],
        optimizer_params=model_params["optimizer_params"],
        ema_params=model_params["ema_params"],
        frequency=model_params["freq"],
        normalization=model_params["normalization"],
        use_lags=model_params["use_lags"],
        use_ema=model_params["use_ema"],
        num_steps=model_params["num_steps"],
        solver=model_params["solver"],
        matching=model_params["matching"],
        info = info
    )
    model.to(model_params["device"])
    return model


def evaluate_conditional(
    model_params,
    model: TSFlowCond,
    test_dataset,
    transformation,
    trainer,
    info,
    num_samples=100,
):
    info(f"Evaluating with {num_samples} samples.")
    results = {}

    transformed_testdata = transformation.apply(test_dataset, is_train=False)
    test_splitter = create_splitter(
        past_length=max(
            model_params["context_length"] + max(model.lags_seq),
            model.prior_context_length,
        ),
        future_length=model_params["prediction_length"],
        mode="test",
    )

    if model.setting == "univariate":
        batch_size = 1024 * 64 // num_samples
        evaluator = Evaluator(num_workers=1)
    else:
        batch_size = 1
        evaluator = MultivariateEvaluator(target_agg_funcs={"sum": np.sum})

    predictor = model.get_predictor(
        test_splitter,
        batch_size=batch_size,
        device=model_params["device"],
    )
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=transformed_testdata,
        predictor=predictor,
        num_samples=num_samples,
    )
    forecasts = list(tqdm(forecast_it, total=len(transformed_testdata)))
    tss = list(ts_it)
    metrics, _ = evaluator(tss, forecasts)
    metrics["CRPS"] = metrics["mean_wQuantileLoss"]
    select = ["CRPS", "ND", "NRMSE"]
    if model.setting == "multivariate":
        metrics["m_sum_CRPS"] = metrics["m_sum_mean_wQuantileLoss"]
        select.append("m_sum_CRPS")
    metrics = filter_metrics(metrics, select)

    for lg in trainer.loggers:
        lg.log_metrics(
            {f"test_{k}": v for k, v in metrics.items()},
            step=trainer.global_step + 1,
        )

    results["test"] = dict(metrics)
    return results


def main(
    model,
    setting,
    model_params,
    dataset_params,
    trainer_params,
    evaluation_params,
    info,
    logdir=None,
    seed=None,
    config=None,
    loggers=[],
):
    if logdir:
        Path(logdir).mkdir(parents=True, exist_ok=True)
    pl.seed_everything(seed)

    # Load dataset
    dataset_name = dataset_params["dataset"]
    freq = model_params["freq"]
    prediction_length = model_params["prediction_length"]

    dataset = get_gts_dataset(dataset_name)
    target_dim = min(2000, int(dataset.metadata.feat_static_cat[0].cardinality))

    # Log dataset info
    info(f"Dataset {dataset_name} has target dimension: {target_dim}")

    model_params['info'] = info
    # Build model
    model = create_model(setting, target_dim, model_params)

    # Sanity checks
    info(f"Dataset metadata freq {dataset.metadata.freq, freq}")
    # assert dataset.metadata.freq == freq
    assert dataset.metadata.prediction_length == prediction_length

    # Figure out rolling eval count
    num_rolling_evals = len(dataset.test) // len(dataset.train)
    time_features = time_features_from_frequency_str(freq)

    # Prepare transforms & data
    if setting == "univariate":
        transformation = create_transforms(
            time_features=time_features,
            prediction_length=prediction_length,
            freq=get_season_length(freq),
            train_length=len(dataset.train),
        )
        training_data = dataset.train
        test_data = dataset.test
    else:
        train_grouper = MultivariateGrouper(max_target_dim=target_dim)
        test_grouper = MultivariateGrouper(
            num_test_dates=num_rolling_evals,
            max_target_dim=target_dim,
        )
        transformation = create_multivariate_transforms(
            time_features=time_features,
            prediction_length=prediction_length,
            target_dim=target_dim,
            freq=get_season_length(freq),
            train_length=len(dataset.train),
        )
        training_data = train_grouper(dataset.train)
        test_data = test_grouper(dataset.test)

    # Instance splitter for training
    training_splitter = create_splitter(
        past_length=max(
            model_params["context_length"] + max(model.lags_seq),
            model.prior_context_length,
        ),
        future_length=prediction_length,
        mode="train",
    )

    callbacks = []

    # Optional validation set
    if evaluation_params.get("use_validation_set", False):
        offset = - prediction_length * num_rolling_evals
        train_val_splitter = OffsetSplitter(offset=offset)
        training_data, val_gen = train_val_splitter.split(training_data)

        transformed_data = transformation.apply(training_data, is_train=True)
        transformed_testdata = transformation.apply(test_data, is_train=False)

        val_data = val_gen.generate_instances(prediction_length, num_rolling_evals)
        transformed_valdata = transformation.apply(ConcatDataset(val_data), is_train=False)

        callbacks.append(
            EvaluateCallback(
                context_length=model_params["context_length"],
                prediction_length=prediction_length,
                model=model,
                datasets={"val": transformed_valdata, "test": transformed_testdata},
                logdir=logdir,
                **evaluation_params,
            )
        )
    else:
        transformed_data = transformation.apply(training_data, is_train=True)

    # Training DataLoader
    data_loader = TrainDataLoader(
        Cached(transformed_data),
        batch_size=dataset_params["batch_size"],
        stack_fn=batchify,
        transform=training_splitter,
        num_batches_per_epoch=dataset_params["num_batches_per_epoch"],
        shuffle_buffer_length=10000,
    )

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        filename=f"{dataset_name}-{{epoch:03d}}-{{train_loss:.3f}}",
        save_last=True,
        save_weights_only=True,
    )
    callbacks.append(checkpoint_callback)

    #Warm start GP 
    callbacks.append(GPWarmStart(data_loader, 30, 1e-2, info))
    # Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=[int(model_params["device"].split(":")[-1])],
        default_root_dir=logdir,
        logger=loggers,
        enable_progress_bar=True,
        callbacks=callbacks,
        **trainer_params,
    )

    info(f"Logging to {logdir}")
    trainer.fit(model, train_dataloaders=data_loader)
    info("Training completed.")

    # Load best checkpoint
    best_ckpt = Path(logdir) / "best_checkpoint.ckpt"
    if not best_ckpt.exists():
        torch.save(
            torch.load(checkpoint_callback.best_model_path)["state_dict"],
            best_ckpt,
        )
    info(f"Loading checkpoint from {best_ckpt}")
    best_state = torch.load(best_ckpt)
    model.load_state_dict(best_state, strict=True)

    # Final evaluation
    if evaluation_params.get("do_final_eval", True):
        metrics = evaluate_conditional(
            model_params, model, test_data, transformation, trainer, info
        )
    else:
        metrics = "Final eval not performed"

    # Dump results
    with open(Path(logdir) / "results.yaml", "w") as fp:
        yaml.dump({
            "config": config,
            "version": trainer.logger.version,
            "metrics": metrics,
        }, fp)

    return metrics


if __name__ == "__main__":
    # Configure root logger to stdout
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Arg parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to yaml config")
    parser.add_argument("--logdir", type=str, default="./logs",
                        help="Path to results dir")
    args, _ = parser.parse_known_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    parser = add_config_to_argparser(config=config, parser=parser)
    args = parser.parse_args()
    updates = vars(args)
    for k in config.keys() & updates.keys():
        if updates[k] != config[k]:
            logging.info(f"Updating config '{k}': {config[k]} -> {updates[k]}")
    config.update(updates)

    # AimLogger setup
    aim_logger = AimLogger(repo="./.aim")
    aim_logger.log_hyperparams(config)
    config["logdir"] = f"{args.logdir}/{aim_logger.version}"

    # Grab Aim run and define info()
    run = aim_logger.experiment
    def info(msg: str):
        logging.info(msg)
        run.track(Text(msg), name="info")  # <- sends your text into Aim

    # Run training + evaluation
    main(
        model=config["model"],
        setting=config["setting"],
        model_params=config["model_params"],
        dataset_params=config["dataset_params"],
        trainer_params=config["trainer_params"],
        evaluation_params=config["evaluation_params"],
        info=info,
        logdir=config["logdir"],
        seed=config.get("seed"),
        config=config,
        loggers=[aim_logger],
    )
