import logging
import sys

from seml.experiment import Experiment

from bin.train_model import main
from tsflow.seml_util import (
    MediaLogger,
    add_default_observer_config,
    create_logdir,
    get_loggers,
)

sys.path.append(".")


ex = Experiment()
add_default_observer_config(ex)


@ex.automain
def sample(
    model,
    setting,
    model_params,
    dataset_params,
    trainer_params,
    evaluation_params,
    logdir=None,
    seed=None,
):
    logdir, array_job_id, array_task_id = create_logdir(logdir, ex)
    hparams = locals()
    logging.info(f"Saving into {logdir}")

    media_logger = MediaLogger(logdir)

    loggers = get_loggers(ex, model, logdir, hparams, media_logger)
    logging.info("Finished Training")
    logging.info(f"Stored model in {logdir}")
    results = main(
        model,
        setting,
        model_params,
        dataset_params,
        trainer_params,
        evaluation_params,
        logdir=logdir,
        seed=seed,
        config=None,
        loggers=loggers,
    )
    return results
