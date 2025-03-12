from dataclasses import asdict

from benchmarks.configs.names import MyExperiments
from tbp.monty.frameworks.config_utils.config_args import (
    SingleCameraMontyConfig,
    LoggingConfig
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    ExperimentArgs,
    get_env_dataloader_per_object_by_idx,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import MontyExperiment
from tbp.monty.simulators.habitat.configs import SinglePTZHabitatDatasetArgs

#####
# To test your env and familiarize with the code, we'll run the simplest possible
# experiment. We'll use a model with a single learning module as specified in
# monty_config. We'll also skip evaluation, train for a single epoch for a single step,
# and only train on a single object, as specified in experiment_args and train_dataloader_args.
#####

first_experiment = dict(
    experiment_class=MontyExperiment,
    logging_config=LoggingConfig(),
    experiment_args=ExperimentArgs(
        do_eval=False,
        max_train_steps=1,
        n_train_epochs=1,
    ),
    monty_config=SingleCameraMontyConfig(),
    # Data{set, loader} config
    dataset_class=ED.EnvironmentDataset,
    dataset_args=SinglePTZHabitatDatasetArgs(),
    train_dataloader_class=ED.EnvironmentDataLoaderPerObject,
    train_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=1),
    eval_dataloader_class=ED.EnvironmentDataLoaderPerObject,
    eval_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=1),
)

experiments = MyExperiments(
    first_experiment=first_experiment,
)
CONFIGS = asdict(experiments)