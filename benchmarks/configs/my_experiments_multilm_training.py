import os

from dataclasses import asdict
from benchmarks.configs.names import MyExperiments

from tbp.monty.frameworks.config_utils.config_args import (
    FiveLMMontyConfig,
    MontyArgs,
    MotorSystemConfigNaiveScanSpiral,
    PretrainLoggingConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
    get_env_dataloader_per_object_by_idx,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_naive_scan_policy_config,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.simulators.habitat.configs import (
    FiveLMMountHabitatDatasetArgs,
)

# Specify directory where an output directory will be created.
project_dir = os.path.expanduser("~/tbp/results/monty/projects")

# Specify a name for the model.
model_name = "dist_agent_5lm_2obj"

# Specify the objects to train on and 14 unique object poses.
object_names = ["mug", "banana"]
train_rotations = get_cube_face_and_corner_views_rotations()

# The config dictionary for the pretraining experiment.
dist_agent_5lm_2obj_train = dict(
    # Specify monty experiment class and its args.
    # The MontySupervisedObjectPretrainingExperiment class will provide the model
    # with object and pose labels for supervised pretraining.
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        do_eval=False,
        n_train_epochs=len(train_rotations),
    ),
    # Specify logging config.
    logging_config=PretrainLoggingConfig(
        output_dir=project_dir,
        run_name=model_name,
    ),
    # Specify the Monty model. The FiveLLMMontyConfig contains all of the sensor module
    # configs, learning module configs, and connectivity matrices we need.
    monty_config=FiveLMMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=500),
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=make_naive_scan_policy_config(step_size=5)
        ),
    ),
    # Set up the environment and agent.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=FiveLMMountHabitatDatasetArgs(),
    # Set up the training dataloader.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations),
    ),
    # Set up the evaluation dataloader. Unused, but required.
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,  # just placeholder
    eval_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=1),
)

experiments = MyExperiments(
    dist_agent_5lm_2obj_train=dist_agent_5lm_2obj_train,
)
CONFIGS = asdict(experiments)