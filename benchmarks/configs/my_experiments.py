import os

from dataclasses import asdict

import numpy as np

from benchmarks.configs.names import MyExperiments

from tbp.monty.frameworks.config_utils.config_args import (
    SingleCameraMontyConfig,
    LoggingConfig,
    MontyArgs,
    MotorSystemConfigCurvatureInformedSurface,
    PatchAndViewMontyConfig,
    PretrainLoggingConfig,
    get_cube_face_and_corner_views_rotations, MontyFeatureGraphArgs,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
    get_env_dataloader_per_object_by_idx,
    OmniglotDatasetArgs,
    OmniglotDataloaderArgs, DebugExperimentArgs,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.embodied_data import OmniglotDataLoader
from tbp.monty.frameworks.environments.two_d_data import OmniglotEnvironment
from tbp.monty.frameworks.experiments import (
    MontySupervisedObjectPretrainingExperiment, MontyObjectRecognitionExperiment,
)
from tbp.monty.frameworks.models.evidence_matching import MontyForEvidenceGraphMatching, EvidenceGraphLM
from tbp.monty.frameworks.models.graph_matching import GraphLM
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    HabitatSurfacePatchSM, HabitatDistantPatchSM,
)
from tbp.monty.simulators.habitat.configs import (
    SurfaceViewFinderMountHabitatDatasetArgs,
)

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

###
## 2D
###

two_d_experiment = dict(
    # experiment_class=MontyExperiment,
    # logging_config=LoggingConfig(),
    # experiment_args=ExperimentArgs(
    #     do_eval=False,
    #     max_train_steps=1,
    #     n_train_epochs=1,
    # ),
    # monty_config=SingleCameraMontyConfig(),
    # # Data{set, loader} config
    # dataset_class=OmniglotEnvironment,
    # dataset_args=OmniglotDatasetArgs(),
    # train_dataloader_class=OmniglotDataLoader,
    # train_dataloader_args=OmniglotDataloaderArgs(),
    # eval_dataloader_class=OmniglotDataLoader,
    # eval_dataloader_args=OmniglotDataloaderArgs(),
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=DebugExperimentArgs(
        do_eval=True,
        do_train=False,
        n_eval_epochs=1,
        # model_name_or_path=model_path_omniglot,
        max_eval_steps=200,
    ),
    logging_config=LoggingConfig(),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        monty_class=MontyForEvidenceGraphMatching,
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    evidence_update_threshold="all",
                    object_evidence_threshold=0.8,
                    # xyz values are in larger range so need to increase mmd
                    max_match_distance=5,
                    tolerances={
                        "patch": {
                            "principal_curvatures_log": np.ones(2),
                            "pose_vectors": np.ones(3) * 45,
                        }
                    },
                    # Point normal always points up so is not usefull
                    feature_weights={
                        "patch": {
                            "pose_vectors": [0, 1, 0],
                        }
                    },
                    # We assume the letter is presented upright
                    initial_possible_poses=[[0, 0, 0]],
                ),
            )
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        "principal_curvatures_log",
                    ],
                    save_raw_obs=True,
                    # Need to set this lower since curvature is generally lower
                    pc1_is_pc2_threshold=1,
                    motor_only_step=False,
                ),
            ),
            sensor_module_1=dict(
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=True,
                ),
            ),
        ),
        # motor_system_config=MotorSystemConfigInformedNoTransStepS1(),
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=OmniglotDatasetArgs(),
    train_dataloader_class=ED.OmniglotDataLoader,
    train_dataloader_args=OmniglotDataloaderArgs(),
    eval_dataloader_class=ED.OmniglotDataLoader,
    # Using versions 1 means testing on same version of character as trained.
    # Version 2 is a new drawing of the previously seen characters. In this
    # small test setting these are 3 characters from 2 alphabets. (for alphabets
    # and characters we use the default of OmniglotDataloaderArgs)
    eval_dataloader_args=OmniglotDataloaderArgs(versions=[1, 1, 1, 1, 1, 1]),
    # eval_dataloader_args=OmniglotDataloaderArgs(versions=[2, 2, 2, 2, 2, 2]),
)

"""
Basic setup
-----------
"""
# Specify directory where an output directory will be created.
project_dir = os.path.expanduser("~/tbp/results/monty/projects")

# Specify a name for the model.
model_name = "surf_agent_1lm_2obj"

"""
Training
----------------------------------------------------------------------------------------
"""
# Here we specify which objects to learn. 'mug' and 'banana' come from the YCB dataset.
# If you don't have the YCB dataset, replace with names from habitat (e.g.,
# 'capsule3DSolid', 'cubeSolid', etc.).
object_names = ["mug", "banana"]
# Get predefined object rotations that give good views of the object from 14 angles.
train_rotations = get_cube_face_and_corner_views_rotations()

# The config dictionary for the pretraining experiment.
my_experiment = dict(
    # Specify monty experiment and its args.
    # The MontySupervisedObjectPretrainingExperiment class will provide the model
    # with object and pose labels for supervised pretraining.
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=len(train_rotations),
        do_eval=False,
    ),
    # Specify logging config.
    logging_config=PretrainLoggingConfig(
        output_dir=project_dir,
        run_name=model_name,
        wandb_handlers=[],
    ),
    # Specify the Monty config.
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=500),
        # sensory module configs: one surface patch for training (sensor_module_0),
        # and one view-finder for initializing each episode and logging
        # (sensor_module_1).
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatSurfacePatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    # a list of features that the SM will extract and send to the LM
                    features=[
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "min_depth",
                        "mean_depth",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        # learning module config: 1 graph learning module.
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=GraphLM,
                learning_module_args=dict(),  # Use default LM args
            )
        ),
        # Motor system config: class specific to surface agent.
        motor_system_config=MotorSystemConfigCurvatureInformedSurface(),
    ),
    # Set up the environment and agent
    dataset_class=ED.EnvironmentDataset,
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations),
    ),
    # For a complete config we need to specify an eval_dataloader but since we only train here, this is unused
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations),
    ),
)

experiments = MyExperiments(
    my_experiment=my_experiment,
    first_experiment=first_experiment,
    two_d_experiment=two_d_experiment,
)
CONFIGS = asdict(experiments)

