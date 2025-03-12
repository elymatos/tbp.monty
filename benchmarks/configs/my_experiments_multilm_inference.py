import copy
import os

import numpy as np

from dataclasses import asdict
from benchmarks.configs.names import MyExperiments

from tbp.monty.frameworks.config_utils.config_args import (
    EvalLoggingConfig,
    FiveLMMontyConfig,
    MontyArgs,
    MotorSystemConfigInformedGoalStateDriven,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
    get_env_dataloader_per_object_by_idx,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import (
    MontyObjectRecognitionExperiment,
)
from tbp.monty.frameworks.loggers.monty_handlers import BasicCSVStatsHandler
from tbp.monty.frameworks.models.evidence_matching import (
    EvidenceGraphLM,
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.goal_state_generation import (
    EvidenceGoalStateGenerator,
)
from tbp.monty.simulators.habitat.configs import (
    FiveLMMountHabitatDatasetArgs,
)

"""
Basic Info
"""

# Specify directory where an output directory will be created.
project_dir = os.path.expanduser("~/tbp/results/monty/projects")

# Specify a name for the model.
model_name = "dist_agent_5lm_2obj"

object_names = ["mug", "banana"]
test_rotations = [np.array([0, 15, 30])] # A previously unseen rotation of the objects

model_path = os.path.join(
    project_dir,
    model_name,
    "pretrained",
)

"""
Learning Module Configs
"""
# Create a template config that we'll make copies of.
evidence_lm_config = dict(
    learning_module_class=EvidenceGraphLM,
    learning_module_args=dict(
        max_match_distance=0.01,  # =1cm
        feature_weights={
            "patch": {
                # Weighting saturation and value less since these might change under
                # different lighting conditions.
                "hsv": np.array([1, 0.5, 0.5]),
            }
        },
        max_nneighbors=10,
        # Use this to update all hypotheses > x_percent_threshold (faster)
        evidence_update_threshold="x_percent_threshold",
        x_percent_threshold=20,
        gsg_class=EvidenceGoalStateGenerator,
        gsg_args=dict(
            goal_tolerances=dict(
                location=0.015,  # distance in meters
            ),  # Tolerance(s) when determining goal-state success
            min_post_goal_success_steps=5,  # Number of necessary steps for a hypothesis
        ),
    ),
)
# We'll also reuse these tolerances, so we specify them here.
tolerance_values = {
    "hsv": np.array([0.1, 0.2, 0.2]),
    "principal_curvatures_log": np.ones(2),
}

# Now we make 5 copies of the template config, each with the tolerances specified for
# one of the five sensor modules.
learning_module_configs = {}
for i in range(5):
    lm = copy.deepcopy(evidence_lm_config)
    lm["learning_module_args"]["tolerances"] = {f"patch_{i}": tolerance_values}
    learning_module_configs[f"learning_module_{i}"] = lm

# The config dictionary for the pretraining experiment.
dist_agent_5lm_2obj_eval = dict(
    #  Specify monty experiment class and its args.
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(test_rotations),
        min_lms_match=3,   # Terminate when 3 learning modules makes a decision.
    ),
    # Specify logging config.
    logging_config=EvalLoggingConfig(
        output_dir=os.path.join(project_dir, model_name),
        run_name="eval",
        monty_handlers=[BasicCSVStatsHandler],
        wandb_handlers=[],
    ),
    # Specify the Monty model. The FiveLLMMontyConfig contains all of the
    # sensor module configs and connectivity matrices. We will specify
    # evidence-based learning modules and MontyForEvidenceGraphMatching which
    # facilitates voting between evidence-based learning modules.
    monty_config=FiveLMMontyConfig(
        monty_args=MontyArgs(min_eval_steps=20),
        monty_class=MontyForEvidenceGraphMatching,
        learning_module_configs=learning_module_configs,
        motor_system_config=MotorSystemConfigInformedGoalStateDriven(),
    ),
    # Set up the environment and agent.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=FiveLMMountHabitatDatasetArgs(),
    # Set up the training dataloader. Unused, but must be included.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=1),
    # Set up the evaluation dataloader.
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
    ),
)

experiments = MyExperiments(
    dist_agent_5lm_2obj_eval=dist_agent_5lm_2obj_eval,
)
CONFIGS = asdict(experiments)

