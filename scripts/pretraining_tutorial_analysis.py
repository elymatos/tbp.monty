import os
import matplotlib.pyplot as plt
from tbp.monty.frameworks.utils.logging_utils import load_stats
from tbp.monty.frameworks.utils.plot_utils import plot_graph

# Specify where pretraining data is stored.
exp_path = os.path.expanduser("~/tbp/results/monty/projects/surf_agent_1lm_2obj")
pretrained_dict = os.path.join(exp_path, "pretrained")

train_stats, eval_stats, detailed_stats, lm_models = load_stats(
    exp_path,
    load_train=False,  # doesn't load train csv
    load_eval=False,  # doesn't try to load eval csv
    load_detailed=False,  # doesn't load detailed json output
    load_models=True,  # loads models
    pretrained_dict=pretrained_dict,
)

# Visualize the mug graph from the pretrained graphs loaded above from
# pretrained_dict. Replace 'mug' with 'banana' to plot the banana graph.
plot_graph(lm_models["pretrained"][0]["banana"]["patch"], rotation=120)
plt.show()