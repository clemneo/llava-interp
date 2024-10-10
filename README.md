# LLaVA interp
Code for [Towards Interpreting Visual Information Processing in Vision-Language Models](https://arxiv.org/abs/2410.07149)

Script for Logit Lens:
* `logit_lens/create_logit_lens.py` Run model and create logit lens on a set of images
* `logit_lens/generate_overview.py` Generate an index.html for a set of logit_lens htmls

Scripts for Token Ablation experiments:
* `save_post_adapter_acts.py` Caches activations of visual tokens
* `esimate_acts_size.py` Estimates the size of the total cache
* `calculate_mean_vector.py` Generates a mean vector using cached visual tokens. This mean vector is used as the ablation vector.

* `ablation_experiment.py` Runs the ablation experiment on LLaVA 1.5 on the generative and polling settings
* `ablation_experiment_phi.py` Runs the ablation experiment on LLaVA-Phi on the generative and polling settings
* `ablation_experiment_curate.py` Runs the ablation experiment on LLaVA-1.5 on the VQA setting
* `ablation_experiment_phi_curate.py` Runs the ablation experiment on LLaVA-Phi on the VQA setting

Script for Attention Blocking experiments:
* `attention_experiment_curate.py` Runs the attention blocking experiments on LLaVA 1.5

## TODOS
* Move scripts folder out
* Add dependencies to install
* Add instructions for dataset download
* Make sure scripts imports work