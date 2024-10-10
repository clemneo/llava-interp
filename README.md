# LLaVA interpretability
This repository contains the code and resources for the paper [Towards Interpreting Visual Information Processing in Vision-Language Models](https://arxiv.org/abs/2410.07149).

## Installation & Data Setup
Coming soon!

## Usage
Logit Lens:
* `scripts/logit_lens/create_logit_lens.py` Run model and create logit lens on a set of images
* `scripts/logit_lens/generate_overview.py` Generate an index.html for a set of logit_lens htmls

Token Ablation experiments:
* `scripts/save_post_adapter_acts.py` Caches activations of visual tokens
* `scripts/esimate_acts_size.py` Estimates the size of the total cache
* `scripts/calculate_mean_vector.py` Generates a mean vector using cached visual tokens. This mean vector is used as the ablation vector.

* `scripts/ablation_experiment.py` Runs the ablation experiment on LLaVA 1.5 on the generative and polling settings
* `scripts/ablation_experiment_phi.py` Runs the ablation experiment on LLaVA-Phi on the generative and polling settings
* `scripts/ablation_experiment_curate.py` Runs the ablation experiment on LLaVA-1.5 on the VQA setting
* `scripts/ablation_experiment_phi_curate.py` Runs the ablation experiment on LLaVA-Phi on the VQA setting

Attention Blocking experiments:
* `scripts/attention_experiment_curate.py` Runs the attention blocking experiments on LLaVA 1.5