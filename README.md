# LLaVA interpretability
This repository contains the code and resources for the paper [Towards Interpreting Visual Information Processing in Vision-Language Models](https://arxiv.org/abs/2410.07149).

## Installation & Data Setup
Coming soon!

## Usage
### 1. Logit Lens
* `scripts/logit_lens/create_logit_lens.py` Run the model and create interative logit lens HTMLs for a set of images
* `scripts/logit_lens/generate_overview.py` Generate an `index.html` to view a set of logit_lens HTMLs files.

### 2. Token Ablation Experiments

**Preparation**

Before running ablation experiments, create the mean vector used for ablation:
1. `scripts/save_post_adapter_acts.py` Caches activations of visual tokens
2. `scripts/esimate_acts_size.py` Estimates the size of the total cache
3. `scripts/calculate_mean_vector.py` Generates a mean vector using cached visual tokens.

The mean vector used in the paper for LLaVA 1.5 and LLaVA-Phi can be found in `data/`.

**Running Experiments**
* `scripts/ablation_experiment.py` Runs ablation experiments on LLaVA 1.5 (generative and polling settings)
* `scripts/ablation_experiment_curate.py` Runs ablation experiments on LLaVA-1.5 (VQA setting)
* `scripts/ablation_experiment_phi.py` Runs ablation experiments on LLaVA-Phi (generative and polling settings)
* `scripts/ablation_experiment_phi_curate.py` Runs ablation experiments on LLaVA-Phi (VQA setting)

### 3. Attention Blocking experiments
* `scripts/attention_experiment_curate.py` Run attention blocking experiments on LLaVA 1.5

## Citation
```
@misc{neo2024interpretingvisualinformationprocessing,
      title={Towards Interpreting Visual Information Processing in Vision-Language Models}, 
      author={Clement Neo and Luke Ong and Philip Torr and Mor Geva and David Krueger and Fazl Barez},
      year={2024},
      eprint={2410.07149},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.07149}, 
}
```
