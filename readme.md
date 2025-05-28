# Text Embedding is Not All You Need: Attention Control for Text-to-Image Alignment


> 📢 **News**: Our paper has been **accepted to CVPR 2025**! 🎉


This repository accompanies the preprint published on arXiv. It introduces a novel approach to improve text-to-image alignment in diffusion models.

> Check out the details on the 👉 [Project Page](https://t-sam-diffusion.github.io)

## Overview

This codebase implements attention control mechanisms to enhance text-to-image alignment in diffusion models. It includes modules for evaluation, model definitions, prompt handling, and utility functions.

## Repository Structure

- `evaluation_codes/`: Scripts for evaluating model performance
- `metrics/`: Custom metrics for assessing alignment quality
- `models/`: Model architectures and related components
- `prompt_classes/`: Classes for handling different prompt types
- `prompt_files/`: Sample prompts and resources
- `sim_optmization/`: Similarity optimization routines
- `utils/`: Utility functions and helpers
- `evaluate.py`: Main evaluation script
- `new_work_desk.py`, `work_desk.py`, `work_desk_latent_update.py`: Latent update and processing tools

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Additional dependencies listed in `requirements.txt`

### Installation

```bash
git clone https://github.com/t-sam-diffusion/code.git
cd code
pip install -r requirements.txt
```
## Usage

To evaluate the model:

```bash
python evaluate.py --config configs/eval_config.yaml
```


## Citation
If you use this code, please cite:
```bash
@article{kim2024text,
  title={Text Embedding is Not All You Need: Attention Control for Text-to-Image Semantic Alignment with Text Self-Attention Maps},
  author={Kim, Jeeyung and Esmaeili, Erfan and Qiu, Qiang},
  journal={arXiv preprint arXiv:2411.15236},
  year={2024}
}
```
