# Memory-Guided Video Object Segmentation

**Memory-Guided Video Object Segmentation with Confidence Gating for Robust Perception Under Occlusion**

## Overview

This repository provides a reproducibility toolkit for evaluating pretrained **SAM-2** on the **DAVIS 2016/2017** video object segmentation benchmarks.

The framework uses temporal memory and confidence-based gating to improve segmentation reliability under occlusion. It supports first-frame prompts (box, centre point, or none) and memory-update intervals of **1, 2, 3, and 5 frames**.

Evaluation covers:
- Segmentation accuracy (J and F metrics)
- Confidence calibration (ECE) and threshold-based gating
- RAFT optical flow and object visibility
- Centroid tracking and Allan variance
- Processing latency and configuration comparisons
- A separate 2-D Cartesian motion simulation

## Repository Structure

```text
src/memory_guided_vos/   Core evaluation library
scripts/                 Evaluation and analysis scripts
configs/                 Experiment configurations
tests/                   Automated tests
docs/                    Reproducibility documentation
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows users can activate the environment with `.venv\Scripts\activate` in Command Prompt.

GPU evaluation additionally requires **SAM-2, RAFT, model checkpoints**, and the dependencies in `requirements-gpu.txt`.

## Dataset

Download **DAVIS-2017-trainval-480p** from [DAVIS](https://davischallenge.org/) and extract it to:

```text
data/raw/DAVIS/
```

## Usage

```bash
# Run tests
python -m pytest

# Run smoke tests
python scripts/smoke_test.py


# Run segmentation evaluation
python scripts/run_segmentation.py --profile

# Run optical flow analysis
python scripts/run_optical_flow.py
```


## Limitations

Evaluation is restricted to DAVIS benchmarks and a separate 2-D point-motion simulation. Physical robotic grasping, identity-verified object recovery, cross-dataset generalization, and guaranteed real-time performance have not been demonstrated.

