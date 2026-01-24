# Human Motion Animation Generation

A pipeline for generating human motion animations using autoregressive models and flow matching, compatible with MoMask's input/output format.

## Overview

This project implements a novel approach to human motion generation using:
- **Autoregressive Context Encoder**: Encodes motion context sequentially
- **Flow Matching Network**: Generates motion sequences using continuous normalizing flows

The pipeline is designed to be compatible with MoMask's data format:
- **Input**: HumanML3D dim-263 feature vectors
- **Output**: Joint positions (nframe, 22, 3) → BVH files

## Project Structure

```
ML-Project/
├── pipeline.ipynb          # Main pipeline notebook with 7 steps
├── config.py                # Configuration and hyperparameters
├── models.py                # Model architectures
├── utils.py                 # Utility functions (data, post-processing, evaluation)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download and prepare the HumanML3D dataset following the instructions from [HumanML3D repository](https://github.com/EricGuo5513/HumanML3D).

Place the dataset in the following structure:
```
dataset/
└── HumanML3D/
    ├── train/
    ├── val/
    └── test/
```

### 3. Configure Settings

Edit `config.py` to adjust hyperparameters, paths, and model settings according to your needs.

## Pipeline Steps

The main pipeline (`pipeline.ipynb`) consists of 7 steps:

1. **Data Preparation (HumanML3D)**: Load and preprocess HumanML3D dataset
2. **Autoregressive Context Encoder**: Initialize and test context encoder
3. **Flow Matching Network**: Initialize and test flow matching model
4. **Training Loop**: Train both models with logging and checkpointing
5. **Inference / Generation**: Generate motion sequences from text prompts
6. **Post-processing**: Convert outputs to BVH format
7. **Evaluation**: Compute metrics and visualize results

## Usage

### Training

Open `pipeline.ipynb` and run the cells sequentially. The notebook contains all 7 steps of the pipeline.

### Generation

After training, use Step 5 (Inference) to generate motions from text prompts:

```python
text_prompt = "A person is walking forward"
generated_motion = generate_motion(context_encoder, flow_matching_net, text_prompt)
```

### Output Format

Generated motions are saved in the following structure:
```
generation/
└── <experiment_name>/
    ├── joints/          # Numpy files with joint positions (nframe, 22, 3)
    └── animation/        # BVH files for visualization
```

## Compatibility with MoMask

This project uses the same input/output format as [MoMask](https://github.com/EricGuo5513/momask-codes):

- **Input Format**: HumanML3D dim-263 feature vectors
- **Output Format**: Joint positions (nframe, 22, 3) → BVH files
- **Dataset**: HumanML3D dataset structure
- **Visualization**: Compatible with MoMask's visualization pipeline

You can use MoMask's evaluation tools and visualization utilities with the outputs from this pipeline.

## Model Architecture

### Autoregressive Context Encoder

- Processes motion sequences sequentially
- Uses transformer encoder architecture
- Outputs contextual embeddings for flow matching
- Handles text conditioning (to be integrated)

### Flow Matching Network

- Implements continuous normalizing flows
- Takes context from encoder as input
- Generates motion sequences through flow matching
- Supports variable-length generation

## Configuration

Key configuration parameters in `config.py`:

- `motion_dim`: 263 (HumanML3D feature dimension)
- `num_joints`: 22 (Number of joints)
- `max_motion_length`: 196 (Maximum motion length)
- `hidden_dim`: 512 (Model hidden dimension)
- `batch_size`: 64 (Training batch size)
- `learning_rate`: 1e-4 (Learning rate)

## TODO

- [ ] Implement actual HumanML3D data loading
- [ ] Implement feature_to_joints conversion
- [ ] Implement joints_to_bvh conversion
- [ ] Integrate text encoder (CLIP or similar)
- [ ] Implement evaluation metrics (FID, diversity, R-precision)
- [ ] Add visualization utilities
- [ ] Implement flow matching training logic
- [ ] Add checkpoint loading/saving utilities

## References

- [MoMask: Generative Masked Modeling of 3D Human Motions (CVPR 2024)](https://github.com/EricGuo5513/momask-codes)
- [HumanML3D Dataset](https://github.com/EricGuo5513/HumanML3D)
- Flow Matching: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)

## License

[Add your license here]

## Contact

[Add your contact information here]
