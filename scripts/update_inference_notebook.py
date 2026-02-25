"""
Script to update the 3d-human-motion-inference.ipynb notebook to use new model interfaces.
"""

import json
from pathlib import Path

# Read the notebook
notebook_path = Path("misc/3d-human-motion-inference.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Define the updates for each cell
updates = {}

# Cell 4: Update imports
updates[
    4
] = """# Install dependencies (uncomment if running on Kaggle/Colab)
# !pip install torch transformers matplotlib tqdm

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from config import Config
from models import MotionHistoryEncoder, FlowMatchingPredictor, HumanMotionGenerator
from utils.dataset import Text2MotionDataset, text2motion_collate_fn, create_dataloader
from utils.text_encoder import CLIPEncoder
from utils.visualization import visualize_motion
from utils.motion_utils import FeatureNormalizer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load configuration
config = Config()
config.device = device
print(f"Dataset path: {config.dataset_path}")
print(f"Checkpoint dir: {config.checkpoint_dir}")
"""

# Cell 5: Update create_dataloader call to capture normalizer
updates[
    5
] = """# Create dataset and dataloader
print("Loading dataset...")
config.dataset_path = Path(
    "/kaggle/input/notebooks/mustafamuhaimin/3d-human-motion-generation/dataset/humanml3d-subset/"
)
config.checkpoint_dir = Path(
    "/kaggle/input/notebooks/mustafamuhaimin/3d-human-motion-generation/checkpoints/"
)

config.batch_size = 1
dataloader, normalizer = create_dataloader(config, split="train", shuffle=True)

print(f"Number of batches: {len(dataloader)}")

# Show a sample batch
sample_batch = next(iter(dataloader))
print(f"\\nSample batch:")
print(f"  Captions: {len(sample_batch['captions'])} samples")
print(f"  Motion shape: {sample_batch['motion'].shape}")  # (B, T, 271)
print(f"  Joints shape: {sample_batch['joints'].shape}")  # (B, T, 22, 3)
print(f"  Text embeddings shape: {sample_batch['text_clip'].shape}")  # (B, 1, 512)
print(f"  Lengths shape: {sample_batch['lengths'].shape}")  # (B,)
print(f"\\nSample caption: '{sample_batch['captions'][0]}'")
"""

# Cell 8: Update load_from_checkpoint to pass normalizer
updates[
    8
] = """# Load the best checkpoint for generation
checkpoint_path = config.checkpoint_dir / "best.pt"

generator = HumanMotionGenerator.load_from_checkpoint(
    checkpoint_path=checkpoint_path,
    config=config,
    device=str(device),
    normalizer=normalizer,
)

print(f"Loaded model from {checkpoint_path}")
print("Model ready for generation!")
"""

# Cell 10: Update CLIPEncoder initialization
updates[
    10
] = """# Initialize CLIP encoder for text-to-embedding conversion
clip_encoder = CLIPEncoder()  # Default model is openai/clip-vit-base-patch32
clip_encoder.to(device)

print("CLIP encoder initialized")
"""

# Cell 15: Update generate_sequence call with dataset text_clip
updates[
    15
] = """# Enter your custom text prompt
print(f"Generating motion for: '{d['captions']}'")
input_features = d["motion"][:, 0:20, :]

# Encode text to CLIP embedding
motions = []
with torch.no_grad():
    text_embedding = d["text_clip"]  # Already (B, 1, 512) from dataset

joint_positions = generator.generate_sequence(
    text=text_embedding,
    # input_features=input_features,
    num_frames=100,
    num_steps=25,
    guidance_scale=1,
    dataset_type="t2m",
)
"""

# Cell 19: Update generate_sequence call with custom prompts
updates[
    19
] = """# Define text prompts for generation
text_prompts = [
    "a person walks forward",
    "a person is running",
    "a person jumps up",
]

# Generation parameters
num_steps = 25  # Number of flow matching steps (higher = better quality, slower)
guidance_scale = 2.5  # CFG scale (1.0 = no guidance, 2.5-3.5 recommended)

print(f"Generating {len(text_prompts)} motions...")
print(f"Flow matching steps: {num_steps}")
print(f"Guidance scale: {guidance_scale}\\n")

generated_motions = []

for i, prompt in enumerate(text_prompts):
    print(f"[{i+1}/{len(text_prompts)}] Generating: '{prompt}'")

    # Generate motion - pass text directly (generator handles encoding internally)
    joint_positions = generator.generate_sequence(
        text=prompt,  # Can pass string directly
        num_frames=100,
        num_steps=10,
        guidance_scale=2.5,
        dataset_type="t2m",
    )

    # Store generated motion
    generated_motions.append(joint_positions.cpu().numpy())
    print(f"  Generated shape: {joint_positions.shape}\\n")

print("All motions generated successfully!")
"""

# Apply updates
for cell_idx, new_source in updates.items():
    if cell_idx < len(nb["cells"]):
        nb["cells"][cell_idx]["source"] = new_source.split("\n")
        # Ensure each line ends with \n for proper notebook format
        nb["cells"][cell_idx]["source"] = [
            line + "\n" for line in new_source.split("\n")
        ][
            :-1
        ]  # Remove last empty line
        print(f"Updated cell {cell_idx}")

# Write the updated notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\nNotebook updated: {notebook_path}")
