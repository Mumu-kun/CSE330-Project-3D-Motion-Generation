# Motion History Encoder

## Input
- Motion features: (B, T_hist, 263) - HumanML3D format
- Text: Union[str, List[str]] (Conditioning)

## Output
- Joint-level features: (B, 22, per_joint_out_dim)

## Parameters
## Parameters (Testing Scale)
- `motion_dim`: 263
- `text_embedding_dim`: 512
- `text_projection_dim`: 16
- `joint_feature_projection_dim`: 32
- `per_joint_out_dim`: 32
- `joint_count`: 22
- `model_dim`: 64
- `num_layers`: 1
- `bidirectional`: True

## Design
### 1. Per-Joint Encoding (Local -> Neuronal)
- **Input splicing** (Standard HumanML3D 263D layout):
    - local features: ric position (3) | ric velocity (3) = 6
    - global features: Root (4) | Foot Contact (4) = 8
    - text embedded: 512 --> linear --> 256
- **MLP Encoder**:
    - Input: (6 + 8 + 256) = 270
    - Logic: `Linear(270, 512) -> SiLU -> Linear(512, 256)`
    - Output: 256 per joint

### 2. Temporal Processing (Sequence Modeling)
- **Flattening**: (B, T_hist, 22 * 256)
- **GRU**:
    - hidden_dim: 512
    - bidirectional: True
- **Summary**: `final_hidden_raw.transpose(0, 1).reshape(B, -1)` (B, L * D * model_dim)

### 3. Per-Joint Decoding (Global -> Spatial)
- **MLP Decoder Head**:
    - Input: Global Summary (B, 1024)
    - Logic: `Linear(1024, 1024) -> SiLU -> Linear(1024, 22 * 64)`
    - Output: (B, 22, 64)
- **Output**: (B, 22, 64)