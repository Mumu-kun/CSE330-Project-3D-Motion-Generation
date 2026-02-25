"""
Script to refactor misc/humanml3d-subset-generator.ipynb to use src/utils imports.

This script:
1. Adds a setup cell with embedded utility files (quaternion.py, motion_utils.py)
2. Replaces inline implementations with imports from utils
"""

import json
from pathlib import Path


def refactor_notebook():
    """Refactor the notebook to use src/utils imports."""

    # Read the source files to embed
    quaternion_py = Path("src/utils/quaternion.py").read_text(encoding="utf-8")
    motion_utils_py = Path("src/utils/motion_utils.py").read_text(encoding="utf-8")

    # Remove the Skeleton import from motion_utils.py since it's not used
    motion_utils_py = motion_utils_py.replace(
        "from .skeleton import Skeleton\n", "# Skeleton import removed - not needed\n"
    )

    # Read the original notebook
    notebook_path = Path("misc/humanml3d-subset-generator.ipynb")
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # Create the setup cell content
    setup_cell_source = [
        "# =========================================================",
        "# CLOUD ENVIRONMENT SETUP - EMBEDDED UTILITIES",
        "# =========================================================",
        "import os",
        "import sys",
        "from pathlib import Path",
        "",
        "IN_COLAB = 'google.colab' in sys.modules",
        "IN_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ",
        "",
        "# Always create utils files (for both cloud and local)",
        "print('Setting up utility files...')",
        "",
        "FILES = {",
        "    'utils/__init__.py': '# Utils module\\n',",
        "    'utils/quaternion.py': '''" + quaternion_py + "''',",
        "    'utils/motion_utils.py': '''" + motion_utils_py + "''',",
        "}",
        "",
        "for filepath, content in FILES.items():",
        "    path = Path(filepath)",
        "    path.parent.mkdir(parents=True, exist_ok=True)",
        "    with open(path, 'w', encoding='utf-8') as f:",
        "        f.write(content)",
        "    print(f'Created {filepath}')",
        "",
        "# Add to Python path",
        "sys.path.insert(0, str(Path.cwd()))",
        "print('Setup Complete!')",
    ]

    # Create the imports cell content
    imports_cell_source = [
        "# --- FEATURE EXTRACTION (Using src/utils imports) ---",
        "import torch",
        "import numpy as np",
        "from utils.motion_utils import (",
        "    T2M_RAW_OFFSETS,",
        "    T2M_KINEMATIC_CHAIN,",
        "    DATASET_CONFIGS,",
        "    get_dataset_config,",
        "    sequence_joints_to_features,  # 271D format - matches notebook dimension",
        ")",
        "from utils.quaternion import (",
        "    qrot,",
        "    qinv,",
        "    qmul,",
        "    quaternion_to_cont6d,",
        "    quaternion_to_cont6d_np,",
        ")",
        "",
        "# Wrapper for 271D feature extraction (matching training pipeline)",
        "def extract_271d_features(positions: np.ndarray, feet_thre: float = 0.002) -> np.ndarray:",
        '    """',
        "    Extract 271D features from joint positions.",
        "    Uses the canonical implementation from motion_utils.py.",
        "    ",
        "    Feature Layout:",
        "        [0:3]   Root height Y, velocity X, velocity Z",
        "        [3:69]  22 RIC positions (22 * 3)",
        "        [69:201] 22 6D rotations (22 * 6)",
        "        [201:267] 22 local velocities (22 * 3)",
        "        [267:271] Foot contacts (4D)",
        '    """',
        "    positions_torch = torch.from_numpy(positions).float()",
        "    features_torch = sequence_joints_to_features(positions_torch, feet_thre=feet_thre)",
        "    return features_torch.numpy()",
        "",
        "print('Feature extraction functions loaded from src/utils/')",
        "print(f'Using correct quaternion rotation: qrot(root_quat, ric) - matches MoMask convention')",
    ]

    # Create setup cell
    setup_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "cell-setup-utils",
        "metadata": {},
        "outputs": [],
        "source": setup_cell_source,
    }

    # Create imports cell
    imports_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "cell-4-refactored",
        "metadata": {},
        "outputs": [],
        "source": imports_cell_source,
    }

    # Insert setup cell after cell-2 (configuration) - at index 3
    cells.insert(3, setup_cell)
    print(f"Inserted setup cell at index 3")

    # Find and replace cell-4 (which has inline implementations)
    for i, cell in enumerate(cells):
        cell_id = cell.get("id", "")
        cell_source = "".join(cell.get("source", []))

        # Check if this is the cell with inline implementations
        if "T2M_RAW_OFFSETS = np.array" in cell_source or cell_id == "cell-4":
            # Replace the source but keep the id
            cells[i]["source"] = imports_cell_source
            cells[i]["id"] = "cell-4-refactored"
            print(f"Replaced cell {i} ({cell_id}) with imports")
            break

    # Write the refactored notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent="\t")

    print(f"Refactored notebook saved to {notebook_path}")
    print(f"Total cells: {len(cells)}")


if __name__ == "__main__":
    refactor_notebook()
