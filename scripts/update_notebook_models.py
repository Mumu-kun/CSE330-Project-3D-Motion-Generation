"""
Script to add a cell to the notebook that writes the new models.py content.
"""

import json

# Read the notebook
with open("misc/3d-human-motion-inference.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Read models.py content
with open("src/models.py", "r", encoding="utf-8") as f:
    models_content = f.read()

# Escape triple quotes in the content
models_content_escaped = models_content.replace("'''", "\\'\\'\\'")

# Create the new cell that writes models.py
new_cell_source = f"""# =========================================================
# UPDATE models.py with new HumanMotionGenerator
# =========================================================
print("Updating models.py with new HumanMotionGenerator...")

models_content = \'\'\'{models_content_escaped}\'\'\'

with open("models.py", "w") as f:
    f.write(models_content)

print("✅ models.py updated successfully!")"""

# Create the cell as a list of lines (notebook format)
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": new_cell_source.split("\n"),
}

# Fix: each line should end with \n except the last
new_cell["source"] = [line + "\n" for line in new_cell["source"][:-1]] + [
    new_cell["source"][-1]
]

# Insert after the cloud setup cell (index 1)
nb["cells"].insert(2, new_cell)

# Write the modified notebook
with open("misc/3d-human-motion-inference.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully!")
print(f"Added cell with {len(models_content)} characters of models.py content")
