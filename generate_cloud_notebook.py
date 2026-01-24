import json
import os
from pathlib import Path

def generate_cloud_notebook():
    # 1. Configuration
    source_dir = Path("src")
    build_dir = Path("build")
    source_notebook = source_dir / "pipeline.ipynb"
    output_notebook = build_dir / "notebook.ipynb"
    support_files = ["config.py", "models.py", "utils.py", "requirements.txt"]
    
    # 2. Extract content from support files
    files_content = {}
    for filename in support_files:
        path = source_dir / filename
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                files_content[filename] = f.read()
        else:
            print(f"Warning: {path} not found.")

    # 3. Create the setup cell source
    setup_source = [
        "# =========================================================\n",
        "# CLOUD ENVIRONMENT SETUP (AUTO-GENERATED)\n",
        "# =========================================================\n",
        "import os\n",
        "import sys\n",
        "\n",
        "IN_COLAB = 'google.colab' in sys.modules\n",
        "IN_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ\n",
        "\n",
        "if IN_COLAB or IN_KAGGLE:\n",
        "    print(\"Running in Cloud Environment\")\n",
        "    \n",
        "    # Write supporting files\n",
        "    FILES = {\n"
    ]
    
    for filename, content in files_content.items():
        setup_source.append(f"        {repr(filename)}: {repr(content)},\n")
        
    setup_source.extend([
        "    }\n",
        "    \n",
        "    for filename, content in FILES.items():\n",
        "        os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None\n",
        "        with open(filename, 'w', encoding='utf-8') as f:\n",
        "            f.write(content)\n",
        "        print(f'Created {filename}')\n",
        "    \n",
        "    # Install dependencies\n",
        "    print(\"Installing dependencies (this may take a minute)...\")\n",
        "    %pip install -r requirements.txt\n",
        "    \n",
        "    print(\"Setup Complete!\")\n",
        "else:\n",
        "    print(\"Running locally. No setup needed.\")\n"
    ])

    # 4. Read source notebook
    if not source_notebook.exists():
        print(f"Error: {source_notebook} not found.")
        return

    with open(source_notebook, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 5. Insert setup cells at the beginning
    setup_markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Project Setup for Colab and Kaggle\n",
            "\n",
            "This notebook was automatically bundled for cloud execution. Run the cell below to reconstruct the project structure and install dependencies."
        ]
    }
    
    setup_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": setup_source
    }
    
    nb['cells'] = [setup_markdown, setup_code] + nb['cells']

    # 6. Save the bundled notebook
    build_dir.mkdir(parents=True, exist_ok=True)
    with open(output_notebook, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully generated bundled notebook: {output_notebook}")
    print("You can now upload this single file to Google Colab or Kaggle.")

if __name__ == "__main__":
    generate_cloud_notebook()
