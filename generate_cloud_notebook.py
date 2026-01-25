import json
import os
import ast
from pathlib import Path

def minify_python(source):
    """Basic minification: removes docstrings and comments via AST round-trip."""
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            # Remove docstrings from modules, classes, and functions
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                if (node.body and isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, (ast.Constant, ast.Str))):
                    # In Python 3.8+, docstrings are ast.Constant. In older, ast.Str.
                    # We check if it's a string constant.
                    val = node.body[0].value
                    if isinstance(val, ast.Str) or (isinstance(val, ast.Constant) and isinstance(val.value, str)):
                        node.body.pop(0)
        return ast.unparse(tree)
    except Exception as e:
        print(f"Warning: Could not minify Python code. Falling back to original. Error: {e}")
        return source

def minify_requirements(content):
    """Removes comments and empty lines from requirements.txt."""
    lines = []
    for line in content.splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            lines.append(line)
    return "\n".join(lines)

def generate_cloud_notebook():
    # 1. Configuration
    script_dir = Path(__file__).parent
    source_dir = script_dir / "src"
    build_dir = script_dir / "build"
    source_notebook = source_dir / "pipeline.ipynb"
    output_notebook = build_dir / "notebook.ipynb"
    support_files = ["config.py", "models.py", "utils.py", "requirements.txt"]
    
    # 2. Extract and minify content from support files
    files_content = {}
    for filename in support_files:
        path = source_dir / filename
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                if filename.endswith('.py'):
                    print(f"Minifying {filename}...")
                    content = minify_python(content)
                elif filename == 'requirements.txt':
                    print(f"Cleaning {filename}...")
                    content = minify_requirements(content)
                files_content[filename] = content
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
