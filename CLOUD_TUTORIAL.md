# Tutorial: Generating a Cloud-Ready Notebook

This project includes a utility script `generate_cloud_notebook.py` that simplifies running your local ML pipeline on cloud platforms like **Google Colab** or **Kaggle**.

## What it Does

When you develop locally, your project is organized as follows:
- `src/pipeline.ipynb`: The main execution flow.
- `src/models.py`, `src/utils.py`, `src/config.py`: Logic and configuration.
- `src/requirements.txt`: List of dependencies.

Cloud platforms usually prefer a single `.ipynb` file. `generate_cloud_notebook.py` (located in the root) bundles everything from the `src/` directory into a single file in the `build/` directory.

## How to Use It

### 1. Development
Work on your project locally using the files inside the `src/` folder.

### 2. Bundle the Project
Run the generator script from your project root:

```bash
python generate_cloud_notebook.py
```

This will:
1. Read `src/pipeline.ipynb`.
2. Extract the content of the supporting files from `src/`.
3. Create a new file named `build/notebook.ipynb`.
4. Inject a specific "Setup" cell at the top of the bundle.

### 3. Upload to Cloud
Upload the newly created `build/notebook.ipynb` to Google Colab or Kaggle.

### 4. Execute in Cloud
When you run the first cell in the cloud environment, it will:
- Detect it's running in Colab/Kaggle.
- Re-create the `.py` files and `requirements.txt` in the cloud's local storage.
- Install all necessary dependencies using `%pip install`.

## Why use this instead of just uploading files?
- **Portability**: You only need to manage one file.
- **Automation**: Dependencies are handled automatically.
- **Consistency**: The cloud environment will have the exact same logic as your local code.

---

*Note: The generated notebook uses `%pip install` which is the recommended magic command for environment-aware installations in Jupyter.*
