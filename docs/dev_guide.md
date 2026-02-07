# Developer Guide: Working with the Cloud Notebook

This project includes a utility to bundle the codebase into a single Jupyter notebook for easy training on cloud platforms like **Kaggle** or **Google Colab**.

## How to Generate the Cloud Notebook

1.  **Ensure your environment is set up**: Make sure you have the required dependencies installed (though the script only uses standard libraries like `json`, `os`, and `pathlib`).
2.  **Run the generator script**:
    ```bash
    python generate_cloud_notebook.py
    ```
3.  **Locate the output**: The bundled notebook will be created at `build/notebook.ipynb`.

## How to Use on Kaggle/Colab

1.  **Upload the notebook**: Upload the generated `build/notebook.ipynb` to Kaggle as a new notebook or to Google Colab.
2.  **Run the Setup Cell**: The first cell in the notebook is an auto-generated setup cell. When run in a cloud environment:
    *   It reconstructs the project file structure (creating the `src` and `src/utils` directories).
    *   It writes all supporting `.py` files and `requirements.txt`.
    *   It installs the necessary Python packages via `%pip`.
    *   It sets up the dataset using `rclone` (on Kaggle).
3.  **Start Training**: Once the setup is complete, you can run the rest of the notebook as if you were working locally.

## Project Structure (Inside the Notebook)

The script bundles the following files into the setup cell:
- `config.py`
- `models.py`
- `utils/`
    - `__init__.py`
    - `utils.py`
    - `dataset.py`
    - `motion_utils.py`
    - `visualization.py`
    - `bvh_utils.py`
    - `metrics.py`
    - `quaternion.py`
    - `skeleton.py`
    - `train_utils.py`

## Working on Code Changes

If you modify any of the supporting `.py` files in `src/`, you **must** re-run `generate_cloud_notebook.py` to update the bundled notebook before uploading it to the cloud.
