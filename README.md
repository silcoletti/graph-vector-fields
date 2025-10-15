# Graph Vector Field: A New Framework for Personalised Risk Assessment

This repository contains the official source code to reproduce the main results for the paper **"Graph Vector Field: A New Framework for Personalised Risk Assessment"** by Silvano Coletti and Francesca Fallucchi.

We introduce **Graph Vector Field (GVF)**, a novel framework that moves beyond traditional scalar scores to represent risk as a dynamic, interpretable vector field on a graph. This work provides a robust proof-of-concept, demonstrating the framework's superiority in modeling complex, heterogeneous, and relational data compared to standard approaches.

![GVF Framework Architecture](figures/overall_view_GVF.png)
*Figure 1: Conceptual architecture of the GVF framework. Please ensure you have a `figures` directory with this image.*

---

## Getting Started

Follow these instructions to set up the environment and reproduce the results presented in the paper.

### Prerequisites

* Python 3.10+
* PyTorch
* Git

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your_username/graph-vector-fields.git](https://github.com/your_username/graph-vector-fields.git)
    cd graph-vector-fields
    ```

2.  **Create a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    This project depends on `torch`, `torch_geometric`, and other common data science libraries. You can install all dependencies using the provided `requirements.txt` file.
    ```sh
    pip install -r requirements.txt
    ```
    *Note: `torch_geometric` installation can be complex. If you encounter issues, please refer to the official [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for instructions specific to your system's PyTorch and CUDA versions.*

---

## Reproducing the Results

The main script `gvf_final_experiment.py` contains the final implementation of the context-aware GVF-MoE model and the synthetic data generator used for the "Ultimate Stress Test".

Running this script will perform the full experimental pipeline:
1.  Generate the synthetic dataset with "super-spreader" and "lockdown" dynamics.
2.  Train the final GVF-MoE model.
3.  Evaluate the trained model on the entire dataset to extract the Gating Network's weights.
4.  Generate and save the plot `gating_weights_analysis.png`, which visualizes the model's adaptive behavior.

To run the full simulation, execute the following command:
```sh
python gvf_final_experiment.py
