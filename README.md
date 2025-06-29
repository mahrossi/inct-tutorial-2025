# INCT Tutorial 2025: ML basics and ML for electronic structure

Files for the first ML tutorial and learning electronic structure

## Run the Tutorial

Jupyter Lab is recommended.
Switch to the repo root directory and run:
```bash
jupyter lab --notebook-dir=.
```

## Requirements for individual tutorials

- In folder ``Basics-GPRandNN-Tutorials`` you will find two jupyter notebooks that are self contained. There is no need for anything fancy to run them. They implement a GPR and a NN from scratch, in order to fit simple data. They serve pedagogical purposes.
    - If you are not using the virtual machine, required Python packages can be installed with:
    ```bash
    pip install numpy matplotlib plotly ipython ipykernel ipywidgets
    ```

- In folder ``SALTED-tutorial`` you will find a jupyter notebook that executes the whole tutorial of SALTED for learning the electronic density of water monomers. For this tutorial, you will need SALTED (https://salted.readthedocs.io/en/latest/installation/) and FHI-aims (https://fhi-aims.org/get-the-code-menu/license-academia) installed. FHI-aims is only required for the extrapolation predictions at the end of the tutorial.
Please follow the installation procedure for these two codes in the respective webpages. For FHI-aims, if necessary ask for an academic license if your work is academic (donation voluntary, free of charge if needed). If you are in a course, it will be provided to you.
Please notice that additional setup is required if you are not using the virtual machine:
    - Navigate to `SALTED-tutorial/tutorial-run` and run `bash decompress_data.sh` to decompress the datasets
    - Install the [SALTED python package](https://github.com/andreagrisafi/SALTED) following the installation instructions
    - Install required dependencies (use non-MPI version of h5py via pip):
    ```bash
    pip install numpy matplotlib py3Dmol salted ase ipywidgets scipy scikit-learn h5py
    ```

- Finally, in folder DeepH-tutorial there is a minimal example of the workings of DeepH with FHI-aims data, also for the water monomer and in order to learn the Hamiltonian. This is meant just as an example in case you are curious (it is not written as a tutorial). You need to install DeepH, following the instructions in their webpage, and read instructions in the folder.

