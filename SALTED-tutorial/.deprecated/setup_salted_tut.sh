#!/bin/bash
echo "WARNING: This script contains commands that should be run individually."
echo "Please copy-paste each command into your terminal one by one to handle potential errors."
echo "Running this script directly will exit after this warning."
exit 1

# setup conda env basic packages
conda create -n salted_tut python=3.10 numpy scipy matplotlib scikit-learn pandas tqdm ipython ipykernel nb_conda_kernels jupyterlab_rise tqdm sympy pyyaml "mpi4py=3.1.4" "ase>=3.23" "spglib>=2.5.0" "phonopy>=2.9.3" "click>=7.1.2" "pytest>=6.2.3" elastic meson ninja -y
# other packages
conda install py3dmol ipywidgets -y
conda activate salted_tut

# setup featomic, remember to setup rust env
python -m pip install git+https://github.com/metatensor/featomic.git@featomic-v0.6.1  # version v0.6.1, date 2025 March

# setup salted
git clone https://github.com/andreagrisafi/SALTED
cd SALTED
git checkout 15e84a8
make
python -m pip install -e . --config-settings editable_mode=compat  # editable mode enabled

