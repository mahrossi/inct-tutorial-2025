echo "WARNING: please copy-paste and run in terminal line by line, for better dealing with corner cases"
exit

### setup conda env
## has to be 3.9, the last python version that pytorch 1.9.1 supports
conda create -n deeph_tut python=3.9 numpy==1.26 scipy matplotlib scikit-learn pandas tqdm ipython ipykernel -y
conda activate deeph_tut
## for pytorch versions https://pytorch.org/get-started/previous-versions/
python -m pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
## for torch-geometric https://pypi.org/project/torch-geometric/1.7.2/
## and https://data.pyg.org/whl/ for buddy-packages' versions
python -m pip install torch-geometric==1.7.2 torch-cluster==1.5.9 torch-scatter==2.0.9 torch-sparse==0.6.10 e3nn==0.3.5 h5py tensorboard pathos psutil asi4py pymatgen
## for problem `AttributeError: module 'distutils' has no attribute 'version'`
python -m pip install setuptools==59.5.0

### install deeph-e3 from
## must not pip install, just download the code and use it directly
git clone https://github.com/Xiaoxun-Gong/DeepH-E3.git
