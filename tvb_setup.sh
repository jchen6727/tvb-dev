#!/bin/bash

# # # CONDA ENVIRONMENT CREATION # # #
#conda create -n tvb python=3.10
#conda activate tvb

# # # PYTHON PACKAGE DEPENDENCIES # # #
pip install xarray pytest pytest-cov pytest-benchmark scikit-learn sqlalchemy numba dill numpy setuptools
pip install --ignore-installed entrypoints
pip install werkzeug==2.0.1
pip install install sympy pyqtgraph lxml pandoc tensorboardX
pip install elephant dill

# # # WORK DIRECTORY # # #
mkdir ~/dev/tvb
cd ~/dev/tvb

# # # TVB-ROOT # # #
git clone --depth 1 --no-single-branch https://github.com/the-virtual-brain/tvb-root.git
cd tvb-root
git pull --allow-unrelated-histories
cd tvb_library
python setup.py develop
cd ../tvb_framework
python setup.py develop
cd ../tvb_storage
python setup.py develop
cd ../tvb_contrib
python setup.py develop

# # # TVB-MULTISCALE # # #
cd ~/dev/tvb
git clone --depth 1 --no-single-branch https://github.com/the-virtual-brain/tvb-multiscale.git
cd tvb-multiscale/
git pull origin --allow-unrelated-histories
python setup.py develop --no-deps
