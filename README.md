# pybnl
Python interface to bnlearn and other probabilistic graphical model libraries

## Pre-requisites

Before you can install this library you have to have a working python3 version and a working R version plus rpy2 pre-installed.

Have a look at `pybnl/environment/Dockerfile` for details. This docker image is based on [jupyter/r-notebook](http://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#core-stacks), which is basically a [miniconda](https://conda.io/miniconda.html) environment plus some R basics like:

    conda install -c r r-essentials rpy2

## Install

Once you have the pre-requisites fulfilled installation can be done as follows:

    python -m pip install --upgrade pip setuptools wheel
    git clone https://github.com/cs224/pybnl.git
    cd pybnl
    pip install -e .

If you do not plan to do development work on this library yourself then you can also install it directly from github:

    python -m pip install --upgrade pip setuptools wheel
    pip --no-cache-dir install git+https://github.com/cs224/pybnl.git@master
