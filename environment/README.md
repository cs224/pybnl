
This folder contains infrastructure to ensure that clean builds work in fresh environments

    make

The above will create a docker image based on [jupyter/r-notebook](http://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#core-stacks), which is basically a [miniconda](https://conda.io/miniconda.html) environment plus some R basics like:

    conda install -c r r-essentials rpy2

It will then install the `pybnl` package via `pip install ...` on top of this base image to ensure that the set-up catches all relevant dependencies.

You then run the unit tests as follows:

    make test

The above will use the created docker container to run the unit tests. If this succeeds the package should be good, both from a pependency definition perspective and a functionality perspective.
