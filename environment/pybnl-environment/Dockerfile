# http://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#core-stacks
# FROM continuumio/anaconda:5.1.0
FROM jupyter/r-notebook:0f73f7488fa0

LABEL maintainer="Christian Schuhegger <christian.schuhegger@gmx.de>"

# https://stackoverflow.com/questions/42510052/pip-install-test-not-working-in-python2-7-9-pip-version-6-0-7
# https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs
# https://stackoverflow.com/questions/20288711/post-install-script-with-python-setuptools
RUN pip install --upgrade pip
RUN conda install -c r rpy2=2.9.1
# https://bioconda.github.io/recipes.html
# RUN conda install -c bioconda bioconductor-biocgraph bioconductor-rbgl
# RUN conda install -c conda-forge r-bnlearn
# https://anaconda.org/kurtwheeler/cran-grbase
# https://stackoverflow.com/questions/45289764/install-r-packages-using-docker-file
# https://stackoverflow.com/questions/6907937/how-to-install-dependencies-when-using-r-cmd-install-to-install-r-packages


# RUN cd /home/jovyan/work && git clone https://github.com/cs224/pybnl.git --branch feature/2018-06-04-install-procedure && cd pybnl && pip install -e .[test]
COPY pybnl $HOME/work/pybnl

USER $root
RUN chown -R jovyan.users $HOME/work
# docker run -it --rm --user root pybnl-test-image:latest /bin/bash
RUN rm -rf /home/jovyan/work/pybnl/pybnl.egg-info

USER $NB_UID
RUN cd /home/jovyan/work && cd pybnl && pip install -e .[test]
