# http://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#core-stacks
# FROM continuumio/anaconda:5.1.0
FROM jupyter/r-notebook:0f73f7488fa0

LABEL maintainer="Christian Schuhegger <christian.schuhegger@gmx.de>"

# https://stackoverflow.com/questions/42510052/pip-install-test-not-working-in-python2-7-9-pip-version-6-0-7
# https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs
# https://stackoverflow.com/questions/20288711/post-install-script-with-python-setuptools
RUN pip install --upgrade pip
RUN conda install -c r rpy2=2.9.1
RUN cd /home/jovyan/work && git clone https://github.com/cs224/pybnl.git --branch feature/2018-06-04-install-procedure && cd pybnl && pip install -e .[test]
