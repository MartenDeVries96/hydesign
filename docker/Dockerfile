FROM continuumio/miniconda3:latest
MAINTAINER Mikkel Friis-Møller <mikf@dtu.dk>
#ENV PATH="/opt/conda/bin:${PATH}"

RUN conda create -n py37 python=3.7
RUN echo "source activate py37" > ~/.bashrc
ENV PATH /opt/conda/envs/py37/bin:$PATH

RUN conda install -n py37 -y -c conda-forge finitediff
RUN conda install -n py37 -y -c conda-forge hdf4
RUN conda install -n py37 -y -c conda-forge hdf5
RUN conda install -n py37 -y -c conda-forge matplotlib
RUN conda install -n py37 -y -c conda-forge numpy
RUN conda install -n py37 -y -c conda-forge pandas
RUN conda install -n py37 -y -c conda-forge pip
RUN conda install -n py37 -y -c conda-forge scikit-learn
RUN conda install -n py37 -y -c conda-forge scipy
RUN conda install -n py37 -y -c conda-forge sphinx=2.2.0
RUN conda install -n py37 -y -c conda-forge wisdem
RUN conda install -n py37 -y -c conda-forge netcdf4
RUN conda install -n py37 -y -c conda-forge zarr 
RUN conda install -n py37 -y -c conda-forge dask
RUN conda install -n py37 -y -c conda-forge xarray
RUN conda install -n py37 -y -c conda-forge jupyterlab
RUN conda install -n py37 -y -c conda-forge openmdao[all]
RUN conda install -n py37 -y -c conda-forge smt
RUN conda install -n py37 -y -c conda-forge pyomo
RUN conda install -n py37 -y -c conda-forge pyomo.extras
RUN conda install -n py37 -y -c conda-forge glpk


RUN /opt/conda/envs/py37/bin/pip install docplex==2.15.194
RUN /opt/conda/envs/py37/bin/pip install numpy-financial==1.0.0
RUN /opt/conda/envs/py37/bin/pip install pvlib
RUN /opt/conda/envs/py37/bin/pip install seaborn
RUN /opt/conda/envs/py37/bin/pip install statsmodels
RUN /opt/conda/envs/py37/bin/pip install py_wake
RUN /opt/conda/envs/py37/bin/pip install rainflow

