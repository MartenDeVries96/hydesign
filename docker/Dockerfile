FROM continuumio/anaconda3:latest
MAINTAINER Mikkel Friis-Møller <mikf@dtu.dk>

# build: 
# docker build . -t dtuwindenergy/hydesign
# tag:
# docker tag dtuwindenergy/hydesign dtuwindenergy/hydesign:latest (if needed)
# docker tag dtuwindenergy/hydesign dtuwindenergy/hydesign:<version number>
# push:
# docker login
# docker push --all-tags dtuwindenergy/hydesign

RUN apt-get update
RUN apt-get -y install make pandoc graphviz

ENV PATH="/opt/conda/bin:${PATH}"

RUN conda install -y -c conda-forge finitediff
RUN conda install -y -c conda-forge numpy
RUN conda install -y -c conda-forge pandas
RUN conda install -y -c conda-forge pip
RUN conda install -y -c conda-forge scikit-learn
RUN conda install -y -c conda-forge scipy
RUN conda install -y -c conda-forge xarray
RUN conda install -y -c conda-forge openmdao[all]
RUN conda install -y -c conda-forge smt
RUN conda install -y -c conda-forge jupyterlab
RUN conda install -y -c conda-forge yaml
RUN conda install -y -c conda-forge glpk

RUN pip install netcdf4
RUN pip install h5netcdf
RUN pip install cplex
RUN pip install docplex
RUN pip install numpy-financial
RUN pip install pvlib
RUN pip install statsmodels
RUN pip install rainflow

RUN pip install --upgrade pip ipython ipykernel
RUN pip install pypandoc
RUN pip install sphinx --upgrade
RUN pip install nbsphinx --upgrade
RUN pip install nbconvert --upgrade
RUN pip install sphinx_rtd_theme


RUN ipython kernel install --name "python3" --user
