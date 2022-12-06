conda create -n hydesign python=3.7
conda activate hydesign

conda install -y -c conda-forge finitediff
conda install -y -c conda-forge hdf4
conda install -y -c conda-forge hdf5
conda install -y -c conda-forge matplotlib
conda install -y -c conda-forge numpy
conda install -y -c conda-forge pandas
conda install -y -c conda-forge pip
conda install -y -c conda-forge scikit-learn
conda install -y -c conda-forge scipy
conda install -y -c conda-forge sphinx=2.2.0
conda install -y -c conda-forge wisdem
conda install -y -c conda-forge netcdf4
conda install -y -c conda-forge zarr
conda install -y -c conda-forge dask
conda install -y -c conda-forge xarray
conda install -y -c conda-forge jupyterlab
conda install -y -c conda-forge openmdao[all]
conda install -y -c conda-forge smt
conda install -y -c conda-forge pyomo
conda install -y -c conda-forge pyomo.extras
conda install -y -c conda-forge glpk

pip install docplex==2.15.194
pip install numpy-financial==1.0.0
pip install pvlib
pip install seaborn
pip install statsmodels
pip install py_wake
pip install rainflow

