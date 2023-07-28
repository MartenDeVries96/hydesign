Welcome to our surrogate model repo. Follow these steps to complete installation:

0. Make sure you have ~ 2 GB of memory available

1. Activate a conda environment

   (more info here: https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

2. unzip the model by running:

   `tar -xzvf model.tar.gz`

   (if tar is not available, you can install it with `conda install -c conda-forge tar`)

3. Make sure GitPython, xarray, pandas, numpy, etc are installed

    `pip install xarray pandas numpy matplotlib jupyter scikit-learn chaospy`

    (if pip isn't available, you can install it with `conda install pip`)

4. Install git dependancies by running

    `python INSTALL.py`


... and that's it! Now open and run the quickstart notebook
