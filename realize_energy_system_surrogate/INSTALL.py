from git import Repo # pip install GitPython
import os

# make directory to install MSVR
msvr_dir = os.sep.join([os.getcwd(), 'msvr'])

# clone MSVR
if not os.path.exists(msvr_dir):
   Repo.clone_from("https://github.com/Analytics-for-Forecasting/msvr", msvr_dir)

# clone country flags for plotting
if not os.path.exists(os.sep.join([os.getcwd(), 'plotting'])):
   os.mkdir('plotting')
   if not os.path.exists(os.sep.join([os.getcwd(), 'plotting', 'country-flags'])):
      Repo.clone_from("https://github.com/hampusborgos/country-flags", os.sep.join([os.getcwd(), 'plotting', 'country-flags']))

