# Code for similarity calculations and $EvoSim$

An example similarity calculation can be found in `sim_calc.ipynb`.

An example of $EvoSim$ application can be found in `evosim.ipynb`.

To properly use `similarity.py` code, users need to first install the following packages.
* Surfgraph; https://surfgraph.readthedocs.io/en/latest/Installation.html
* PyTorch; https://pytorch.org
* Numpy version < 2.0
* scikit-learn; https://scikit-learn.org/stable/install.html
* Joblib; for parallel processing speed-up; https://joblib.readthedocs.io/en/stable/installing.html

Finally,
* the `similarity.py` needs to be added to the python path for similarity score calculations.
* the `utility.py` needs to be added to the python path for $EvoSim$.