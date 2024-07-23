# TPOT2 Imputer Experiments

TPOT stands for Tree-based Pipeline Optimization Tool. TPOT2 is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming. 

TPOT2 is a rewrite of TPOT with some additional functionality. Notably, we added support for graph-based pipelines and additional parameters to better specify the desired search space. 
TPOT2 is currently in Alpha. This means that there will likely be some backwards incompatible changes to the API as we develop. Some implemented features may be buggy. There is a list of known issues written at the bottom of this README. Some features have placeholder names or are listed as "Experimental" in the doc string. These are features that may not be fully implemented and may or may not work with all other features.

This repository contains experimental functions for the purposes of testing the role of missing value imputation in AutoML, and allowing you to recreate these experiments yourself. Most of these functions are not intended to be implemented in API releases of TPOT2. 

If you are interested in using the current stable release of TPOT, you can do that here: [https://github.com/EpistasisLab/tpot/](https://github.com/EpistasisLab/tpot/). 


## License

Please see the [repository license](https://github.com/EpistasisLab/tpot2/blob/main/LICENSE) for the licensing and usage information for TPOT2.
Generally, we have licensed TPOT2 to make it as widely usable as possible.

## Documentation

[The documentation webpage for TPOT2 can be found here.](https://epistasislab.github.io/tpot2/)

This repository README serves as reproucability documentation for an as of yet untitled paper.  

## Installation

TPOT2 requires a working installation of Python.

To replicate this experiment: 
1. Fork this repository into a folder on your computer.
2. In the terminal,type 'cd ..' into the command line to enter the parent folder.
3. Rename the 'tpot2_addition' folder to 'tpot2'
4. Create and activate a virtual environment as desribed below.
5. Check that you are in a venv folder by typing 'which pip'.
7. In the terminal, type 'pip install -e tpot2' to install the experimental tpot2 version from this repository.
8. If you get an error on installation regarding greenlet not installing, run
```
pip install --only-binary :all: greenlet
pip install --only-binary :all: Flask-SQLAlchemy
```
and repeat step 7. 
9. If you are using a MacOS computer, delete or hash-mark out line 3 in requirements_.txt. scikit-learn-intelex>=2023.2.1 is not compatible with some verisons of MacOS.
10. Then  type 'pip install -r requirements_.txt' to install additional experimental packages.
11. Run the test.ipynb notebook to check that TPOT2 is correctly installed.


### Creating a venv environment.

While we recommend using conda environments for installing TPOT2, multiple conda versions cause versioning issues in MacOs.
Here we suggest 

```
python3.10 -m venv tpot2env
source tpot2env/bin/activate
```

### Packages Used

python version <3.12
numpy
scipy
scikit-learn
update_checker
tqdm
stopit
pandas
joblib
xgboost
matplotlib
traitlets
lightgbm
optuna
baikal
jupyter
networkx>
dask
distributed
dask-ml
dask-jobqueue
func_timeout
configspace

#experiemental packages:
skrebate
scikit-mdr

Many of the hyperparameter ranges used in our configspaces were adapted from either the original TPOT package or the AutoSklearn package. 

## Usage 

See the Experiments Folder for more instructions and analysis details.


## Contributing to TPOT2

We welcome you to check the existing issues for bugs or enhancements to work on. If you have an idea for an extension to TPOT2, please file a new issue so we can discuss it.


### Support for TPOT2

TPOT2 was developed in the [Artificial Intelligence Innovation (A2I) Lab](http://epistasis.org/) at Cedars-Sinai with funding from the [NIH](http://www.nih.gov/) under grants U01 AG066833 and R01 LM010098. We are incredibly grateful for the support of the NIH and the Cedars-Sinai during the development of this project.

The TPOT logo was designed by Todd Newmuis, who generously donated his time to the project.
