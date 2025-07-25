# EquiNNx is an equivariant neural network model designed for molecules and periodic systems
==========================================================================================
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This package integrates an equivariant neural network model to predict molecular and/or periodic systems. It can predict any tensor of rank ≤ 3, including scalars, vectors, matrices, and (3,3,3) tensors. It has been tested on the prediction of energy, forces, dipole moments, polarizability, and hyperpolarizability.

## Requirement
 - jax
 - e3x 
 - ase

After installation of conda, and activation of your environnement,  Type : 
```console
pip install -U jax
# If you have a NVIDIA GPU
pip install -U "jax[cuda12]" 
# If you have a TPU
pip install -U "jax[tpu]"
pip install --upgrade e3x
pip install ase
```

## Installation

Using git, under a terminal, type : 
```console
git clone https://github.com/allouchear/EquiNNx.git
```
You can also download the .zip file of EquiNNx : Click on Code and Download ZIP

## Main programs
### buildData.py
**Build data for training.**\
See xbuildData script in examples directory.

### train.py
**Using the database created by buildData.py, make the training**\
See train.inp and ./xtrain bash script in examples directory.
See getArgments() function in [src/Utils/UtilsFunctions.py](https://github.com/allouchear/EquiNNx/blob/master/src/Utils/UtilsFunctions.py) to obtain all input parameters.

### evaluation.py
**Test the models (one or an ensemble of models) using a database**\
See  xevaluation bash script in examples directory.
See getArgments() function in [src/evaluator.py](https://github.com/allouchear/EquiNNx/blob/master/src/evaluator.py) to obtain all input parameters.

### predict.py
**Predict the properties using a .xyz (with multiple geometries) file**\
See xpredict bash script in examples directory.
See getArgments() function in [src/predict.py](https://github.com/allouchear/EquiNNx/blob/master/src/predict.py) to obtain all input parameters.

### aseMD.py
**Make Molecular dynamic with ase and EquiNNx using a .xyz or a POSCAR file**\
see xaseMD bash script in examples directory.
See getArgments() function in [tools/aseMD.py](https://github.com/allouchear/EquiNNx/blob/master/tools/aseMD.py) to obtain all input parameters.


## Examples
see examples directory

## Contributors
The code is written by Abdul-Rahman Allouche.

## License
This software is licensed under the [GNU General Public License version 3 or any later version (GPL-3.0-or-later)](https://www.gnu.org/licenses/gpl.txt).

