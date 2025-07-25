# EquiNNx  An Equivariant neural network model to predict properties of molecules and periodic systems
======================================================================================================

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Requirement
 - jax, 
 - E3x 
 - ase

After installation of conda, and activation of your environnement,  Type : 
```console
pip install -U jax
# If you have a NVIDIA GPU
pip install -U "jax[cuda12]" 
# If you have a TPU
pip install -U "jax[tpu]"
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
see train.inp and ./xtrain bash script in examples directory.

### evaluation.py
**Test the models (one or an ensemble of models) using a database**\
see  ./xevaluation bash script in examples directory.

### predict.py
**Predict the properties using a .xyz (with multiple geometries) file**\
see ./xpredict bash script in examples directory.

### aseMD.py
**Make Molecular dynamic with ase and EquiNNx using a .xyz or a POSCAR file**\
see ./xaseMD bash script in examples directory.


## Examples
see examples directory

## Contributors
The code is written by Abdul-Rahman Allouche.

## License
This software is licensed under the [GNU General Public License version 3 or any later version (GPL-3.0-or-later)](https://www.gnu.org/licenses/gpl.txt).

