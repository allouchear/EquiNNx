from ase import io
from ase.units import Bohr,Hartree,kcal,mol,GPa
# Electron volts (eV), Ångström (Ang), the atomic mass unit and Kelvin are defined as 1.0.
import sys

import numpy as np
from Utils.Predictor import *
import os
import shutil

def remove_directory(dir_path):
	try:
		shutil.rmtree(dir_path)
	except OSError as e:
		print("Error: %s : %s" % (dir_path, e.strerror))

def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument('--list_models', type=str, nargs='+', help="list of directory containing fitted models (at least one file), ....")
	parser.add_argument('--input_file_name', default="molecule.xyz", type=str, help="xyz file")
	parser.add_argument('--unit_models', default="Hartree,Bohr", type=str, help="the unit of model, that used during train")

	#if no command line arguments are present, config file is parsed
	config_file='config.txt'
	fromFile=False
	if len(sys.argv) == 1:
		fromFile=False
	if len(sys.argv) == 2 and sys.argv[1].find('--') == -1:
		config_file=sys.argv[1]
		fromFile=True

	if fromFile is True:
		print("Try to read configuration from ",config_file, "file")
		if os.path.isfile(config_file):
			args = parser.parse_args(["@"+config_file])
		else:
			args = parser.parse_args(["--help"])
	else:
		args = parser.parse_args()

	return args

args = getArguments()
lmodels=args.list_models
lmodels=lmodels[0].split(',')
print("---------------------------------------------")
print("Models = ", lmodels)


#atoms = io.read('POSCAR')

print(lmodels)
atoms = io.read(args.input_file_name)
if args.unit_models.upper()=="KCAL/MOL,ANG":
	atoms.calc = Predictor(lmodels,
				scale_distance=1.0,
				scale_output={'energy':kcal/mol,'forces':kcal/mol}
				)
elif args.unit_models.upper()=="HARTREE,BOHR":
	atoms.calc = Predictor(lmodels,
				scale_distance=1.0/Bohr,
				scale_output={'energy':Hartree,'forces':Hartree/Bohr, 'stress':Hartree/(Bohr*Bohr*Bohr)}
				)
elif args.unit_models.upper()=="EV,ANG":
	atoms.calc = Predictor(lmodels,
				scale_distance=1.0,
				scale_output={'energy':1.0,'forces':1.0, 'stress':1.0}
				)
else:
		raise ValueError('Unknown unit_models. Please use kcal/mol,Ang or Hartree,Bohr or eV,Ang')
	
e = atoms.get_potential_energy()
print("Energy=",e)
f = atoms.get_forces()
print(f)
periodic = atoms.pbc.any()
if periodic :
	print("stress en eV/A^3 : ",atoms.get_stress())
	print("stress en GPa : ",atoms.get_stress()/GPa)
	print("stress en bar : ",atoms.get_stress()/GPa*1e4)

from ase.optimize import BFGS

BFGS(atoms).run(fmax=0.001)

print("Geometry after optimization")
print(atoms.get_positions())

fn=''.join(args.input_file_name.split(".")[0:-1])
opt_xyz=fn+"_opt.xyz"
io.write(opt_xyz, atoms, format='extxyz')
#io.write(opt_xyz, atoms, format='xyz')
print("See ", opt_xyz , "file",flush=True)
if periodic :
	print("stress en eV/A^3 : ",atoms.get_stress())
	print("stress en GPa : ",atoms.get_stress()/GPa)
	print("stress en bar : ",atoms.get_stress()/GPa*1e4)

"""
from ase.vibrations import Vibrations
vib = Vibrations(atoms)
vib.run()
print("End run")
vib.summary()
"""
