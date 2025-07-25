import numpy as np
import sys
import warnings
import re
import h5py
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import write
import codecs

def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument("--input_file_name", default='file.h5', type=str, help="Name of the input file in h5 format. Default=file.h5")
	parser.add_argument("--prefix", default=None, type=str, help="the prefix name for the output files. Default=prefix of input_file_name")

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

def getmol(R, Z, dkeys):
	atoms = Atoms(Z)
	atoms.set_positions(R)
	for key in dkeys:
		if key=='cells':
			atoms.set_cell(dkeys[key])
		else:
			atoms.info[key] = dkeys[key]
	return  atoms

def convtoase(data):
	keys = ['potential', 'forces', 'cells', 'name', 'theory']
	mols=[]
	for i in range(data['N'].shape[0]):
		print(i,'/',data['N'].shape[0],end='\r',flush=True)
		R=data['R'][i,0:data['N'][i]]
		Z=data['Z'][i,0:data['N'][i]]
		dkeys={}
		for key in keys:
			dkeys[key]=data[key][i] if  key in data.keys() else None
		mol = getmol(R, Z, dkeys)
		mols.append(mol)
	return mols

args = getArguments()
print("Load data....",flush=True)
dataset = np.load(args.input_file_name)
nmols=dataset['E'].shape[0]
natoms=dataset['z'].shape[0]
data={}
data['name'] = np.asarray([codecs.decode(dataset['name'])]*nmols)
data['theory'] = np.asarray([dataset['theory']]*nmols)
data['N'] = np.asarray([natoms]*nmols)
data['cells'] = np.asarray([[False, False, False]]*nmols)
data['Z'] = np.asarray([dataset['z']]*nmols)  # same number of atoms in all molecules in dataset assumed
data['potential'] = np.asarray(dataset['E'])
data['R'] = np.asarray(dataset['R'])
data['forces'] = np.asarray(dataset['F'])
mols = convtoase(data)
fn = args.prefix
if fn is None:
	fn = args.input_file_name.split('.')[0]
print("Save data in a xyz file  ...",flush=True)
data_xyz=fn+'.xyz'
write(data_xyz, mols, format='extxyz')
print("See ", data_xyz , "file",flush=True)

print("That's all ",flush=True)
