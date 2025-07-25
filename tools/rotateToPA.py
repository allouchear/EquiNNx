import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

import argparse
import sys
from ase import io
from ase import Atoms
from ase.io import write,read
from ase.data import chemical_symbols

def get_principal_axes(atoms: Atoms, n_embeded_atoms=-1):
	nAtoms=n_embeded_atoms
	if nAtoms<0 or nAtoms> atoms.info['N']:
		nAtoms = atoms.info['N']
	new_atoms = Atoms(atoms.get_atomic_numbers())
	new_atoms.set_atomic_numbers(atoms.get_atomic_numbers())
	new_atoms.set_positions(atoms.get_positions())
	inertia_tensor = new_atoms.get_moments_of_inertia(vectors=True)[1]
	# Each column is a principal axis
	R = inertia_tensor.T  # Make it a rotation matrix: world → principal axes
	#print(R.dot(R.T))
	return R

def rotate_tensor(tensor, R):
	if tensor.ndim == 1:
		return R @ tensor
	elif tensor.ndim == 2:
		return R @ tensor @ R.T
	elif tensor.ndim == 3:
		# β_ijk' = R_ia R_jb R_kc β_abc
		return np.einsum('ia,jb,kc,abc->ijk', R, R, R, tensor)
	else:
		raise ValueError("Tensor rank not supported.")

def rotate_molecule(mol, R):
    mol.translate(-mol.get_center_of_mass())
    rotated_positions = mol.get_positions() @ R.T  # or np.dot(pos, R.T)
    mol.set_positions(rotated_positions)
    return mol

def rotate_molecule_and_info(mol, n_embeded_atoms=-1):
	Rot=get_principal_axes(mol, n_embeded_atoms=n_embeded_atoms)
	for key in mol.info.keys():
		if '_predict' in key or '_target' in key:
			mol.info[key] =rotate_tensor(mol.info[key],Rot)
	mol = rotate_molecule(mol, Rot)
	return mol

def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument('--input_file_name', default="evaluation.npz", type=str, help="input data file, Default=evaluation.npz")
	parser.add_argument('--n_embeded_atoms', default=-1, type=int, help="Compute the rot matrix usinf the first n_embeded_atoms. Default = -1=> all atoms")
	parser.add_argument("--output_file_name", default='evaluation_rotated.npz', type=str, help="Name of the output file in npz format. Default=evaluation.npz")

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

def getmol(R, Z,dkeys):
	atoms = Atoms(Z)
	atoms.set_positions(R)
	for key in dkeys:
		if key=='cells':
			atoms.set_cell(dkeys[key])
		else:
			atoms.info[key] = dkeys[key]
	return atoms

def convtoase(data):
	keys = []
	for key in data.keys():
		if '_predict' in key:
			k=key.split('_')[0]
			keys.append(k+'_predict')
			keys.append(k+'_target')
	keys.append('ID')
	mols=[]
	nmols=len(data['N'])
	ibegin=0
	for i in range(nmols):
		print("{:<5d} / {:>5d}.......\r".format(i+1,nmols),flush=True,end=' ')
		N = len(data['Z'])
		R=data['R'][ibegin:ibegin+data['N'][i]]
		Z=data['Z'][ibegin:ibegin+data['N'][i]]
		dkeys={}
		for key in keys:
			dkeys[key] = data[key][i]
		dkeys['cells'] = [0.0,0.0,0.0]
		dkeys['N'] = data['N'][i] # this is the real number of atoms. mol can contain ghost atoms
		mol = getmol(R, Z,dkeys)
		mols.append(mol)
		ibegin += data['N'][i]
	return mols

def build_mols(filename):
	data   = np.load(args.input_file_name,allow_pickle=True)['values']
	dkeys=data.item().keys()
	data=data.item()
	mols = convtoase(data)
	for i in range(len(mols)):
		mols[i].info['ID']=i+1
		mols[i].set_pbc([False,False,False])
	return mols

def build_data(mols):
	keys=[]
	for key in mols[0].info.keys():
		if '_predict' in key or '_target' in key:
			keys.append(key)
	keys.append('N')
	keys.append('ID')
	data={}
	data['R'] = []
	data['Z'] = []
	data['pbc'] = []
	data['cells'] = []
	for key in keys:
		data[key]=[]
	for mol in mols:
		data['Z'].append(mol.get_atomic_numbers())
		data['pbc'].append(mol.get_pbc())
		data['cells'].append(mol.get_cell()[:])
		data['R'].append(np.asarray(mol.get_positions()))
		for key in keys:
			data[key].append(np.asarray(mol.info[key]))
	for key in keys:
		data[key]=np.asarray(data[key])
	return data
	
def rotate_molecules(mols,n_embeded_atoms=-1):
	rmols=[]
	for mol in mols:
		rmol=rotate_molecule_and_info(mol,n_embeded_atoms=n_embeded_atoms)
		rmols.append(rmol)
	return rmols

args = getArguments()
print("build molecules ....")
mols = build_mols(args.input_file_name)
print("rotate molecules ....")
mols = rotate_molecules(mols, n_embeded_atoms=args.n_embeded_atoms)
print("build data ....")
data = build_data(mols)
print("Results saved in ", args.output_file_name, "file"); 
np.savez(args.output_file_name, values=data)
print("-"*80)
