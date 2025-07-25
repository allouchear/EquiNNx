import numpy as np
import os
import sys
from ase import io


def rotate(fi, fo, angle, vector):
	atoms=io.read(fi, format='extxyz')
	edipole=atoms.info['edipole']
	atoms.append('Ag')
	atoms.positions[-1]=edipole
	print(edipole)
	print(atoms)
	atoms.rotate(angle, vector)
	print(atoms.positions[-1])
	atoms.info['edipole']=atoms.positions[-1]
	del atoms[-1]
	io.write(fo, atoms,format='extxyz')
	print('see file ', fo)
	
fi='mol.xyz'
fo='molr.xyz'
rotate('mol.xyz', 'molr.xyz', 45, (1,1,1))
rotate('mol.xyz', 'molx.xyz', 90, (1,0,0))
rotate('mol.xyz', 'moly.xyz', 90, (0,1,0))
rotate('mol.xyz', 'molz.xyz', 90, (0,0,1))



