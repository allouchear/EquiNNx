import ase
from ase.neighborlist import neighbor_list
import functools
import os
import urllib.request
import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import sys
from pympler import tracker
from pympler import muppy
import psutil
from pympler import asizeof
import logging
from Utils.UtilsFunctions import *
#from memory_profiler import profile
from ase import Atoms
from ase.io import write,read
from ase.data import chemical_symbols
from Utils.DataContainer import *
from Utils.DataProvider import *
import re
from types import SimpleNamespace
from Utils.PeriodicTable import *

periodicTable=PeriodicTable()		

def add_values(values, results, data):
	if values is None:
		values = {}
		n=jnp.sum(jnp.asarray(data['N']))
		for key in  results.keys():
			if 'forces' in key:
				values[key] = []
				values[key].append(results[key][0:n])
			else:
				values[key] = results[key]
		values['ID'] = data['ID']
		'''
		values['N'] = jnp.asarray(data['N'])
		values['R'] = []
		values['R'].append(jnp.asarray(data['R'][0:n]))
		values['Z'] = []
		values['Z'].append(jnp.asarray(data['Z'][0:n]))
		'''
	else:
		n=jnp.sum(jnp.asarray(data['N']))
		for key in  results.keys():
			if 'forces' in key:
				values[key].append(results[key][0:n])
			else:
				values[key]=jnp.concatenate([values[key], results[key]], axis=0)
		values['ID'] = jnp.concatenate([values['ID'], data['ID']], axis=0)
		'''
		values['N'] = jnp.concatenate([values['N'], jnp.asarray(data['N'])], axis=0)
		values['R'].append(jnp.asarray(data['R'][0:n]))
		values['Z'].append(jnp.asarray(data['Z'][0:n]))
		'''
	return values

def sparse_pairwise_indices(num):
	if num < 1:
		raise ValueError(f'num must be larger than 0, received {num}')

	dst_idx = np.repeat(np.arange(num), num)
	src_idx = np.tile(np.arange(num), num)
	return dst_idx, src_idx

def gather_idx(R, idx):
	return np.take(R, idx, axis=0, out=None, mode='raise')

def get_distances(R, dst_idx, src_idx):
	#dst_R = e3x.ops.gather_dst(R, dst_idx=dst_idx)
	#src_R = e3x.ops.gather_src(R, src_idx=src_idx)
	dst_R = gather_idx(R, dst_idx)
	src_R = gather_idx(R, src_idx)
	distances = np.linalg.norm(src_R - dst_R, axis=-1)
	return distances

def cut_idx(R, dst_idx, src_idx, cutoff=None):
	if cutoff is None:
		return dst_idx, src_idx
	distances = get_distances(R, dst_idx, src_idx)
	dst_idx = dst_idx[distances<=cutoff]
	src_idx = src_idx[distances<=cutoff]
	return dst_idx, src_idx

def get_idx_list(atoms, cutoff=None):
	if any(atoms.get_pbc()):
		if cutoff is None:
			print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			print("cutoff  is needed for periodic system")
			print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			sys.exit()
		else:
			srcal=False
			dst_idx, src_idx, S = neighbor_list('ijS', atoms, cutoff, self_interaction=False)
			offsets = np.dot(S, atoms.get_cell())
	else:
		N=len(atoms.get_atomic_numbers())
		#dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(N)
		dst_idx, src_idx = sparse_pairwise_indices(N)
		offsets=None
		if cutoff is not None:
			dst_idx, src_idx =  cut_idx(atoms.get_positions(), dst_idx, src_idx, cutoff=cutoff)
	return dst_idx, src_idx, offsets

def set_idx(mols, cutoff, iBegin, iEnd, iWorker, verbose=0):
	a_dst_idx =[]
	a_src_idx =[]
	a_offsets =[]
	n=(iEnd-iBegin)
	ns=(iEnd-iBegin)//100
	ns = 1 if ns<1 else ns
	for im in range(iBegin, iEnd):
		if im%ns==0 and verbose>0:
			print("\t {:<5d} / {:>5d}   Wroker #{:<5d}.......".format(im-iBegin+1,n,iWorker),flush=True)

		atoms=mols[im]
		dst_idx, src_idx, offsets = get_idx_list(atoms, cutoff=cutoff)
		a_dst_idx.append(dst_idx[:])
		a_src_idx.append(src_idx[:])
		a_offsets.append(offsets)

	return  a_dst_idx, a_src_idx, a_offsets
	

def build_idx_list(mols, nmols, cutoff=None, verbose=0):
	all_dst_idx =[]
	all_src_idx =[]
	all_offsets =[]
	if verbose>0:
		print("Setting of neighbor lists.......",flush=True);
	if verbose>0:
		print("Number of structures =",nmols)
	dst_idx, src_idx, offsets = set_idx(mols, cutoff, 0,nmols,1, verbose=verbose)
	all_dst_idx.extend(dst_idx)
	all_src_idx.extend(src_idx)
	all_offsets.extend(offsets)
	if verbose>0:
		print("End setting of neighbor lists",flush=True);
	if np.all(np.array(all_offsets) == None):
		all_offsets = None
	return all_dst_idx, all_src_idx, all_offsets

def getData(mols):
	positions=[]
	Z=[]
	pbc=[]
	cells=[]
	dst_idx=[]
	src_idx=[]
	offsets=[]
	ID=[]
	n_idx=[] # number of idx by molecule
	for atoms in mols:
		Z.append(atoms.get_atomic_numbers())
		pbc.append(atoms.get_pbc())
		cells.append(atoms.get_cell()[:])
		positions.append(np.asarray(atoms.get_positions()))
		if 'ID' in atoms.info.keys():
			ID.append(atoms.info['ID'])
		if 'dst_idx' in atoms.info.keys():
			dst_idx.extend(np.asarray(atoms.info['dst_idx']))
			n_idx.append(len(atoms.info['dst_idx']))
		if 'src_idx' in atoms.info.keys():
			src_idx.extend(np.asarray(atoms.info['src_idx']))
		if 'offsets' in atoms.info.keys():
			if atoms.info['offsets'] is None or not isinstance( atoms.info['offsets'], np.ndarray):
				offsets.extend([None])
			else:
				offsets.extend(np.asarray(atoms.info['offsets']))

	data ={}
	data['N'] = []
	for z in Z:
		data['N'].append(z.shape[0])
	data['N']=np.asarray(data['N'])

	nmax=0
	for z in Z:
		if nmax<z.shape[0]:
			nmax= z.shape[0]
	ZZ = []
	for i,z in enumerate(Z):
		nres=nmax-z.shape[0]
		if nres>0:
			zz =z.tolist()
			zz += [0]*nres
			ZZ.append(zz)
		else:
			ZZ.append(z.tolist())
	Z = np.asarray(ZZ)
	R = []
	for i,r in enumerate(positions):
		nres=nmax-r.shape[0]
		if nres>0:
			zeros = np.zeros((nres, 3))
			R.append(np.vstack((r, zeros)))  # or use np.append(r, zeros, axis=0)
		else:
			R.append(r)
	R = np.asarray(R)

	pbc=np.asarray(pbc)
	#print(cells)
	cells=np.asarray(cells)
	data['Z'] = Z
	data['R'] = R
	if len(ID)>=1:
		data['ID'] = np.asarray(ID)
	else:
		data['ID'] = None

	data['cells'] = cells
	if len(n_idx)>=1:
		data['n_idx'] = np.asarray(n_idx)
	else:
		data['n_idx'] = None
	if len(dst_idx)>=1:
		data['dst_idx'] = (dst_idx)
	else:
		data['dst_idx'] = None

	if len(src_idx)>=1:
		data['src_idx'] = (src_idx)
	else:
		data['src_idx'] = None

	
	if len(offsets)>=1:
		if np.all(np.array(offsets) == None):
			data['offsets'] = None
		else:
			data['offsets'] = (offsets)
	else:
		data['offsets'] = None
	
	
	return data


def saveDatah5(data, filename):
	with h5py.File(filename, "w") as h5file:
		for key, value in data.items():
			if value is not None:
				h5file.create_dataset(key, data=value, compression="lzf")

def get_JSON_format(tensor):
	st = np.array2string(tensor, threshold=np.inf, separator=', ')
	st=re.sub(r"\s+", " ", st.replace('\n',' '))
	return "_JSON "+st

def build_mols(filename):
	mols=read(filename, format='extxyz',index=':') 
	for i in range(len(mols)):
		mols[i].info['ID']=i+1
		mols[i].set_pbc([False,False,False])

	return mols

def remove_index(mols):
	nmols=len(mols)

	for im in range(nmols):
		atoms=mols[im]
		if 'dst_idx' in atoms.info.keys():
			atoms.info['dst_idx'] = 'NONE'
			atoms.info['src_idx'] = 'NONE'
			atoms.info['offsets'] = 'NONE'
	return mols


def build_index(mols, cutoff=5, verbose=0):
	nmols=len(mols)
	all_dst_idx, all_src_idx, all_offsets =  build_idx_list(mols, nmols, cutoff=cutoff, verbose=verbose)
	if verbose>0:
		print("End build index")

	for im in range(nmols):
		atoms=mols[im]
		atoms.info['dst_idx'] = (np.array(all_dst_idx[im]))
		atoms.info['src_idx'] = (np.array(all_src_idx[im]))
		if all_offsets is not None and len(all_offsets)>0:
			#print(all_offsets[im])
			#atoms.info['offsets'] = get_JSON_format(all_offsets[im])
			atoms.info['offsets'] = (all_offsets[im])
		else:
			atoms.info['offsets'] = 'NONE'
	return mols


@functools.partial(jax.jit, static_argnames=('model_apply'))
def apply_one_step(model_apply, data, params):
	results = model_apply(params, data)
	return results


class Predictor:
	def __init__(self,
		models_directories,  # directories containing fitted models (can also be a list for ensembles)
		scale_output=None, # dictionary to scale output values
		scale_distance=1.0, # scale distance to model unit
		):

		if(type(models_directories) is not list):
			self._models_directories=[models_directories]
		else:
			self._models_directories=models_directories

		self._scale_output = scale_output
		self._scale_distance = scale_distance
		self._models = []
		self._restoreds = []
		n=0
		cutoff = None
		for directory in self._models_directories:
			model, restored = load_model(directory)
			if model is None:
				raise ValueError("I cannot read model from  {}".format(directory))
				
			config=restored['config']
			if cutoff is None:
				cutoff = config['cutoff']
				self._cutoff = cutoff
			elif abs(config['cutoff'] - self._cutoff)>0.01 :
				raise ValueError("The cutoff is not the same on all models")

			self._models.append(model)
			self._restoreds.append(restored)
		self._last_atoms = None
		self._reset_index = True

	def add_prop_to_mols(self):
		remove_index(self._mols)
		for i in range(len(self._mols)):
			self._mols[i].calc = None
			for key in self._results.keys():
				#self._mols[i].info[key] = get_JSON_format(self._results[key])
				self._mols[i].info[key] = np.asarray(self._results[key][i])

	def _scale_mols(self, mols, scale):
		for mol in mols:
			positions = mol.get_positions()
			mol.set_positions(positions * scale)
			cell = mol.get_cell()
			mol.set_cell(cell * scale)
		return mols

	def _get_properties(self, config):
		xyzfile=config.input_file_name
		configModel = self._models[0].get_config()
		cutoff=self.cutoff
		fn=xyzfile.split('.')[0]
		data_h5file=fn+'_'+str(cutoff)+".h5"
		mols=build_mols(xyzfile)
		mols = self._scale_mols(mols, self._scale_distance)
		mols = build_index(mols, cutoff=cutoff, verbose=config.verbose)
		data=getData(mols)
		saveDatah5(data, data_h5file)

		config.n_embeded_atoms=configModel['n_embeded_atoms']
		config.cutoff=cutoff
		config.dataset=data_h5file
		config.seed=-1

		config.num_train = len(mols)
		config.num_valid = 0
		config.num_test = 0
		if config.batch_size<-1 :
			config.batch_size=1

		data=DataContainer(config)
		dataProvider = DataProvider(data, config)
		size=len(data)
		if config.batch_size > size:
			config.batch_size=size

		n_batches = dataProvider.get_nsteps_batch()
		ncharall=45
		nchartrain = ncharall/n_batches
		ndigits=len(str(n_batches))

		all_values = None
		for i in range(len(self._models)):
			values = None
			keys = None
			dataProvider.reset_train_batch()
			if config.verbose>0:
				print("Model number ",i+1)
			for j in range(n_batches):
				data = dataProvider.next_train_batch()
				r = apply_one_step(
						model_apply=self._models[i].apply,
        					data=data,
						params=self._restoreds[i]['params']
					)
				if config.verbose>0:
					ieq=int((j+1)*nchartrain)
					isp=abs(ncharall-ieq)
					print("{:{width}d}/{:{width}d} [{}{}]".format(j+1,n_batches, "="*ieq, " "*isp, width=ndigits),end="\r",flush=True)
				values = add_values(values, r, data)
				keys=list(r.keys())
			if config.verbose>0:
				print("")
			if i == 0:
				all_values  = values
			else:
				for key in keys:
					all_values[key] +=  (values[key]-all_values[key])/(j+1)
		
		if self._scale_output is not None:
			for key in keys:
				if key in self._scale_output.keys():
					all_values[key] *= self._scale_output[key]
		mols = self._scale_mols(mols, 1.0/self._scale_distance)
		self._mols = mols
		self._results = all_values
		self.add_prop_to_mols()
		return self._mols

	def _get_data_atoms(self, mols):
		mol = mols[0]
		data = {}
		data['ID'] = np.array([1])
		N = len(mol)
		data['N']  = [N]
		data['Z']  = np.array(mol.get_atomic_numbers())
		data['R']  = np.array(mol.get_positions())
		if 'masses' in mol.info.keys():
			data['masses'] = np.array(mol.info['masses'])
		else:
			masses = []
			Z =data['Z']
			for ia in range(N):
				mass=periodicTable.elementZ(int(Z[ia])).isotopes[0].rMass
				masses.append(mass)
			data['masses'] = np.asarray(masses)
		data['energy'] = np.nan
		data['energybyatom'] = np.nan
		for key in mol.info.keys():
			if key == 'totalcharge':
				if mol.info[key] is not None:
					data[key]=([mol.info[key]]*N)
				else:
					data[key]=([0]*N)
			elif key == 'spinmultiplicity':
				if mol.info[key] is not None:
					data[key]=([mol.info[key]]*N)
				else:
					data[key]=([1]*N)
			elif key == 'offsets' and not isinstance(mol.info['offsets'], np.ndarray):
				data[key] = None
			else:
				data[key] = np.array(mol.info[key])
		if 'efield' not in mol.info.keys():
			data['efield'] = None
		data['batch_seg'] = [0]*N
		data['n_idx'] = np.array([len(data['dst_idx'])])
		data['cells'] = np.array([mol.get_cell()[:]])
		data['pbc'] = np.array([mol.get_pbc()])
		return data

	def _get_properties_atoms_scaling(self, model_apply, params, data, scaling=None):
		if scaling is not None and data["cells"] is not None:
			data["scaling"] = scaling
			data["R"] = data["R"] @ scaling
			if data["offsets"] is not None:
				data["offsets"] = data["offsets"] @ scaling
			if data["cells"] is not None:
				data["cells"] = data["cells"] @ scaling
		r = apply_one_step(
				model_apply=model_apply,
        			data=data,
				params=params
			)
		energy = None
		if 'energy' in r.keys():
			energy = jnp.sum(r['energy'])
		return energy,r


	def _get_properties_atoms(self, atoms, stress=False, verbose=0):
		#store copy of atoms
		configModel = self._models[0].get_config()
		cutoff=self.cutoff
		mols=[atoms.copy()]
		mols=self._scale_mols(mols, self._scale_distance)
		# DEBUG
		#self._reset_index=True
		if self._reset_index:
			mols = build_index(mols, cutoff=cutoff, verbose=verbose)
			self._reset_index = False
			self._last_atoms = atoms.copy()
			self._last_atoms.info=mols[0].info
		else:
			mols[0].info = self._last_atoms.info

		data = self._get_data_atoms(mols)
		scaling = None
		if stress and 'cells' in data.keys():
			cell=data['cells'][0].copy()
		periodic = any(data['pbc'][0])
		if stress and not periodic:
			raise ValueError("\nError : I cannot compute stress for a non periodic system\n")

		all_values = None
		for i in range(len(self._models)):
			values = None
			keys = None
			if verbose>0:
				print("Model number ",i+1)

			if stress and 'cells' in data.keys():
				#scaling = jnp.eye(3, dtype=data['R'].dtype)
				scaling = jnp.eye(3)

			if stress and periodic:
				properties_and_stress = jax.value_and_grad(self._get_properties_atoms_scaling, argnums=3, has_aux=True)
				(_,r), rstress = properties_and_stress(
					self._models[i].apply, 
					self._restoreds[i]['params'], 
					data, 
					scaling)
			else:
				(_,r) = self._get_properties_atoms_scaling(
					self._models[i].apply, 
					self._restoreds[i]['params'], 
					data
					)
			values = add_values(values, r, data)
			keys=list(r.keys())
			if i == 0:
				all_values  = values
			else:
				for key in keys:
					all_values[key] +=  (values[key]-all_values[key])/(j+1)

		if self._scale_output is not None:
			for key in keys:
				if key in self._scale_output.keys():
					if isinstance(all_values[key], list):
						all_values[key] = all_values[key][0] # forces is a list, one for each molecule. Here we have only one molecule
					all_values[key] *= self._scale_output[key]

		self._results = all_values
		
		self._energy = None
		self._forces = None
		self._stress = None
		if 'energy' in self._results.keys():
			self._energy = self._results['energy'][0]
		if 'forces' in self._results.keys():
			self._forces = np.asarray(self._results['forces']) # compatibality with ase
		for key in keys:
			if self._results[key].shape[0]==1:
				self._results[key]=float(self._results[key][0])
		if stress and data['cells'] is not None:
			self._stress = np.asarray(rstress)/np.abs(np.linalg.det(cell))*self._scale_output['stress']
			self._results['stress']=self._stress

	def calculation_required(self, atoms, quantities=None):
		return atoms != self._last_atoms  # the operator __eq__ already defined in Atoms class

	def get_potential_energy(self, atoms, force_consistent=False):
		if self.calculation_required(atoms):
			self._get_properties_atoms(atoms)
		return self._energy

	def get_energy_forces(self, atoms, force_consistent=False):
		if self.calculation_required(atoms):
			self._get_properties_atoms(atoms)
		return self._energy, self._forces

	def get_forces(self, atoms):
		if self.calculation_required(atoms):
			self._get_properties_atoms(atoms)
		return self._forces

	def get_stress(self, atoms, force_consistent=False):
		if self.calculation_required(atoms) or self._stress is None:
			self._get_properties_atoms(atoms, stress=True)
		return self._stress

	def reset_neighbor_list(self):
		self._reset_index = True

	@property
	def results(self):
		#return SimpleNamespace(**self._results)
		return self._results

	def __call__(self, config):
		return self._get_properties(config)

	@property
	def model(self):
        	return self._models

	@property
	def mols(self):
		return self._mols

	@property
	def cutoff(self):
		return self._cutoff


	@property
	def energy(self):
		return self._energy

	@property
	def forces(self):
		return self._forces


