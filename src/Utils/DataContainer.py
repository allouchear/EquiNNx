import sys
import jax
import jax.numpy as jnp
import numpy as np
from ase.io import read
from Utils.UtilsFunctions import *
from Utils.PeriodicTable import *
import ase
from ase import Atoms
from ase.neighborlist import neighbor_list
import h5py
import warnings

KEYS = [
	'ID','N','Z','R','forces','masses', 'energy', 'energybyatom', 'totalcharge','spinmultiplicity', 'efield',
	'offsets','n_idx','dst_idx','src_idx',
	'edipole', 'polarizability', 'hyperpolarizability',
	'sdipole','alfa','beta'
]

def build_batch(container, idx, keys, n_embeded_atoms):
	"""Build a padded-free batch dict from a container (dict of arrays indexed by molecule).

	Pure function extracted from DataContainer.__getitem__ so that both the in-memory
	path and buildDataBatching produce bit-identical batches.
	"""
	if type(idx) is int or type(idx) is np.int64:
		idx = [idx]
	data = {}
	for key in keys:
		data[key] = []
	data['batch_seg'] = []

	Ntot = 0 #total number of atoms
	for k, i in enumerate(idx):
		N = int(container['N'][i]) #number of atoms
		for key in keys:
			if key=='efield' and container[key] is not None:
				M = np.tile(container[key][i], (N, 1))
				if len(data['efield'])==0:
					data['efield'] = M
				else:
					data['efield'] = np.vstack((data['efield'], M))
			elif key=='cells' and container[key] is not None:
				data[key].extend(container[key][i:i+1,:,:].tolist())
			elif key in ['Z','masses'] and container[key] is not None:
				data[key].extend(container[key][i,:N].tolist())
			elif key in ['R','forces'] and container[key] is not None:
				data[key].extend(container[key][i,:N,:].tolist())
			elif key in ['dst_idx','src_idx'] and container[key] is not None and len(container[key])==len(container['Z']):
				data[key].extend(np.reshape(container[key][i]+Ntot,[-1]).tolist())
			elif key in ['totalcharge','spinmultiplicity'] and container[key] is not None:
				data[key].extend([container[key][i]]*N)
			elif key=='offsets' and container[key] is not None and container['dst_idx']  and len(container['dst_idx'])==len(container['Z']):
				data['offsets'].extend(container[key][i].tolist())
			elif container[key] is not None:
				data[key].append(container[key][i])

		skeys = ['energybyatom', 'energy']
		for key in skeys:
			if key in keys and container[key] is None: 
				data[key].extend([(np.nan)])
		skeys = ['spinmultiplicity', 'masses']
		for key in skeys:
			if key in keys and container[key] is None: 
				data[key].extend([1.0]*N)
		key = 'totalcharge'
		if key in keys and container[key] is None: 
			data[key].extend([0.0]*N)
		key = 'ID'
		if key in keys and container[key] is None: 
			data[key].append(i+1)
		key = 'Z'
		if key in keys and container[key] is None: 
			data[key].append(0)

		if len(container['dst_idx'])!=len(container['Z']): 
			data['dst_idx'].extend(np.reshape(container['dst_idx'][:N,:N-1]+Ntot,[-1]).tolist())
			data['src_idx'].extend(np.reshape(container['src_idx'] [:N,:N-1]+Ntot,[-1]).tolist())
			data['offsets']=[]
		#offsets could be added in case they are need
		#data['batch_seg'].extend([k]*N)
		if n_embeded_atoms<=0 or n_embeded_atoms>=N:
			data['batch_seg'].extend([2*k]*N)
		else:
			data['batch_seg'].extend([2*k]*n_embeded_atoms)
			data['batch_seg'].extend([2*k+1]*(N-n_embeded_atoms))
		#increment totals
		Ntot += N

	listform =['batch_seg','N']
	for key in keys:
		if key not in listform:
			if data[key] is not None and len(data[key])>0:
				data[key] = jnp.asarray(data[key])
			else:
				data[key] = None
	return data

def saveDatah5(data, filename):
	with h5py.File(filename, "w") as h5file:
		for key, value in data.items():
			if value is not None:
				h5file.create_dataset(key, data=value, compression="lzf")

def loadDatah5(filename):
	with h5py.File(filename, "r") as h5file:
		data = {key: h5file[key][()] for key in h5file.keys()}
	return data

def build_idx_format(data):
	if 'dst_idx' not in data  or 'src_idx' not in data  :
		return data
	if 'n_idx' in data.keys():
		ibegin=0
		dst_idx = []
		src_idx = []
		offsets= []
		for k in range(data['n_idx'].shape[0]):
			iend=ibegin+data['n_idx'][k]
			dst_idx.append(data['dst_idx'][ibegin:iend])
			src_idx.append(data['src_idx'][ibegin:iend])
			if 'offsets' in data and data['offsets'] is not None:
				offsets.append(data['offsets'][ibegin:iend])
			ibegin = iend
		if 'offsets' not in data:
			offsets =  None
	else:
		dst_idx = None
		src_idx = None
		offsets=None
	data['dst_idx'] = dst_idx
	data['src_idx'] = src_idx
	data['offsets'] = offsets

	return data

def build_idx(data, idx,datanew):
	dst_idx = []
	src_idx = []
	for k in idx:
		dst_idx.extend(data['dst_idx'][k].tolist())
		src_idx.extend(data['src_idx'][k].tolist())
	datanew['dst_idx']= np.asarray(dst_idx)
	datanew['src_idx']= np.asarray(src_idx)

	if data['offsets'] is not None:
		offsets = []
		for k in idx:
			offsets.extend(data['offsets'][k])
		datanew['offsets']= np.asarray(offsets)
	return datanew


class DataContainer:
	def __repr__(self):
		return "DataContainer"
	def __init__(self, config):
		if config.verbose>0 :
			print("Reading of data.....",flush=True)
		filename =config.dataset
		data = loadDatah5(filename)

		self._n_embeded_atoms=config.n_embeded_atoms
		if config.num_train<=0:
			#print(data['N'])
			config.num_train = len(data['N'])
		ns=config.num_train+ config.num_valid
		if config.num_test>0:
			ns=config.num_test
		num_struct=ns
		seed=config.seed
		#read in data
		nAll=len(data['N'])
		ns=-1
		if num_struct>0:
			ns=num_struct
		else:
			ns=nAll

		if seed is not None and seed>-1:
			random_state = np.random.RandomState(seed=seed)
			idx = random_state.permutation(np.arange(nAll))[0:ns]
		else:
			idx = np.arange(ns)

		idx = idx.tolist()
		idx = jnp.asarray(idx)

		self._container = {}

		self._keys = KEYS
		for key in self._keys:
			if key in data and data[key] is not None: 
				self._container[key] = data[key][idx]
			else:
				self._container[key] = None
		if self._container['totalcharge'] is None:
			self._container['totalcharge'] = jnp.asarray([0.0]*len(idx))
		if self._container['spinmultiplicity'] is None:
			self._container['spinmultiplicity'] = jnp.asarray([1.0]*len(idx))
		if self._container['energy'] is None:
			self._container['energybyatom']
		else:
			self._container['energybyatom'] = self._container['energy']/self._container['N'] # energy by atom

		if self._container['energy'] is not None : 
			meane=np.mean(self._container['energy'])
			print("-"*60)
			print("mean/energy        =", jnp.mean(jnp.asarray(self._container['energy'])))
			print("std/energy         = ",jnp.std(jnp.asarray(self._container['energy'])))
			print("mean/energybyatom  =", jnp.mean(jnp.asarray(self._container['energybyatom'])))
			print("std/energybyatom   = ",jnp.std(jnp.asarray(self._container['energybyatom'])))
			print("-"*60)

		if self._container['forces'] is not None: 
			print("std/forces         =", np.std(self._container['forces']))
			print("-"*60)

		if self._container['masses'] is None and self._container['Z'] is not None:
			periodicTable=PeriodicTable()		
			NMax = 0
			for imol in range(len(idx)):
				if self._container['N'][imol]>NMax:
					NMax = self._container['N'][imol]
			masses = []
			for imol in range(len(idx)):
				Z =self._container['Z'][imol]
				mMol=[]
				for ia in range(self._container['N'][imol]):
					mass=periodicTable.elementZ(int(Z[ia])).isotopes[0].rMass
					mMol.append(mass)
				nrest=NMax-self._container['N'][imol]
				if nrest>0:
					mMol.extend([1.0]*nrest)
				masses.append(mMol)
			self._container['masses'] = jnp.asarray(masses)

		if self._container['offsets'] is not None:
			self._container['offsets']=0 # to be take into account in build_idx_format
		self._datafile = data
		self.build_idx_format(data, idx)
		self._idx = idx

	def build_idx_format(self, data, idx):
		if self._container['offsets'] is None:
			dataoffsets=None
		else:
			dataoffsets=data['offsets']
		if 'n_idx' in data.keys():
			ibegin=0
			dst_idx = []
			src_idx = []
			offsets= []
			for k in range(data['n_idx'].shape[0]):
				iend=ibegin+data['n_idx'][k]
				dst_idx.append(data['dst_idx'][ibegin:iend])
				src_idx.append(data['src_idx'][ibegin:iend])
				if self._container['offsets'] is not None:
					offsets.append(dataoffsets[ibegin:iend])
				ibegin = iend
			self._container['dst_idx' ] = []
			self._container['src_idx' ] = []
			for k in idx:
				self._container['dst_idx' ].append(dst_idx[k])
				self._container['src_idx' ].append(src_idx[k])
			if self._container['offsets'] is not None:
				self._container['offsets'] = []
				for k in idx:
					self._container['offsets'].append(offsets[k])

	def save(self,filename, idx, datatype):
		data = self._datafile.copy()
		datanew = {}
		for key in data:
			if data[key] is not None:
				datanew[key] = data[key][self._idx][idx]

		data = build_idx_format(data)
		datanew = build_idx(data, idx, datanew)
		print("Saving data ", datatype, " in ", filename, 'file....',flush=True)
		saveDatah5(datanew, filename)

	@property
	def Z(self):
		return self._container['Z']

	@property
	def container(self):
		return self._container

	def __len__(self): 
		return self._container['Z'].shape[0]

	def __getitem__(self, idx):
		return build_batch(self._container, idx, self._keys, self._n_embeded_atoms)

