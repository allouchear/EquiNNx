from ase.io import read
import numpy as np
import sys
from joblib import Parallel, delayed
from joblib import cpu_count
import warnings
from ase.neighborlist import neighbor_list
import re
import h5py
import pandas as pd
import os
from ase import Atoms
from ase.io import write,read
from ase.data import chemical_symbols
import argparse



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

def getNBNE(njobs, nmols):
	m =  nmols//njobs
	M =[m]*njobs
	nr=nmols%njobs
	for i in range(nr):
		M[i] += 1
	NB =[0]*njobs
	NE =[0]*njobs
	NE[0] = M[0]
	for i in range(1,njobs):
		NB[i] = NE[i-1]
		NE[i] = NB[i]+M[i]
	'''
	print("NB=",NB,flush=True)
	print("NE=",NE,flush=True)
	'''
	return NB, NE

def set_idx(mols, cutoff, iBegin, iEnd, iWorker):
	a_dst_idx =[]
	a_src_idx =[]
	a_offsets =[]
	n=(iEnd-iBegin)
	ns=(iEnd-iBegin)//100
	ns = 1 if ns<1 else ns
	for im in range(iBegin, iEnd):
		if im%ns==0:
			print("\t {:<5d} / {:>5d}   Wroker #{:<5d}.......".format(im-iBegin+1,n,iWorker),flush=True)

		atoms=mols[im]
		dst_idx, src_idx, offsets = get_idx_list(atoms, cutoff=cutoff)
		a_dst_idx.append(dst_idx[:])
		a_src_idx.append(src_idx[:])
		a_offsets.append(offsets)

	return  a_dst_idx, a_src_idx, a_offsets
	

def build_idx_list(mols, nmols, njobs=1, cutoff=None):
	all_dst_idx =[]
	all_src_idx =[]
	all_offsets =[]
	print("Setting of neighbor lists.......",flush=True);
	print("Number of structures =",nmols)
	if njobs>nmols:
		njobs= nmols
	print("Number of workers =",njobs)
	if njobs==1:
		dst_idx, src_idx, offsets = set_idx(mols, cutoff, 0,nmols,1)
		all_dst_idx.extend(dst_idx)
		all_src_idx.extend(src_idx)
		all_offsets.extend(offsets)
	else:
		NB,NE = getNBNE(njobs, nmols)
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore")
			r = Parallel(n_jobs=njobs, verbose=0)(
				delayed(set_idx)(mols,cutoff, NB[i],NE[i],i+1) for i in range(njobs)
				) 
		for a in r:
			dst_idx, src_idx, offsets = a
			all_dst_idx.extend(dst_idx)
			all_src_idx.extend(src_idx)
			all_offsets.extend(offsets)
	print("End setting of neighbor lists",flush=True);
	if np.all(np.array(all_offsets) == None):
			all_offsets = None
	return all_dst_idx, all_src_idx, all_offsets


def symMatrix(a):
	return (a+a.T)/2
def sym3DArray(b):
	for i in range(3):
		b[i] = (b[i]+b[i].T)/2
	return b
def symmetrize(t):
	if len(t.shape)==1:
		return t
	elif len(t.shape)==2:
		return symMatrix(t)
	elif len(t.shape)==3:
		return sym3DArray(t)
	else:
		print(" unknown symmetrization ",file=sys.stderr)
		exit(1)

def getData(mols):
	positions=[]
	edipole=[]
	polar=[]
	alfa=[]
	beta=[]
	hpolar=[]
	Z=[]
	pbc=[]
	cells=[]
	dst_idx=[]
	src_idx=[]
	offsets=[]
	ID=[]
	n_idx=[] # number of idx by molecule
	energy=[]

	for atoms in mols:
		Z.append(atoms.get_atomic_numbers())
		pbc.append(atoms.get_pbc())
		cells.append(atoms.get_cell()[:])
		positions.append(np.asarray(atoms.get_positions()))
		if 'energy' in atoms.info.keys():
			energy.append(atoms.info['energy'])
		if 'ID' in atoms.info.keys():
			ID.append(atoms.info['ID'])
		if 'edipole' in atoms.info.keys():
			edipole.append((atoms.info['edipole']))
		if 'polarizability' in atoms.info.keys():
			#polar.append(symmetrize(atoms.info['polarizability']))
			polar.append((atoms.info['polarizability']))
		if 'hyperpolarizability' in atoms.info.keys():
			#hpolar.append(symmetrize(atoms.info['hyperpolarizability']))
			hpolar.append((atoms.info['hyperpolarizability']))
		if 'dst_idx' in atoms.info.keys():
			dst_idx.extend(np.asarray(atoms.info['dst_idx']))
			n_idx.append(len(atoms.info['dst_idx']))
		if 'src_idx' in atoms.info.keys():
			src_idx.extend(np.asarray(atoms.info['src_idx']))
		if 'offsets' in atoms.info.keys():
			if atoms.info['offsets'] is None or atoms.info['offsets'] == 'NONE':
				offsets.extend([None])
			else:
				offsets.extend(np.asarray(atoms.info['offsets']))
		if 'alfa' in atoms.info.keys():
			alfa.append(atoms.info['alfa'])
		if 'beta' in atoms.info.keys():
			beta.append(atoms.info['beta'])

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

	if len(energy)>=1:
		data['energy'] = np.asarray(energy)
	else:
		data['energy'] = None

	if len(alfa)>=1:
		data['alfa'] = np.asarray(alfa)
	else:
		data['alfa'] = None
	if len(beta)>=1:
		data['beta'] = np.asarray(beta)
	else:
		data['beta'] = None
	if len(edipole)>=1:
		edipole=np.asarray(edipole)
		data['edipole'] = edipole
		data['sdipole'] = np.linalg.norm(edipole, ord=None, axis=1, keepdims=False)
	else:
		data['sdipole'] = None

	if len(edipole)>=1:
		edipole=np.asarray(edipole)
		data['edipole'] = edipole
	else:
		data['edipole'] = None
	if len(polar)>=1:
		polar=np.asarray(polar)
		data['polarizability'] = polar
	else:
		data['polarizability'] = None
	if len(hpolar)>=1:
		hpolar=np.asarray(hpolar)
		data['hyperpolarizability'] = hpolar
	else:
		data['hperpolarizability'] = None
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


def loadDatah5(filename):
	with h5py.File(filename, "r") as h5file:
		data = {key: h5file[key][()] for key in h5file.keys()}
	return data



def get_JSON_format(tensor):
	st = np.array2string(tensor, threshold=np.inf, separator=', ')
	st=re.sub(r"\s+", " ", st.replace('\n',' '))
	return "_JSON "+st

def build_mols(filename):
	mols=read(filename, format='extxyz',index=':') 
	for i in range(len(mols)):
		mols[i].info['ID']=i+1
		mols[i].set_pbc([False,False,False])
		mols[i].info['energy'] = mols[i].info['potential']
		mols[i].info['edipole'] = mols[i].info['mu']
		mols[i].info['sdipole'] = np.linalg.norm(mols[i].info['mu'])
		mols[i].info['polarizability'] = mols[i].info['alpha'].reshape(-1,3)
		mols[i].info['alfa'] = (mols[i].info['polarizability'][0,0]+mols[i].info['polarizability'][1,1]+mols[i].info['polarizability'][1,1])/3
		if 'beta' in mols[i].info.keys() and len(mols[i].info['beta'])>0:
			b=mols[i].info['beta'].reshape(3,-1)
			bb=mols[i].info['beta'].reshape(3,3,3)
			mols[i].info['hyperpolarizability'] = mols[i].info['beta'].reshape(3,3,3)
			# https://doi-org.docelec.univ-lyon1.fr/10.1063/5.0010231
			hyperpolarizability = mols[i].info['hyperpolarizability']
			beta=np.zeros(3)
			for k in range(3):
				for j in range(3):
					beta[k] += hyperpolarizability[k,j,j]+hyperpolarizability[j,k,j]+hyperpolarizability[j,j,k]
				beta[k]/=5
			betatot=np.linalg.norm(beta)
			mols[i].info['beta']=betatot

	return mols

def mae_dipoles(mols):
	dipoles=[]
	for atoms in mols:
		dipoles.append(atoms.info['edipole'])
	dipoles= np.asarray(dipoles)
	mdip = np.mean(np.abs(dipoles),axis=0)
	print("dipoles mean ", mdip)


def remove_index(mols):
	nmols=len(mols)

	for im in range(nmols):
		atoms=mols[im]
		if 'dst_idx' in atoms.info.keys():
			atoms.info['dst_idx'] = 'NONE'
			atoms.info['src_idx'] = 'NONE'
			atoms.info['offsets'] = 'NONE'
	return mols


def build_index(mols, cutoff=5):
	nmols=len(mols)
	njobs = cpu_count()
	all_dst_idx, all_src_idx, all_offsets =  build_idx_list(mols, nmols, njobs=njobs, cutoff=cutoff)
	print("End build index")

	for im in range(nmols):
		atoms=mols[im]
		atoms.info['dst_idx'] = (np.array(all_dst_idx[im]))
		atoms.info['src_idx'] = (np.array(all_src_idx[im]))
		if all_offsets is not None and len(all_offsets)>1:
			#print(all_offsets[im])
			#atoms.info['offsets'] = get_JSON_format(all_offsets[im])
			atoms.info['offsets'] = (all_offsets[im])
		else:
			atoms.info['offsets'] = 'NONE'
	return mols


def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument('--build_index', type=int, default=1, help="1=> true, 0=> false")
	parser.add_argument('--save_xyz_with_index', type=int, default=1, help="1=> true, 0=> false")
	parser.add_argument('--save_xyz_without_index', type=int, default=1, help="1=> true, 0=> false")
	parser.add_argument('--cutoff', type=float, default=-1, help="cutoff to build index. if <0 : not cutoff for molecule, 5 Ang for bulk. Default=-1")
	parser.add_argument('--num_structures', default=-1, type=int, help="Number of values to take in the data file. Default = -1=> all structures")
	parser.add_argument("--input_file_name", default='coordinates.xyz', type=str, help="Name of the input file in xyz format. Default=coordinates.xyz")
	parser.add_argument("--prefix", default=None, type=str, help="the prefic name for the output files. Default=prefix of input_file_name")

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

cutoff_idx=None
if args.cutoff >0:
	cutoff_idx = args.cutoff
mols=build_mols(args.input_file_name)
fn = args.prefix
if args.prefix is None:
	fn = args.input_file_name.split('.')[0]

if args.cutoff<0 and any(mols[0].get_pbc()):
	cutoff_idx = 5.0
data_name=fn+"_n_structures_"+str(len(mols))+'_cutidx_'+str(cutoff_idx)
data_name = data_name.replace(".", "p") 
if args.build_index:
	data_h5file=data_name+".h5"
	print("Building idx....", flush=True)
	mols = build_index(mols, cutoff=cutoff_idx)
	print("Building data....",flush=True)
	data=getData(mols)
	print("Save data....",flush=True)
	saveDatah5(data, data_h5file)
	print("See ", data_h5file, "file",flush=True)
	print("Load data....",flush=True)
	data2= loadDatah5(data_h5file)
	#print('ID=',data2['ID'],flush=True)

if args.save_xyz_with_index:
	data_xyzfile=data_name+"_with_index.xyz"
	# Save all molecules into one extended XYZ file
	print("Save data in a xyz file with index ...",flush=True)
	write(data_xyzfile, mols, format='extxyz')
	print("See ", data_xyzfile, "file",flush=True)
	#print("Test read molecules from ", data_xyzfile, "file",flush=True)
	#mols=read(data_xyzfile, format='extxyz', index=":") 
	#print("Mol # 0",flush=True)
	#print(mols[0])

if args.save_xyz_without_index:
	remove_index(mols)
	print("Save data in a xyz file without idx ...",flush=True)
	data_xyzshort=data_name+"_without_index.xyz"
	write(data_xyzshort, mols, format='extxyz')
	print("See ", data_xyzshort , "file",flush=True)
	#print("Test read molecules from ", data_xyzshort, "file",flush=True)
	#mols=read(data_xyzshort, format='extxyz', index=":") 
	#print("Mol # 0",flush=True)
	#print(mols[0])

print("Computing of mae for dipoles ...",flush=True)
mae_dipoles(mols)
print("That's all ",flush=True)

