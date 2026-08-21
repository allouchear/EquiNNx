import sys
import os
import collections
import numpy as np
import jax
import jax.numpy as jnp
import h5py
from ase.io import read
from Utils.UtilsFunctions import *
from Utils.DataContainer import KEYS


def add_padding(data, maxidx, maxlen, verbose=0):
	"""Pad dst_idx/src_idx (with maxidx+1) and offsets (with zeros) up to maxlen.
	Extracted from DataProvider.addpading."""
	if data['dst_idx'] is not None:
		ldata=len(data['dst_idx'])
	else:
		ldata=0
	if ldata>0 and maxlen>ldata:
		pad=jnp.asarray([maxidx+1]*(maxlen-ldata))
		data['dst_idx']=jnp.concatenate([data['dst_idx'], pad])
		data['src_idx']=jnp.concatenate([data['src_idx'], pad])
		pad= jnp.asarray([0]*3*(maxlen-ldata)).reshape(-1,3)
		if data['offsets'] is not None:
			data['offsets']=jnp.concatenate([data['offsets'], pad])


def add_ghost_atoms(data, maxnatoms, verbose=0):
	"""Pad per-atom arrays with ghost atoms (R=0, Z=Z[0], batch_seg += 1) up to maxnatoms.
	Extracted from DataProvider.addghostatoms."""
	if data['R'] is not None:
		ldata=len(data['R'])
	else:
		ldata=0
	if ldata>0 and maxnatoms>ldata:
		pad = jnp.asarray([0]*3*(maxnatoms-ldata)).reshape(-1,3)
		data['R'] = jnp.concatenate([data['R'], pad])
		pad = jnp.asarray([data['Z'][0]]*(maxnatoms-ldata)).reshape(-1)
		data['Z'] = jnp.concatenate([data['Z'], pad])
		if data['efield'] is not None:
			pad = jnp.asarray([0]*3*(maxnatoms-ldata)).reshape(-1,3)
			data['efield'] = jnp.concatenate([data['efield'], pad])
		if data['forces'] is not None:
			pad = jnp.asarray([0]*3*(maxnatoms-ldata)).reshape(-1,3)
			data['forces'] = jnp.concatenate([data['forces'], pad])
		pad = jnp.asarray([data['totalcharge'][0]]*(maxnatoms-ldata)).reshape(-1)
		data['totalcharge'] = jnp.concatenate([data['totalcharge'], pad])
		pad = jnp.asarray([data['spinmultiplicity'][1]]*(maxnatoms-ldata)).reshape(-1)
		data['spinmultiplicity'] = jnp.concatenate([data['spinmultiplicity'], pad])
		pad = jnp.asarray([data['masses'][1]]*(maxnatoms-ldata)).reshape(-1)
		data['masses'] = jnp.concatenate([data['masses'], pad])

		data['batch_seg'] = data['batch_seg']+[1]*(maxnatoms-ldata)


def load_batches(path, split, all_keys):
	"""Load pre-built (already padded) batches of a split from a buildDataBatching h5 file.
	Handles both the flat format (batches directly in the split group) and the binned
	format (batches grouped in bin_%d subgroups, read bin by bin)."""
	if not os.path.exists(path):
		raise ValueError("Batching file {} does not exist".format(path))
	with h5py.File(path, "r") as f:
		if split not in f:
			return []
		grp = f[split]
		bin_names = sorted(n for n in grp if n.startswith("bin_"))
		if not bin_names:
			bin_names = [""]
		batches = []
		for bname in bin_names:
			bg = grp[bname] if bname else grp
			n_batches = bg.attrs["n_batches"] if "n_batches" in bg.attrs else len(bg)
			for i in range(n_batches):
				g2 = bg["batch_%06d" % i]
				batch = {}
				for k in all_keys:
					if k in g2:
						batch[k] = np.asarray(g2[k][()])
					else:
						batch[k] = None
				batches.append(batch)
		return batches


class _LazyBatches:
	"""List-like proxy over the pre-built (already padded) batches of one split of a
	batching h5 file. Batches are read on demand from the h5 file and kept in an LRU
	cache of cache_size batches (-1 = keep all batches, i.e. warm-up on first epoch)."""

	def __init__(self, path, split, all_keys, cache_size=4):
		self._path = path
		self._split = split
		self._keys = all_keys
		self._cache_size = cache_size
		self._fh = None
		self._n_batches = -1
		self._cache = collections.OrderedDict()
		self._hits = 0
		self._misses = 0

	def _file(self):
		if self._fh is None:
			self._fh = h5py.File(self._path, "r")
		return self._fh

	def __len__(self):
		if self._n_batches < 0:
			f = self._file()
			if self._split not in f:
				self._n_batches = 0
			else:
				grp = f[self._split]
				self._n_batches = grp.attrs["n_batches"] if "n_batches" in grp.attrs else len(grp)
		return self._n_batches

	def __getitem__(self, i):
		if i in self._cache:
			self._cache.move_to_end(i)
			self._hits += 1
			return self._cache[i]
		self._misses += 1
		f = self._file()
		bg = f[self._split]["batch_%06d" % i]
		batch = {}
		for k in self._keys:
			if k in bg:
				batch[k] = np.asarray(bg[k][()])
			else:
				batch[k] = None
		self._store(i, batch)
		return batch

	def _store(self, i, batch):
		if self._cache_size == 0:
			return
		if self._cache_size < 0 or len(self._cache) < self._cache_size:
			self._cache[i] = batch
			self._cache.move_to_end(i)
			return
		self._cache.popitem(last=False)
		self._cache[i] = batch
		self._cache.move_to_end(i)

	@property
	def stats(self):
		return (self._hits, self._misses)


class _BinnedBatches:
	"""List-like proxy over the batches of one split of a binned batching h5 file.
	Batches are ordered bin by bin (all batches of bin_0, then bin_1, ...) and each bin
	is read lazily through a _LazyBatches with its own LRU cache. A flat (un-binned)
	split is handled as a single implicit bin, so this class works for both formats."""

	def __init__(self, path, split, all_keys, cache_size=4):
		self._path = path
		self._split = split
		self._bins = []
		with h5py.File(path, "r") as f:
			if split not in f:
				names = []
			else:
				grp = f[split]
				names = sorted(n for n in grp if n.startswith("bin_"))
		if not names:
			names = [""]
		for n in names:
			self._bins.append(_LazyBatches(path, split + "/" + n if n else split, all_keys, cache_size))
		self._lens = np.asarray([len(b) for b in self._bins], dtype=np.int64)
		self._offsets = np.concatenate([[0], np.cumsum(self._lens)])

	def __len__(self):
		return int(self._offsets[-1])

	def __getitem__(self, i):
		b = int(np.searchsorted(self._offsets, i, side='right') - 1)
		return self._bins[b][i - self._offsets[b]]

	@property
	def stats(self):
		hits = sum(b.stats[0] for b in self._bins)
		misses = sum(b.stats[1] for b in self._bins)
		return (hits, misses)


class DataProvider:
	def __repr__(self):
		return "DataProvider"

	def __init__(self, data, config, batching_file=None, memory_mode='ram', cache_size=4):
		self._verbose = config.verbose
		if self.verbose>0 :
			print("Preparing datasets.......",flush=True)
		self._seed = config.seed
		self._data = data
		self._n_embeded_atoms = config.n_embeded_atoms
		self._data_keys = None
		self._memory_mode = memory_mode
		self._cache_size = cache_size

		#for retrieving batches
		self._idx_in_epoch = 0 
		self._valid_idx = 0
		self._test_idx = 0

		self._idx_in_epoch_old = 0 

		self._i_train_data = 0 
		self._i_valid_data = 0 
		self._i_test_data = 0 

		if batching_file is not None:
			# Pre-built batches : the split/batching/padding parameters are read from the
			# batching file (built by buildDataBatching.py) and override the train.inp values.
			# The raw dataset is NOT read.
			self._load_batching(batching_file, config)
		else:
			ntrain= config.num_train
			if ntrain<0:
				ntrain = len(data.Z)
			nvalid=config.num_valid
			ntest=config.num_test
			batch_size=config.batch_size
			valid_batch_size=config.batch_size
			seed=config.seed
			self._seed = seed
			self._ndata  = len(data.Z)
			self._ntrain = ntrain
			if self.ntrain> self.ndata:
				print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
				print("Error : ntrain = ", self.ntrain," > ndata =",self.ndata)
				print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
				sys.exit(1)
			self._nvalid = nvalid
			if self.ntrain+self.nvalid> self.ndata:
				print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
				print("Error : ntrain+nvalid = ", self.ntrain+self.nvalid," > ndata =",self.ndata)
				print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
				sys.exit(1)
			self._ntest  = len(data.Z)-self.ntrain-self.nvalid
			if ntest>=0 and ntest<self._ntest:
				self._ntest  = ntest

			self._batch_size = batch_size
			self._valid_batch_size = valid_batch_size
			self._test_batch_size  = valid_batch_size
			if batch_size>ntrain :
				self._batch_size = ntrain
			if valid_batch_size>nvalid and nvalid>0:
				self._valid_batch_size = nvalid
			if self.test_batch_size>self.ntest :
				self._test_batch_size = self.ntest

			if seed is not None and seed>-1:
				#random state parameter, such that random operations are reproducible if wanted
				self._random_state = np.random.RandomState(seed=seed)

				#create shuffled list of indices
				idx = self._random_state.permutation(np.arange(len(self.data.Z)))
			else:
				idx = np.arange(len(self.data))

			#store indices of training, validation and test data
			self._idx_train = idx[0:self.ntrain]
			self._idx_valid = idx[self.ntrain:self.ntrain+self.nvalid]
			self._idx_test  = idx[self.ntrain+self.nvalid:self.ntrain+self.nvalid+self.ntest]
			idxall  = idx[0:self.ntrain+self.nvalid+self.ntest]

			# number_of_batches = len([_ for _ in iter(self._dataset_train)])
			#self._data.putInHostMemory(idxall)
			if self.verbose>0 :
				print("Building train data .......",flush=True)
			self._build_train_data()
			if self.verbose>0 :
				print("Building validation data .......",flush=True)
			self._build_valid_data()
			if self.verbose>0 :
				print("Building test data .......",flush=True)
			self._build_test_data()
			if self.verbose>0 :
				print("Setting of padding to avoid jax.jit recompilation .......",flush=True)
			self.set_padding() # to avoid recompilation jig.jax

	def _load_batching(self, batching_file, config):
		if not os.path.exists(batching_file):
			raise ValueError("Batching file {} does not exist".format(batching_file))
		if self.verbose>0:
			print("Loading pre-built batches from ", batching_file, "file .......",flush=True)
		with h5py.File(batching_file, "r") as f:
			num_train   = int(f.attrs["num_train"])   if "num_train"   in f.attrs else -1
			num_valid   = int(f.attrs["num_valid"])   if "num_valid"   in f.attrs else -1
			num_test    = int(f.attrs["num_test"])    if "num_test"    in f.attrs else -1
			batch_size  = int(f.attrs["batch_size"])  if "batch_size"  in f.attrs else -1
			n_embeded_atoms = int(f.attrs["n_embeded_atoms"]) if "n_embeded_atoms" in f.attrs else -1
			seed        = int(f.attrs["seed"])        if "seed"        in f.attrs else -1
			keys_str    = f.attrs["keys"] if "keys" in f.attrs else ",".join(KEYS)
			num_bins    = int(f.attrs["num_bins"])    if "num_bins"    in f.attrs else 1
			bin_edges   = np.asarray(f.attrs["bin_edges"], dtype=np.int64) if "bin_edges" in f.attrs else np.zeros(0, dtype=np.int64)
			bin_batch_sizes = np.asarray(f.attrs["bin_batch_sizes"], dtype=np.int64) if "bin_batch_sizes" in f.attrs else np.zeros(1, dtype=np.int64)
		# The batching file is the source of truth : its split/batching parameters override
		# the train.inp values so that the training uses exactly the parameters used when the
		# batching file was created by buildDataBatching.py.
		self._ntrain = num_train
		self._nvalid = num_valid
		self._ntest  = num_test
		self._ndata  = num_train + num_valid + num_test
		self._batch_size = batch_size
		self._valid_batch_size = batch_size
		self._test_batch_size  = batch_size
		self._n_embeded_atoms = n_embeded_atoms
		self._seed = seed
		self._num_bins = num_bins
		self._bin_edges = bin_edges
		self._bin_batch_sizes = bin_batch_sizes
		self._data_keys = [k for k in keys_str.split(",") if k != ""]
		config.num_train = num_train
		config.num_valid = num_valid
		config.num_test  = num_test
		config.batch_size = batch_size
		config.n_embeded_atoms = n_embeded_atoms
		config.seed = seed
		if self.verbose>0:
			print("Batching file parameters : num_train =", num_train,
				  " num_valid =", num_valid, " num_test =", num_test,
				  " batch_size =", batch_size, " n_embeded_atoms =", n_embeded_atoms,
				  " seed =", seed, flush=True)
			print("Bins : num_bins =", num_bins, " pairs edges =", list(bin_edges),
				  " per-bin batch sizes =", list(bin_batch_sizes), flush=True)

		all_keys = KEYS + ["batch_seg"]
		if self._memory_mode == 'ram':
			if self.verbose>0:
				print("Loading all batches in memory (memory_mode=ram) .......",flush=True)
			self._data_train = load_batches(batching_file, "train", all_keys)
			self._data_valid = load_batches(batching_file, "valid", all_keys)
			self._data_test  = load_batches(batching_file, "test",  all_keys)
		else:
			if self.verbose>0:
				print("Loading batches on demand from the batching file (memory_mode=disk, cache_size={}) .......".format(self._cache_size),flush=True)
			self._data_train = _BinnedBatches(batching_file, "train", all_keys, self._cache_size)
			self._data_valid = _BinnedBatches(batching_file, "valid", all_keys, self._cache_size)
			self._data_test  = _BinnedBatches(batching_file, "test",  all_keys, self._cache_size)
		n_train = len(self._data_train)
		n_valid = len(self._data_valid)
		n_test  = len(self._data_test)
		if n_train != self.get_nsteps_batch():
			raise ValueError("Number of train batches in batching file ({}) != expected ({})".format(n_train, self.get_nsteps_batch()))
		if self.nvalid>0 and n_valid != self.get_nsteps_valid_batch():
			raise ValueError("Number of valid batches in batching file ({}) != expected ({})".format(n_valid, self.get_nsteps_valid_batch()))
		if self.ntest>0 and n_test != self.get_nsteps_test_batch():
			raise ValueError("Number of test batches in batching file ({}) != expected ({})".format(n_test, self.get_nsteps_test_batch()))

	def set_padding(self):
		if self.verbose>0 :
			print("Setting of padding for idx .......",flush=True)
		self.padding_idx()
		if self.verbose>0 :
			print("Setting of padding for positions & atomic numbers .......",flush=True)
		self.padding_positions_atomic_numbers()

	def save_data(self, filename, idx, datatype):
		self._data.save(filename, idx, datatype)

	@property
	def verbose(self):
		return self._verbose

	@property
	def data(self):
		return self._data

	@property
	def data_keys(self):
		return self._data_keys

	@property
	def train_data(self):
		return self.get_all_train_data()

	@property
	def ndata(self):
		return self._ndata

	@property
	def ntrain(self):
		return self._ntrain
	
	@property
	def nvalid(self):
		return self._nvalid
	
	@property
	def ntest(self):
		return self._ntest

	@property
	def num_bins(self):
		return self._num_bins

	@property
	def random_state(self):
		return self._random_state

	@property
	def idx_train(self):
		return self._idx_train

	@property
	def idx_valid(self):
		return self._idx_valid   

	@property
	def idx_test(self):
		return self._idx_test

	@property
	def idx_in_epoch(self):
		return self._idx_in_epoch

	@property
	def idx_in_epoch_old(self):
		return self._idx_in_epoch_old


	@property
	def valid_idx(self):
		return self._valid_idx

	@property
	def test_idx(self):
		return self._test_idx


	@property
	def batch_size(self):
		return self._batch_size

	@property
	def valid_batch_size(self):
		return self._valid_batch_size

	@property
	def test_batch_size(self):
		return self._test_batch_size

	def addpading(self, dataall, maxidx, maxlen):
		n = len(dataall)
		i = 1
		for data in dataall:
			if self._verbose>0:
				print(i,'/',n, end='\r',flush=True)
			i += 1
			add_padding(data, maxidx, maxlen)

	def getmaxlenidx(self,dataall,maxlen):
		n=len(dataall)
		i = 1
		for data in dataall:
			if self._verbose>0:
				print(i,'/',n, end='\r',flush=True)
			i += 1
			if data['dst_idx'] is not None:
				maxlen=max(len(data['dst_idx']), maxlen)
		return maxlen

	def getmaxidx(self,dataall,maxidx):
		n = len(dataall)
		i = 1
		for data in dataall:
			if self._verbose>0:
				print(i,'/',n, end='\r',flush=True)
			i += 1
			if data['dst_idx'] is not None and len(data['dst_idx'])>0:
				maxidx=max(jnp.max(data['dst_idx']), maxidx)
				maxidx=max(jnp.max(data['src_idx']), maxidx)
		return maxidx


	def padding_idx(self):
		maxidx=-1
		if self._verbose>0:
			print("Get max idx value of train set .......",flush=True)
		maxidx=self.getmaxidx(self._data_train,maxidx)
		if self._verbose>0:
			print("Get max idx value of valid set .......",flush=True)
		maxidx=self.getmaxidx(self._data_valid,maxidx)
		if self._verbose>0:
			print("Get max idx value of test set .......",flush=True)
		maxidx=self.getmaxidx(self._data_test,maxidx)
		maxlen=-1
		if self._verbose>0:
			print("Get max length of dst_idx and src_idx  for train set.......",flush=True)
		maxlen=self.getmaxlenidx(self._data_train,maxlen)
		if self._verbose>0:
			print("Get max length of dst_idx and src_idx  for valid set.......",flush=True)
		maxlen=self.getmaxlenidx(self._data_valid,maxlen)
		if self._verbose>0:
			print("Get max length of dst_idx and src_idx  for test set.......",flush=True)
		maxlen=self.getmaxlenidx(self._data_test,maxlen)
		if self._verbose>0:
			print("Setting of idx padding for train set .......",flush=True)
		self.addpading(self._data_train, maxidx, maxlen)
		if self._verbose>0:
			print("Setting of idx padding for valid set .......",flush=True)
		self.addpading(self._data_valid, maxidx, maxlen)
		if self._verbose>0:
			print("Setting of idx padding for test set .......",flush=True)
		self.addpading(self._data_test, maxidx, maxlen)

	def addghostatoms(self, dataall, maxnatoms):
		n = len(dataall)
		i = 1
		for data in dataall:
			if self._verbose>0:
				print(i,'/',n, end='\r',flush=True)
			i += 1
			add_ghost_atoms(data, maxnatoms)

	def getmaxnatoms(self,dataall,maxnatoms):
		n = len(dataall)
		i = 1
		for data in dataall:
			if self._verbose>0:
				print(i,'/',n, end='\r',flush=True)
			i += 1
			if data['R'] is not None and len(data['R'])>0:
				maxnatoms=max(len(data['R']), maxnatoms)
		return maxnatoms

	def padding_positions_atomic_numbers(self):
		maxnatoms=-1
		if self._verbose>0:
			print("Get max number of atoms for train set.......",flush=True)
		maxnatoms=self.getmaxnatoms(self._data_train,maxnatoms)
		if self._verbose>0:
			print("Get max number of atoms for valid set.......",flush=True)
		maxnatoms=self.getmaxnatoms(self._data_valid,maxnatoms)
		if self._verbose>0:
			print("Get max number of atoms for test set.......",flush=True)
		maxnatoms=self.getmaxnatoms(self._data_test,maxnatoms)
		if self._verbose>0:
			print("Add ghost atoms to molecules of train set.......",flush=True)
		self.addghostatoms(self._data_train,maxnatoms)
		if self._verbose>0:
			print("Add ghost atoms to molecules of valid set.......",flush=True)
		if self._verbose>1:
			print("maxnatoms=", maxnatoms, flush=True)

		self.addghostatoms(self._data_valid,maxnatoms)
		if self._verbose>0:
			print("Add ghost atoms to molecules of test set.......",flush=True)
		self.addghostatoms(self._data_test,maxnatoms)

	#shuffle the training data
	def shuffle(self):
		newidx = self._random_state.permutation(np.arange(len(self._data_train)))
		self._data_train = self._data_train [newidx]

	def _build_train_data(self):
		self._data_train = []
		self._data_train.append(self.next_train_batch_byidx())

		nbatch = self.ntrain//self.batch_size
		if self.ntrain%self.batch_size>0:
			nbatch += 1
		i=1
		while  self.idx_in_epoch < self.ntrain:
			if self._verbose>0:
				print(i,'/',nbatch, end='\r',flush=True)
			self._data_train.append(self.next_train_batch_byidx())
			i += 1

	def next_train_batch(self):
		i = self._i_train_data
		self._i_train_data += 1 
		if self._i_train_data > len(self._data_train)-1:
			#self.shuffle()
			self._i_train_data = 0 
		return self._data_train[i]

	def reset_train_batch(self):
		self._i_train_data = 0 

	def reset_valid_batch(self):
		self._i_valid_data = 0 

	def reset_test_batch(self):
		self._i_test_data = 0 

	def _build_valid_data(self):
		self._data_valid = []
		self._data_valid.append(self.next_valid_batch_byidx())

		nbatch = self.nvalid//self.valid_batch_size
		if self.nvalid%self.valid_batch_size>0:
			nbatch += 1
		i=1
		while  self.valid_idx < self.nvalid:
			if self._verbose>0:
				print(i,'/',nbatch, end='\r',flush=True)
			self._data_valid.append(self.next_valid_batch_byidx())
			i += 1

	def next_valid_batch(self):
		i = self._i_valid_data
		self._i_valid_data += 1 
		if self._i_valid_data > len(self._data_valid)-1:
			self._i_valid_data = 0 
		return self._data_valid[i]

	def _build_test_data(self):
		self._data_test = []
		self._data_test.append(self.next_test_batch_byidx())
		while  self.test_idx < self.ntest:
			self._data_test.append(self.next_test_batch_byidx())

	def next_test_batch(self):
		i = self._i_test_data
		self._i_test_data += 1 
		if self._i_test_data > len(self._data_test):
			self._i_test_data = 0 
		return self._data_test[i]
		

	#returns a batch of samples from the training set
	def next_train_batch_byidx(self):
		start = self.idx_in_epoch
		self._idx_in_epoch += self.batch_size
		if start >= self.ntrain:
			start = 0
			self._idx_in_epoch = self.batch_size
		# case where ntrain is not a multiple of batch_size
		if self.idx_in_epoch > self.ntrain:
			self._idx_in_epoch = self.ntrain

		end = self.idx_in_epoch   
		self._idx_in_epoch_old = start
		return self.data[self.idx_train[start:end]]

	#returns a batch of samples from the validation set
	def next_valid_batch_byidx(self):
		start = self.valid_idx
		self._valid_idx += self.valid_batch_size
		#finished one pass-through, reset index
		if start >= self.nvalid:
			start = 0
			self._valid_idx = self.valid_batch_size

		# case where nvalid is not a multiple of valid_batch_size
		if self.valid_idx > self.nvalid:
			self._valid_idx = self.nvalid

		end =  self.valid_idx
		return self.data[self.idx_valid[start:end]]

	#returns a batch of samples from the test set
	def next_test_batch_byidx(self):
		start = self.test_idx
		self._test_idx += self.test_batch_size
		#finished one pass-through, reset index
		if start >= self.ntest:
			start = 0
			self._test_idx = self.test_batch_size

		# case where ntest is not a multiple of test_batch_size
		if self.test_idx > self.ntest:
			self._test_idx = self.ntest

		end =  self.test_idx
		return self.data[self.idx_test[start:end]]

	#returns the current batch of samples from the training set
	def current_batch(self):
		if self._data is None:
			return self._data_train[self._i_train_data]
		start = self.idx_in_epoch_old
		end = self.idx_in_epoch   
		return self.data[self.idx_train[start:end]]

	#returns the current batch of samples from the training set
	def first(self):
		if self._data is None:
			return self._data_train[self._i_train_data]
		start = self.idx_in_epoch_old
		end = start+1
		return self.data[self.idx_train[start:end]]

	def set_train_idx_to_end(self):
		self._idx_in_epoch=self.ntrain


	def reset_train_batch_byidx(self):
		start = 0
		self._idx_in_epoch = self.batch_size

	def reset_valid_batch_byidx(self):
		start = 0
		self._valid_idx = self.valid_batch_size

	def rest_test_batch_byidx(self):
		start = 0
		self._test_idx = self.test_batch_size

	def get_nsteps_batch(self):
		if self._data is None:
			return len(self._data_train)
		nsteps = 0
		if self.batch_size != 0:
			nsteps=self.ntrain//self.batch_size
			if self.ntrain%self.batch_size != 0:
				nsteps += 1
		return nsteps

	def get_nsteps_valid_batch(self):
		if self._data is None:
			return len(self._data_valid)
		nsteps = 0
		if self.valid_batch_size != 0:
			nsteps=self.nvalid//self.valid_batch_size
			if self.nvalid%self.valid_batch_size != 0:
				nsteps += 1
		return nsteps

	def get_nsteps_test_batch(self):
		if self._data is None:
			return len(self._data_test)
		nsteps = 0
		if self.test_batch_size !=0:
			nsteps=self.ntest//self.test_batch_size
			if self.ntest%self.test_batch_size !=0:
				nsteps += 1
		return nsteps

	def get_data(self, idx):
		return self.data[idx]

	def get_train_data(self, i):
		idx = self.idx_train[i]
		return self.data[idx]

	def get_all_train_data(self):
		return self.data[self.idx_train]

	def get_valid_data(self, i):
		idx = self.idx_valid[i]
		return self.data[idx]

	def get_all_valid_data(self):
		return self.data[self.idx_valid]

	def get_test_data(self, i):
		idx = self.idx_test[i]
		return self.data[idx]
	
	def get_all_test_data(self):
		return self.data[self.idx_test]
