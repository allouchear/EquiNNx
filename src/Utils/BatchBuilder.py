import os
import numpy as np
import h5py
from joblib import Parallel, delayed
from Utils.PeriodicTable import PeriodicTable

# Keys that are stored flat (concatenated over molecules) in the raw h5 file.
_FLAT_KEYS = {'dst_idx', 'src_idx', 'offsets'}

# Same key list as Utils/DataContainer.KEYS (kept local so that this module has NO
# dependency on jax/flax/e3x : it runs on plain CPU numpy, the GPU code is untouched).
KEYS = [
	'ID', 'N', 'Z', 'R', 'forces', 'masses', 'energy', 'energybyatom', 'totalcharge',
	'spinmultiplicity', 'efield', 'offsets', 'n_idx', 'dst_idx', 'src_idx',
	'edipole', 'polarizability', 'hyperpolarizability', 'sdipole', 'alfa', 'beta'
]


def build_batch(container, idx, keys, n_embeded_atoms):
	"""Build a padded-free batch dict from a container (dict of arrays indexed by molecule).

	Numpy-only replica of Utils.DataContainer.build_batch (no jax). The model converts
	every batch array with jnp.asarray, so numpy arrays loaded from the h5 file produce
	the exact same values as the jax arrays built in memory."""
	if type(idx) is int or type(idx) is np.int64:
		idx = [idx]
	data = {}
	for key in keys:
		data[key] = []
	data['batch_seg'] = []

	Ntot = 0  # total number of atoms
	for k, i in enumerate(idx):
		N = int(container['N'][i])  # number of atoms
		for key in keys:
			if key == 'efield' and container[key] is not None:
				M = np.tile(container[key][i], (N, 1))
				if len(data['efield']) == 0:
					data['efield'] = M
				else:
					data['efield'] = np.vstack((data['efield'], M))
			elif key == 'cells' and container[key] is not None:
				data[key].extend(container[key][i:i + 1, :, :].tolist())
			elif key in ['Z', 'masses'] and container[key] is not None:
				data[key].extend(container[key][i, :N].tolist())
			elif key in ['R', 'forces'] and container[key] is not None:
				data[key].extend(container[key][i, :N, :].tolist())
			elif key in ['dst_idx', 'src_idx'] and container[key] is not None and len(container[key]) == len(container['Z']):
				data[key].extend(np.reshape(container[key][i] + Ntot, [-1]).tolist())
			elif key in ['totalcharge', 'spinmultiplicity'] and container[key] is not None:
				data[key].extend([container[key][i]] * N)
			elif key == 'offsets' and container[key] is not None and container['dst_idx'] and len(container['dst_idx']) == len(container['Z']):
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
				data[key].extend([1.0] * N)
		key = 'totalcharge'
		if key in keys and container[key] is None:
			data[key].extend([0.0] * N)
		key = 'ID'
		if key in keys and container[key] is None:
			data[key].append(i + 1)
		key = 'Z'
		if key in keys and container[key] is None:
			data[key].append(0)

		if len(container['dst_idx']) != len(container['Z']):
			data['dst_idx'].extend(np.reshape(container['dst_idx'][:N, :N - 1] + Ntot, [-1]).tolist())
			data['src_idx'].extend(np.reshape(container['src_idx'][:N, :N - 1] + Ntot, [-1]).tolist())
			data['offsets'] = []
		# offsets could be added in case they are needed
		if n_embeded_atoms <= 0 or n_embeded_atoms >= N:
			data['batch_seg'].extend([2 * k] * N)
		else:
			data['batch_seg'].extend([2 * k] * n_embeded_atoms)
			data['batch_seg'].extend([2 * k + 1] * (N - n_embeded_atoms))
		# increment totals
		Ntot += N

	listform = ['batch_seg', 'N']
	for key in keys:
		if key not in listform:
			if data[key] is not None and len(data[key]) > 0:
				data[key] = np.asarray(data[key])
			else:
				data[key] = None
	return data


def add_padding(data, maxidx, maxlen, verbose=0):
	"""Pad dst_idx/src_idx (with maxidx+1) and offsets (with zeros) up to maxlen.
	Numpy-only replica of Utils.DataProvider.add_padding."""
	if data['dst_idx'] is not None:
		ldata = len(data['dst_idx'])
	else:
		ldata = 0
	if ldata > 0 and maxlen > ldata:
		pad = np.full(maxlen - ldata, maxidx + 1)
		data['dst_idx'] = np.concatenate([data['dst_idx'], pad])
		data['src_idx'] = np.concatenate([data['src_idx'], pad])
		pad = np.zeros((maxlen - ldata, 3))
		if data['offsets'] is not None:
			data['offsets'] = np.concatenate([data['offsets'], pad])


def add_ghost_atoms(data, maxnatoms, verbose=0):
	"""Pad per-atom arrays with ghost atoms (R=0, Z=Z[0], batch_seg += 1) up to maxnatoms.
	Numpy-only replica of Utils.DataProvider.add_ghost_atoms."""
	if data['R'] is not None:
		ldata = len(data['R'])
	else:
		ldata = 0
	if ldata > 0 and maxnatoms > ldata:
		data['R'] = np.concatenate([data['R'], np.zeros((maxnatoms - ldata, 3))])
		data['Z'] = np.concatenate([data['Z'], np.full(maxnatoms - ldata, data['Z'][0])])
		if data['efield'] is not None:
			data['efield'] = np.concatenate([data['efield'], np.zeros((maxnatoms - ldata, 3))])
		if data['forces'] is not None:
			data['forces'] = np.concatenate([data['forces'], np.zeros((maxnatoms - ldata, 3))])
		data['totalcharge'] = np.concatenate([data['totalcharge'], np.full(maxnatoms - ldata, data['totalcharge'][0])])
		data['spinmultiplicity'] = np.concatenate([data['spinmultiplicity'], np.full(maxnatoms - ldata, data['spinmultiplicity'][1])])
		data['masses'] = np.concatenate([data['masses'], np.full(maxnatoms - ldata, data['masses'][1])])

		data['batch_seg'] = data['batch_seg'] + [1] * (maxnatoms - ldata)


def getNBNE(njobs, n):
	m = n // njobs
	M = [m] * njobs
	nr = n % njobs
	for i in range(nr):
		M[i] += 1
	NB = [0] * njobs
	NE = [0] * njobs
	NE[0] = M[0]
	for i in range(1, njobs):
		NB[i] = NE[i - 1]
		NE[i] = NB[i] + M[i]
	return NB, NE


def compute_splits(nAll, num_train, num_valid, num_test, seed):
	"""Replicate exactly the DataContainer + DataProvider split logic.

	DataContainer selects the first ns = num_train+num_valid (or num_test) molecules of a
	seeded permutation of the full dataset, then DataProvider applies a second seeded
	permutation over the selected container and slices train/valid/test.
	"""
	ns = num_train + num_valid
	if num_test > 0:
		ns = num_test
	if ns > nAll:
		raise ValueError("num_train+num_valid (or num_test) ({}) > number of structures ({})".format(ns, nAll))
	# DataContainer selection
	if seed is not None and seed > -1:
		rng = np.random.RandomState(seed=seed)
		sel = rng.permutation(np.arange(nAll))[0:ns]
	else:
		sel = np.arange(ns)
	# DataProvider split over the selected container
	ndata = ns
	ntrain = num_train if num_train >= 0 else ndata
	nvalid = num_valid
	if ntrain > ndata:
		raise ValueError("num_train ({}) > number of selected structures ({})".format(ntrain, ndata))
	if ntrain + nvalid > ndata:
		raise ValueError("num_train+num_valid ({}) > number of selected structures ({})".format(ntrain + nvalid, ndata))
	ntest = ndata - ntrain - nvalid
	if num_test >= 0 and num_test < ntest:
		ntest = num_test
	if seed is not None and seed > -1:
		rng2 = np.random.RandomState(seed=seed)
		idx = rng2.permutation(np.arange(ndata))
	else:
		idx = np.arange(ndata)
	idx_train = sel[idx[0:ntrain]]
	idx_valid = sel[idx[ntrain:ntrain + nvalid]]
	idx_test = sel[idx[ntrain + nvalid:ntrain + nvalid + ntest]]
	return idx_train, idx_valid, idx_test, (ntrain, nvalid, ntest)


def batch_slices(idx, batch_size):
	batches = []
	n = len(idx)
	nsteps = n // batch_size + (1 if n % batch_size else 0)
	for j in range(nsteps):
		start = j * batch_size
		end = min(start + batch_size, n)
		batches.append(idx[start:end])
	return batches


def make_tasks(idx_train, idx_valid, idx_test, batch_size):
	"""Flatten (split, batch_index, mol_indices) tasks in a global order:
	train batches first, then valid, then test. bin_idx=-1 (flat/un-binned format)."""
	tasks = []
	for split_name, idx in [("train", idx_train), ("valid", idx_valid), ("test", idx_test)]:
		for b_idx, mol_idx in enumerate(batch_slices(idx, batch_size)):
			tasks.append((split_name, -1, b_idx, np.asarray(mol_idx)))
	return tasks


def compute_bin_edges(pairs, k, method='geometric'):
	"""Return the k-1 interior cut points (strictly increasing integers) that split the
	per-molecule pair-count distribution into k bins. k<=1 -> [] (single bin).
	method='geometric' : log-spaced edges over [pmin, pmax] (default : handles skewed
	distributions where the largest bin must be split finely so huge molecules are not
	mixed with the bulk). method='quantile' : equal-population (percentile) edges."""
	if k <= 1:
		return []
	p = np.asarray(pairs, dtype=np.float64)
	if len(p) == 0:
		return list(range(1, k))  # degenerate : no data, still k-1 distinct edges
	if method == 'quantile':
		edges = np.unique(np.maximum(np.ceil(np.percentile(p, np.linspace(0.0, 100.0, k + 1)[1:-1])), 0.0))
	else:  # geometric : log-spaced over the pair-count range
		pmin = float(p.min())
		pmax = float(p.max())
		lo = np.log10(max(pmin, 1.0))
		hi = np.log10(max(pmax, 2.0))
		if hi - lo < 1e-9:
			edges = np.full(k - 1, np.ceil(pmax))
		else:
			edges = np.power(10.0, np.linspace(lo, hi, k + 1)[1:-1])
			edges = np.ceil(edges)
		edges = np.unique(edges)
	out = []
	prev = -1
	for e in edges:
		if int(e) > prev:
			out.append(int(e))
			prev = int(e)
	while len(out) < k - 1:
		out.append((out[-1] if out else -1) + 1)
	return out


def assign_bins(pairs, edges):
	"""Map each molecule (by its pair count) to a bin id in 0..len(edges).
	Bin b covers pairs in [edges[b-1], edges[b]) with edges[-1] treated as +inf."""
	edges_arr = np.asarray(edges)
	if len(edges_arr) == 0:
		return np.zeros(len(pairs), dtype=np.int64)
	return np.searchsorted(edges_arr, np.asarray(pairs), side='right')


def flat_maxlen(pairs, batch_size):
	"""Max over batches (sequential packing of batch_size molecules) of the sum of real
	per-molecule pair counts : the maxlen a flat (global padding) build would produce for
	this batch_size."""
	pairs = np.asarray(pairs)
	n = len(pairs)
	if n == 0:
		return 0
	nsteps = (n + batch_size - 1) // batch_size
	m = 0
	for j in range(nsteps):
		s = int(pairs[j * batch_size:(j + 1) * batch_size].sum())
		if s > m:
			m = s
	return m


def bin_batch_sizes(pairs, bin_ids, batch_size, bins, budget=None):
	"""Per-bin batch size : the largest batch size such that the worst-case number of
	pairs per batch of the bin stays <= budget (bounded padded shape / memory), capped by
	the user --batch_size and floored at 1. budget=None => the flat (global padding)
	maxlen over the given pairs, so that no bin can produce a larger padded graph than the
	flat build (worst-case graphs stay comparable while small bins pack full batches)."""
	pairs = np.asarray(pairs)
	if budget is None:
		budget = flat_maxlen(pairs, batch_size)
	if budget < 1:
		budget = 1
	pmax = np.zeros(bins, dtype=np.int64)
	for b in range(bins):
		m = bin_ids == b
		if m.any():
			pmax[b] = int(pairs[m].max())
	sizes = []
	for b in range(bins):
		if pmax[b] <= 0:
			sizes.append(max(1, int(batch_size)))
		else:
			sizes.append(min(int(batch_size), max(1, int(budget // pmax[b]))))
	return sizes


def make_binned_tasks(idx_splits, bin_ids, batch_sizes):
	"""Flatten (split, bin_idx, batch_index, mol_indices) tasks : for each split and each
	bin, pack the bin molecules with the bin batch_size. Only non-empty (split, bin)
	combinations produce tasks."""
	tasks = []
	for split_name, idx in idx_splits.items():
		idx = np.asarray(idx)
		for b in range(len(batch_sizes)):
			sel = idx[bin_ids[idx] == b]
			if len(sel) == 0:
				continue
			for b_idx, mol_idx in enumerate(batch_slices(sel, batch_sizes[b])):
				tasks.append((split_name, b, b_idx, np.asarray(mol_idx)))
	return tasks


def make_chunks(tasks, num_workers):
	num_workers = max(1, min(num_workers, len(tasks)))
	if len(tasks) == 0:
		return [[]]
	nb, ne = getNBNE(num_workers, len(tasks))
	chunks = []
	for k in range(num_workers):
		chunks.append(tasks[nb[k]:ne[k]])
	return chunks


def _mol_max_range(raw, cum, i0, i1):
	"""Per-molecule max of dst_idx and src_idx over the contiguous molecule range [i0,i1)."""
	with h5py.File(raw, "r") as f:
		dst = f["dst_idx"][cum[i0]:cum[i1]]
		src = f["src_idx"][cum[i0]:cum[i1]]
	nidx = np.diff(cum[i0:i1 + 1])
	mol_ids = np.repeat(np.arange(i1 - i0), nidx)
	md = np.full(i1 - i0, -1, dtype=np.int64)
	ms = np.full(i1 - i0, -1, dtype=np.int64)
	if len(mol_ids) > 0:
		np.maximum.at(md, mol_ids, dst)
		np.maximum.at(ms, mol_ids, src)
	return i0, i1, np.maximum(md, ms)


def compute_mol_max_pairs(raw, nAll, num_workers):
	"""Parallel reduce over molecule ranges -> per-molecule max(dst_idx, src_idx).
	-1 for molecules without any neighbor pair."""
	with h5py.File(raw, "r") as f:
		n_idx = f["n_idx"][()]
	cum = np.concatenate([[0], np.cumsum(n_idx)])
	num_workers = max(1, min(num_workers, nAll))
	nb, ne = getNBNE(num_workers, nAll)
	results = Parallel(n_jobs=num_workers)(
		delayed(_mol_max_range)(raw, cum, nb[i], ne[i]) for i in range(num_workers)
	)
	max_pair = np.full(nAll, -1, dtype=np.int64)
	for i0, i1, mp in results:
		max_pair[i0:i1] = mp
	return max_pair


def compute_padding_params(raw, tasks, num_workers, max_pair=None):
	"""PRE-PASS over the given batches (train+valid+test): compute the padding parameters
	maxidx, maxlen, maxnatoms, identically to DataProvider.padding_idx / padding_positions_atomic_numbers.
	When called per-bin, max_pair is the file-wide per-molecule max idx (computed once)."""
	with h5py.File(raw, "r") as f:
		N = f["N"][()]
		has_idx = ("n_idx" in f) and ("dst_idx" in f) and ("src_idx" in f)
		n_idx = f["n_idx"][()] if has_idx else None
	nAll = len(N)
	if has_idx and max_pair is None:
		max_pair = compute_mol_max_pairs(raw, nAll, num_workers)

	maxnatoms = -1
	maxlen = -1
	maxidx = -1
	for _split_name, _bin_idx, _b_idx, mol_indices in tasks:
		N_b = N[mol_indices]
		n = int(N_b.sum())
		if n > maxnatoms:
			maxnatoms = n
		if has_idx:
			nidx_b = n_idx[mol_indices]
			l = int(nidx_b.sum())
			if l > maxlen:
				maxlen = l
			prefix = np.concatenate([[0], np.cumsum(N_b[:-1])])
			mp = max_pair[mol_indices]
			mask = mp >= 0  # molecules without pairs never appear in dst/src
			if mask.any():
				v = prefix[mask] + mp[mask]
				m = int(v.max())
				if m > maxidx:
					maxidx = m
	return maxidx, maxlen, maxnatoms


def _split_lists(arr, boundaries):
	if len(boundaries) == 0:
		if len(arr) == 0:
			return []
		return [np.asarray(arr)]
	return list(np.split(arr, boundaries))


def read_container(f, mol_indices, N_full, n_idx_full, cum):
	"""Build a container (dict of arrays indexed by molecule) for the given molecules,
	with the same structure as DataContainer._container.

	h5py requires increasing indices for fancy indexing, so reads are done in sorted
	molecule order and the result is reordered back to the batch order."""
	mol_indices = np.asarray(mol_indices)
	n = len(mol_indices)
	order = np.argsort(mol_indices)
	inv = np.argsort(order)
	sorted_idx = mol_indices[order]

	container = {k: None for k in KEYS}
	container["N"] = N_full[mol_indices]
	for key in KEYS:
		if key in f and key not in _FLAT_KEYS:
			vals = np.asarray(f[key][sorted_idx])
			container[key] = vals[inv]

	if container["energy"] is not None:
		container["energybyatom"] = np.asarray(container["energy"], dtype=np.float64) / np.asarray(container["N"], dtype=np.float64)
	if container["totalcharge"] is None:
		container["totalcharge"] = np.zeros(n)
	if container["spinmultiplicity"] is None:
		container["spinmultiplicity"] = np.ones(n)
	if container["masses"] is None and container["Z"] is not None:
		# Replica of DataContainer.__init__ : real atomic masses (PeriodicTable) when the
		# raw dataset does not provide 'masses' (1.0 for the padded entries, as in memory).
		periodicTable = PeriodicTable()
		Nz = np.asarray(container["N"])
		NMax = int(Nz.max()) if n > 0 else 0
		masses = np.ones((n, NMax), dtype=np.float64)
		Zc = np.asarray(container["Z"])
		for j in range(n):
			for ia in range(int(Nz[j])):
				masses[j, ia] = periodicTable.elementZ(int(Zc[j, ia])).isotopes[0].rMass
		container["masses"] = masses

	if n_idx_full is not None and "dst_idx" in f and "src_idx" in f:
		nidx = n_idx_full[mol_indices]
		nidx_sorted = n_idx_full[sorted_idx]
		sel = np.concatenate(
			[np.arange(int(cum[int(i)]), int(cum[int(i)] + n_idx_full[int(i)])) for i in sorted_idx]
		)
		dst = np.asarray(f["dst_idx"][sel])
		src = np.asarray(f["src_idx"][sel])
		boundaries = np.cumsum(nidx_sorted)[:-1]
		dst_list = _split_lists(dst, boundaries)
		src_list = _split_lists(src, boundaries)
		container["dst_idx"] = [dst_list[inv[j]] for j in range(n)]
		container["src_idx"] = [src_list[inv[j]] for j in range(n)]
		container["n_idx"] = nidx
		if "offsets" in f:
			# offsets follow the same flat-over-pairs layout as dst_idx/src_idx :
			# slice by pair ranges (sorted order) and reorder the per-molecule blocks back.
			offs = np.asarray(f["offsets"][sel])
			offs_list = _split_lists(offs, boundaries)
			container["offsets"] = [offs_list[inv[j]] for j in range(n)]
	return container


def write_batch(g, split_name, bin_idx, b_idx, batch, bin_attrs=None):
	grp = g.require_group(split_name)
	if bin_idx is not None and bin_idx >= 0:
		grp = grp.require_group("bin_%d" % bin_idx)
		if bin_attrs is not None and len(grp.attrs) == 0:
			for k, v in bin_attrs.items():
				grp.attrs[k] = v
	bg = grp.create_group("batch_%06d" % b_idx)
	for k, v in batch.items():
		if v is None:
			continue
		bg.create_dataset(k, data=np.asarray(v), compression="lzf")


def build_pad_chunk(raw_h5, chunk, n_embeded_atoms, pad_params, bin_attrs, out_path, verbose=0):
	"""Worker: build each batch of the chunk, pad it immediately with the padding
	parameters of its bin (pad_params[bin_idx] = (maxidx, maxlen, maxnatoms)), and write
	the PADDED batch into out_path (inside the bin group when bin_idx >= 0)."""
	with h5py.File(raw_h5, "r") as f:
		N_full = f["N"][()]
		n_idx_full = f["n_idx"][()] if "n_idx" in f else None
		cum = None
		if n_idx_full is not None:
			cum = np.concatenate([[0], np.cumsum(n_idx_full)])
		with h5py.File(out_path, "w") as g:
			for it, (split_name, bin_idx, b_idx, mol_indices) in enumerate(chunk):
				container = read_container(f, mol_indices, N_full, n_idx_full, cum)
				batch = build_batch(container, list(range(len(mol_indices))), KEYS, n_embeded_atoms)
				maxidx, maxlen, maxnatoms = pad_params[bin_idx]
				add_padding(batch, maxidx, maxlen)
				add_ghost_atoms(batch, maxnatoms)
				write_batch(g, split_name, bin_idx, b_idx, batch, bin_attrs.get(bin_idx))
				if verbose > 0:
					print("    batch {}/{}".format(it + 1, len(chunk)), end="\r", flush=True)
	if verbose > 0:
		print("")


def merge(out_path, tmp_paths, attrs):
	with h5py.File(out_path, "w") as g:
		for k, v in attrs.items():
			g.attrs[k] = v
		for tmp in tmp_paths:
			with h5py.File(tmp, "r") as tf:
				for split_name in tf:
					src = tf[split_name]
					dst = g.require_group(split_name)
					bin_names = sorted(n for n in src if n.startswith("bin_"))
					if bin_names:
						for bname in bin_names:
							bdst = dst.require_group(bname)
							if len(bdst.attrs) == 0:
								for k, v in src[bname].attrs.items():
									bdst.attrs[k] = v
							for name in src[bname]:
								src[bname].copy(name, bdst, name=name)
					else:
						for name in src:
							src.copy(name, dst, name=name)
		for split_name in ["train", "valid", "test"]:
			if split_name not in g:
				continue
			grp = g[split_name]
			bin_names = sorted(n for n in grp if n.startswith("bin_"))
			if bin_names:
				total = 0
				for bname in bin_names:
					bg = grp[bname]
					nb = bg.attrs["n_batches"] if "n_batches" in bg.attrs else len(bg)
					bg.attrs["n_batches"] = nb
					total += nb
				grp.attrs["n_batches"] = total
			else:
				grp.attrs["n_batches"] = len(grp)


def save_split(raw, out_path, mol_indices, verbose=0):
	"""Save the given molecules (raw indices) as a raw-format h5 file : same keys as the
	raw dataset (per-molecule arrays), with dst_idx/src_idx/offsets sliced accordingly.
	The result is loadable by DataContainer/loadDatah5 and can be used as a dataset.

	Ordering : per-molecule arrays and n_idx follow the split order (mol_indices); the flat
	dst_idx/src_idx/offsets are the concatenation of the selected molecules' arrays in the
	same order (so that build_idx_format can slice them positionally from n_idx)."""
	mol_indices = np.asarray(mol_indices, dtype=np.int64)
	n = len(mol_indices)
	if n == 0:
		if verbose > 0:
			print("   empty split : nothing to save")
		return
	order = np.argsort(mol_indices)
	sorted_idx = mol_indices[order]
	inv = np.argsort(order)
	row_lo = int(sorted_idx[0])
	row_hi = int(sorted_idx[-1]) + 1
	with h5py.File(raw, "r") as f:
		n_idx = f["n_idx"][()] if "n_idx" in f else None
		with h5py.File(out_path, "w") as g:
			for key in f.keys():
				if key in _FLAT_KEYS:
					continue
				# contiguous row-block read + numpy gather : much faster than h5py fancy indexing
				vals = np.asarray(f[key][row_lo:row_hi])
				g.create_dataset(key, data=vals[sorted_idx - row_lo][inv], compression="lzf")
			if n_idx is not None and "dst_idx" in f and "src_idx" in f:
				n_idx = np.asarray(n_idx, dtype=np.int64)
				lens_sorted = n_idx[sorted_idx]
				total = int(lens_sorted.sum())
				if total > 0:
					# flat pair indices of the selected molecules, in sorted order (contiguous,
					# chunk-friendly read) and the permutation mapping them to the split order.
					cum = np.concatenate([[0], np.cumsum(n_idx)])
					prefix_sorted = np.concatenate([[0], np.cumsum(lens_sorted)])[:-1]
					sel_sorted = np.repeat(cum[sorted_idx], lens_sorted) + (
						np.arange(total) - np.repeat(prefix_sorted, lens_sorted)
					)
					lens_split = lens_sorted[inv]
					starts_split = np.cumsum(lens_split) - lens_split
					perm = np.repeat(prefix_sorted[inv], lens_split) + (
						np.arange(total) - np.repeat(starts_split, lens_split)
					)
					# raw (file) flat indices, in split order
					raw_sel_split = sel_sorted[perm]
					pair_lo = int(sel_sorted[0])
					pair_hi = int(sel_sorted[-1]) + 1
					for key in ["dst_idx", "src_idx"]:
						block = np.asarray(f[key][pair_lo:pair_hi])
						g.create_dataset(key, data=block[raw_sel_split - pair_lo], compression="lzf")
					if "offsets" in f:
						offs = f["offsets"]
						if len(offs) != len(n_idx):  # flat over pairs -> same vectorized path
							block = np.asarray(offs[pair_lo:pair_hi])
							g.create_dataset("offsets", data=block[raw_sel_split - pair_lo], compression="lzf")
						else:  # per-molecule storage -> flatten (split order)
							parts = [np.asarray(offs[int(i)]) for i in mol_indices]
							try:
								data = np.concatenate(parts) if len(parts) else None
							except Exception:
								data = None
							if data is not None:
								g.create_dataset("offsets", data=data, compression="lzf")
				else:  # no pair at all for this split
					for key in ["dst_idx", "src_idx"]:
						g.create_dataset(key, data=np.asarray(f[key][:0]), compression="lzf")


def keys_present(tmp_paths):
	keys = set()
	for tmp in tmp_paths:
		with h5py.File(tmp, "r") as tf:
			for split_name in tf:
				grp = tf[split_name]
				for name in grp:
					node = grp[name]
					if isinstance(node, h5py.Group):
						for member in node:
							sub = node[member]
							if isinstance(sub, h5py.Group):
								keys.update(sub.keys())  # binned : batch groups inside a bin
							else:
								keys.add(member)         # flat : datasets directly
					else:
						keys.add(name)
	return sorted(keys)


def default_output_name(raw, seed, ntrain, nvalid, ntest, batch_size, n_bins=1):
	stem = os.path.basename(raw)
	if stem.endswith(".h5"):
		stem = stem[:-3]
	return "{}_batched_{}_{}_{}_{}_{}_bins{}.h5".format(stem, seed, ntrain, nvalid, ntest, batch_size, n_bins)