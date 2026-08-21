import argparse
import os
import shutil
import sys
import tempfile

import h5py
import numpy as np
from joblib import Parallel, delayed
from joblib import cpu_count

from Utils.BatchBuilder import (
	compute_splits,
	make_tasks,
	make_binned_tasks,
	make_chunks,
	compute_padding_params,
	compute_mol_max_pairs,
	compute_bin_edges,
	assign_bins,
	bin_batch_sizes,
	flat_maxlen,
	build_pad_chunk,
	merge,
	keys_present,
	save_split,
	default_output_name,
)


def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument('--dataset', type=str, help="Raw data file (h5) built by buildData.py")
	parser.add_argument('--seed', default=-1, type=int, help="Seed for the train/valid/test split. Default=-1 => no shuffle.")
	parser.add_argument('--num_train', type=int, default=1, help="Number of training samples")
	parser.add_argument('--num_valid', type=int, default=1, help="Number of validation samples")
	parser.add_argument('--num_test', type=int, default=-1, help="Number of test samples. Default : nAll-num_valid-num_train")
	parser.add_argument('--batch_size', type=int, default=10, help="Batch size. With --bins>1 it is the batch size of the smallest (first) bin : the batch size of each larger bin is reduced so that the worst-case number of pairs per batch stays within --batch_size_budget (Default=10)")
	parser.add_argument('--bins', type=int, default=1, help="Number of size bins (per-molecule pair counts). Each bin is padded to its own maxima -> one fixed jax.jit shape per bin. --bins=1 => single bin (old flat format, one global shape). Default=1")
	parser.add_argument('--bin_edges', type=str, default='geometric', choices=['geometric', 'quantile'], help="Bin boundaries (only used when --bins>1) : geometric => log-spaced over the pair-count range (default, splits the skewed tail so huge molecules are not mixed with the bulk) ; quantile => equal-population percentiles")
	parser.add_argument('--batch_size_budget', type=int, default=0, help="Target maximum number of pairs per batch (bounds the padded maxlen / compilation shape of every bin, only used when --bins>1). 0 => automatic : the maxlen of a flat (global padding) build, so bins never produce larger graphs than the flat build. Default=0")
	parser.add_argument('--n_embeded_atoms', default=-1, type=int, help="Number of embedded atoms. Default -1=> we take all atoms")
	parser.add_argument('--num_workers', default=-1, type=int, help="Number of CPU workers. Default=-1 => cpu_count()")
	parser.add_argument('--output', default=None, type=str, help="Name of the output file in h5 format. Default : derived from the dataset name")
	parser.add_argument('--verbose', default=1, type=int, help="Verbose. 0=> minimum of output  Default=1")
	parser.add_argument('--keep_temp', default=0, type=int, help="1=> keep temporary worker files for debug, 0=> remove (Default=0)")
	parser.add_argument('--save_data', default=0, type=int, help="1=> also save the train, valid and test splits as raw-format h5 files (in addition to the batching file), 0=> no (Default=0)")
	parser.add_argument('--output_train', default=None, type=str, help="Name of the train split h5 file. Default : {dataset_stem}_train.h5")
	parser.add_argument('--output_valid', default=None, type=str, help="Name of the valid split h5 file. Default : {dataset_stem}_valid.h5")
	parser.add_argument('--output_test', default=None, type=str, help="Name of the test split h5 file. Default : {dataset_stem}_test.h5")

	#if no command line arguments are present, config file is parsed
	config_file='config.txt'
	fromFile=False
	if len(sys.argv) == 1:
		fromFile=True
	if len(sys.argv) == 2 and sys.argv[1].find('--') == -1:
		config_file=sys.argv[1]
		fromFile=True

	if fromFile is True:
		print("Try to read configuration from ",config_file, "file")
		if os.path.isfile(config_file):
			config = parser.parse_args(["@"+config_file])
		else:
			config = parser.parse_args(["--help"])
	else:
		config = parser.parse_args()

	return config


def main():
	config = getArguments()

	raw = config.dataset
	seed = config.seed
	num_train = config.num_train
	num_valid = config.num_valid
	num_test = config.num_test
	batch_size = config.batch_size
	n_bins = config.bins
	batch_size_budget = config.batch_size_budget if config.batch_size_budget > 0 else None
	n_embeded_atoms = config.n_embeded_atoms
	num_workers = config.num_workers
	verbose = config.verbose

	if num_workers < 0:
		num_workers = cpu_count()

	if raw is None or not os.path.exists(raw):
		print("ERROR : dataset file {} does not exist".format(raw))
		sys.exit(1)

	with h5py.File(raw, "r") as f:
		nAll = len(f["N"])
		# per-molecule pair counts : the size criterion for the bins (fallback : atoms)
		if "n_idx" in f:
			pairs = np.asarray(f["n_idx"][()], dtype=np.int64)
		else:
			pairs = np.asarray(f["N"][()], dtype=np.int64)
		has_idx = ("n_idx" in f) and ("dst_idx" in f) and ("src_idx" in f)

	if num_train <= 0:
		num_train = nAll

	if verbose > 0:
		print("-" * 80)
		print("dataset       = ", raw)
		print("seed          = ", seed)
		print("num_train     = ", num_train)
		print("num_valid     = ", num_valid)
		print("num_test      = ", num_test)
		print("batch_size    = ", batch_size)
		print("bins          = ", n_bins)
		print("bin_edges     = ", config.bin_edges)
		if batch_size_budget is not None:
			print("batch_size_budget = ", batch_size_budget)
		print("num_workers   = ", num_workers)
		print("-" * 80)

	print("Computing train/valid/test splits (seed={}) .......".format(seed))
	idx_train, idx_valid, idx_test, (ntrain, nvalid, ntest) = compute_splits(
		nAll, num_train, num_valid, num_test, seed
	)
	print("   ntrain =", len(idx_train), " nvalid =", len(idx_valid), " ntest =", len(idx_test))

	sel = np.concatenate([idx_train, idx_valid, idx_test])
	if n_bins > 1:
		edges = compute_bin_edges(pairs[sel], n_bins, method=config.bin_edges)
		bin_ids = assign_bins(pairs, edges)
		batch_sizes = bin_batch_sizes(pairs[sel], assign_bins(pairs[sel], edges), batch_size, n_bins, budget=batch_size_budget)
		tasks = make_binned_tasks({"train": idx_train, "valid": idx_valid, "test": idx_test}, bin_ids, batch_sizes)
		print("   bins by per-molecule pairs : edges =", edges, " batch_sizes =", batch_sizes,
			  " (budget =", batch_size_budget if batch_size_budget is not None else flat_maxlen(pairs[sel], batch_size), ")")
	else:
		edges = []
		bin_ids = assign_bins(pairs, [])
		batch_sizes = [int(batch_size)]
		tasks = make_tasks(idx_train, idx_valid, idx_test, batch_size)

	n_batches_train = sum(1 for t in tasks if t[0] == "train")
	n_batches_valid = sum(1 for t in tasks if t[0] == "valid")
	n_batches_test = sum(1 for t in tasks if t[0] == "test")
	if verbose > 0:
		print("   number of batches : train =", n_batches_train, " valid =", n_batches_valid, " test =", n_batches_test)

	print("Pre-pass over ALL batches : computing padding parameters per bin (maxidx, maxlen, maxnatoms) .......")
	max_pair = compute_mol_max_pairs(raw, nAll, num_workers) if has_idx else None
	pad_params = {}
	bin_attrs = {}
	bins = list(set(b for _s, b, _i, _m in tasks))
	for b in sorted(bins):
		btasks = [t for t in tasks if t[1] == b]
		pad_params[b] = compute_padding_params(raw, btasks, num_workers, max_pair=max_pair)
		if b >= 0:
			m = bin_ids[sel] == b
			p_lo, p_hi = (int(pairs[sel][m].min()), int(pairs[sel][m].max())) if m.any() else (-1, -1)
			bin_attrs[b] = {
				"batch_size": int(batch_sizes[b]),
				"maxidx": int(pad_params[b][0]),
				"maxlen": int(pad_params[b][1]),
				"maxnatoms": int(pad_params[b][2]),
				"pairs_lo": int(p_lo),
				"pairs_hi": int(p_hi),
			}
			print("   bin {:d} : {} batches : maxidx = {:d}, maxlen = {:d}, maxnatoms = {:d}, pairs in [{:d}, {:d}], batch_size = {:d}".format(
				b, len(btasks), pad_params[b][0], pad_params[b][1], pad_params[b][2], p_lo, p_hi, batch_sizes[b]))

	maxidx_global = max(pad_params[b][0] for b in pad_params)
	maxlen_global = max(pad_params[b][1] for b in pad_params)
	maxnatoms_global = max(pad_params[b][2] for b in pad_params)
	print("   global (all bins) : maxidx =", maxidx_global, " maxlen =", maxlen_global, " maxnatoms =", maxnatoms_global)

	chunks = make_chunks(tasks, num_workers)
	num_workers = len(chunks)
	tmpdir = tempfile.mkdtemp(prefix="buildDataBatching_")
	tmp_paths = [os.path.join(tmpdir, "worker_%03d.h5" % k) for k in range(num_workers)]
	try:
		if verbose > 0:
			print("Building and padding batches in parallel ({} workers) .......".format(num_workers))
		Parallel(n_jobs=num_workers)(
			delayed(build_pad_chunk)(
				raw, chunks[k], n_embeded_atoms, pad_params, bin_attrs, tmp_paths[k], verbose=verbose
			)
			for k in range(num_workers)
		)

		if config.output is None:
			output = default_output_name(raw, seed, ntrain, nvalid, ntest, batch_size, n_bins)
		else:
			output = config.output

		keys = keys_present(tmp_paths)
		attrs = {
			"dataset": os.path.basename(raw),
			"seed": int(seed) if seed is not None else -1,
			"num_train": int(ntrain),
			"num_valid": int(nvalid),
			"num_test": int(ntest),
			"batch_size": int(batch_size),
			"n_embeded_atoms": int(n_embeded_atoms),
			"maxidx": int(maxidx_global),
			"maxlen": int(maxlen_global),
			"maxnatoms": int(maxnatoms_global),
			"keys": ",".join(keys),
			"num_bins": int(n_bins),
			"bin_edges": np.asarray(edges, dtype=np.int64),
			"bin_batch_sizes": np.asarray(batch_sizes, dtype=np.int64),
		}

		if verbose > 0:
			print("Merging into ", output, "file .......")
		merge(output, tmp_paths, attrs)
		print("Done. See ", output, "file")

		if config.save_data > 0:
			stem = os.path.splitext(os.path.basename(raw))[0]
			split_files = [
				("train", idx_train, config.output_train if config.output_train is not None else stem + "_train"+'_'+str(num_train)+'.h5'),
				("valid", idx_valid, config.output_valid if config.output_valid is not None else stem + "_valid"+'_'+str(num_valid)+'.h5'),
				("test",  idx_test,  config.output_test  if config.output_test  is not None else stem + "_test"+'_'+str(num_test)+'.h5'),
			]
			for name, idx, out in split_files:
				if len(idx) == 0:
					if verbose > 0:
						print("Split ", name, " is empty : nothing to save")
					continue
				if verbose > 0:
					print("Saving ", name, " data ({} structures) in ".format(len(idx)), out, "file .......")
				save_split(raw, out, idx, verbose=verbose)
			print("Done. See the 3 split files and the batching file")
	finally:
		if config.keep_temp <= 0:
			shutil.rmtree(tmpdir, ignore_errors=True)
		else:
			print("Temporary files kept in ", tmpdir)


if __name__ == "__main__":
	main()
