import pickle
from ase.io import read
import numpy as np
import sys
import warnings
from ase.neighborlist import neighbor_list
from ase.io import write
import re
import h5py
import argparse


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


def splitData(data, seed, num_train, num_test):
	ns=num_train + num_test
	nAll=len(data['N'])
	if ns>nAll:
		ns=nAll

	if seed is not None and seed>-1:
		random_state = np.random.RandomState(seed=seed)
		idx = random_state.permutation(np.arange(nAll))[0:ns]
	else:
		idx = np.arange(ns)
	idx_train=idx[0:num_train]
	idx_test=idx[num_train:ns]

	data_train = {}
	data_test  = {}
	for key in data:
		data_train[key] = data[key][idx_train]
		data_test[key]  = data[key][idx_test]

	data = build_idx_format(data)

	data_train = build_idx(data, idx_train, data_train)
	data_test = build_idx(data, idx_test, data_test)

	return data_train, data_test

def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument("--input_file_name", default='file.h5', type=str, help="Name of the input file in h5 format. Default=file.h5")
	parser.add_argument('--seed', type=int, default=-1, help="seed to shuffle data before spliting. Default=-1 (no shuffle)")
	parser.add_argument('--train_size', type=float, default=-1, help="float. If between 0.0 and 1.0 that represent the proportion of the dataset to include in the train split. If int , represents the absolute number of train samples. Dafault=-1 : 1-test_size or / number of structures - test_size")
	parser.add_argument('--test_size', type=float, default=0.2, help="float. If between 0.0 and 1.0 that represent the proportion of the dataset to include in  the train split. If int , represents the absolute number of train samples. Default 0.2")
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

seed=args.seed

print("Load data....",flush=True)
data= loadDatah5(args.input_file_name)

if args.test_size<=1.:
	num_test= int(len(data['N'])*args.test_size)
else:
	num_test= int(args.test_size)

if args.train_size<0:
	num_train= len(data['N'])-num_test
elif args.train_size<1:
	num_train= int(len(data['N'])*args.train_size)
else:
	num_train= int(args.train_size)

if num_test+num_train>len(data['N']):
	print("Error : the total number of structures must be <= ", len(data['N']))
	sys.exit(1) 

data_train, data_test= splitData(data, seed, num_train, num_test)
fn = args.prefix
if args.prefix is None:
	fn = args.input_file_name.split('.')[0]
filename_train = fn + "_n_structures_"+str(num_train)+ '_train.h5'
filename_test  = fn +  "_n_structures_"+str(num_test)+ '_test.h5'
print("Saving data train in ", filename_train, 'file....',flush=True)
saveDatah5(data_train, filename_train)
print("Saving data test in ", filename_test, 'file....',flush=True)
saveDatah5(data_test, filename_test)
datat= loadDatah5(filename_test)

print("That's all ",flush=True)


