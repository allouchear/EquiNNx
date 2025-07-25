import numpy as np
import sys
import warnings
import re
import h5py
import argparse
import seaborn as sns
import matplotlib.pyplot as plt


def saveDatah5(data, filename):
	with h5py.File(filename, "w") as h5file:
		for key, value in data.items():
			if value is not None:
				h5file.create_dataset(key, data=value, compression="lzf")


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



def loadDatah5(filename):
	with h5py.File(filename, "r") as h5file:
		data = {key: h5file[key][()] for key in h5file.keys()}
	return data


def getToDelete(data, key, prefix, c_size=5.0):
	names=['x','y','z']
	#print(data[key].shape)
	x=0.1
	y=data[key].shape[0]/30
	tpos=0.9
	toDelete=np.array([False]*data[key].shape[0])
	sc = str(c_size)
	sc = sc.replace('.','p')
	sc = '_csize_'+sc
	sn = '_n_structures_'+str(data[key].shape[0])
	if len(data[key].shape)==2: # dipole
		for i in range(3):
			mean=np.mean(abs(data[key][:,i]))
			std=np.std(abs(data[key][:,i]))
			xcut=mean+c_size*std
			td=abs(data[key][:,i])>xcut
			toDelete=toDelete|td
			sns.displot(abs(data[key][:,i]))
			s=names[i]
			unique, counts = np.unique(td, return_counts=True)
			ndel=0 if len(counts)<2 else counts[1]
			ss='$\\mu_{:s}$\nMAV   = {:0.5}\nstd = {:0.5}\ncutoff=MAV+{:0.5}*std\ncuoff={:0.5}\n# to delete ={:d}'.format(s,mean,std,c_size, xcut,ndel)
			x=mean+c_size*std/4
			plt.text(x=x, y=y, s=ss,  horizontalalignment='left', verticalalignment='center', bbox=dict(facecolor='lightsteelblue', alpha=0.5))
			plt.vlines(xcut, 0, y, linestyles ="dotted", colors ="k")
			filename=prefix+"_"+s+sn+sc+'.pdf'
			plt.savefig(filename)
			plt.close()
	if len(data[key].shape)==3: # polar
		for i in range(3):
			for j in range(3):
				mean=np.mean(abs(data[key][:,i,j]))
				std=np.std(abs(data[key][:,i,j]))
				xcut=mean+c_size*std
				td=abs(data[key][:,i,j])>xcut
				toDelete=toDelete|td
				sns.displot(abs(data[key][:,i,j]))
				s=names[i]+names[j]
				x=mean+c_size*std/4
				unique, counts = np.unique(td, return_counts=True)
				ndel=0 if len(counts)<2 else counts[1]
				ss='$\\alpha_{:s}{:s}{:s}$\nMAV   = {:0.5}\nstd = {:0.5}\ncutoff=MAV+{:0.5}*std\ncuoff={:0.5}\n# to delete ={:d}'.format('{',s,'}',mean,std,c_size, xcut,ndel)
				plt.text(x=x, y=y, s=ss,  horizontalalignment='left', verticalalignment='center', bbox=dict(facecolor='lightsteelblue', alpha=0.5))
				plt.vlines(xcut, 0, y, linestyles ="dotted", colors ="k")
				filename=prefix+"_"+s+sn+sc+'.pdf'
				plt.savefig(filename)
				plt.close()
	if len(data[key].shape)==4: # hyppolar
		for i in range(3):
			for j in range(3):
				for k in range(3):
					mean=np.mean(abs(data[key][:,i,j,k]))
					std=np.std(abs(data[key][:,i,j,k]))
					xcut=mean+c_size*std
					td=abs(data[key][:,i,j,k])>xcut
					toDelete=toDelete|td
					s=names[i]+names[j]+names[k]
					sns.displot(abs(data[key][:,i,j,k]))
					unique, counts = np.unique(td, return_counts=True)
					ndel=0 if len(counts)<2 else counts[1]
					ss='$\\beta_{:s}{:s}{:s}$\nMAV   = {:0.5}\nstd = {:0.5}\ncutoff=MAV+{:0.5}*std\ncuoff={:0.5}\n# to delete ={:d}'.format('{',s,'}',mean,std,c_size, xcut,ndel)
					x=mean+c_size*std/4
					plt.text(x=x, y=y, s=ss,  horizontalalignment='left', verticalalignment='center', bbox=dict(facecolor='lightsteelblue', alpha=0.5))
					plt.vlines(xcut, 0, y, linestyles ="dotted", colors ="k")
					filename=prefix+"_"+s+sn+sc+'.pdf'
					plt.savefig(filename)
					plt.close()

	return toDelete

def getToDeleteAll(data, prefix, c_size):
	names=['edipole', 'polarizability', 'hyperpolarizability']
	toDelete={}
	for name in names:
		toDelete[name] = getToDelete(data, name, prefix, c_size=c_size)
		unique, counts = np.unique(toDelete[name], return_counts=True)
		print("Number of structures to delete for ",name, " = ", 0 if len(counts)<2 else counts[1],flush=True)
		toDelete["All"]=toDelete["All"]|toDelete[name] if "All" in toDelete.keys() else toDelete[name]

	unique, counts = np.unique(toDelete["All"], return_counts=True)
	print("Number of structures to delete = ", 0 if len(counts)<2 else counts[1],flush=True)
	idx = np.arange(data[names[0]].shape[0])
	idxToNoDelete = idx[toDelete['All']==False]
	idxToDelete = idx[toDelete['All']==True]
	return toDelete, idxToNoDelete, idxToDelete

def removeOutliers(data, prefix, c_size):
	toDelete, idx, idxToDelete = getToDeleteAll(data, prefix, c_size)
	newdata = {}
	for key in data:
		newdata[key] = data[key][idx]
	datatoDelete = {}
	for key in data:
		datatoDelete[key] = data[key][idxToDelete]

	print("Rebuild idx....",flush=True)
	data = build_idx_format(data)

	newdata = build_idx(data, idx, newdata)
	
	ns = idx.shape[0]
	s = str(c_size)
	s = s.replace('.','p')

	filename = prefix + "_n_structures_"+str(ns)+'_csize_'+s+'.h5'
	print("Saving new data  in ", filename, 'file....',flush=True)
	saveDatah5(newdata, filename)
	datat= loadDatah5(filename) # to test

	ns = idxToDelete.shape[0]
	s = str(c_size)
	s = s.replace('.','p')
	datatoDelete = build_idx(data, idxToDelete, datatoDelete)
	filename = prefix + "_deleted_n_structures_"+str(ns)+'_csize_'+s+'.h5'
	print("Saving new data  in ", filename, 'file....',flush=True)
	saveDatah5(datatoDelete, filename)
	datat= loadDatah5(filename) # to test

	return newdata, ns

def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument("--input_file_name", default='file.h5', type=str, help="Name of the input file in h5 format. Default=file.h5")
	parser.add_argument("--prefix", default=None, type=str, help="the prefix name for the output files. Default=prefix of input_file_name")
	parser.add_argument("--c_size", default=5.0, type=float, help="cutoff = MAV+c_size*STD")
	parser.add_argument("--remove_outliers", default=0, type=int, help="0=> Non, 1=> yes remove outliers")

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

print("Load data....",flush=True)
data= loadDatah5(args.input_file_name)
fn = args.prefix
if args.prefix is None:
	fn = args.input_file_name.split('.')[0]
if args.remove_outliers>0:
	newdata, ns = removeOutliers(data, fn, c_size=args.c_size)
else:
	toDelete, idxToNoDelete, idxToDelete = getToDeleteAll(data, fn, args.c_size)

print("That's all ",flush=True)
