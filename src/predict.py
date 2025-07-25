import numpy as np
from Utils.Predictor import *
from Utils.UtilsFunctions import *
import os
import argparse
import sys
from ase import io
from ase.io import write,read



def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument('--list_models', type=str, nargs='+', help="list of directory containing fitted models (at least one file), ....")
	parser.add_argument('--input_file_name', default="mol.xyz", type=str, help="xyz file")
	parser.add_argument('--batch_size', default=1, type=int, help="Batch size. Default =1")
	parser.add_argument("--verbose", default=0, type=int, help="Verbose. 0=> minimum of output  Default=1")
	parser.add_argument("--seed", default=-1, type=int, help="-1=> no shuffle.")

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
lmodels=args.list_models
lmodels=lmodels[0].split(',')
print("---------------------------------------------")
if args.verbose>0:
	print("Models = ", lmodels)

predictor = Predictor(lmodels)
mols=predictor(args)
foutxyz=args.input_file_name.split('.')[0]+'_predict.xyz'
write(foutxyz, mols,  format='extxyz')
print("See results in file : ", foutxyz)
print("---------------------------------------------")
