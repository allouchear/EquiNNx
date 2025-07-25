import numpy as np
from Utils.Evaluator import *
from Utils.UtilsFunctions import *
import os
import argparse
import sys
from ase import io

def get_mae(results, name):
	mae  ={}
	pred=name+'_predict'
	targ=name+'_target'
	mae=None
	if pred in results.keys() and targ in results.keys():
		mae = mean_absolute_error(results[pred], results[targ])
	return mae

def checkArguments(args, output_names):
	n=len(output_names)
	if args.output_weights is None:
		args.output_weights = [1.0]*n
	else:
		output_weights=args.output_weights.lower().replace(',',' ').lower().split()
		args.output_weights=[]
		for i,w in enumerate(output_weights):
			args.output_weights.append(abs(float(w)))
		args.output_weights /= np.sum(args.output_weights)
	if len(output_names) != len(args.output_weights) :
		sm=''.join(['-']*100)
		st = '\nThe number of output_weights must be equal to that of args.output_types.\n'
		st += 'output_names={}\n'.format(output_names)
		st = "\n"+sm+st+sm
		raise ValueError(st)
	weights ={}
	for i,key in enumerate(output_names):
		weights[key] = args.output_weights[i]
	args.output_weights = weights
	return args


def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument('--list_models', type=str, nargs='+', help="List of directory containing fitted models (at least one file), ....")
	parser.add_argument('--dataset', default="mols.h5", type=str, help="Data with index in h5 format. See buildData.py")
	parser.add_argument('--batch_size', default=1, type=int, help="Batch size. Default =1")
	parser.add_argument('--num_structures', default=-1, type=int, help="Number of values to take in the data file. Default = -1=> all structures")
	parser.add_argument("--output_weights", default=None, type=str, help="Weights for output properties.  with the same order than output_types. (default=1.0 for each one)")
	parser.add_argument("--output_file_name", default='evaluation.npz', type=str, help="Name of the output file in npz format. Default=evaluation.npz")
	parser.add_argument("--verbose", default=1, type=int, help="Verbose. 0=> minimum of output  Default=1")
	parser.add_argument("--seed", default=-1, type=int, help="-1=> no shuffle.")
	parser.add_argument("--loss_type", default='mse', type=str, help="loss type : mse (mean squared error) or mea (mean absolute error) or r2mean (1-R2(mean)). Default=mse")

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
print("-"*80)
if args.verbose>0:
	print("Models = ", lmodels)

evaluator = Evaluator(lmodels)
model = evaluator.models[0]
modelconfig=model.get_config()
args = checkArguments(args, modelconfig['output_names'])
results=evaluator(args)
print("-"*80)
names=[]
for name in results.keys():
	if '_predict' in name:
		names.append(name.split('_')[0])
for name in names:
	mae=get_mae(results,name)
	if mae is not None:
		print('{:<s}{:<20s}{:14.6f}'.format("MAE/",name ,mae))
print("-"*80)
print("Results saved in ", args.output_file_name, "file"); 
np.savez(args.output_file_name, values=results)
print("-"*80)
