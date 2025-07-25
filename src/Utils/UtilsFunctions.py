import jax
import sys
import os
import argparse
import string
import random
from datetime import datetime
import numpy  as np
import logging
from flax.training import orbax_utils
import orbax.checkpoint
from Model.Model import *

def checkArguments(args):
	lv=args.output_types.lower().replace(',',' ').lower().split()
	# output names and options
	args.output_names = []
	args.output_options = []
	for i,v in enumerate(lv):
		if not v.isdigit():
			args.output_names.append(v)
		if i<len(lv)-1 and lv[i+1].isdigit():
			args.output_options.append(int(v[i+1].isdigit()))
		else:
			args.output_options.append(-1)
	n=len(args.output_names)
	# output ranks
	if args.output_ranks is not None and len(args.output_ranks) != n:
		output_ranks=args.output_ranks.lower().replace(',',' ').lower().split()
		if len(output_ranks) != n:
			sm=''.join(['-']*100)
			st='\nThe number of output_types must be the same of --output_ranks.\nIn your input file #output_types ={}, #output_ranks={}\n'.format(len(args.output_names), len(args.output_ranks))
			st = "\n"+sm+st+sm
			raise ValueError(st)
		else:
			args.output_ranks=[]
			for i in range(len(output_ranks)):
				if output_ranks[i].isdigit() and int(output_ranks[i])>0 and int(output_ranks[i])<=args.max_basis_degree:
					args.output_ranks.append(int(output_ranks[i]))
				else:
					sm=''.join(['-']*100)
					st = '\nThe output_ranks must be a list of positive integers and <= max_basis_degree.\n'
					st = "\n"+sm+st+sm
					raise ValueError(st)
				
	if args.output_ranks is None:
		args.output_ranks = [0]*n
		for i,name in enumerate(args.output_names):
			if name=="forces":
				args.output_ranks[i] = 0 # in fact the eenergy is rank 0, used to compute the derivatives to obtain forces
			if name=="edipole":
				args.output_ranks[i] = 1
			if name=="polarizability":
				args.output_ranks[i] = 2
			if name=="hyperpolarizability":
				args.output_ranks[i] = 3

	for i in range(len(args.output_ranks)):
		if args.output_ranks[i]>args.max_basis_degree:
			sm=''.join(['-']*100)
			st = '\nThe rank tensor must be <= max_basis_degree.'
			st += '\nExamples : for edipole max_basis_degree must >=1.'
			st += '\n         : for polarizability max_basis_degree must >=2.'
			st += '\n         : for hyperpolarizability max_basis_degree must >=3.\n'
			st = "\n"+sm+st+sm
			raise ValueError(st)
		if args.output_ranks[i]>3:
			sm=''.join(['-']*100)
			st = '\nTTensor with rank >3 is not ye implemented.'
			st = "\n"+sm+st+sm
			raise ValueError(st)
	# weights
	n=len(args.output_names)
	if args.output_weights is None:
		args.output_weights = [1.0]*n
	else:
		output_weights=args.output_weights.lower().replace(',',' ').lower().split()
		args.output_weights=[]
		for i,w in enumerate(output_weights):
			args.output_weights.append(abs(float(w)))
		args.output_weights /= np.sum(args.output_weights)
	if len(args.output_names) != len(args.output_weights) :
		sm=''.join(['-']*100)
		st = '\nThe number of output_weights must be equal to that of args.output_types.\n'
		st += 'output_names={}\n'.format(args.output_names)
		st = "\n"+sm+st+sm
		raise ValueError(st)
	weights ={}
	for i,key in enumerate(args.output_names):
		weights[key] = args.output_weights[i]
	args.output_weights = weights
	return args
	
		

def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument("--restart", type=str, default='train',  help="restart training from a specific folder")
	parser.add_argument("--dataset", type=str,   help="file name of the dataset (including path)")
	parser.add_argument("--num_features", type=int,   help="dimensionality of feature vectors")
	parser.add_argument("--seed", default=42, type=int,   help="seed for splitting dataset into training/validation/test")
	parser.add_argument("--num_epochs", type=int,   help="maximum number of epocs")
	parser.add_argument("--batch_size", type=int,  default=10, help="batch size (default=10)")
	parser.add_argument("--learning_rate", default="0.001,-1", type=str, help="starting learning rate used by the optimizer and learning rate limit to stop the training. -1=> no stopping.Default, 0.001,-1.")
	parser.add_argument("--cutoff", default=5.0, type=float, help="cutoff (5.0 by default)")
	parser.add_argument("--max_basis_degree", default=3, type=int, help="degre for basis set : min = 2 for alpha and min=3 for beta (default=3)")
	parser.add_argument("--num_iterations", default=3, type=int, help="number of MP iterations(default 3)")
	parser.add_argument("--num_basis_functions", default=8, type=int, help="number of basis functions (default=8)")
	parser.add_argument("--max_atomic_number", default=118, type=int, help="max atomic number (default=118)")
	parser.add_argument("--num_train", type=int, default=1,  help="number of training samples")
	parser.add_argument("--num_valid", type=int,  default=1, help="number of validation samples")
	parser.add_argument("--num_test", type=int, default=-1,  help="number of test samples. Default : nAll-num_valid-num_train")
	parser.add_argument("--loss_type", default='mse', type=str, help="loss type : mse (mean squared error) or mea (mean absolute error) or r2mean (1-R2(mean)). Default=mse")
	parser.add_argument("--model_to_load", default='last_step', type=str, help="best_train, best_valid, last_step (default=last_step)")
	parser.add_argument("--output_types", default='edipole,polarizability,hyperpolarizability', type=str, help="List of properties to predict. Examples: edipole,polarizability,hyperpolarizability, energy, energybyatom, forces, sdipole, edipole, ..... For tensor property of rank 2, you can add 6 (alpha symmetric), for tensor of rank 3, you can add 10 (beta with 10 differents elements) or 18 (beta with 18 elements). Example : energy, edipole,polarizability,6,hyperpolarizability,18. Default : edipole,polarizability,hyperpolarizability (so general tensor for polarizability and hyperpolarizability without any symmetry")
	parser.add_argument("--output_ranks", default=None, help="List of output ranks, with the same order than output_types. Default : 1 for edipole, 2 for polarizability, 3 for hyperpolarizability and 0 for others")
	parser.add_argument("--output_weights", default=None, type=str, help="Weights for output properties.  with the same order than output_types. (default=1.0 for each one)")
	parser.add_argument("--n_embeded_atoms", default=-1, type=int, help="number of the embeded atoms. Default -1=> we take all atoms")
	parser.add_argument("--verbose", default=1, type=int, help="Verbose. 0=> minimum of output  Default=1")
	parser.add_argument("--embed_type", default=-1, type=int, help=" 0 : standard : Z, One NN by value : efield, charge or spin multiplicity  , 1: element mode : DNN with Z and optionnally efield , charge or spin multiplicy. Default = 0")
	parser.add_argument("--embed_num_hidden_layers", default=1, type=int, help="number of hidden layers if embed_type=1. Default = 1")

	parser.add_argument('--embed_inputs', type=str, default=None, nargs='+', help="List of embed inputs : efield(for electric field)   and/or totalcharge and/or spinmultiplicity and/or masses. Use comma (,) as seperator")

	parser.add_argument("--lr_schedule", type=str,   default='constant', help="Learning Rate Schedules type  and decay. Default=constant. Examples :  Cosine,0.95 ; Exponential,0.98 ; Linear,0.95 ; Piecewise,0.95 ; Warmupcosine,1.2,0.95")
	parser.add_argument("--optimizer", type=str,   default='adam', help="optimizer : adam, adamw, adamax, adamaxw, amsgrad, rmsprop. You can give b1,b2,decay... Example : adam,0.9,0.999 ; adamw,0.9,0.999,1e-4.Default=adam")
	parser.add_argument("--save_data", type=int, default=0,  help="1(or >0)=> save train&test data, 0=> nosave (Default=0)")
	parser.add_argument('--kernel_init', type=str, default="glorot_uniform", help="kernel_init for nn : zeros, uniform, normal, glorot_uniform, glorot_normal. Default=glorot_uniform")
	parser.add_argument('--radial_basis_fn', type=str, default="reciprocal_bernstein", help="radial basis function : basic_bernstein, exponential_bernstein, reciprocal_bernstein, basic_gaussian, exponential_gaussian, reciprocal_gaussian, basic_chebyshev, exponential_chebyshev, reciprocal_chebyshev and others (see https://e3x.readthedocs.io/v1.0.1/_autosummary/e3x.nn.functions.html). Default=reciprocal_bernstein")
	parser.add_argument('--num_mlps_post_residual', type=int, default=0, help="number  of post residual mlps. Default=0")

	parser.add_argument("--num_residuals_interaction", default=3, type=int,   help="# number of residuals for interaction block (1 at least, default=3)")
	parser.add_argument("--num_residuals_output", default=1, type=int,   help="# number of residuals for interaction block (0 at least, default=1)")
	parser.add_argument("--num_residuals_atomic", default=2, type=int,   help="# number of residuals module block (0 at least, default=2)")
	parser.add_argument("--model_type", default=0, type=int, help="Model type : 0=> Simple, 1=> more flexible)")
	parser.add_argument("--interaction_type", default=0, type=int, help="Interaction type : 0=> Message passing, 1=>SelfAttention. (default=0)")

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

	config = checkArguments(config)

	return config

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

def build_logger(filename):
	print("filename=",filename)
	log = logging.getLogger('mylogger')
	log.setLevel(logging.DEBUG)

	formatter = logging.Formatter('%(message)s')

	fh = logging.FileHandler(filename)
	fh.setLevel(logging.INFO)
	fh.setFormatter(formatter)
	log.addHandler(fh)

	ch = logging.StreamHandler()
	ch.setLevel(logging.ERROR)
	ch.setFormatter(formatter)
	log.addHandler(ch)
	log.propagate = False
	return log

#used for creating a "unique" id for a run (almost impossible to generate the same twice)
def id_generator(size=8, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

def setOutputConfig(config):
	#create directories
	#a unique directory name is created for this run based on the input
	if config.restart is None:
		directory=datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + id_generator() +"_F"+str(config.num_features)+"K"+str(config.num_basis_fonctions)+"i"+str(config.num_iterations)+"cut"+str(config.cutoff)
	else:
		directory=config.restart

	directory = os.path.abspath(directory)

	print("creating directories...")
	if not os.path.exists(directory):
		os.makedirs(directory)

	train_dir = os.path.join(directory, 'train')
	if not os.path.exists(train_dir):
		os.makedirs(train_dir)
	valid_dir = os.path.join(directory, 'validation')
	if not os.path.exists(valid_dir):
		os.makedirs(valid_dir)
	metrics_dir = os.path.join(directory, 'metrics')
	if not os.path.exists(metrics_dir):
		os.makedirs(metrics_dir)
	best_train_checkpoint = os.path.join(train_dir, 'best')
	best_valid_checkpoint = os.path.join(valid_dir, 'best')

	train_data_filename = os.path.join(train_dir, 'train.h5')
	valid_data_filename = os.path.join(valid_dir, 'valid.h5')

	step_checkpoint = os.path.join(train_dir,  'model')
	best_train_loss_file  = os.path.join(train_dir, 'best_loss.npz')
	best_valid_loss_file  = os.path.join(valid_dir, 'best_loss.npz')

	best_train_train_set_file  = os.path.join(metrics_dir, 'best_train_train_set.npz')
	best_train_valid_set_file  = os.path.join(metrics_dir, 'best_train_valid_set.npz')
	best_valid_train_set_file  = os.path.join(metrics_dir, 'best_valid_train_set.npz')
	best_valid_valid_set_file  = os.path.join(metrics_dir, 'best_valid_valid_set.npz')

	logfile= os.path.join(train_dir, 'train.log')
	logger = build_logger(logfile)
	#logging.basicConfig(filename=logfile,level=logging.INFO)
	#logging.basicConfig(filename=logfile,level=logging.DEBUG)
	logger.info("creating directories...")
	#logging.basicConfig(filename=logfile,level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
	#logging.basicConfig(filename=logfile,level=logging.INFO, format='%(message)s')

	s="{:<64s}".format("-"*140)
	print(s,flush=True)
	print("Main output directory          : ", directory)
	print("Logging file                   : ", logfile)
	print("Best train checkpoint          : ", best_train_checkpoint)
	print("Best valid checkpoint          : ", best_valid_checkpoint)
	print("Step checkpoint                : ", step_checkpoint)
	print("Best train loss file           : ", best_train_loss_file)
	print("Best valid loss file           : ", best_valid_loss_file)
	print("Best train / train set file    : ", best_train_train_set_file)
	print("Best train / valid set file    : ", best_train_valid_set_file)
	print("Best valid / train set file    : ", best_valid_train_set_file)
	print("Best valid / valid set file    : ", best_valid_valid_set_file)
	if config.save_data>0:
		print("train data file    : ", train_data_filename)
		print("valid data file    : ", valid_data_filename)

	print(s,flush=True)
	outputConfigs={}
	outputConfigs['directory'] = directory
	outputConfigs['logger'] = logger
	outputConfigs['best_train_checkpoint'] = best_train_checkpoint
	outputConfigs['best_valid_checkpoint'] = best_valid_checkpoint
	outputConfigs['step_checkpoint'] = step_checkpoint
	outputConfigs['best_train_loss_file'] = best_train_loss_file
	outputConfigs['best_valid_loss_file'] = best_valid_loss_file
	outputConfigs['best_train_train_set_file'] = best_train_train_set_file
	outputConfigs['best_train_valid_set_file'] = best_train_valid_set_file
	outputConfigs['best_valid_train_set_file'] = best_valid_train_set_file
	outputConfigs['best_valid_valid_set_file'] = best_valid_valid_set_file
	outputConfigs['train_data_filename'] = None
	outputConfigs['valid_data_filename'] = None
	if config.save_data>0:
		outputConfigs['train_data_filename'] = train_data_filename
		outputConfigs['valid_data_filename'] = valid_data_filename
	
	return outputConfigs 

def save_chk(filename, params, model_config, data):
	ckpt = {'params':params, 'config': model_config, 'data':data}
	checkpointer = orbax.checkpoint.PyTreeCheckpointer()
	save_args = orbax_utils.save_args_from_target(ckpt)
	checkpointer.save(filename, ckpt, save_args=save_args, force=True)
	'''
	from flax.training import checkpoints ## Here not on the top ! problem with logging
	checkpoints.save_checkpoint(ckpt_dir=filename,
                            target=ckpt,
                            step=0,
                            overwrite=True,
                            keep=1)
	'''
	
def read_chk(filename):
	if os.path.exists(filename):
		checkpointer = orbax.checkpoint.PyTreeCheckpointer()
		restored = checkpointer.restore(filename)
		return restored
	else:
		return None

def load_model(filename):
	restored = read_chk(filename)
	if restored is not None:
		config=restored['config']
		model = Model(
			num_features = config['num_features'], 
			max_basis_degree= config['max_basis_degree'],
			num_iterations = config['num_iterations'],
			num_basis_functions = config['num_basis_functions'],
			cutoff = config['cutoff'],
			max_atomic_number = config['max_atomic_number'],
			batch_size=config['batch_size'],
			model_type=config['model_type'],
			output_names=config['output_names'],
			output_ranks=config['output_ranks'],
			output_options=config['output_options'],
			embed_num_hidden_layers=config['embed_num_hidden_layers'],
			embed_inputs=config['embed_inputs'],
			embed_type=config['embed_type'],
			kernel_init=config['kernel_init'],
			n_embeded_atoms=config['n_embeded_atoms'],
			radial_basis_fn=config['radial_basis_fn'],
			num_mlps_post_residual=config['num_mlps_post_residual'],
			num_residuals_interaction=config['num_residuals_interaction'],
			num_residuals_output=config['num_residuals_output'],
			num_residuals_atomic=config['num_residuals_atomic'],
			interaction_type=config['interaction_type'],
			)
	else:
		model = None
	return model, restored
		

def create_model(outputConfig, args):
	print("Begin creation of model .....",flush=True)
	if args.model_to_load =="best_train":
		filename = outputConfig['best_train_checkpoint']
	elif args.model_to_load == "best_valid":
		filename = outputConfig['best_valid_checkpoint']
	else:
		filename = outputConfig['step_checkpoint']

	restored = read_chk(filename)
	if restored is not None:
		config=restored['config']
		model = Model(
			num_features = config['num_features'], 
			max_basis_degree= config['max_basis_degree'],
			num_iterations = config['num_iterations'],
			num_basis_functions = config['num_basis_functions'],
			cutoff = config['cutoff'],
			max_atomic_number = config['max_atomic_number'],
			batch_size=config['batch_size'],
			model_type=config['model_type'],
			output_names=config['output_names'],
			output_ranks=config['output_ranks'],
			output_options=config['output_options'],
			embed_type=config['embed_type'],
			kernel_init=config['kernel_init'],
			embed_num_hidden_layers=config['embed_num_hidden_layers'],
			embed_inputs=config['embed_inputs'],
			n_embeded_atoms=config['n_embeded_atoms'],
			radial_basis_fn=config['radial_basis_fn'],
			num_mlps_post_residual=config['num_mlps_post_residual'],
			num_residuals_interaction=config['num_residuals_interaction'],
			num_residuals_output=config['num_residuals_output'],
			num_residuals_atomic=config['num_residuals_atomic'],
			interaction_type=config['interaction_type'],
			)
	else:
		restored=None
		config=args
		print(config.output_names)
		model = Model(
			num_features = config.num_features, 
			max_basis_degree= config.max_basis_degree,
			num_iterations = config.num_iterations,
			num_basis_functions = config.num_basis_functions,
			cutoff = config.cutoff,
			max_atomic_number = config.max_atomic_number,
			batch_size=config.batch_size,
			model_type=config.model_type,
			output_names=config.output_names,
			output_ranks=config.output_ranks,
			output_options=config.output_options,
			embed_type=config.embed_type,
			kernel_init=config.kernel_init,
			embed_num_hidden_layers=config.embed_num_hidden_layers,
			embed_inputs=config.embed_inputs,
			n_embeded_atoms=config.n_embeded_atoms,
			radial_basis_fn=config.radial_basis_fn,
			num_mlps_post_residual=config.num_mlps_post_residual,
			num_residuals_interaction=config.num_residuals_interaction,
			num_residuals_output=config.num_residuals_output,
			num_residuals_atomic=config.num_residuals_atomic,
			interaction_type=config.interaction_type,
			)
	return model, restored

