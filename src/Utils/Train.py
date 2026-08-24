import functools
import os
import urllib.request
import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import sys
import logging
from Utils.UtilsFunctions import *
from Utils.ValueAccumulator import ValueAccumulator
from flax.training.train_state import TrainState
#from memory_profiler import profile
from optax import contrib
import time

class TrainStateWithValue(TrainState):
	def apply_gradients(self, *, grads, value, **kwargs):
		"""Applies gradients and updates the optimizer using the given value."""
		updates, new_opt_state = self.tx.update(
			grads,
			self.opt_state,
			params=self.params,
			value=value  # Pass the monitored metric here
		)
		new_params = optax.apply_updates(self.params, updates)
		return self.replace(
			step=self.step + 1,
			params=new_params,
			opt_state=new_opt_state,
			**kwargs
		)

def getProps(values):
	dkeys=values.keys()
	props={}
	props['R2']={}
	props['R2(mean)']={}
	props['R2(min)']={}
	props['RMSE']={}
	props['STD target']={}
	#props['MABS target']={}
	for key in dkeys:
		if '_predict' in key:
			name=key.split('_')[0]
			name_predict=name+"_predict"
			name_target=name+"_target"
			predict=values[name_predict].reshape(-1)
			target=values[name_target].reshape(-1)
			ss_res = jnp.sum((target - predict) ** 2)
			ss_tot = jnp.sum((target - jnp.mean(target)) ** 2)
			r2 = 1 - (ss_res / ss_tot)
			props['R2'][name]=r2
			rmse = jnp.sqrt(jnp.mean((predict - target) ** 2))
			props['RMSE'][name]=rmse
			std_target=float(jnp.std(target))
			props['STD target'][name]=std_target
			predict=values[name_predict]
			target=values[name_target]
			ss_res = jnp.sum((target - predict) ** 2,axis=0)
			ss_tot = jnp.sum((target - jnp.mean(target,axis=0)) ** 2,axis=0)
			r2all = 1 - (ss_res / ss_tot)
			r2min = r2all.min()
			r2mean = r2all.mean()
			props['R2(mean)'][name]=r2mean
			props['R2(min)'][name]=r2min
			#meanabs_target=jnp.mean(jnp.abs(target))
			#props['MABS target'][name]=meanabs_target

	return props

def getRMSE(values):
	dkeys=values.keys()
	R2={}
	for key in dkeys:
		if '_predict' in key:
			name=key.split('_')[0]
			name_predict=name+"_predict"
			name_target=name+"_target"
			predict=values[name_predict].reshape(-1)
			target=values[name_target].reshape(-1)
			ss_res = jnp.sum((target - predict) ** 2)
			ss_tot = jnp.sum((target - jnp.mean(target)) ** 2)
			r2 = 1 - (ss_res / ss_tot)
			R2[name]=r2
	return R2


def save_best_loss(best_loss_file, loss, mae):
		props={}
		props["Loss"]=loss
		for key in mae.keys():
			props[key] = mae[key]
		np.savez(best_loss_file, props=props)

def load_best_recorded_loss(best_loss_file):
	#save/load best recorded loss (only the best model is saved)
	print("Trying to load best recorded loss from ", best_loss_file)
	if os.path.isfile(best_loss_file):
		loss_file   = np.load(best_loss_file,allow_pickle=True)
		props=loss_file['props'].item()
		print("Old result with best loss :", end=' ')
		for key in props.keys():
			print("{:s}={:0.10f}".format(key, props[key]), end=' ') 
		print()
		lossbest = props['Loss']
		#[ print("best[", key,"]=", loss_file[key].item()) for key in props ]
	else:
		props = {'Step':0,'Loss':np.inf,'edipole':np.inf,'polarisability':np.inf, 'hyperpolarisability':np.inf,
		'sdipole':np.inf,'alfa':np.inf, 'beta':np.inf, 'energybyatom':np.inf, 'energy':np.inf,}
		np.savez(best_loss_file, props=props)
		lossbest=None
	return lossbest, props

def add_values(values, results, data):
	if values is None:
		values = {}
		n=jnp.sum(jnp.asarray(data['N']))
		for key in  results.keys():
			if 'forces' in key:
				values[key+'_predict'] = results[key][0:n]
				values[key+'_target'] = jnp.asarray(data[key])[0:n]
			else:
				values[key+'_predict'] = results[key]
				values[key+'_target'] = jnp.asarray(data[key])
		values['ID'] = data['ID']
		values['N'] = jnp.asarray(data['N'])
		values['R'] = jnp.asarray(data['R'][0:n])
		values['Z'] = jnp.asarray(data['Z'][0:n])
	else:
		n=jnp.sum(jnp.asarray(data['N']))
		for key in  results.keys():
			if 'forces' in key:
				values[key+'_predict']=jnp.concatenate([values[key+'_predict'], results[key][0:n]], axis=0)
				values[key+'_target'] =jnp.concatenate([values[key+'_target'], jnp.asarray(data[key])[0:n]], axis=0)
			else:
				values[key+'_predict']=jnp.concatenate([values[key+'_predict'], results[key]], axis=0)
				values[key+'_target'] =jnp.concatenate([values[key+'_target'], jnp.asarray(data[key])], axis=0)
		values['ID'] = jnp.concatenate([values['ID'], data['ID']], axis=0)
		values['N'] = jnp.concatenate([values['N'], jnp.asarray(data['N'])], axis=0)
		values['R'] = jnp.concatenate([values['R'], jnp.asarray(data['R'][0:n])], axis=0)
		values['Z'] = jnp.concatenate([values['Z'], jnp.asarray(data['Z'][0:n])], axis=0)
	return values


def add_mae(mae, pmae, i):
	if mae is None:
		mae = pmae
	else:
		for key in pmae.keys():	
			mae[key] += (pmae[key] - mae[key])/(i+1)
	return mae

def mean_squared_loss(results, data, weights):
	loss = 0
	for key in  results.keys():
		loss += weights[key]*jnp.mean(optax.l2_loss(results[key], jnp.asarray(data[key])))
	return loss

def mean_loss(results, data, weights):
	loss = 0
	for key in  results.keys():
		loss += weights[key]*jnp.mean(jnp.abs(results[key] - jnp.asarray(data[key])))
	return loss

def r2mean_loss(results, data, weights):
	loss = 0
	for key in  results.keys():
		predict = results[key]
		target  = jnp.asarray(data[key])
		ss_res  = jnp.sum((target - predict) ** 2,axis=0)
		ss_tot  = jnp.sum((target - jnp.mean(target,axis=0)) ** 2,axis=0)
		#jax.debug.print("sstot={}",ss_tot)
		ss_tot = jnp.where(abs(ss_tot) < 1e-14, 1e-14, ss_tot)
		r2all   = 1.0 - (ss_res / ss_tot)
		r2mean  = r2all.mean()
		loss   += weights[key]*(1.0-r2mean)
	return loss

def mean_absolute_error(prediction, target):
	return jnp.mean(jnp.abs(prediction - jnp.asarray(target)))

def get_mae(results, data):
	mae  ={}
	for key in  results.keys():
		mae[key] = mean_absolute_error(results[key], data[key])
	return mae

@functools.partial(jax.jit, static_argnames=('model_apply', 'loss_type'))
#@profile
def train_step(model_apply, data, weights, loss_type, opt_state, transform_value):
	def loss_fn(params):
		results = model_apply(params, data)
		if loss_type=="MAE": # MAE
			loss = mean_loss(results, data, weights)
		elif loss_type=="MSE": # MSE
			loss = mean_squared_loss(results, data, weights) 
		else:
			loss = r2mean_loss(results, data, weights) #r2mean
		return loss, results
	(loss, results), grads = jax.value_and_grad(loss_fn, has_aux=True)(opt_state.params)
	opt_state = opt_state.apply_gradients(grads=grads,value=transform_value)
	mae = get_mae(results, data)
	return opt_state, loss, mae, results

@functools.partial(jax.jit, static_argnames=('model_apply', 'loss_type'))
def eval_step(model_apply, data, weights, loss_type,  params):
	def loss_fn(params):
		results = model_apply(params, data)
		if loss_type=="MAE": # MAE
			loss = mean_loss(results, data, weights)
		elif loss_type=="MSE": # MSE
			loss = mean_squared_loss(results, data, weights) 
		else:
			loss = r2mean_loss(results, data, weights) #r2mean
		return loss, results
	loss, results =loss_fn(params)
	mae = get_mae(results, data)
	return loss, mae, results

def create_schedule(model, params, first_learning_rate, optimizer, n_train_batches,  typeschedule='constant,0.95',total_steps=-1, loss_type='MSE'):
	losstype=loss_type.upper()
	s="{:<64s}".format("-"*140)
	print(s,flush=True)
	if losstype not in ['MSE','MAE','R2MEAN']:
		print(['*']*160)
		print("ERROR :" , loss_type, " is not a known loss type. Known types : mse , mae or r2mean")
		print(['*']*160)
		sys.exit(1)
	print('Loss type =', loss_type)
	t = typeschedule.split(',')
	if len(t)<2:
		t = typeschedule.split()
	ts=t[0].upper()
	typeplateau = None
	if ts=='COSINE':
		alpha = float(t[1]) if len(t)>1 else 0.9
		mes='{:s}\n\t{:20s}{:s}{:14.8f}\n\t{:20s}{:s}{:14.8f}'.format(
				'Cosine learning rate schedule : ', 
				'First learning rate','=', first_learning_rate, 
				'Decay rate', '=', alpha 
				)
		print(mes)
		lr_schedule = optax.cosine_decay_schedule(first_learning_rate, decay_steps=total_steps, alpha=alpha)
	elif ts=='EXPONENTIAL':
		alpha = float(t[1]) if len(t)>1 else 0.9
		tb = float(t[2])/100 if len(t)>2 else 0.25
		mes='{:s}\n\t{:20s}{:s}{:14.8f}\n\t{:20s}{:s}{:14.8f}\n\t{:20s}{:s}{:14.0f}'.format(
				'Exponential decay learning rate schedule : ', 
				'First learning rate','=', first_learning_rate, 
				'Decay rate', '=', alpha, 
				'Transition begin(%)','=',tb*100)
		print(mes)
		lr_schedule = optax.exponential_decay(
			first_learning_rate, 
			transition_steps=total_steps, 
			transition_begin=int(total_steps*tb), 
			decay_rate=alpha)
	elif ts=='LINEAR':
		alpha = float(t[1]) if len(t)>1 else 0.9
		tb = float(t[2])/100 if len(t)>2 else 0.25
		end_value=first_learning_rate*alpha
		mes='{:s}\n\t{:30s}{:s}{:14.8f}\n\t{:30s}{:s}{:14.8f}\n\t{:30s}{:s}{:14.0f}'.format(
				'Linear learning rate schedule :', 
				'First learning rate','=', first_learning_rate, 
				'End learning rate value', '=', end_value, 
				'Transition begin(%)','=',tb*100)
		print(mes)
		lr_schedule = optax.linear_schedule(
			init_value=first_learning_rate, 
			end_value=end_value,
			transition_steps=total_steps, 
			transition_begin=int(total_steps*tb), 
			)
	elif ts=='PIECEWISE':
		alpha = float(t[1]) if len(t)>1 else 0.9
		tb = float(t[2])/100 if len(t)>2 else 0.25
		mes='{:s}\n\t{:30s}{:s}{:14.8f}\n\t{:30s}{:s}{:4d}{:s}{:4d}{:s}{:4d}'.format(
				'Piecewise Constant Schedule :', 
				'Alpha','=', alpha, 
				'boundaries(%)', '=', int(tb*100),",",int(tb*2*100),",",int(tb*3*100)) 
		print(mes)
		lr_schedule = optax.piecewise_constant_schedule(
			init_value=first_learning_rate,
			boundaries_and_scales={
				int(total_steps*tb):alpha,
				int(total_steps*tb*2):alpha,
				int(total_steps*tb*3):alpha
				}
			)
	elif ts=='WARMUPCOSINE':
		alpha = float(t[1]) if len(t)>1 else 0.9
		beta = float(t[2]) if len(t)>2 else 1.0/alpha
		tb = float(t[3])/100 if len(t)>3 else 0.25
		end_value=first_learning_rate*alpha
		peak_value=first_learning_rate*beta
		mes='{:s}\n\t{:30s}{:s}{:14.8f}\n\t{:30s}{:s}{:14.8f}\n\t{:30s}{:s}{:14.8f}\n\t{:30s}{:s}{:14.0f}'.format(
				'Warmup cosine decay_schedule :', 
				'First learning rate','=', first_learning_rate, 
				'Peak learning rate value', '=', peak_value,
				'End learning rate value', '=', end_value, 
				'Transition begin(%)','=',tb*100)
		print(mes)
		lr_schedule = optax.warmup_cosine_decay_schedule(
			init_value=first_learning_rate,
			end_value=end_value,
			peak_value=peak_value,
			warmup_steps=int(total_steps*tb),
			decay_steps=total_steps,
			)
	elif ts=='WARMUPEXPONENTIAL':
		alpha = float(t[1]) if len(t)>1 else 0.9
		beta = float(t[2]) if len(t)>2 else 1.0/alpha
		tbt = float(t[3])/100 if len(t)>3 else 0.25
		tbw = float(t[4])/100 if len(t)>4 else tbt
		end_value=first_learning_rate*alpha
		peak_value=first_learning_rate*beta
		mes='{:s}\n\t{:30s}{:s}{:14.8f}\n\t{:30s}{:s}{:14.8f}\n\t{:30s}{:s}{:14.8f}\n\t{:30s}{:s}{:14.8f}\n\t{:30s}{:s}{:14.0f}\n\t{:30s}{:s}{:14.0f}'.format(
				'Warmup exponential decay schedule :', 
				'First learning rate','=', first_learning_rate, 
				'Decay_rate','=', alpha, 
				'Peak learning rate value', '=', peak_value,
				'End learning rate value', '=', end_value, 
				'Transition begin(%)','=',tbt*100,
				'Warmup steps(%)','=',tbw*100)
		print(mes)
		lr_schedule = optax.warmup_exponential_decay_schedule(
			warmup_steps=int(total_steps*tbw),
			init_value=first_learning_rate,
			end_value=end_value,
			peak_value=peak_value,
			transition_begin=int(total_steps*tbt),
			transition_steps=total_steps,
			decay_rate=alpha,
			)
	elif ts=='CONSTANT':
		mes='{:s}{:20s}{:s}{:14.8f}'.format(
				'Constant learning rate : ', 
				'First learning rate','=', first_learning_rate
				)
		print(mes)
		lr_schedule = optax.constant_schedule(first_learning_rate)
	elif ts=='REDUCEONPLATEAU':
		typeplateau = str(t[1]) if len(t)>1 else 'Validation'
		typeplateau = typeplateau.upper()[0]
		if typeplateau not in ['T','V']:
			print("ERROR : Unknown type of reduceonplateau")
			print("        The firt parameter must me Train or Validation")
			print("Examples :")
			print("      --lr_schedule=reduceonplateau,Validation,5,0.1,1e-4,0 # use validation loss, patience, factor, rtol, cooldown")
			print("      --lr_schedule=reduceonplateau,Train,5,0.1,1e-4,0 # use train loss, patience, factor, rtol, cooldown (they are the default values)")
			print("      --lr_schedule=reduceonplateau # use validation loss, patience=5, factor=0.5, rtol=1e-4, cooldown=0 (they are the default values)")
			print("      --lr_schedule=reduceonplateau,V # use validation loss, patience=5, factor=0.5, rtol=1e-4, cooldown=0")
			print("      --lr_schedule=reduceonplateau,T # use train loss, patience=5, factor=0.5, rtol=1e-4, cooldown=0")
			sys.exit(1)
		patience = int(t[2]) if len(t)>2 else 5
		factor = float(t[3]) if len(t)>3 else 0.5
		rtol   = float(t[4]) if len(t)>4 else 1e-4
		cooldown = int(t[5]) if len(t)>5 else 0
		accumulation_size = n_train_batches 
		typeplateaushow= 'Train' if typeplateau=='T' else 'Validation'
		mes='{:s}\n\t{:30s}{:s}{:>14s}\n\t{:30s}{:s}{:14d}\n\t{:30s}{:s}{:14.8f}\n\t{:30s}{:s}{:14.8f}\n\t{:30s}{:s}{:14d}'.format(
				'Reduce on plateau schedule :', 
				'Use loss of',' ', typeplateaushow,
				'patience','=', patience,
				'factor','=', factor, 
				'rtol', '=', rtol ,
				'cooldown', '=', cooldown, 
				)
		print(mes)
		lr_schedule = optax.constant_schedule(first_learning_rate)
	else:
		print("ERROR : Unknown schedule type")
		print("Known schedule types are :", "constant, cosine, exponential, linear, piecewise, warmupcosine")
		print("Examples :")
		print("      --lr_schedule=cosine,0.9")
		print("      --lr_schedule=exponential,0.9,20")
		print("      --lr_schedule=linear,0.9,20")
		print("      --lr_schedule=piecewise,0.9,20")
		print("      --lr_schedule=warmupcosine,0.9,1.5,20")
		print("      --lr_schedule=warmupexponential,0.9,1.5,10,20")
		print("      --lr_schedule=reduceonplateau,Validation,5,0.1,1e-4,0 # use validation loss, patience, factor, rtol, cooldown")
		print("      --lr_schedule=reduceonplateau,Train,5,0.1,1e-4,0 # use train loss, patience, factor, rtol, cooldown (they are the default values)")
		print("      --lr_schedule=constant")
		print(s,flush=True)
		sys.exit(1)

	opt = optimizer.split(',')
	if len(opt)<2:
		opt = optimizer.split()
	txt=opt[0].upper()
	if txt=='ADAM':
		b1 = float(opt[1]) if len(opt)>1 else 0.9
		b2 = float(opt[2]) if len(opt)>2 else 0.999
		mes='{:s}\n\t{:20s}{:s}{:14.8f}\n\t{:20s}{:s}{:14.8f}'.format(
				'adam optimizer :', 
				'b1', '=', b1,
				'b2', '=', b2 
				)
		print(mes)
		tx=optax.adam(lr_schedule,b1=b1, b2=b2)
	elif txt=='ADAMW':
		b1 = float(opt[1]) if len(opt)>1 else 0.9
		b2 = float(opt[2]) if len(opt)>2 else 0.999
		weight_decay = float(opt[3]) if len(opt)>3 else 0.0001
		mes='{:s}\n\t{:20s}{:s}{:14.8f}\n\t{:20s}{:s}{:14.8f}\n\t{:20s}{:s}{:14.8f}'.format(
				'adamw optimizer :', 
				'b1', '=', b1,
				'b2', '=', b2 ,
				'weight decay', '=', weight_decay ,
				)
		print(mes)
		tx=optax.adamw(lr_schedule, b1=b1, b2=b2, weight_decay=weight_decay)
	elif txt=='ADAMAX':
		b1 = float(opt[1]) if len(opt)>1 else 0.9
		b2 = float(opt[2]) if len(opt)>2 else 0.999
		mes='{:s}\n\t{:20s}{:s}{:14.8f}\n\t{:20s}{:s}{:14.8f}'.format(
				'adamax optimizer :', 
				'b1', '=', b1,
				'b2', '=', b2 ,
				)
		print(mes)
		tx=optax.adamax(lr_schedule, b1=b1, b2=b2)
	elif txt=='ADAMAXW':
		b1 = float(opt[1]) if len(opt)>1 else 0.9
		b2 = float(opt[2]) if len(opt)>2 else 0.999
		weight_decay = float(opt[3]) if len(opt)>3 else 0.0001
		mes='{:s}\n\t{:20s}{:s}{:14.8f}\n\t{:20s}{:s}{:14.8f}\n\t{:20s}{:s}{:14.8f}'.format(
				'adamaxw optimizer :', 
				'b1', '=', b1,
				'b2', '=', b2 ,
				'weight decay', '=', weight_decay ,
				)
		print(mes)
		tx=optax.adamaxw(lr_schedule, b1=b1, b2=b2, weight_decay=weight_decay)
	elif txt=='AMSGRAD':
		b1 = float(opt[1]) if len(opt)>1 else 0.9
		b2 = float(opt[2]) if len(opt)>2 else 0.999
		mes='{:s}\n\t{:20s}{:s}{:14.8f}\n\t{:20s}{:s}{:14.8f}'.format(
				'amsgrad optimizer :', 
				'b1', '=', b1,
				'b2', '=', b2 ,
				)
		print(mes)
		tx=optax.amsgrad(lr_schedule, b1=b1, b2=b2)
	elif txt=='RMSPROP':
		decay = float(opt[1]) if len(opt)>1 else 0.9
		mes='{:s}\n\t{:20s}{:s}{:14.8f}'.format(
				'rmsprop optimizer :', 
				'decay', '=', decay,
				)
		print(mes)
		tx=optax.rmsprop(lr_schedule)
	else:
		print("ERROR : Unknown optimizer")
		print("Known optimizers are :", "adam, adamw, adamax, adamaxw, amsgrad, rmsprop")
		print("Examples :")
		print("      --optimizer=adam")
		print("      --optimizer=adam,0.9,0.999")
		print("      --optimizer=adamw,0.9,0.999,1e-4")
		print("      --optimizer=adamax,0.9,0.999")
		print("      --optimizer=adamaxw,0.9,0.999,1e-4")
		print("      --optimizer=amsgrad,0.9,0.999")
		print("      --optimizer=rmsprop,0.9")
		print(s,flush=True)
		sys.exit(1)

	if ts=='REDUCEONPLATEAU':
		tx = optax.chain(
			tx, 
			contrib.reduce_on_plateau(
				patience=patience, 
				cooldown=cooldown,
				factor=factor,
				rtol=rtol,
				accumulation_size=accumulation_size
				),
			)

	# Initialize TrainState with Optax optimizer
	state = TrainStateWithValue.create(
		apply_fn=model.apply,
		params=params,
		tx=tx,
		)
	s= '\t{:20s}{:s}{:>14s}'.format("Loss type","=",loss_type)
	print(s)
	s="{:<64s}".format("-"*140)
	print(s)
	return state, lr_schedule, typeplateau, losstype

def print_weights(weights):
	s="{:<64s}".format("-"*140)
	mes ="Weights:"
	for key in weights.keys():
		mes +="\n\t{:20s}{:s}{:14.8f}".format(key, "=",weights[key])
	print(mes)
	print(s,flush=True)

def train_model(key, model, restored, dataProvider, config,  outputConfig):
	# Initialize model parameters and optimizer state.
	s="{:<64s}".format("-"*140)
	print(s,flush=True)
	best_train_loss, _ = load_best_recorded_loss(outputConfig["best_train_loss_file"])
	best_valid_loss, _ = load_best_recorded_loss(outputConfig["best_valid_loss_file"])
	print(s,flush=True)
	print("Begin training ......", flush=True)
	model_config=model.get_config()
	data0 = dataProvider.first()
	print("Init model......", flush=True)
	if restored is None:
		key, init_key = jax.random.split(key)
		#params = model.init(init_key, data=data0)
		params = model.init(init_key, data=dataProvider.next_train_batch())
		#params = model.init(init_key, data=dataProvider.train_data)
	else:
		params = restored['params']

	n_train_batches = dataProvider.get_nsteps_batch()
	n_valid_batches = dataProvider.get_nsteps_valid_batch()
	lrconfig=config.learning_rate.replace(',',' ').split()
	first_learning_rate=float(lrconfig[0])
	final_learning_rate=-1
	if len(lrconfig)>1: 
		final_learning_rate=float(lrconfig[1])
	
	opt_state, lr_schedule , type_plateau, loss_type = create_schedule(model, params,  first_learning_rate, config.optimizer, n_train_batches, total_steps=n_train_batches*config.num_epochs, typeschedule=config.lr_schedule, loss_type=config.loss_type)

	weights = config.output_weights
	print_weights(weights)

	nparams=sum(p.size for p in jax.tree_util.tree_leaves(params))
	print("Number of parameters of the model =",nparams,flush=True)
	logger = outputConfig['logger']
	logger = logging.getLogger('mylogger')
	logger.info("Number of parameters of the model ={}\n".format(np))
	# Train for 'num_epochs' epochs.
	ncharall=45
	nchartrain = ncharall/n_train_batches
	ncharvalid = ncharall/n_valid_batches
	ndigitstrain=len(str(n_train_batches))
	ndigitsvalid=len(str(n_valid_batches))
	ndigits=ndigitstrain
	if ndigits<ndigitsvalid:
		ndigits=ndigitsvalid

	if config.verbose>0:
		print('Total number of epochs =',config.num_epochs,'\njax.jit compilation ......', flush=True)
		s="{:<s}".format("="*77)
		print(s,flush=True)
	transform_value=jnp.inf
	for epoch in range(1, config.num_epochs + 1):
		# Loop over train batches.
		train_loss = 0.0
		train_mae = None
		train_accum = ValueAccumulator()
		dataProvider.reset_train_batch()
		dataProvider.reset_valid_batch()

		start = time.time()
		ieqold=0
		for i in range(n_train_batches):
			data = dataProvider.next_train_batch()
			opt_state, loss, mae, results = train_step(
					model_apply=model.apply,
        				data=data,
					weights=weights,
					loss_type=loss_type,
					opt_state=opt_state,
					transform_value=transform_value,
				)
			train_loss += (loss - train_loss)/(i+1)
			train_mae = add_mae(train_mae, mae, i)
			ieq=int((i+1)*nchartrain)
			isp=abs(ncharall-ieq)
			if config.verbose>0 and ieq>ieqold:
				print("Train : {:{width}d}/{:{width}d} [{}{}] loss={:0.8f}".format(i+1,n_train_batches, "="*ieq, " "*isp, train_loss, width=ndigits),end="\r",flush=True)
				ieqold=ieq
			train_accum.add_batch(results, data)
			del results
		if config.verbose>0:
			#print(f"\r{' ':80}", end="\r", flush=True)
			print("",flush=True)
		end = time.time()
		train_time = end-start

		# Evaluate on validation set.
		valid_mae = None
		valid_loss = 0.0
		valid_accum = ValueAccumulator()
		start = time.time()
		ieqold=0
		for i in range(n_valid_batches):
			data = dataProvider.next_valid_batch()
			loss, mae, results = eval_step(
					model_apply=model.apply,
        				data=data,
					weights=weights,
					loss_type=loss_type,
					params=opt_state.params,
				)
			valid_loss += (loss - valid_loss)/(i+1)
			valid_mae = add_mae(valid_mae, mae, i)
			ieq=int((i+1)*ncharvalid)
			isp=abs(ncharall-ieq)
			if config.verbose>0 and ieq>ieqold:
				print("Valid : {:{width}d}/{:{width}d} [{}{}] loss={:0.8f}".format(i+1,n_valid_batches, "="*ieq, " "*isp, valid_loss, width=ndigits),end="\r",flush=True)
				ieqold=ieq
			valid_accum.add_batch(results, data)
			del results
		if config.verbose>0:
			#print(f"\r{' ':80}", end="\r", flush=True)
			print("",flush=True)
		end = time.time()
		valid_time = end-start

		scale = optax.tree_utils.tree_get(opt_state,'scale')
		current_lr = lr_schedule(opt_state.step)
		if scale is not None:
			current_lr *= scale
		'''
		for v in opt_state.opt_state:
			if 'ReduceLROnPlateauState' in type(v).__name__:
				current_lr = v.scale
				print(v)
				print(current_lr)
		'''
		if type_plateau=='T':
			transform_value = train_loss
		else:
			transform_value = valid_loss
		
    		# Print progress.
		s="{:6s} {:5d}/{:<5d} {:16s} {:>11s} {:>14s} {:>14s}".format("Epoch:",epoch,config.num_epochs," ", "lr", "Train","Validation")
		print(s,flush=True)
		logger.info(s)

		s="   {:32s} {:11.8f} {:14.8f} {:>14.8f}".format("Loss",current_lr,train_loss, valid_loss)
		print(s,flush=True)
		logger.info(s)

		s="   {:12s}/{:<20s} {:10s} {:14.8f} {:>14.8f}".format("Time","second", " ", train_time, valid_time)
		print(s,flush=True)
		logger.info(s)

		s="   {:<s}".format("-"*74)
		print(s,flush=True)
		for key in mae.keys():	
			s="   {:12s}/{:<20s} {:10s} {:14.8f} {:>14.8f}".format("MAE",key, " ", train_mae[key], valid_mae[key])
			print(s)
			logger.info(s)
		if config.verbose>=2:
			train_props=getProps(train_accum.to_dict())
			valid_props=getProps(valid_accum.to_dict())
			for key in train_props.keys():	
				s="   {:<s}".format("-"*74)
				print(s,flush=True)
				logger.info(s)
				for keyp in train_props[key].keys():	
					s="   {:12s}/{:<20s} {:10s} {:14.8f} {:>14.8f}".format(key,keyp, " ", train_props[key][keyp], valid_props[key][keyp])
					print(s)
					logger.info(s)
		s="{:<s}".format("="*77)
		print(s,flush=True)
		logger.info(s)
		if best_train_loss is None or train_loss < best_train_loss:
			best_train_loss =  train_loss
			save_best_loss(outputConfig["best_train_loss_file"], train_loss, train_mae)
			save_chk(outputConfig["best_train_checkpoint"], opt_state.params, model_config, data0)
			np.savez(outputConfig['best_train_train_set_file'], values = train_accum.to_dict())
			np.savez(outputConfig['best_train_valid_set_file'], values = valid_accum.to_dict())
		if best_valid_loss is None or valid_loss < best_valid_loss:
			best_valid_loss =  valid_loss
			save_best_loss(outputConfig["best_valid_loss_file"], valid_loss, valid_mae)
			save_chk(outputConfig["best_valid_checkpoint"], opt_state.params, model_config, data0)
			np.savez(outputConfig['best_valid_train_set_file'], values=train_accum.to_dict())
			np.savez(outputConfig['best_valid_valid_set_file'], values=valid_accum.to_dict())
		save_chk(outputConfig["step_checkpoint"], opt_state.params, model_config, data0)
		if current_lr<final_learning_rate:
			s="Current learning rate = {:0.8e} < final_learning_rate = {:0.8e}".format(current_lr,final_learning_rate)
			print(s)
			ts='Training stopped'
			l=(len(s)-len(ts))//2
			print("{x:{l}}{ts:s}".format(x=" ",l=l, ts=ts))
			s="{:<s}".format("="*77)
			print(s,flush=True)
			break

	# Return final model parameters.
	return opt_state.params
