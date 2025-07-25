import ase
from ase.neighborlist import neighbor_list
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
from pympler import tracker
from pympler import muppy
import psutil
from pympler import asizeof
import logging
from Utils.UtilsFunctions import *
from Utils.DataContainer import *
from Utils.DataProvider import *
#from memory_profiler import profile

def add_mae(mae, pmae, i):
	if mae is None:
		mae = pmae
	else:
		for key in pmae.keys():	
			mae[key] += (pmae[key] - mae[key])/(i+1)
	return mae

def print_weights(weights):
	s="{:<64s}".format("-"*140)
	print(s,flush=True)
	mes ="Weights:"
	for key in weights.keys():
		mes +="\n\t{:20s}{:s}{:14.8f}".format(key, "=",weights[key])
	print(mes)
	print(s,flush=True)

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
def eval_step(model_apply, data, loss_type,  weights, params):
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

class Evaluator:
	def __init__(self,
		models_directories,               # directories containing fitted models (can also be a list for ensembles)
		):
		if(type(models_directories) is not list):
			self._models_directories=[models_directories]
		else:
			self._models_directories=models_directories

		self._models = []
		self._restoreds = []
		cutoff = None
		for directory in self._models_directories:
			model, restored = load_model(directory)
			if model is None:
				raise ValueError("I cannot read model from  {}".format(directory))
				
			config=restored['config']
			if cutoff is None:
				cutoff = config['cutoff']
			elif abs(config['cutoff'] - cutoff)>0.01 :
				raise ValueError("The cutoff is not the same on all models")

			self._models.append(model)
			self._restoreds.append(restored)

	def get_properties(self, config):
		loss_type=config.loss_type.upper()
		s="{:<64s}".format("-"*140)
		if loss_type not in ['MSE','MAE','R2MEAN']:
			print(['*']*160)
			print("ERROR :" , config.loss_type, " is not a known loss type. Known types : mse , mae or r2mean")
			print(['*']*160)
			sys.exit(1)
		print(s,flush=True)
		print('Loss type =', config.loss_type)
		print(s,flush=True)
		config.num_train = config.num_structures
		config.num_valid = 0
		config.num_test = 0
		if config.batch_size<-1 :
			config.batch_size=1

		configModel=self._models[0].get_config()
		config.n_embeded_atoms=configModel['n_embeded_atoms']
		config.verbose=config.verbose
		data=DataContainer(config)
		dataProvider = DataProvider(data, config)
		size=len(data)
		if config.batch_size > size:
			config.batch_size=size
		weights = config.output_weights
		print_weights(weights)

		n_batches = dataProvider.get_nsteps_batch()
		ncharall=45
		nchartrain = ncharall/n_batches
		ndigits=len(str(n_batches))

		all_values = None
		all_mae = None
		all_loss = 0.0
		for i in range(len(self._models)):
			mae = None
			loss = 0.0
			values = None
			keys = None
			dataProvider.reset_train_batch()
			if config.verbose>0:
				print("Model number ",i+1)
			for j in range(n_batches):
				data = dataProvider.next_train_batch()
				l, m, r = eval_step(
						model_apply=self._models[i].apply,
						loss_type=loss_type,
        					data=data,
						weights=weights,
						params=self._restoreds[i]['params']
					)
				loss += (l - loss)/(j+1)
				mae = add_mae(mae, m, j)
				if config.verbose>0:
					ieq=int((j+1)*nchartrain)
					isp=abs(ncharall-ieq)
					print("{:{width}d}/{:{width}d} [{}{}] loss={:0.8f}".format(j+1,n_batches, "="*ieq, " "*isp, loss, width=ndigits),end="\r",flush=True)
				values = add_values(values, r, data)
				keys=list(r.keys())
			if config.verbose>0:
				print("")
			all_loss += (loss - all_loss)/(i+1)
			all_mae = add_mae(all_mae, mae, i)
			print("Model # {} : {}/{} loss={:0.3f}".format(i+1, i+1, len(self._models), loss),flush=True)
			print("All Models # 1-{} : loss={:0.3f}".format(i+1, all_loss),flush=True)
			if i == 0:
				all_values  = values
			else:
				for key in keys:
					all_values[key+'_predict'] +=  (values[key+'_predict']-all_values[key+'_predict'])/(j+1)
		self._results = all_values
		return all_values

	def __call__(self, config):
		return self.get_properties(config)

	@property
	def models(self):
        	return self._models

