import functools
import jax
import e3x
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import sys
from Model.MLP import *
import jax.nn.initializers as init

def get_kernel_init_fn(kernel_init):
	kernel_init=kernel_init.lower()
	if kernel_init in ['zeros','ones']:
		kernel_init_fn=getattr(init, kernel_init)
	elif kernel_init=='constant':
		kernel_init_fn=getattr(init, kernel_init)(1.0)
	else:
		kernel_init_fn=getattr(init, kernel_init)()
	return kernel_init_fn

class nnBasicModel(nn.Module):
	num_iterations: int = 3 # Number of iterations (MP+atom-wise refinement)
	interaction_type : int = 0 # interaction type : 0=> Message passing, 1=>SelfAttention. (default=0)
	kernel_init : str = 'glorot_uniform' # kernel_init for nn : zeros, uniform, normal, glorot_uniform, glorot_normal. Default='glorot_uniform'
	num_mlps_post_residual : int =0 # num of post residual mlps

	@nn.compact
	def __call__(self, inputs: jnp.ndarray, basis,  dst_idx,  src_idx) -> jnp.ndarray:
		kernel_init_fn=get_kernel_init_fn(self.kernel_init)
		x = inputs
		ft = x.shape[-1]
		for i in range(self.num_iterations):
			if self.interaction_type == 0:
				y = e3x.nn.MessagePass()(x, basis, dst_idx=dst_idx, src_idx=src_idx)
			else:
				y = e3x.nn.SelfAttention()(x, basis, dst_idx=dst_idx, src_idx=src_idx)

			y = e3x.nn.add(x, y)
			# Atom-wise refinement MLP.
			y = e3x.nn.Dense(ft)(y)
			y = e3x.nn.silu(y)
			y = e3x.nn.Dense(ft, kernel_init=kernel_init_fn)(y)
			# Residual connection.
			x = e3x.nn.add(x, y)

			# Apply post residual MLPs.
			for _ in range(self.num_mlps_post_residual):
				y = e3x.nn.Dense(self.num_features)(x)
				y = e3x.nn.silu(y)
				y = e3x.nn.Dense(self.num_features, kernel_init=kernel_init_fn)(y)
				x = e3x.nn.add(x, y)

		return x

class Interaction(nn.Module):
	num_residuals_atomic: int = 2
	num_residuals_interaction: int = 3
	activation_fn: Callable = silu
	kernel_init : str = 'glorot_uniform'
	use_bias: bool = False
	interaction_type : int = 0 # interaction type : 0=> Message passing, 1=>SelfAttention. (default=0)

	@nn.compact
	def __call__(self, inputs: jnp.ndarray, basis,  dst_idx,  src_idx) -> jnp.ndarray:
		x = inputs
		kernel_init_fn=get_kernel_init_fn(self.kernel_init)
		ft = x.shape[-1]
		# interaction 
		if self.interaction_type == 0:
			m = e3x.nn.MessagePass()(x, basis, dst_idx=dst_idx, src_idx=src_idx)
		else:
			m = e3x.nn.SelfAttention()(x, basis, dst_idx=dst_idx, src_idx=src_idx)
		for i in range(self.num_residuals_interaction):
			m = Residual(num_blocks=2, activation_fn=self.activation_fn, use_bias=self.use_bias)(m)
		m = e3x.nn.Dense(ft,use_bias=self.use_bias)(m)
		
		#u = self.param('u', lambda rng, shape: jnp.ones(shape), x.shape)
		u = self.param('u', lambda rng, shape: jnp.ones(shape), (ft))
		x = x*u + m
		#jax.debug.print("u={}",u)


		# post interaction 
		for i in range(self.num_residuals_atomic):
			x = Residual(num_blocks=2, activation_fn=self.activation_fn, use_bias=self.use_bias)(x)

		return x

class Output(nn.Module):
	num_residuals: int = 3
	activation_fn: Callable = silu
	kernel_init : str = 'glorot_uniform'
	use_bias: bool = False

	@nn.compact
	def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
		x = inputs
		kernel_init_fn=get_kernel_init_fn(self.kernel_init)
		ft = x.shape[-1]
		for i in range(self.num_residuals):
			x = Residual(num_blocks=2, activation_fn=self.activation_fn, use_bias=self.use_bias)(x)
		if self.activation_fn is not None:
			x= self.activation_fn(x)
		x = e3x.nn.Dense(ft, kernel_init=kernel_init_fn, use_bias=False)(x)
		return x

class nnFlexibleModel(nn.Module):
	num_iterations: int = 3
	num_residuals_atomic: int = 2
	num_residuals_interaction: int = 3
	num_residuals_output: int = 1
	activation_fn: Callable = silu
	kernel_init : str = 'glorot_uniform'
	use_bias: bool = False
	interaction_type : int = 0 # interaction type : 0=> Message passing, 1=>SelfAttention. (default=0)

	@nn.compact
	def __call__(self, inputs: jnp.ndarray, basis,  dst_idx,  src_idx) -> jnp.ndarray:
		x = inputs
		kernel_init_fn=get_kernel_init_fn(self.kernel_init)
		ft = x.shape[-1]
		W = self.param('Woutput', lambda rng, shape: jnp.ones(shape), (self.num_iterations))
		#b = self.param('boutput', lambda rng, shape: jnp.zeros(shape), x.shape)
		#b = self.param('boutput', lambda rng, shape: jnp.zeros(shape), (ft))
		outputs=0
		for i in range(self.num_iterations):
			InteractionBlock = Interaction(
				num_residuals_atomic = self.num_residuals_atomic, 
				num_residuals_interaction = self.num_residuals_interaction,
				activation_fn=self.activation_fn,
				kernel_init=self.kernel_init,
				use_bias=self.use_bias,
				interaction_type=self.interaction_type
				)
			OutputBlock = Output(
				num_residuals = self.num_residuals_output, 
				activation_fn = self.activation_fn,
				kernel_init = self.kernel_init,
				use_bias = self.use_bias,
				)
			x = InteractionBlock(x,  basis, dst_idx,  src_idx)
			out = OutputBlock(x)
			#outputs += out * W[i]+b[i]
			outputs += out * W[i]
		#jax.debug.print("W={}",W)
		#jax.debug.print("b={}",b)

		return outputs
