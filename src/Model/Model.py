import functools
import os
import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import optax
import sys
from Model.Modules import *
#from memory_profiler import profile
from typing import List


def add_traces1(beta, V):
	M = jnp.einsum('ijk,k->ij', beta, V)
	beta = jnp.einsum('ij,k->ijk', M, V)  # Resulting tensor of shape (3, 3, 3)
	perm1 = jnp.transpose(beta, (2, 0, 1))  # A_{kij}
	perm2 = jnp.transpose(beta, (1, 2, 0))  # A_{jki}
	beta = (beta + perm1 + perm2) / 3.0
	perm3 = jnp.transpose(beta, (0, 1, 2))
	beta = (beta + perm3) / 2.0
	return beta

def add_traces(beta, A): # best
	betaV = (A[:, None, None] * A[None, :, None] * A[None, None, :]) \
     + (A[:, None, None] * A[None, None, :] * A[None, :, None]) \
     + (A[None, :, None] * A[None, None, :] * A[:, None, None])

	betaV = betaV / 3
	beta += betaV
	return beta

def build_alpha(x,cg):
	return jnp.einsum('...n,lmn->lm', x, cg)

def build_beta(x,cg1, cg2):
	return jnp.einsum('...p,lmn,nop->...lmo', x, cg1, cg2)

def symmetrize_matrix(m):
	return 0.5 * (m+m.T)
def symmetrize_tensor_3(beta):
	return 0.5 * (beta + jnp.swapaxes(beta, 1, 2))

def symmetrize_Kleinman(beta):
    return (beta + 
            jnp.transpose(beta, (1, 0, 2)) + 
            jnp.transpose(beta, (0, 2, 1)) + 
            jnp.transpose(beta, (2, 0, 1)) + 
            jnp.transpose(beta, (2, 1, 0)) + 
            jnp.transpose(beta, (1, 2, 0))) / 6.0

def add_traces3(beta, V):
	"""
	Constructs a rank-3 tensor T_{ijk} from a vector V,
	which is equivariant under SO(3) and has:
	cyclic symmetry: T_{ijk} = T_{jki} = T_{kij}
	T_{ijk} = T_{ikj}
	- T_{ijk} != T_{jik}
	"""
	# Levi-Civita symbol (antisymmetric tensor)
	eps = jnp.array([
        [[ 0,  0,  0],
         [ 0,  0,  1],
         [ 0, -1,  0]],
        
        [[ 0,  0, -1],
         [ 0,  0,  0],
         [ 1,  0,  0]],
        
        [[ 0,  1,  0],
         [-1,  0,  0],
         [ 0,  0,  0]],
	])  # shape (3, 3, 3)

	# Build the tensor
	T = (
		jnp.einsum('ijl,l,k->ijk', eps, V, V) +
		jnp.einsum('jkl,l,i->ijk', eps, V, V) +
		jnp.einsum('kil,l,j->ijk', eps, V, V)
	)
	beta += T
	return beta
def approximately_unique_floats(tensor, tol=1e-6):
	flat = tensor.flatten()
	sorted_vals = jnp.sort(flat)
	diffs = jnp.diff(sorted_vals)
	# Count how many differences exceed the tolerance
	num_unique = jnp.sum(diffs > tol) + 1  # +1 for the first unique value
	return num_unique

class Model(nn.Module):
	num_features: int = 32, #dimensionality of feature vector
	max_basis_degree: int = 3, # max basis degree Harmonic spheric
	num_iterations: int = 3, # Number of iterations (MP+atom-wise refinement)
	num_basis_functions: int = 8, # #number of basis functions
	cutoff: float = 5.0, # cutoff distance
	max_atomic_number: int = 118,   # Max atomic number. This is overkill for most applications.
	batch_size: int = 10,   # batch size
	output_names: List = ['edipole','polarizability','hyperpolarzibality'],   # list of properties to compute :  edipole,polarizability,hyperpolarizability, energy, energybyatom, forces, sdipole, edipole, ..... 
	output_ranks: List = [1,2,3],  # ranks of output properties
	output_options: List = [-1,-1,-1],  # option for properties. Example 6=>symmetric polarizability tensor, 10 or 18=> for symmetric hypolarizability. -1 not symmetric tensor
	embed_type: int = 0,  # 0 : standard : Z, One NN by value : efield, charge or spin multiplicity  , 1: element mode : DNN with Z and optionnally efield , charge or spin multiplicy
	embed_num_hidden_layers: int = 1   # number of hidden layers if embed_type=1
	embed_inputs: str = None # List of embed inputs : efield(for electric field) and/or totalcharge and/or spinmultiplicity and/or masses. Use comma (,) as seperator
	n_embeded_atoms : int = -1 # number of the embeded atoms. -1=> we take all atoms
	kernel_init : str = 'glorot_uniform' # kernel_init for nn : zeros, uniform, normal, glorot_uniform, glorot_normal. Default='glorot_uniform'
	radial_basis_fn: str = 'reciprocal_bernstein' # radial basis function : basic_bernstein, exponential_bernstein, reciprocal_bernstein, basic_gaussian, exponential_gaussian, reciprocal_gaussian, basic_chebyshev, exponential_chebyshev, reciprocal_chebyshev and others (see https://e3x.readthedocs.io/v1.0.1/_autosummary/e3x.nn.functions.html)
	num_mlps_post_residual : int =0 # num of post residual mlps
	num_residuals_interaction : int = 3 # number of residuals for interaction block (1 at least, default=3)
	num_residuals_output : int = 1 # number of residuals for interaction block (0 at least, default=1)
	num_residuals_atomic : int = 2 # number of residuals module block (0 at least, default=2)
	interaction_type : int = 0 # interaction type : 0=> Message passing, 1=>SelfAttention. (default=0)
	model_type: int = 0   #  model type : 0=> Simple, 1=> more flexible)

	def _kernel_init(self):
		if self.kernel_init.lower() =="glorot_uniform":
			return jax.nn.initializers.glorot_uniform()
		elif self.kernel_init.lower() =="glorot_normal":
			return jax.nn.initializers.glorot_normal()
		elif self.kernel_init.lower() =="normal":
			return jax.nn.initializers.normal()
		elif self.kernel_init.lower() =="uniform":
			return jax.nn.initializers.uniform()
		else:
			return jax.nn.initializers.zeros

	def get_config(self):
		config={}
		config['num_features'] = self.num_features
		config['max_basis_degree'] = self.max_basis_degree
		config['num_iterations'] = self.num_iterations
		config['num_basis_functions'] = self.num_basis_functions
		config['cutoff'] = self.cutoff
		config['max_atomic_number'] = self.max_atomic_number
		config['batch_size'] = self.batch_size
		config['model_type'] = self.model_type
		config['output_names'] = self.output_names
		config['output_ranks'] = self.output_ranks
		config['output_options'] = self.output_options
		config['embed_type'] = self.embed_type
		config['embed_num_hidden_layers'] = self.embed_num_hidden_layers
		config['embed_inputs'] = self.embed_inputs
		config['kernel_init'] = self.kernel_init
		config['n_embeded_atoms'] = self.n_embeded_atoms
		config['radial_basis_fn'] = self.radial_basis_fn
		config['num_mlps_post_residual'] = self.num_mlps_post_residual
		config['num_residuals_interaction'] = self.num_residuals_interaction
		config['num_residuals_output'] = self.num_residuals_output
		config['num_residuals_atomic'] = self.num_residuals_atomic
		config['interaction_type'] = self.interaction_type
		return config

	def get_xscalar(self, x, atomic_numbers, shift=None, name='energy'):
		xscalar = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)
		xscalar = e3x.nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)(xscalar)
		xscalar = jnp.squeeze(xscalar, axis=(-2, -3))

		element_scale = self.param('element_scale'+name, lambda rng, shape: jnp.ones(shape), (self.max_atomic_number+1))
		xscalar = xscalar.at[:,0].multiply(element_scale[atomic_numbers])

		if shift is not None:
			element_bias = self.param('element_bias'+name, lambda rng, shape: jnp.ones(shape)*shift, (self.max_atomic_number+1))
		else:
			element_bias = self.param('element_bias'+name, lambda rng, shape: jnp.zeros(shape), (self.max_atomic_number+1))
		xscalar = xscalar.at[:,0].add(element_bias[atomic_numbers])

		return xscalar

	def get_displacements(self, R, dst_idx, src_idx, offsets=None):
		#calculate interatomic displacements
		dst_R = e3x.ops.gather_dst(R, dst_idx=dst_idx)
		src_R = e3x.ops.gather_src(R, src_idx=src_idx)
		if offsets is not None:
			src_R += offsets
		displacements = src_R - dst_R  # Shape (num_pairs, 3)
		return displacements 

	#@profile
	def get_energies(self, data, x):
		batch_size = len(data['N'])
		results ={}
		energies = None
		shift=0
		if 'energybyatom'  in self.output_names:
			shift= jnp.mean(jnp.asarray(data['energybyatom']))
		else:
			shift= jnp.sum(jnp.asarray(data['energy']))/jnp.sum(jnp.asarray(data['N']))
		x = self.get_xscalar(x, data['Z'],shift=shift, name='energy')
		energies = x[:,0]
		energies = jax.ops.segment_sum(energies, segment_ids=jnp.asarray(data['batch_seg']),num_segments=2*batch_size)[0:2*batch_size:2]
		Na_per_batch = jax.ops.segment_sum(jnp.ones_like(jnp.asarray(data['batch_seg']), dtype=x.dtype), segment_ids=jnp.asarray(data['batch_seg']),num_segments=2*batch_size)[0:2*batch_size:2]
		energiesbyatom = energies/Na_per_batch 
		return energies, energiesbyatom

	def get_vector(self,data,x):
		batch_size = len(data['N'])
		p=1 # 1=> odd, 0=>even
		f=0 # feature, last index
		l=1
		#x1t = e3x.nn.TensorDense(features=1, max_degree=l, dense_kernel_init=jax.nn.initializers.glorot_uniform())(x)
		x1t = e3x.nn.TensorDense(features=1, max_degree=l, dense_kernel_init=self._kernel_init())(x)
		#print(x1t.shape) # ..., 2, 4, 1, ...,p,l2:(l=1)2,f
		edipole = jnp.squeeze(x1t[:,p:p+1,l**2:(l+1)**2,f:f+1], axis=(-1, -3)) # 1:4 => dipole i=1,2,3
		edipole = jax.ops.segment_sum(edipole, segment_ids=jnp.asarray(data['batch_seg']),num_segments=2*batch_size)[0:2*batch_size:2]
		return edipole

	def get_asym_matrix(self,data,x):
		batch_size = len(data['N'])
		p=0 # 1=> odd, 0=>even
		f=0 # feature, last index
		l=2
		cg = e3x.so3.clebsch_gordan(1, 1, 2)
		#x2t = e3x.nn.TensorDense(features=1, max_degree=l, dense_kernel_init=jax.nn.initializers.glorot_uniform())(x)
		x2t = e3x.nn.TensorDense(features=1, max_degree=l, dense_kernel_init=self._kernel_init())(x)
		# https://e3x.readthedocs.io/stable/constructing_cartesian_tensors.html
		batched_update = jax.vmap(build_alpha, in_axes=(0,None))
		alpha = None
		for li in range(3):
			a = batched_update(x2t[...,p, li**2:(li+1)**2, f], cg[1:4, 1:4, li**2:(li+1)**2])
			if alpha is None:
				alpha = a
			else:
				alpha += a
				
		alpha = jax.ops.segment_sum(alpha, segment_ids=jnp.asarray(data['batch_seg']),num_segments=2*batch_size)[0:2*batch_size:2]
		return alpha

	def get_asym_tensor_3(self,data,x):
		batch_size = len(data['N'])
		p=1 # 1=> odd, 0=>even
		f=0 # feature, last index
		l=3
		cg = e3x.so3.clebsch_gordan(2, 1, 3)
		# https://e3x.readthedocs.io/stable/constructing_cartesian_tensors.html
		# 3 features needed
		#x3t = e3x.nn.TensorDense(features=3, max_degree=l, dense_kernel_init=jax.nn.initializers.glorot_uniform())(x)
		x3t = e3x.nn.TensorDense(features=3, max_degree=l, dense_kernel_init=self._kernel_init())(x)
		tensor_components = []
		for li, f in [(0, 0), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0)]:
			tensor_components.append(x3t[...,p, li**2:(li+1)**2, f])
		batched_update = jax.vmap(build_beta, in_axes=(0,None,None))
		coupling_paths = [
			(1, 0), #1
			(0, 1), #2
			(1, 1), #3
			(2, 1), #4
			(1, 2), #5
			(2, 2), #6
			(2, 3), #7
		]

		beta = None
		for i, (l1, l2) in enumerate(coupling_paths):
			b = batched_update(tensor_components[i],
				cg[1:4, 1:4, l1**2:(l1+1)**2],
				cg[l1**2:(l1+1)**2, 1:4, l2**2:(l2+1)**2]
			)
			if beta is None:
				beta = b
			else:
				beta += b
		
		beta = jax.ops.segment_sum(beta, segment_ids=jnp.asarray(data['batch_seg']),num_segments=2*batch_size)[0:2*batch_size:2]
		return beta

	#@profile
	def get_matrix(self, data, x, nelements=9):
		if nelements == -1:
			nelements=9
		m =  self.get_asym_matrix(data,x)
		if nelements==6:
			symetrize = jax.vmap(symmetrize_matrix)
			m = symetrize(m)
		return  m

	#@profile
	def get_tensor_3(self, data, x, nelements=27):
		if nelements==-1:
			nelements=27
		T =  self.get_asym_tensor_3(data,x)
		if nelements==18:
			symetrize = jax.vmap(symmetrize_tensor_3)
			T = symetrize(T)
		if nelements==10:
			symetrize = jax.vmap(symmetrize_Kleinman)
			T = symetrize(T)
		return T

	#@profile
	def get_tensor(self, data, x, name, rank, nelements=-1):
		if rank == 0:
			return self.get_one_scalar(data, x,name)
		elif rank == 1:
			return self.get_vector(data,x)
		elif rank == 2:
			return self.get_matrix(data,x, nelements=nelements)
		elif rank == 3:
			return self.get_tensor_3(data,x, nelements=nelements)
		else:
			raise ValueError('Tensors with rank>3 are not yet implemented')

	#@profile
	def get_one_scalar(self, data, x, name='sdipole'):
		batch_size = len(data['N'])
		if 'byatom'  in name:
			shift= jnp.mean(jnp.asarray(data[name]))
		else:
			shift= jnp.sum(jnp.asarray(data[name]))/jnp.sum(jnp.asarray(data['N']))
		x = self.get_xscalar(x, data['Z'],shift=shift, name=name)
		s = x[:,0]
		s = jax.ops.segment_sum(s, segment_ids=jnp.asarray(data['batch_seg']),num_segments=2*batch_size)[0:2*batch_size:2]
		#s = s.reshape(-1,1)
		return s

	def combine_x_v(self, x, inputs, embed_inputs_list, atomic_numbers, typetoadd):
		if inputs[typetoadd] is None and embed_inputs_list is not None and typetoadd in embed_inputs_list:
			raise ValueError("{} is not available in data. I cannot use embed it".format(typetoadd)) 
		if inputs[typetoadd] is not None and embed_inputs_list is not None and typetoadd  in embed_inputs_list:
			xv = e3x.nn.Embed(num_embeddings=self.max_atomic_number+1, features=self.num_features,dtype=x.dtype)(atomic_numbers)
			# v : shape [batch, n]
			if len(inputs[typetoadd].shape)==1:
				v_sh = inputs[typetoadd][:,None,None,None] #  shape [batch, 1, 1, 1]
			else:
				v_sh = inputs[typetoadd][:,None,:,None] #  shape [batch, 1, n, 1]
			v = xv*v_sh 
			if x.shape[2]>=v.shape[2]:
				x = x+v
			else:
				x = jnp.concatenate([x, v], axis=2) #  shape [batch, 1, 4, features]
		return x
	def get_embed_inputs_list(self):
		embed_inputs_list = self.embed_inputs[0].lower().split(',') if self.embed_inputs is not None else None
		return embed_inputs_list

	def build_input_embed(self, atomic_numbers, inputs, dtype):
		embed_inputs_list = self.get_embed_inputs_list()
		# Embed atomic numbers in feature space, x has shape (num_atoms, 1, 1, features).
		x = e3x.nn.Embed(num_embeddings=self.max_atomic_number+1, features=self.num_features,dtype=dtype)(atomic_numbers)
		if embed_inputs_list is not None:
			for typetoadd in embed_inputs_list:
				x = self.combine_x_v(x, inputs, embed_inputs_list, atomic_numbers, typetoadd)
		return x

	def add_inputs(self, x, inputs, embed_inputs_list, typetoadd):
		if inputs[typetoadd] is None and embed_inputs_list is not None and typetoadd in embed_inputs_list:
			raise ValueError("{} is not available in data. I cannot use embed it".format(typetoadd)) 
		if inputs[typetoadd] is not None and embed_inputs_list is not None and typetoadd  in embed_inputs_list:
			if len(inputs[typetoadd].shape)==1:
				v = inputs[typetoadd][:,None,None,None] #  shape [batch, 1, 1, 1]
			else:
				v = inputs[typetoadd][:,None,:,None] #  shape [batch, 1, n, 1]
			# x shape [batch, 1, n, features]
			A = v
			for i in range(x.shape[-1]-1):
				A = jnp.concatenate([A, v], axis=-1)
			# A shape [batch, 1, n, features]
			if x.shape[2]>=v.shape[2]:
				x = jnp.concatenate([x, A], axis=-1) #  shape [batch, 1, n, features+1]
			else:
				x = jnp.concatenate([x, A], axis=2) #  shape [batch, 1, n+1, features]
		return x

	def build_input_element_mod(self, atomic_numbers, inputs, dtype, num_hidden_nodes=None, activation_fn=None, use_bias=True):
		embed_inputs_list = self.get_embed_inputs_list()
		a = atomic_numbers[:,None,None,None]
		x = a 
		x = jnp.asarray(x,dtype=dtype)
		if embed_inputs_list is not None:
			maxsize=0
			for typetoadd in embed_inputs_list:
				if len(inputs[typetoadd].shape)>maxsize:
					maxsize=len(inputs[typetoadd].shape)
			for isize in range(maxsize):
				for typetoadd in embed_inputs_list:
					if isize+1==len(inputs[typetoadd].shape):
						x = self.add_inputs(x, inputs, embed_inputs_list,typetoadd)

		nhn=num_hidden_nodes if num_hidden_nodes is not None else self.num_features
		for i in range(self.embed_num_hidden_layers):
			#x = e3x.nn.Dense(nhn, use_bias=use_bias, kernel_init=jax.nn.initializers.glorot_uniform())(x)  
			x = e3x.nn.Dense(nhn, use_bias=use_bias, kernel_init=self._kernel_init())(x)  
			if activation_fn is not None:
				x = activation_fn(x)
		#x = e3x.nn.Dense(self.num_features, use_bias=True, kernel_init=jax.nn.initializers.glorot_uniform())(x)  
		x = e3x.nn.Dense(self.num_features, use_bias=True, kernel_init=self._kernel_init())(x)  
		return x



	#@profile
	def _get_properties(self, data, R):
		batch_size = len(data['N'])
		atomic_numbers = data['Z']
		dst_idx = data['dst_idx']
		src_idx = data['src_idx']
		batch_seg = data['batch_seg']
		inputs={}
		embed_inputs_list = self.get_embed_inputs_list()
		if embed_inputs_list is not None:
			for typetoadd in embed_inputs_list:
				inputs[typetoadd]=data[typetoadd] if typetoadd in data.keys() else None
		'''
		print(data['ID'])
		print(data['energybyatom'])
		print('E=',R)
		print('dstx=',dst_idx)
		print('srcx=',src_idx)
		print('batch_seg=',batch_seg)
		sys.exit()
		'''
		offsets = data['offsets']
		displacements = self.get_displacements(R, dst_idx, src_idx, offsets=offsets)
		# Expand displacement vectors in basis functions.
		basis = e3x.nn.basis(  # Shape (num_pairs, 1, (max_basis_degree+1)**2, num_basis_functions).
			displacements,
			num=self.num_basis_functions,
			max_degree=self.max_basis_degree,
			radial_fn=getattr(e3x.nn, self.radial_basis_fn),
			#radial_fn=e3x.nn.reciprocal_gaussian,
			#radial_fn=e3x.nn.basic_gaussian,
			cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff)
		)
		if self.embed_type==0:
			x = self.build_input_embed(atomic_numbers, inputs, basis.dtype)
		else:
			x = self.build_input_element_mod(atomic_numbers, inputs, basis.dtype, activation_fn=None)

		if self.model_type==0:
			nnModel = nnBasicModel(
				num_iterations=self.num_iterations, 
				interaction_type=self.interaction_type, 
				kernel_init=self.kernel_init,
				num_mlps_post_residual=self.num_mlps_post_residual)
		else:
			nnModel = nnFlexibleModel(
				num_iterations=self.num_iterations, 
				num_residuals_atomic = 2,
				num_residuals_interaction = 3,
				num_residuals_output = 1, 
				activation_fn= silu,
				use_bias = True,
				interaction_type=self.interaction_type, 
				kernel_init=self.kernel_init,
				)


		x = nnModel(x,  basis,  dst_idx,  src_idx)

		Na_per_batch = jax.ops.segment_sum(jnp.ones_like(jnp.asarray(data['batch_seg']), dtype=x.dtype), segment_ids=jnp.asarray(data['batch_seg']),num_segments=2*batch_size)[0:2*batch_size:2]

		results = {}
		for name in self.output_names:
			if name!='forces': # forces are derivative of energies along positions
				i = self.output_names.index(name)
				results[name]  = self.get_tensor(data, x, name, self.output_ranks[i], nelements=int(self.output_options[i]))
				if 'byatom' in name:
					results[name]  /= Na_per_batch
		sumenergies = None
		for name in self.output_names: # get sumenergies if energies are already calculated
			if 'energ' in name:
				energies = results[name]
				if 'byatom' in name:
					energies = results[name]*Na_per_batch
				sumenergies = jnp.sum(energies)
				break
		
		if sumenergies is None and 'forces' in self.output_names:# we need sumenergies to compute forces
			name = 'energy'
			for key in data.keys():
				if 'energ' in key:
					name=key
					break
			energies  = self.get_tensor(data, x, name, 0)
			if 'byatom' in name:
				energies = results[name]*Na_per_batch
			sumenergies = jnp.sum(energies)

		return sumenergies, results

	#@profile
	def get_properties(self, data):
		R = data['R'] # here not in _get_properties, needed if forces have to calculated
		if 'forces' in self.output_names:
				energy_and_grads = jax.value_and_grad(self._get_properties, argnums=1, has_aux=True)
				(_,results), grads = energy_and_grads(data, R)
				results['forces']=-grads
				#jax.debug.print('results={}',results)
				return results
		else:
			_ , results = self._get_properties(data,R)
			return results

	@nn.compact
	def __call__(self, data):
		return self.get_properties(data)

	def __repr__(self):
		config =self.get_config()
		str_list=[]
		title=' Model hyperparameters '
		str_list.append('-'*60+title+'-'*60)
		for key in  config.keys():
			str_list.append(
			'{:<25s} = {} '.format(key, config[key])
			)
		str_list.append('-'*(120+len(title)))
		return "\n".join(str_list)

