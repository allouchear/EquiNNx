import numpy as np


class ValueAccumulator:
	"""Collect per-batch predict/target arrays on host, single concat at end.

	Replaces the old add_values() that did jnp.concatenate every batch
	(O(N^2) device memory). This keeps everything in host numpy and
	concatenates only once per epoch via to_dict().
	"""

	def __init__(self):
		self._lists = None

	def add_batch(self, results, data):
		"""Append one batch's results/targets + metadata (ID, N, R, Z)."""
		n = int(np.sum(np.asarray(data['N'])))
		if self._lists is None:
			self._lists = {}
			for key in results.keys():
				if 'forces' in key:
					self._lists[key+'_predict'] = [np.asarray(results[key][:n])]
					self._lists[key+'_target'] = [np.asarray(data[key][:n])]
				else:
					self._lists[key+'_predict'] = [np.asarray(results[key])]
					self._lists[key+'_target'] = [np.asarray(data[key])]
			self._lists['ID'] = [np.asarray(data['ID'])]
			self._lists['N'] = [np.asarray(data['N'])]
			self._lists['R'] = [np.asarray(data['R'][:n])]
			self._lists['Z'] = [np.asarray(data['Z'][:n])]
		else:
			for key in results.keys():
				if 'forces' in key:
					self._lists[key+'_predict'].append(np.asarray(results[key][:n]))
					self._lists[key+'_target'].append(np.asarray(data[key][:n]))
				else:
					self._lists[key+'_predict'].append(np.asarray(results[key]))
					self._lists[key+'_target'].append(np.asarray(data[key]))
			self._lists['ID'].append(np.asarray(data['ID']))
			self._lists['N'].append(np.asarray(data['N']))
			self._lists['R'].append(np.asarray(data['R'][:n]))
			self._lists['Z'].append(np.asarray(data['Z'][:n]))

	def to_dict(self):
		"""Single concatenation, returns dict compatible with getProps/np.savez."""
		return {k: np.concatenate(v, axis=0) for k, v in self._lists.items()}

	def reset(self):
		self._lists = None

	@property
	def is_empty(self):
		return self._lists is None
