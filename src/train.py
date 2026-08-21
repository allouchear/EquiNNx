import jax

from Utils.UtilsFunctions import *
from Utils.Train import *
from Utils.DataContainer import *
from Utils.DataProvider import *
from Model.Model import *


config = getArguments()

outputConfig=setOutputConfig(config) 

batching_file = None
if is_batching_file(config.dataset):
	# The dataset is a pre-built batching file : the raw dataset is NOT read. DataProvider
	# reads the split/batching/padding parameters (num_train, num_valid, num_test,
	# batch_size, n_embeded_atoms, seed) directly from the file and overrides config.
	batching_file = config.dataset

if batching_file is not None:
	dataProvider = DataProvider(None, config, batching_file=batching_file, memory_mode=config.data_loading, cache_size=config.data_cache)
	# check if output_names in the batching file
	for key in config.output_names:
		if key not in dataProvider.data_keys:
			sm=''.join(['-']*100)
			st='\nThe output {} required is not in your batching data.\nCheck your input and your batching data file.\n'.format(key)
			st+='In your batching file, you have :\n{}\n'.format(dataProvider.data_keys)
			st = "\n"+sm+st+sm
			raise ValueError(st)
	if outputConfig['train_data_filename'] is not None or outputConfig['valid_data_filename'] is not None:
		print("WARNING : --save_data is ignored when the dataset is a batching file (no raw dataset is read)", flush=True)
else:
	data=DataContainer(config)
	# check if output_names in data
	for key in config.output_names:
		if key not in data.container.keys():
			sm=''.join(['-']*100)
			st='\nThe output {} required is not in your data.\nCheck your input and your data files.\n'.format(key)
			st+='In your data files, you have :\n{}\n'.format(list(data.container.keys()))
			st = "\n"+sm+st+sm
			raise ValueError(st)
		
	dataProvider = DataProvider(data, config)

	if outputConfig['train_data_filename'] is not None:
		dataProvider.save_data(outputConfig['train_data_filename'], dataProvider.idx_train, 'train')
	if outputConfig['valid_data_filename'] is not None:
		dataProvider.save_data(outputConfig['valid_data_filename'], dataProvider.idx_valid, 'validation')

model, restored = create_model(outputConfig, config)
print(model) # print hyperparameters of the model
# Create PRNGKeys.
seed = 0 if config.seed is None or config.seed>-1 else config.seed
data_key, train_key = jax.random.split(jax.random.PRNGKey(seed), 2)
train_key = jax.random.PRNGKey(seed)

params = train_model(key=train_key, model=model, restored=restored, dataProvider=dataProvider, config=config, outputConfig=outputConfig)
