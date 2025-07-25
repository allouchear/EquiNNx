import jax

from Utils.UtilsFunctions import *
from Utils.Train import *
from Utils.DataContainer import *
from Utils.DataProvider import *
from Model.Model import *


config = getArguments()

outputConfig=setOutputConfig(config) 
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
