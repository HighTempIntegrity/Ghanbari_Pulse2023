# Use modern python to run this. Notepadd++ F5 command:
# cmd /k cd /d $(CURRENT_DIRECTORY) && title NN && python $(FILE_NAME)
# cmd /k cd /d $(CURRENT_DIRECTORY) && python -i $(FILE_NAME)

## Quick links
# Here you can find more information about the packages used here:
# torch.nn # https://pytorch.org/docs/stable/nn.html
# LBFGS singularity https://github.com/pytorch/pytorch/issues/5953


## Imports
# Generic stuff
import os
import sys
import shutil
import numpy as np
import pandas as pd
import math
import pickle
import itertools
import copy
import random
from datetime import datetime
from sklearn.model_selection import train_test_split

# Pytorch stuff
import torch
import torch.optim as optim				# https://pytorch.org/docs/stable/optim.html
from torch.utils.data import DataLoader	# https://pytorch.org/docs/stable/data.html

# Behold my stuff
import _pylib.tools as tl
import _pylib.laser as lsr
import _deeplearn as dl	#.loss|network|utility

## Initializations
STG = { # Settings
	'dict_name':'Script settings',
	'seed':128,				# Random sampling seed
	'plot_theme':'notebook',		# notebook|chic; different sets of RC parameters
	'csv_root':'FE_CSV',			# Directory to read the csv files from
	'pickle_root':'FE_PICKLE_TRN',			# Directory to read the csv files from
	'sample_root':'SMP_PICKLE',			# Directory to read the csv files from
	'laser_root':'FE_LASER',			# Directory to read the csv files from
	'save_FE_data':True,		# Bool; whether to save FE as pickle or not
	'save_sampled_data':True,	# True|False
	'source_FE_data':'pk_file',	# csv_file|pk_file
	'source_sampled_data':'pk_file',	# fe_data|pk_file
	'time_file':'9_time_pulse.log',	# File name for saving all script duration values
	'overview_file':'9_overview.txt',
	'device_type':'cuda',		# cpu|cuda
	'reproducible':True,
	'plot_notebook':'3_jpy_plots.ipynb',	# Template plotter
}
SMP = { # Sampler properties
	'dict_name':'FE Sampler properties',#
	'target_FE_models':['G00L8mm_HXTdep_7t2mm'],
	# 'target_FE_models':['G00L8mm_1pulse'],
	# 'target_FE_models':['G00L8mm_1t2mm'],
	# 'target_FE_models':['G00L8mm_7t2mm'],
	# 'type':'contour_legacy',	# contour|contour_legacy
	'type':'contour',	# contour
	'time_subset':[(1e-6,1e3)],	# range of times to consider in training, [] for full time
	'space_subset':[],
	'LT_range':[], # Layer Track range
	'test_ratio':0.2,
	'max_batch_size':2e6,
	'contour':{
		'levels':[[30]], # list of temperatures
		'aliasing_factor':[1,5],
		'batch_size':[100e3,100e3],
		# 'batch_weight':[[1]*2],
	},
}
ARCH  = { # Network properties
	'dict_name':'Network architecture',
	'model_type':'pulse',	# pulse|geomtery
	'hidden_layers':6,
	'neurons':24,
	'activation':'tanh',	# tanh|relu|lrelu|sigmoid|softplus|celu|swish|sin|gaussian
	'last_act':'softplus',	# softplus|exp|identity
	# 'weight_seed':STG['seed'],				# Random sampling seed for reproducibility
	'weight_seed':128,				# Random sampling seed for reproducibility
	'pulse_length':0.08,			# (mm) Used to determine the number of pulse events
	'scale_type':'HX', # constant|linear|piecewise|HX
	'multi_model':False,	# Whether multiple FE models are used in training, False by default
	'time_filter':{
		'cutoff':True,
		'cutoff_start':-2e-5,
	},
	'space_filter':{
		'x_dis':2,
		'y_dis':2,
		'z_dis':1.5,
		'cutoff':False,
		'cutoff_value':0.1,
	},
}
LSS = { # Loss function properties
	'dict_name':'Loss function properties',
	'loss_type':'MSE',	# L1|MSE/L2|aymMSE|RMSE|L3|L4
	'model_evaluation_type':'internal',	# internal|memory
	'model_type':ARCH['model_type'],	# pulse|geomtery
	'reg_power':2,			# 1|2; regularization degree
	'reg_factor':0,		# 0<float<1; Contribution to loss
	'ysymm':{
		'factor':4,
		'x_bounds':[-2.0,2.0],
		'z_bounds':[-1.0,0.0],
		't_bounds':[ 0.0,1.0],
		'points_num':1e4,
		# 'seed':STG['seed'],
		'seed':128,
	},
}
OPT = { # Optimization parameters
	'dict_name':'Optimizer properties',
	'algorithm':'LBFGS',	# LBFGS; Type of optimizer
	'LBFGS':{
		# 'learning_rate':(.9,.5,.2,.1),		# <1; Step size factor in moving downhill
		'learning_rate':1,		# <1; Step size factor in moving downhill
		'tolerance_grad':1e-5,	# default:1e-5; stop iterations when changes in loss is low
		'max_iter':100,			# int(default:20)
	},
	'drop_last_batch':True,
	'epochs':250,			# Number of epochs
	'batch_num':4,
	'max_loss_jump':5,	# Used in divergence check
	'min_loss_change':1e-8,	# Used in early stopping
	'scheduler_gamma':0.8,
	'patience':6,
	'save_dir_name':'pulse',
}
RST = { # Restart options
	'dict_name':'Restart options',
	'restart_folder':[],# Folder to restart, [] to disable
	# 'restart_folder':['pulse_20230610_Tdep_7t2mm_Ll0.9_res1'],# Folder to restart, [] to disable
	# 'restart_folder':['geom_20230326_102856'],# Folder to restart, [] to disable
	'load_model':True,	# True|False
	'load_optimizer':False,	# True|False
}
PRC = {	# Precision degrees for rounding values where needed
	'dict_name':'Rounding precision',
	'time':8,
	'space':7,
}
GEOM = { # Geometry training info
	'dict_name':'Geometry training info',
	'geom_training':True,	# True|False
	'pulse_tag':'B1',	# B1|B3
	'aliasing_factor':[1,5],
	'template_batch_size':[20e3,5e3,5e3],
	# 'layers_M':[ii*2+1 for ii in range(17)],
	# 'layers_CS':[ii*4+1 for ii in range(9)],
	# 'layers_CE':[1,*[ii*4+3 for ii in range(8)],33],
	'layers_M':[1],
	'layers_CS':[],
	'layers_CE':[],
}
if GEOM['geom_training']: ## Geometry training switch
	STG['save_FE_data'] = True
	STG['save_sampled_data'] = True
	STG['source_FE_data'] = 'pk_file' # csv_file|pk_file
	STG['source_sampled_data'] = 'pk_file' # fe_data|pk_file
	
	# del SMP['contour']['levels']
	SMP['contour']['levels'] = [30,-30]

	ARCH['model_type'] = 'geomtery' # pulse|geomtery
	ARCH['hidden_layers'] = 3
	ARCH['neurons'] = 24
	ARCH['last_act'] = 'identity' # softplus|identity
	
	LSS['model_type'] = ARCH['model_type'] 
	LSS['ysymm']['factor'] = 0
	
	OPT['save_dir_name'] = 'geom'
	OPT['batch_num'] = 4
	
	ARCH['space_filter']['x_dis']=2
	ARCH['space_filter']['y_dis']=2
	ARCH['space_filter']['z_dis']=1
	ARCH['space_filter']['cutoff']=True
	
	SMP['max_batch_size'] = 200e3
	# Definint FE models
	lvl_len = len(GEOM['aliasing_factor'])+1

	LTH = 0.03
	GEOM['model_list'] = []
	for L_num in GEOM['layers_M']:
		cur_name = '%s_8t05mm_M_%s'%('G%02dL4mm'%(L_num),GEOM['pulse_tag'])
		GEOM['model_list'].append({
			'name':cur_name,
			'centre':(0,0,L_num*LTH)
		})
	for L_num in GEOM['layers_CS']:
		cur_name = '%s_8t05mm_CS_%s'%('G%02dL4mm'%(L_num),GEOM['pulse_tag'])
		GEOM['model_list'].append({
			'name':cur_name,
			'centre':(-2+0.5/2,0,L_num*LTH)
		})
	for L_num in GEOM['layers_CE']:
		cur_name = '%s_8t05mm_CE_%s'%('G%02dL4mm'%(L_num),GEOM['pulse_tag'])
		GEOM['model_list'].append({
			'name':cur_name,
			'centre':(2-0.5/2,0,L_num*LTH)
		})
	xd = ARCH['space_filter']['x_dis']
	yd = ARCH['space_filter']['y_dis']
	zd = ARCH['space_filter']['z_dis']
	for model in GEOM['model_list']:
		cur_name = model['name']
		laser_name = '_'.join(cur_name.split('_')[:-1])
		coords = model['centre']
		model['dict_name'] = 'FE model; %s'%(cur_name)
		model['directory'] = 'run_%s'%(cur_name)
		model['laser'] = '1_AM_laser_%s.inp'%(laser_name)
		model['batch_size'] = GEOM['template_batch_size']
		model['batch_weight'] = [1]*lvl_len
		model['space_subset'] = [
			(coords[0]-0.5/2-xd, coords[1]-yd, coords[2]-zd),
			(coords[0]+0.5/2+xd, coords[1]+yd, coords[2]+0.3*LTH)
			]
		del model['centre']


# cmd window title
if sys.platform == 'win32':
	if GEOM['geom_training']:
		os.system('title Geometry training')
	else:
		os.system('title Pulse training')


# Get cpu or gpu device for training.
if STG['device_type'] == 'cpu':
	device = 'cpu'
else:
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using %s device'%(device))

# Assembling of all setting into one "options" variable
options = {
	'settings':STG,
	'sampling':SMP,
	'architecture':ARCH,
	'loss_info':LSS,
	'optimizer':OPT,
	'restart':RST,
	'precision':PRC,
	'geom':GEOM,
}

# List of dicts containing info about target ensemble
variables = []
# Go through option dictionaries
for opt_key in options:
	# Check their keys
	for dict_key in options[opt_key]:
		# for tuples which need separate consideration
		if type(options[opt_key][dict_key]) == tuple:
			variables.append({
				'source':opt_key,
				'key':dict_key,
				'values':options[opt_key][dict_key],
			})
		elif type(options[opt_key][dict_key]) == dict: # Dict in a dict in a dict...
			for sub_key in options[opt_key][dict_key]:
				if type(options[opt_key][dict_key][sub_key]) == tuple:
					variables.append({
						'source':opt_key,
						'key':dict_key,
						'sub_key':sub_key,
						'values':options[opt_key][dict_key][sub_key],
					})

# List of options for multiple runs
runs = []
if variables: # In case of ensemble training this is not empty
	for run_variables in itertools.product(*[var['values'] for var in variables]):
		cur_options = copy.deepcopy(options)
		var_tags = []
		# run_variables is tuple with current iteration variables in order
		for var_id, var_value in enumerate(run_variables):
			# initialization
			var_source = variables[var_id]['source']
			var_key = variables[var_id]['key']
			if 'sub_key' in variables[var_id]:
				var_sub_key = variables[var_id]['sub_key']
				cur_options[var_source][var_key][var_sub_key]=var_value
				var_tags.append('%s%s%s'%(var_key[0],var_sub_key[0],str(var_value)))
			else:
				cur_options[var_source][var_key]=var_value
				# A tag using the first letter of the key and its value is created for folder name
				var_tags.append('%s%s'%(var_key[0],str(var_value)))
			
		runs.append({
			'opts':cur_options,
			'tag':'_'.join(var_tags)
		})
else:	# In case of single run
	runs.append({
		'opts':options,
	})

## Script execution
if __name__ == '__main__':
	time_string = datetime.now().strftime("%Y%m%d_%H%M%S")
	# Prepare output folder
	for run in runs: # run options: ropts

		ropts = run['opts']

		# Set random seed for reproducibility
		if ropts['settings']['reproducible']:
			torch.manual_seed(ropts['settings']['seed'])
			random.seed(ropts['settings']['seed'])
			np.random.seed(ropts['settings']['seed'])
			rng = np.random.default_rng(ropts['settings']['seed'])
			if torch.cuda.is_available():
				torch.backends.cudnn.deterministic = True
				torch.backends.cudnn.benchmark = False	
		else:
			rng = np.random
		
		if ropts['restart']['restart_folder']:
			save_folder = '%s_%s_res'%(ropts['optimizer']['save_dir_name'],time_string)
		else:
			save_folder = '%s_%s'%(ropts['optimizer']['save_dir_name'],time_string)
		if variables: # In case of ensemble training
			save_root = '%s_ens'%(save_folder)
			save_folder = os.path.join(save_root,run['tag'])
			
			with open(STG['plot_notebook'], 'r') as file:
				jpy_contents = file.readlines()
			old_line ='    "twd = os.path.dirname(cwd.__str__())\\n",\n'
			new_line ='    "twd = os.path.dirname(os.path.dirname(cwd.__str__()))\\n",\n'
			search_id = jpy_contents.index(old_line)
			jpy_contents[search_id] = new_line
			jpy_ens_name = '9_jpy_plots_ensemble.ipynb'
			with open(jpy_ens_name,'w') as file:
				for line in jpy_contents:
					file.write(line)
			ropts['settings']['plot_notebook'] = jpy_ens_name
		if not os.path.exists(save_folder):
			os.makedirs(save_folder)
			
		time_log = tl.TimeLog(os.path.join(save_folder,ropts['settings']['time_file']))
		def print_title(title, time_log):
			title = '# %s #'%(title)
			time_log.out('\n'.join(['', '#'*len(title), title, '#'*len(title)]))

		# Write the overview file and make a copt of the plotting notebook
		tl.write_overview([ropts[key] for key in ropts], os.path.join(save_folder,ropts['settings']['overview_file']))
		shutil.copy(ropts['settings']['plot_notebook'],os.path.join(save_folder,'3_plot.ipynb'))
		
		# Initialize time logger
		time_log.start()
		## Data preparation
		print_title('Data Preparation', time_log)

		
		## Loading FE data
		# Pulse training
		if not ropts['geom']['geom_training']:
			fe_names = ropts['sampling']['target_FE_models']
			model_list = []
			lvl_len = len(ropts['sampling']['contour']['aliasing_factor'])+1
			for cur_name in fe_names:
				model_list.append({
				'dict_name':'FE model; %s'%(cur_name),
				'name':cur_name,
				'directory':'run_%s'%(cur_name),
				'laser':'1_AM_laser_%s.inp'%(cur_name),

			})
		# For geomtery training I pre-evalulate the pulse functions
		# so only the perror (pulse error) is loaded here
		else:
			model_list = ropts['geom']['model_list']
			
		# Import FE data
		time_log.start()
		fe_model_packs = tl.readModels(
			models_given = model_list, 
			csv_root = ropts['settings']['csv_root'], 
			pickle_root = ropts['settings']['pickle_root'], 
			source = ropts['settings']['source_FE_data'],
			save = ropts['settings']['save_FE_data'])
		time_log.append('Imported FE model data from %i model(s).'%(len(fe_model_packs)))
		
		
		# Pulse training: Additional info for model packs
		if not ropts['geom']['geom_training']:
			for mpack in fe_model_packs:
				mpack['space_subset'] = ropts['sampling']['space_subset']
				mpack['batch_size']=ropts['sampling']['contour']['batch_size']
				mpack['batch_weight']=[1]*lvl_len		
		

		# Reformatting space and time
		time_log.start()
		time_log.out('Started sampling the temperature distribution')
		time_log.in_right()
		for ii in range(len(fe_model_packs)):
			time_log.out('%s'%(fe_model_packs[ii]['name']))
			time_log.in_right()
			time_log.start()
			sampler_input = {
				'model_pack':fe_model_packs[ii],
				'pickle_root':ropts['settings']['sample_root'], 
				'source':ropts['settings']['source_sampled_data'],
				'save':ropts['settings']['save_sampled_data'],
			}
			fe_model_packs[ii] = dl.utility.full_sampler(**sampler_input)
			time_log.append('Performed full sampling.')
			
			
			if ropts['sampling']['type'] == 'contour':
				time_log.start()
				if 'levels' in ropts['sampling']['contour']:
					cur_levels = ropts['sampling']['contour']['levels']
				else:
					cur_levels = dl.utility.define_levels(
						fe_model_packs[ii]['nt11'],
						ropts['optimizer']['batch_num'],
						fe_model_packs[ii]['batch_size'], 
						ropts['sampling']['contour']['aliasing_factor'])
					# ropts['sampling']['contour']['levels'] = [cur_levels]
				sampler_input = {
					'model_pack':fe_model_packs[ii],
					'levels':cur_levels,
					'space_limits':fe_model_packs[ii]['space_subset'],
					'time_range':ropts['sampling']['time_subset'],
				}
				fe_model_packs[ii] = dl.utility.contour_divider(**sampler_input)
				time_log.append('Separated into contours.')
			elif ropts['sampling']['type'] == 'contour_legacy':
				sampler_input = {
					'model_pack':fe_model_packs[ii],
					'levels':ropts['sampling']['contour']['levels'],
					'time_range':ropts['sampling']['time_subset'],
					'space_limits':fe_model_packs[ii]['space_subset'],
				}
				dl.utility.contour_sampler_new(**sampler_input)
			time_log.in_left()
		time_log.in_left()
		time_log.append('Generated samples and labels from FE data.')
		
		
		# Lighten the batches
		time_log.start()
		for ii in range(len(fe_model_packs)):
			fe_model_packs[ii] = dl.utility.lighten_samples(fe_model_packs[ii], rng, ropts['sampling']['max_batch_size'])
		time_log.append('Trimmed batches to %d'%(ropts['sampling']['max_batch_size']))
		
		# Batch info table
		width_left = 24
		width_mid = 16
		nlvl = lvl_len
		total_points = 0
		header = ['   {0: <{1}}'.format('model', width_left)]
		header_items = ['{0: >{1}}'.format('total %i'%(ii+1), width_mid) for ii in range(nlvl)]
		header.extend(header_items)
		header = ' | '.join(header)
		time_log.out('   '+'-'*(len(header)-3))					 
		time_log.out(header)
		time_log.out('   '+'-'*(len(header)-3))					 
		for mpack in fe_model_packs:
			row = ['   {0: <{1}}'.format(mpack['name'], width_left)]
			for lvl in mpack['data']:
				cur_points_len = len(mpack['data'][lvl]['label_ar'])
				total_points += cur_points_len
				lvl_nfo = '{0}: {1:,}'.format(lvl,cur_points_len)
				row.append('{0: >{1}}'.format(lvl_nfo, width_mid))
			row = ' | '.join(row)
			time_log.out(row)
		time_log.out('   '+'-'*(len(header)-3))
		time_log.out('There are {0:,} points in memory in total.'.format(total_points))
		time_log.out('')		


		# Assign batch size
		for ii , mpack in enumerate(fe_model_packs):
			for jj, lvl in enumerate(mpack['data']):
				mpack['data'][lvl]['batch_size'] = int(mpack['batch_size'][jj])
				mpack['data'][lvl]['batch_weight'] = int(mpack['batch_weight'][jj])

		# Scale the data
		time_log.start()
		
		if   ropts['architecture']['model_type'] == 'pulse':
			label_transformer = dl.network.LabelTransform(ropts['architecture']['scale_type'])
		elif ropts['architecture']['model_type'] == 'geomtery':
			if ropts['architecture']['last_act'] != 'identity': # Filter negatives
				for mpack in fe_model_packs:
					for lvl in mpack['data']:
						mpack['data'][lvl]['label_ar'][mpack['data'][lvl]['label_ar']<0]=0
			label_transformer = dl.network.LabelTransform(ropts['architecture']['scale_type'], ref_temp = 0)
		
		for mpack in fe_model_packs:
			for lvl in mpack['data']:
				mpack['data'][lvl]['label_ar'] = label_transformer.T2u(mpack['data'][lvl]['label_ar'])
		
		time_log.append('Scaled the data.')

		## Dividing into train and test sets
		time_log.start()
		
		# Determine test size
		for mpack in fe_model_packs:
			for ii, lvl in enumerate(mpack['data']):
				sample_len = len(mpack['data'][lvl]['sampl_ar'])
				batch_len = mpack['data'][lvl]['batch_size']
				mpack['data'][lvl]['test_ratio'] = min(batch_len/sample_len,
					ropts['sampling']['test_ratio'])
					
		
		# Make tensors of samples and labels
		for mpack in fe_model_packs:
			for lvl in mpack['data']:
				mpack['data'][lvl]['sampl_tns'] = torch.tensor(mpack['data'][lvl]['sampl_ar'], device=device)
				mpack['data'][lvl]['label_tns'] = torch.tensor(mpack['data'][lvl]['label_ar'], device=device)
		
		# Splitting the data based on test size
		for mpack in fe_model_packs:
			for lvl in mpack['data']:
				cur_lvl = mpack['data'][lvl]
				ss = cur_lvl['sampl_tns']
				ll = cur_lvl['label_tns']
				tr = cur_lvl['test_ratio']
				cur_lvl['x_train'], cur_lvl['x_test'], cur_lvl['y_train'], cur_lvl['y_test']=\
				train_test_split( ss, ll, test_size=tr, random_state=1)
				cur_lvl['len_train'] = len(cur_lvl['x_train'])
				cur_lvl['len_test']  = len(cur_lvl['x_test'])
		
		# Combine test sets
		test_dict = {'x_test':[],'y_test':[],'subsets':[]}
		for mpack in fe_model_packs:
			cur_subsets = [0]
			for ii, lvl in enumerate(mpack['data']):
				cur_lvl = mpack['data'][lvl]
				if ii == 0:
					test_set = {
						'x_test':cur_lvl['x_test'],
						'y_test':cur_lvl['y_test'],
						# 'set_lengths':[int(cur_lvl['len_test'])],
					}
				else:
					test_set['x_test'] = torch.cat((test_set['x_test'],
						cur_lvl['x_test']),0)
					test_set['y_test'] = torch.cat((test_set['y_test'],
						cur_lvl['y_test']),0)
					# test_set['set_lengths'].append(int(cur_lvl['len_test']))
				cur_subsets.append(cur_subsets[-1]+int(cur_lvl['len_test']))
					
			test_dict['x_test'].append(test_set['x_test'])
			test_dict['y_test'].append(test_set['y_test'])
			test_dict['subsets'].append(cur_subsets)
			
		time_log.append('Set aside %i data for testing.'%(len(test_set['x_test'])))
		# test size table
		width_left = 24
		width_mid = 16
		header = ['   {0: <{1}}'.format('model', width_left)]
		header_items = ['{0: >{1}}'.format('total %i'%(ii+1), width_mid) for ii in range(nlvl)]
		header.extend(header_items)
		header = ' | '.join(header)
		time_log.out('   '+'-'*(len(header)-3))					 
		time_log.out(header)
		time_log.out('   '+'-'*(len(header)-3))					 
		for mpack in fe_model_packs:
			row = ['   {0: <{1}}'.format(mpack['name'], width_left)]
			for lvl in mpack['data']:
				lvl_nfo = '{0}: {1:,}'.format(lvl,mpack['data'][lvl]['len_test'])
				row.append('{0: >{1}}'.format(lvl_nfo, width_mid))
			row = ' | '.join(row)
			time_log.out(row)
		time_log.out('   '+'-'*(len(header)-3))					 
		time_log.out('')

		
		
		time_log.start()
		# If the first batch is too small, can't drop the only batch
		for mpack in fe_model_packs:
			for ii, lvl in enumerate(mpack['data']):
				cur_lvl = mpack['data'][lvl]
				if ii == 0:
					if cur_lvl['batch_size']>cur_lvl['len_train']:
						ropts['optimizer']['drop_last_batch'] = False
					
		# Wrap the data for training
		training_packs = []
		for mpack in fe_model_packs:
			loaders = []
			loader_lens = []
			loader_weights = []
			for ii, lvl in enumerate(mpack['data']):
				cur_lvl = mpack['data'][lvl]
				cur_loader = DataLoader(torch.utils.data.TensorDataset(cur_lvl['x_train'],cur_lvl['y_train']),
					batch_size = cur_lvl['batch_size'],	shuffle=True,
					drop_last=ropts['optimizer']['drop_last_batch'])
				loaders.append(cur_loader)
				
				cur_loader_len = min(cur_lvl['len_train'],cur_lvl['batch_size'])
				loader_lens.append(cur_loader_len)
				loader_weights.append(cur_lvl['batch_weight'])
			
			training_packs.append({
				'data_loaders':loaders,
				'loader_lens':loader_lens,
				'loader_weights':loader_weights,
				'batch_num':min(len(loaders[0]),ropts['optimizer']['batch_num']),
				})
		# Update the batch number for optimizer based on the first dataset
		ropts['optimizer']['batch_num'] = training_packs[0]['batch_num']
		
		# Add subset lengths to loss properties
		ropts['loss_info']['subsets'] = []
		ropts['loss_info']['weights'] = []
		for tp in training_packs:
			subsets = [0]
			weights = []
			for ii, loader_len in enumerate(tp['loader_lens']):
				subsets.append(subsets[-1]+loader_len)
				weights.append(tp['loader_weights'][ii])
			ropts['loss_info']['subsets'].append(subsets)
			ropts['loss_info']['weights'].append(weights)
			
		
		time_log.append('Primary set has %i batches of %i.'%(training_packs[0]['batch_num'],training_packs[0]['loader_lens'][0]))
		time_log.out('First %i batches will be used for training.'%(ropts['optimizer']['batch_num']))
		
		time_log.start()
		## Define events
		event_packs = []
		for mpack in fe_model_packs:
			laser_tracker = lsr.LaserTracker(os.path.join(STG['laser_root'],mpack['laser']))
			pulse_tracker = lsr.PulseTracker(LEO=laser_tracker, pulse_length=ropts['architecture']['pulse_length'], rounding_digits=ropts['precision'])
			dl.network.append_transform(pulse_tracker.events, device)
			event_packs.append(pulse_tracker.events)
		time_log.append('Defined %i events based on laser data.'%(len(event_packs[0])))
		
		## Calculate computation cost
		width_left = 24
		width_mid = 10
		header = '   {0: <{1}} | {2: >{3}} | {4: >{3}} | {5: >{3}}'.format(
			'model', width_left, 'length', width_mid, 'events','total')
		time_log.out('')					 
		time_log.out('Evaluation costs:')					 
		time_log.out('   '+'-'*(len(header)-3))					 
		time_log.out(header)
		time_log.out('   '+'-'*(len(header)-3))					 
		total_cost = 0
		for ii in range(len(fe_model_packs)):
			mname = fe_model_packs[ii]['name']
			mlen = sum(training_packs[ii]['loader_lens'])
			mevents = len(event_packs[ii])
			mcost = mevents*mlen
			total_cost+=mcost
			# time_log.out('%15s | %10d | %10d | %10d'%(
				# mname, mlen, mevents, mcost))
			time_log.out('   {0: <{1}} | {2:{3},} | {4:{3},} | {5:{3},}'.format(
				mname, width_left, mlen, width_mid, mevents, mcost))
		time_log.out('   '+'-'*(len(header)-3))					 
		time_log.out('Each iteration involves {0:,} evaluations.'.format(total_cost))
		time_log.out('')					 
		
		
		## Network definition
		print_title('Network Definition', time_log)
		
		# Decide if it's a case of multi-model training
		if len(fe_model_packs)>1:
			ropts['architecture']['multi_model'] = True
		else:
			event_packs = event_packs[0]
			# test_packs = test_packs[0]
			# training_packs = training_packs[0]
			# fe_model_packs = fe_model_packs[0]

		time_log.start()
		# Passing the inputs for the model as dictionary for easy saving later
		init_pars = {
			'Din':cur_lvl['x_train'].shape[1],
			'Dout':cur_lvl['y_train'].shape[1],
			'device':device,
			'net_arch':ropts['architecture'],
			'events':event_packs,
		}
		if ropts['geom']['geom_training']: # Geometry switch
			nn_model = dl.network.NeuralNetGeometry(**init_pars)
		else:
			nn_model = dl.network.NeuralNetPulse(**init_pars)
		nn_model.to(device)

		## Initialize weights and biases
		# Random Seed for weight initialization
		# retrain_seed = 128
		# Xavier weight initialization
		dl.utility.init_xavier(nn_model, ropts['architecture']['weight_seed'])
		time_log.append('Prepared network.')
		
		## Choose optimizer
		time_log.start()
		if ropts['optimizer']['algorithm'] == 'LBFGS':
			init_lbfgs = { #https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#torch.optim.LBFGS
				'params':nn_model.parameters(),
				'lr':ropts['optimizer']['LBFGS']['learning_rate'],
				'max_iter':ropts['optimizer']['LBFGS']['max_iter'],
				# 'max_eval':None,
				'tolerance_grad':ropts['optimizer']['LBFGS']['tolerance_grad'],
				'tolerance_change':1.0 * np.finfo(float).eps,
				# 'history_size':10, # 100 default
				# 'line_search_fn':'strong_wolfe', # default: None|'strong_wolfe'
			}
			nn_model.optimizer = optim.LBFGS(**init_lbfgs)
		elif ropts['optimizer']['algorithm'] == 'ADAM':
			init_adam = { #https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
				'params':nn_model.parameters(),
				'lr':ropts['optimizer']['ADAM']['learning_rate'],	# default: 0.001
				'betas':(0.9, 0.999),	# default: (0.9, 0.999)
				'eps':1e-08,		# default: 1e-08
				'weight_decay':0,	# default: 0
				'amsgrad':False,	# default: False
				# 'maximize':False,	# default: False
			}
			nn_model.optimizer = optim.Adam(**init_adam)
		else:
			raise ValueError('Optimizer not recognized!')
		time_log.append('Prepared optimizer.')
		
		## Loading old training data
		if ropts['restart']['restart_folder']:
			last_epoch = sorted([f for f in os.listdir(ropts['restart']['restart_folder'][0]) if (f.split('.')[-1]=='pth')])[-1]
			ropts['restart']['checkpoint'] = torch.load(os.path.join(ropts['restart']['restart_folder'][0],last_epoch))
		else:
			ropts['restart']['checkpoint'] = {}
		
		time_log.out('')
		time_log.append('Finished pre-processing.')
		time_log.out('')
		
		## Train the model
		print_title('Training', time_log)
		time_log.start()
		train_input_dict = {
			'model':nn_model,
			'training_packs':training_packs,
			'test_dict':test_dict,
			'opt_prop':ropts['optimizer'],
			'loss_prop':ropts['loss_info'],
			'res_prop':ropts['restart'],
			'time_log':time_log,
			'save_folder':save_folder,
			'fe_packs':fe_model_packs,
		}
		epoch_history = dl.loss.train(**train_input_dict)
		time_log.append('Finished training.')

		time_log.closure()
		
		## Save performance metrics in case of ensemble training
		if variables:
			result_file = os.path.join(save_root,'results_overview.csv')
			run_dict = {
				'tag':run['tag'],
				'loss':epoch_history[-1]['loss'],
				}
			for key in epoch_history[-1]['test']:
				if key != 'loss':
					run_dict[key] = epoch_history[-1]['test'][key]
				else:
					for sub_key in epoch_history[-1]['test'][key]:
						run_dict[sub_key] = epoch_history[-1]['test'][key][sub_key]
			run_df = pd.DataFrame([run_dict])
			run_df.index.name='id'
			if not os.path.exists(result_file):
				run_df.to_csv(result_file)
			else:
				old_df = pd.read_csv(result_file, index_col='id')
				old_df = pd.concat([old_df,run_df], ignore_index=True)
				old_df.index.name='id'
				old_df.to_csv(result_file)
		
	if sys.platform == 'win32':
		import _pylib.notify as ntf
		ntf.balloon_tip('Network trained!','Finished training the network.')