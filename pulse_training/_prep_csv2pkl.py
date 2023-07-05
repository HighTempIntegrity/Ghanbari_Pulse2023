import os
import pickle
import _pylib.tools as tl

## Initializations
STG = { # Settings
	'csv_root':'FE_CSV',			# Directory to read the csv files from
	'pickle_root':'FE_PICKLE_TRN',			# Directory to read the pickle files from
	'target_models':[],	
	# 'target_models':['run_G00L8mm_1pulse'],	
	# 'target_models':['run_G00L8mm_15t8mm'],	
	'time_file':'9_time_pickler.log',	# File name for saving all timing values
}

## Script execution
if __name__ == '__main__':
	time_log = tl.TimeLog(STG['time_file'])
	
	# List models in CSV folder
	if not STG['target_models']:
		for cur_dir in os.listdir(STG['csv_root']):
			STG['target_models'].append(cur_dir)

	for cur_dir in STG['target_models']:
		time_log.start()
		cur_segments = cur_dir.split('_')
		if 'perror' in cur_segments:
			cur_segments.remove('perror')
		cur_name = '_'.join(cur_segments[1:])
		model_list = [{
		'name':cur_name,
		'directory':cur_dir,
		'laser':'1_AM_laser_%s.inp'%(cur_name),
		}]
		model_packs = tl.readModels(
			models_given = model_list, 
			csv_root = STG['csv_root'], 
			pickle_root = STG['pickle_root'], 
			source = 'pk_file',
			save = True)
		time_log.append('Pickled %s'%(cur_dir))

	time_log.closure()
	