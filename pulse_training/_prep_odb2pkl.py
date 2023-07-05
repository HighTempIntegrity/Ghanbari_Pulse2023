import os
import pickle
import _pylib.tools as tl

## Initializations
STG = { # Settings
	'odb_root':'FE_ODB',			# Directory to read the odb files from
	'csv_root':'FE_CSV',			# Directory to read the csv files from
	'pickle_root':'FE_PICKLE_TRN',			# Directory to read the pickle files from
	'target_models':[],	
	# 'target_models':['run_G00L8mm_7t2mm'],	
	'odb_filter':'_',
	'csv_filter':'_',
	'time_file':'9_time_odb2pkl.log',	# File name for saving all timing values
}

## Script execution
if __name__ == '__main__':
	time_log = tl.TimeLog(STG['time_file'])
	
	# Filter models in ODB folder
	if STG['odb_filter'] != '_':
		for cur_dir in os.listdir(STG['odb_root']):
			if STG['odb_filter'] in cur_dir:
				STG['target_models'].append(os.path.splitext(cur_dir)[0])

	# Extract ODB data into CSV files
	time_log.start()
	abqpy_script = '_abqprep_odb2csv.py'
	if STG['target_models']:
		command_call = ' '.join([abqpy_script, *STG['target_models']])
	else:
		command_call = abqpy_script
	tl.runAbaqusPython(command_call)
	time_log.append('Finished ODB to CSV')
	
	
	# Filter models in CSV folder
	if STG['csv_filter'] != '_':
		for cur_dir in os.listdir(STG['csv_root']):
			if STG['csv_filter'] in cur_dir:
				STG['target_models'].append(cur_dir)
				
	# List models in CSV folder
	if not STG['target_models']:
		for cur_dir in os.listdir(STG['csv_root']):
			STG['target_models'].append(cur_dir)
	
	# Transform CSV data into pickle files
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
	