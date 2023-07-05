# Use modern python to run this. Notepadd++ F5 command:
# cmd /k cd /d $(CURRENT_DIRECTORY) && python -i $(FILE_NAME)

## Imports
# Generic stuff
import numpy as np
import pandas as pd
import os
import sys

# Pytorch stuff
import torch
from pathlib import Path

# Behold my stuff
import _pylib.laser as lsr
import _pylib.tools as tl
import _deeplearn.network as dln

# cmd window title
if sys.platform == 'win32':
	os.system('title perror generation')
else:
	twd = './'

STG = { # Settings
	'csv_root':os.path.join(twd,'FE_CSV'),
	'laser_root':os.path.join(twd,'FE_LASER'),
	'pickle_root':os.path.join(twd,'FE_PICKLE_TRN'),			# Directory to read the csv files from
    'data_source':'pk_file',	# csv_file|pk_file
    'save_data':True,	# Bool; whether to save FE as pickle or not
	'time_file':'9_time_perror.log',	# File name for saving all script duration values
	'pulse_files':['B1_1pulse_e0249.pth','B3_7t2mm_e0249.pth'],	# .pth file containing model_state_dict
	'pulse_tags':['B1','B3'],	# used for naming data in vtk files
	'frequency':1,
	'time_range':[],	# Set to [] to skip
	'device_type':'cuda',		# cpu|cuda
	'target_FE_models':[],
	'filter_string':'L4mm',
	# 'filter_string':'_',
	# 'target_FE_models':[
		# 'G01L4mm_8t05mm_M',
		# 'G01L4mm_8t05mm_CS',
		# 'G01L4mm_8t05mm_CE',
		# ],
}
PRC = {	# Precision degrees for rounding values where needed
	'time':8,
	'space':7,
}

# Get cpu or gpu device for training.
if STG['device_type'] == 'cpu':
	device = 'cpu'
else:
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# Initialize time logger
## Script execution
if __name__ == '__main__':

	time_log = tl.TimeLog(STG['time_file'])
	
	# Load NNs
	time_log.start()
	pulse_models = []
	for ii, pfile in enumerate(STG['pulse_files']):
		cur_pnfo = torch.load(pfile)
		pulse_network = dln.NeuralNetPulse(**cur_pnfo['init_pars'])
		pulse_network.load_state_dict(cur_pnfo['model_state_dict'])
		# pulse_network.sf['cutoff'] = True
		pulse_network.device = device
		pulse_network.to(device)
		pulse_network.eval()
		pulse_models.append({
			'nn':pulse_network,
			'tag':STG['pulse_tags'][ii],
		})
	time_log.append('Loaded pulse models')
	
	# Determine what FE models to go through
	if not STG['target_FE_models']:
		for cur_file in os.listdir(STG['pickle_root']):
			segs = cur_file.split('-')[0].split('_')[1:]
			name = '_'.join(segs)
			if STG['filter_string'] in name:
				if name not in STG['target_FE_models']:
					STG['target_FE_models'].append(name)
	
	for fe_name in STG['target_FE_models']:
		time_log.out('Processing %s'%(fe_name))
		time_log.in_right()
		
		# Reading FE Data
		time_log.start()
		model_list = [{
			'name':fe_name,
			'directory':'run_%s'%(fe_name),
			'laser':'1_AM_laser_%s.inp'%(fe_name),
		}]
		model_packs = tl.readModels(
			models_given = model_list, 
			csv_root = STG['csv_root'], 
			pickle_root = STG['pickle_root'],
			source = STG['data_source'],
			save = STG['save_data'],)
		cur_model = model_packs[0]
		time_log.append('Loaded FE data')

		# Prepare pulses in the network for current model
		time_log.start()
		laser_tracker = lsr.LaserTracker(os.path.join(STG['laser_root'],cur_model['laser']))
		for pulse_model in pulse_models:
			p_len = pulse_model['nn'].init_pars['net_arch']['pulse_length']
			pulse_tracker = lsr.PulseTracker(LEO=laser_tracker, pulse_length=p_len, rounding_digits=PRC)
			pulse_events = pulse_tracker.events
			dln.append_transform(pulse_events, device)
			pulse_model['nn'].update_events(pulse_events)
			pulse_model['error_df'] = pd.DataFrame(index=cur_model['nt11'].index.values)
		
		# Look at info about frames
		frame_series = []
		if STG['time_range']:
			for range in STG['time_range']:
				cur_frames = cur_model['frames']
				cur_frames = cur_frames[cur_frames>=range[0]]
				cur_frames = cur_frames[cur_frames<=range[1]]
				frame_series.append(cur_frames)
		else:
			frame_series.append(cur_model['frames'])
		
		frame_series = pd.concat(frame_series)
		for num, frame in enumerate(frame_series):
			if num%STG['frequency'] != 0:
				frame_series.pop(num)
		
		time_log.append('Prepared everything.')
		
		# Get simulation data
		# time_log.in_right()
		nodes_tns = torch.from_numpy(cur_model['coords'].to_numpy()).to(device)
		time_log.start()
		for loop_id, frame_time_og in enumerate(frame_series):
			frame_time_rndd = round(float(frame_time_og),PRC['time'])
			# time_log.out('%sProcessing frame %f (%i/%i)...'%(time_log.in_str(),frame_time_og,loop_id+1,len(frame_series)))
			# time_log.in_right()

			# Prepare input tensor
			# time_log.start()
			time_tns = torch.tensor(()).new_full(size=(len(nodes_tns), 1), fill_value=frame_time_og,device=device)
			nn_input_tns = torch.cat([nodes_tns,time_tns], 1).float().to(device)
			# time_log.append('Prepared network input.')
			
			
			# Evaluating Pulse NN for the current time frame
			for ii, pulse_model in enumerate(pulse_models):
				pulse_tag = STG['pulse_tags'][ii]
				# time_log.start()
				pulse_response_tns = pulse_model['nn'].pulsum_memory(nn_input_tns)
				# pulse_response_tns = pulse_model['nn'].pulsum(nn_input_tns)
				# pulse_response_tns = pulse_model['nn'].pulsum_unique(nn_input_tns)
				# time_log.append('Evaluated the pulse network.')
			
				# Post processing
				# time_log.start()
				pulse_transformer = dln.LabelTransform(pulse_model['nn'].init_pars['net_arch']['scale_type'])
				pulse_ar = pulse_transformer.u2T(pulse_response_tns.cpu().detach().numpy())
				pulse_ar = np.float64(pulse_ar.flatten())
				pulse_error_ar = pulse_ar-cur_model['nt11'][frame_time_og].values
				pulse_error_df = pd.DataFrame(data=pulse_error_ar, index=cur_model['nt11'].index.values)
				pulse_model['error_df'] = pd.concat([pulse_model['error_df'],pulse_error_df],axis=1)
				# time_log.append('Transformed network output to a frame.')
			# time_log.in_left()
		# time_log.in_left()
		time_log.append('Evaluated error for all frames.')
		
		# Save evaluated errors
		time_log.start()
		for pulse_model in pulse_models:
			pulse_model['error_df'].columns = [round(float(tt),PRC['time']) for tt in frame_series]
			pulse_model['error_df'] = -pulse_model['error_df']
			error_pack = {
				'name':'%s_%s'%(cur_model['name'], pulse_model['tag']),
				'directory':'%s_%s'%(cur_model['directory'], pulse_model['tag']),
				'laser':cur_model['laser'],
				'coords':cur_model['coords'],
				'elcons':cur_model['elcons'],
				'nt11':pulse_model['error_df'],
			}
			tl.blosc_save(error_pack, STG['pickle_root'])
		time_log.append('Saved error packs.')
			
		time_log.in_left()
	
	if sys.platform == 'win32':
		import _pylib.notify as ntf
		ntf.balloon_tip('Error','Finished evaluating the errors.')

	time_log.closure()
