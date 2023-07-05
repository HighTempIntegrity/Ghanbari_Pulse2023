# Use Abaqus python to run this. Notepadd++ F5 command:
# cmd /k cd /d $(CURRENT_DIRECTORY) && title odbProc && abaqus python $(FULL_CURRENT_PATH)

from odbAccess import *			# Package from Abaqus
from abaqusConstants import *	# Package from Abaqus
import os	# For interatcing with listing and deleting files such as .lck
import sys
import csv	# For writing csv files
from time import time	# For tracking and printing time during running the script

import _pylib.abq_laser as lsr


## User-assigned parameters
STG = {
	'odb_root':'FE_ODB',	# Directory where the odb files are copied
	'csv_root':'FE_CSV',	# Directory where the processed csv files are stored
	'sep':'-',			# Character used in csv file names
	'overwrite':False,	# Whether to overwrite old CSV folders or not
	'export_frames':False,	# Write the frame info to a CSV file
	'keys':['NT11'],	# List of variables to save from ODB
	'time_file':'9_time_odbProc_sub.log',
	'frame_file':'3_frames.csv',
	'odb_targets':[], # No ODB extension!
	# 'laser_file':['1_AM_laser_40L_3tracks.inp'],
	'laser_file':[],
	'ins_ID':[0], # Instance index among abaqus assemblies
	'time_shift':0,
	'id_len':4,				# Number of characters to fill with leading zeros
	'save_style':'step_wise',	# step_wise|LTr_based
}
FIL = {	# Filters to save specific data
	'interval':0,			# Time interval to save the frame data; 0 to record all frames
	'time_range':[],		# List of float tuples
	# 'time_range':[(51.151,100)],		# List of float tuples
	'LTr_opts':{
		'type':None,	#'sections'|'range'|None
		'Layer_id':0,
		'ratio':0.80,
		'Tr_range':({'T':0,'r':0.00},{'T':0,'r':0.10}),
		'track_count':16,
		'skip_mid_steps':False,
		'starting_track_id':160,
	},
	'frames_time1':[],		# List of floats; values of time1 frames to save
	'frames_time2':[],	# List of floats; values of time2 frames to save
}
PRC = {	# Precision degrees for rounding values where needed
	'time':7,
	'space':7,
	'comp':5
}
IN_S = ' > '
in_i = 0 # For controlling the level of indentation in the IO window

## Tools for logging time
def timeWrite(message):
	print(message)
	with open(STG['time_file'], 'w') as time_log:
		time_log.write(message+'\n')

def timeAppend(message):
	print(message)
	with open(STG['time_file'], 'a') as time_log:
		time_log.write(message+'\n')

def export_fieldData(fieldData, file_name):
	for output_key in fieldData:
		# Tranpose data to have [point_ID][frame_ID] indices
		# (The purpose of saving all the data into lists is to do the following operation)
		fieldData[output_key]['data']= map(list, zip(*fieldData[output_key]['data']))
		with open(os.path.join(cur_csv_dir, file_name.format(key=output_key)), mode='wb') as output_file:
			csv_writer = csv.writer(output_file, delimiter=',')
			csv_writer.writerow(['label']+fieldData[output_key]['time'])
			csv_writer.writerows(fieldData[output_key]['data'])

def znum(id):
	return str(id).zfill(STG['id_len'])

# Get the laser tracker for finding the right time values
if STG['laser_file']:
	laser_tracker = lsr.LaserTracker(STG['laser_file'][0])

# Get the time_range based on LT ids
if FIL['LTr_opts']['type']=='range':
	FIL['LTr_opts']['idx'] = (
		{'L':FIL['LTr_opts']['Layer_id'],'T':FIL['LTr_opts']['Tr_range'][0]['T'], 'r':FIL['LTr_opts']['Tr_range'][0]['r']},
		{'L':FIL['LTr_opts']['Layer_id'],'T':FIL['LTr_opts']['Tr_range'][1]['T'], 'r':FIL['LTr_opts']['Tr_range'][1]['r']})
	FIL['time_range'] = [(laser_tracker.LTr_to_time2(**FIL['LTr_opts']['idx'][0]),
		laser_tracker.LTr_to_time2(**FIL['LTr_opts']['idx'][1]))]
		
	LT_start = 'L%iT%ir%.2f'%(FIL['LTr_opts']['idx'][0]['L'],FIL['LTr_opts']['idx'][0]['T'],FIL['LTr_opts']['idx'][0]['r'])
	LT_end = 'L%iT%ir%.2f'%(FIL['LTr_opts']['idx'][1]['L'],FIL['LTr_opts']['idx'][1]['T'],FIL['LTr_opts']['idx'][1]['r'])
	LTr_name = '_'.join([LT_start, LT_end])
	STG['save_style'] = 'LTr_based'

if FIL['LTr_opts']['type']=='sections':
	LTr_name = 'Lr%i'%(FIL['LTr_opts']['Layer_id'])
	STG['save_style'] = 'LTr_based'

## Script execution
if __name__ == '__main__':

	time_alpha = time() # The timestamp for the beginning
	
	# Check to see if a list of ODBs has been passed in the call
	if len(sys.argv)>1:
		STG['odb_targets'] = sys.argv[1:]
		STG['overwrite'] = False
	
	# Find all odb files in current directory
	odb_files = []	# Contains ODB names as strings
	if STG['odb_targets']:
		odb_files = [os.path.join(STG['odb_root'],file_name+'.odb') for file_name in STG['odb_targets']]
		# odb_files.append(os.path.join(STG['odb_root'],STG['odb_targets'][0]))
	else:	
		for file in os.listdir(STG['odb_root']):
			if file.endswith('.odb'):
				odb_files.append(os.path.join(STG['odb_root'],file))

	# Post-processing every odb in current directory
	in_i += 1
	for cur_odb_file in odb_files:
		cur_name = os.path.basename(cur_odb_file).split('.')[0]
		cur_odb_path = os.path.splitext(cur_odb_file)[0]
		cur_csv_dir = os.path.join(STG['csv_root'],cur_name)
		
		if (not STG['overwrite']) and os.path.exists(cur_csv_dir):
			print('%s already processed'%(cur_odb_file))
			continue
		
		print('%sProcessing %s'%(IN_S*(in_i-1),cur_name))
		
		# Delete .lck files if they exist
		lck_was = False
		if os.path.isfile(cur_odb_path+'.lck'):
			os.remove(cur_odb_path+'.lck')
			lck_was=True
		if lck_was:
			print('%sRemoved *.lck files'%(IN_S*in_i))
		
		# Create a directory for CSV files of current simulation
		if not os.path.exists(cur_csv_dir):
			os.makedirs(cur_csv_dir)

		# Open ODB
		time_start=time()
		odb = openOdb(path=cur_odb_file)
		timeWrite('%sOpened ODB - %.2f sec'%(IN_S*in_i,time()-time_start))
		in_i += 1	# Going through instances
		for ins_ID in STG['ins_ID']:
			ins_name, ins = odb.rootAssembly.instances.items()[ins_ID]
			print('%sProcessing instance %s...'%(IN_S*in_i, ins_name))
			
					
			# Write2CSV available frames
			step_start_time2 = 0
			if STG['export_frames']:
				with open(STG['frame_file'], 'wb') as file:
					csv_writer = csv.writer(file, delimiter=',')
					csv_writer.writerow(['Kstep','Kinc','Time1','Time2'])
					for stepkey in odb.steps.keys():
						step = odb.steps[stepkey]
						for frameID in range(len(step.frames)):
							# Frame processing
							frame = step.frames[frameID] # get Abaqus frame object
							frame_time = round(frame.frameValue,PRC['time'])
							cur_time2 = frame_time + step_start_time2
							# last frame of current step
							if frameID == len(step.frames)-1:
								step_start_time2 = cur_time2
							csv_writer.writerow([step.number,frame.frameId,frame_time,cur_time2])
				print('%sFrame info exported to CSV.'%(IN_S*in_i))

			## Mesh information
			# Write node coordinates
			time_start=time()
			nodeCoord_name = STG['sep'].join([cur_name, 'i%03d'%(ins_ID), 'node', 'coords.csv'])
			nodeCoord_path = os.path.join(cur_csv_dir, nodeCoord_name)
			if os.path.exists(nodeCoord_path):
				timeAppend('%sNode coord already available - %.2f sec'%(IN_S*in_i,time()-time_start))
			else:
				print('%sProcessing node coordinates...'%(IN_S*in_i))
				with open(nodeCoord_path, mode='wb') as output_file:
					csv_writer = csv.writer(output_file, delimiter=',')
					head_row = ['label','x','y','z']
					csv_writer.writerow(head_row)
					for node_ID in range(len(ins.nodes)):
						cur_coords = [round(ins.nodes[node_ID].coordinates[0].item(),PRC['space']),
									  round(ins.nodes[node_ID].coordinates[1].item(),PRC['space']),
									  round(ins.nodes[node_ID].coordinates[2].item(),PRC['space'])]
						cur_row = [ins.nodes[node_ID].label]+cur_coords
						csv_writer.writerow(cur_row)
				timeAppend('%sWrote node coord file - %.2f sec'%(IN_S*in_i,time()-time_start))
			
			# Write element connectivity file
			time_start = time()
			elCon_name = STG['sep'].join([cur_name, 'i%03d'%(ins_ID), 'element', 'cons.csv'])
			elCon_path = os.path.join(cur_csv_dir, elCon_name)
			if os.path.exists(elCon_path):
				timeAppend('%sElement connectivity already available - %.2f sec'%(IN_S*in_i,time()-time_start))
			else:
				with open(elCon_path, mode='wb') as output_file:
					csv_writer = csv.writer(output_file, delimiter=',')
					head_row = ['label','connection']
					csv_writer.writerow(head_row)
					for el_ID in range(len(ins.elements)):
						cur_row = [ins.elements[el_ID].label,ins.elements[el_ID].connectivity]
						csv_writer.writerow(cur_row)
				timeAppend('%sWrote element connectivity file - %.2f sec'%(IN_S*in_i,time()-time_start))
			
			## Simulation information
			
			fieldData = {}
			in_i += 1 # Iterate over steps in the instance
			step_start_time2 = 0	# Starting time of current step, gets updated
			
			# if STG['laser_file']:
				# step_start_time2 = laser_tracker.LTr_to_time2(FIL['LTr_opts']['Layer_id'])
			
			# Set list of searched steps
			if FIL['LTr_opts']['type']=='range':
				track_start = laser_tracker.LTr_to_trackID(**FIL['LTr_opts']['idx'][0])
				track_end = laser_tracker.LTr_to_trackID(**FIL['LTr_opts']['idx'][1])
				step_keys = ['Step_T%s'%(znum(id+FIL['LTr_opts']['starting_track_id'])) for id in range(track_start,track_end+1)]
				step_start_time2 = laser_tracker.LTr_to_time2(FIL['LTr_opts']['idx'][0]['L'],FIL['LTr_opts']['idx'][0]['T'])
			else:
				step_keys = odb.steps.keys()
			
			# step_start_time2 = 0	# Starting time of current step, gets updated
			# step_keys = odb.steps.keys()
			
			if FIL['LTr_opts']['type']=='sections':
				for T_id in range(FIL['LTr_opts']['track_count']):
					cur_LTr = {
						'L':FIL['LTr_opts']['Layer_id'],
						'T':T_id,
						'r':FIL['LTr_opts']['ratio'],
					}
					cur_time2 = laser_tracker.LTr_to_time2(**cur_LTr)
					FIL['frames_time2'].append(round(cur_time2,PRC['time']))
				
				if FIL['LTr_opts']['skip_mid_steps']:
					track_start = laser_tracker.LTr_to_trackID(FIL['LTr_opts']['Layer_id'],0)
					track_end = laser_tracker.LTr_to_trackID(FIL['LTr_opts']['Layer_id'],FIL['LTr_opts']['track_count']-1)
					step_keys = ['Step_T%s'%(znum(id+FIL['LTr_opts']['starting_track_id'])) for id in range(track_start,track_end+1)]
			
			step_start_time2+=STG['time_shift']
			
			for stepkey in step_keys:
				step = odb.steps[stepkey]
				
				print('%sIn %s...'%(IN_S*in_i, stepkey))
				
				# Dictionary of dictionries for parameters that have been saved in the simulations
				# Structure is {{dict NT11},{dict FV1}}
				# Each sub-dictionary contains:
				#	'data' : 	list of data sorted by frames
				#   'time' :    list of frame times
				time_start = time()
				
				
				
				in_i += 1	# For each frame in the step
				for frameID in range(len(step.frames)):
					# Frame processing
					frame = step.frames[frameID] # get Abaqus frame object
					frame_time1 = round(frame.frameValue,PRC['time'])
					frame_time2 = frame_time1 + step_start_time2
					# last frame of current step
					if frameID == len(step.frames)-1:
						step_start_time2 = frame_time2
					
					## Frame filters
					# If an interval is defined
					if FIL['interval'] > 0: 
						# If current frame time is not a multiple of given interval
						if (round(frame.frameValue/FIL['interval'],PRC['comp']))%1!=0:
							continue	
					
					# In case a range is defined, check the current frame
					if FIL['time_range']:
						out_of_range = False
						for cur_range in FIL['time_range']:
							if frame_time2 >= cur_range[0] and frame_time2 <= cur_range[1]:		
								out_of_range = False
								break
							else:
								out_of_range = True
						if out_of_range:
							continue
					
					# If we are given some frame times based on step time		
					if FIL['frames_time1']:	
						if round(frame.frameValue,PRC['time']) not in FIL['frames_time1']:
							continue
					
					# If we are given some frame times based on total time		
					if FIL['frames_time2']:
						if round(frame_time2,PRC['time']) not in FIL['frames_time2']:
							continue
					
					
					## Frame data extraction
					print('%sStep %s | Frame %f - Time2: %f'%(IN_S*in_i,stepkey,frame_time1,frame_time2))
					frame_time1 = round(frame_time1,PRC['time'])
					
					# Checking the variables that were saved during the simulation
					for output_key in frame.fieldOutputs.keys():
						if output_key not in STG['keys']: # Skip unwanted keys
							continue
						data_list = [value.data for value in frame.fieldOutputs[output_key].values if value.instance.name == ins_name]
						if output_key not in fieldData: # See if the key is already in fieldData
							print('%sProcessing %s'%(IN_S*in_i,output_key))
							fieldData[output_key]={ # Add new field for current key
								'data':[range(1,1+len(data_list))],
								'time':[],
								} 
						fieldData[output_key]['data'].append(data_list)
						fieldData[output_key]['time'].append(frame_time2)
				in_i -= 1	# Processed all frames in current step
				
				if len(fieldData.keys())==0:
					continue
				
				timeAppend('%sExtracted %i frames - %.2f sec'%(IN_S*in_i, len(fieldData[STG['keys'][0]]['time']), time()-time_start))
				
				## Write data to CSV
				if STG['save_style'] == 'step_wise':
					time_start = time()
					print('%sProcessing CSV...'%(IN_S*in_i))
					output_format = STG['sep'].join([cur_name, 'i%03d'%(ins_ID), '{key}', 's%05d.csv'%int(step.number)])
					export_fieldData(fieldData, output_format)
					timeAppend('%sWrote csv files - %.2f sec'%(IN_S*in_i,time()-time_start))
					fieldData = {}
					
			if STG['save_style'] == 'LTr_based':
				time_start = time()
				print('%sProcessing CSV...'%(IN_S*in_i))
				output_format = STG['sep'].join([cur_name, 'i%03d'%(ins_ID), '{key}', LTr_name+'.csv'])
				export_fieldData(fieldData, output_format)
				timeAppend('%sWrote csv files - %.2f sec'%(IN_S*in_i,time()-time_start))
				fieldData = {}
			in_i -= 1	# Finished all steps in the instance
		in_i -= 1	# Finished all instances
		odb.close()
	in_i -= 1	# Finshed all odb files

	timeAppend('Total time: %.2f sec'%(time()-time_alpha))
