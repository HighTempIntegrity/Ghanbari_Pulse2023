# Use modern python to run this. Notepadd++ F5 command:
# cmd /k cd /d $(CURRENT_DIRECTORY) && python $(FULL_CURRENT_PATH)

import pandas as pd
import _inpyt.experiment as inpr
import _pylib.tools as tl

STG = {	# Settings
	'model_tag':'Exp',			# This will be used in the beginning of all input files for local models
	'zeros':3,
	'time_file':'4_time.log',
}
TPL = {	# Templates
	'experiment':'2_exp.csv',
	'laser':'1_AM_laser_01L_15t8mm.inp',
	'input':'1_input.inp',
}
SYS = { # For running abaqus
	# 'command':'abaqus',
	# 'cpus':12,			# Number of logical processors to run the simulations
}
time_log = tl.TimeLog(STG['time_file'])

# Read the experimental design vectors
exp_df = pd.read_csv(TPL['experiment'])

time_log.in_right()
for exp_id, exp_info  in exp_df.iterrows():
	# exp_id:	index of experiments starting from 0
	# exp_info:	contatining individual setting of current experiment
	time_log.start()
	
	exp_id = int(exp_info['id'])
	exp_name = '%s%s'%(STG['model_tag'],str(exp_id).zfill(STG['zeros']))

	# Create new laser file
	layer_count = int(exp_info['layer'])
	laserf = inpr.TextFile(TPL['laser'], exp_name)
	laserf.swap_string(',0.030000',',%f'%(layer_count*0.03))
	laserf.update_file_name('1_AM_laser_%02dL_15t8mm.inp'%(layer_count))
	laserf.write_file()
	
	# Create new input file
	input = inpr.InputFile(TPL['input'], exp_name)
	input.set_laser(laserf.filename)
	input.swap_string('step=1','step=%d'%(layer_count))
	input.swap_line(
		'*Include,input=1_mesh_01L15T8mm_om40x35.inp\n',
		'*Include,input=1_mesh_%02dL15T8mm_om40x35.inp\n'%(layer_count))
	input.write_file(exp_id)

	time_log.append('Prepared input files for experiment %i'%(exp_id))
	
	# Run simulation
	time_log.start()
	# abaqus_tags = {
		# 'job':'run_%s'%(exp_name),
		# 'input':input.filename,
	# }
	# tl.runAbaqus(system=SYS,**abaqus_tags)
	time_log.append('Simulation %i finished'%(exp_id))
time_log.in_left()

time_log.closure()
