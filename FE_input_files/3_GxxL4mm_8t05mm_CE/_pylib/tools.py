import time
import subprocess
import pandas as pd
import numpy as np
import sys, os
import pickle



STG = { # Settings
	'indent_string':'   ',	# Used for indenting the output info in command line
}

# def runAbaqus(job, input, system):
def runAbaqus(system, **kwargs):
	# Decide cpu number
	if 'cpus' in system:
		run_cpus = system['cpus']
	else:
		if sys.platform == 'win32':
			run_cpus = 6
		elif sys.platform == 'linux':
			run_cpus = 12
	
	if 'command' in system:
		arg_list = [system['command']]
	else:
		arg_list = ['abaqus']
	
	for key in kwargs:
		arg_list.append('%s=%s'%(key,kwargs[key]))
	
	if  sys.platform == 'win32':
		command_frt = 'ifortvars intel64\r\n'
		command_job = '%s cpus=%i ask_delete=OFF\r\n'%(' '.join(arg_list), run_cpus)
		command = command_frt + ';' + command_job
		process = subprocess.Popen('cmd.exe', stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=None, shell=False)
		out = process.communicate(command.encode('utf-8', 'ignore'))[0]
		print(out.decode('utf-8', 'ignore'))
		
	elif sys.platform == 'linux':
		arg_list.append('cpus=%i'%(run_cpus))
		process = subprocess.Popen(arg_list)
		process.communicate()

def extractModels(MODELS_ZIP, CSV_ROOT):
	model_packs = []
	for model in MODELS_ZIP:
		# * in folder name means that there are a list of enumareted folders
		if '*' in model['directory']:
			template_name = model['directory'].replace('*','')
			
			for item in os.listdir(CSV_ROOT):
				if template_name in item:
					new_copy = model.copy()
					new_copy['directory'] = item
					new_copy['name'] = '_'.join(item.split('_')[1:])
					new_copy['id'] = int(item.split('_')[1][1:])
					model_packs.append(new_copy)
		else:
			model_packs.append(model)
	
	return model_packs

def readModels(models_given, root, source = 'csv_file', save = False, LT_string='' ):
	in_i = 0
	spacer = STG['indent_string']
	model_packs = []
	if source == 'csv_file':
		for model in models_given:
			# * in folder name means that there are a list of enumareted folders
			if '*' in model['directory']:
				template_name = model['directory'].replace('*','')
				
				for item in os.listdir(root):
					if template_name in item:
						new_copy = model.copy()
						new_copy['directory'] = item
						new_copy['name'] = item.split('_')[-1]
						model_packs.append(new_copy)
			else:
				model_packs.append(model)
			
		print('%sReading CSV files into dataframes...'%(spacer*in_i))
		in_i += 1
		for cur_model in model_packs:
			print('%sModel %s'%(spacer*in_i, cur_model['directory']))
			
			model_dir = os.path.join(root,cur_model['directory'])
			# Iterate in files of current directory
			in_i+=1
			for curfile in sorted(os.listdir(model_dir)):
				print('%sFile %s'%(spacer*in_i, curfile))
				# curfile_segments = curfile.split('.')[0].split('-')
				curfile_segments = os.path.splitext(curfile)[0].split('-')
				curpath = os.path.join(model_dir,curfile)
				
				if curfile_segments[2] == 'node':
					cur_model['coords'] = pd.read_csv(curpath, index_col=0)
					
				elif curfile_segments[2] == 'element':
					cur_model['elcons'] = pd.read_csv(curpath, index_col=0)
				
				elif curfile_segments[2] == 'NT11':
					if curfile_segments[3][0]=='s':	# Data is saved per step
						nt_df = pd.read_csv(curpath, index_col=0)
						if 'nt11' not in cur_model: # Step 1
							cur_model['nt11'] = nt_df
						else: # Steps 2,3,...
							nt_df = nt_df.drop(columns=[nt_df.columns[0]])
							# Shifting the step time to total time to combine all temperature data
							# shifted_columns = [str(float(x)+float(cur_model['nt11'].columns[-1])) for x in nt_df.columns]
							# nt_df.columns = shifted_columns
							cur_model['nt11'] = pd.concat([cur_model['nt11'],nt_df],axis=1)
						cur_model['nt11_file'] = '-'.join(curfile_segments)+'.csv'
					elif curfile_segments[3][0]=='L':	# Data is saved per Layer Track
						if curfile_segments[3] == LT_string:
							cur_model['nt11'] = pd.read_csv(curpath, index_col=0)
							cur_model['nt11_file'] = curfile
							
				
				elif curfile_segments[2] == 'EACTIVE':
					cur_model['eactive'] = pd.read_csv(curpath, index_col=0)
			in_i-=1
		
			# Remove duplicate columns that have very close frame times
			if 'nt11' in cur_model:
				for time_str in cur_model['nt11'].columns:
					if time_str.count('.') == 2: # Anything that's like '5.024596214.1'
						cur_model['nt11'].drop(time_str, inplace=True, axis=1)
				cur_model['frames'] = pd.Series(cur_model['nt11'].columns)
			print('%sFinished importing all data.'%(spacer*in_i))
		in_i-=1
		
		if save:
			pickle.dump( model_packs, open( '3_model_packs.pk', 'wb' ) )
		print('%sFinished importing data from CSV files.'%(spacer*in_i))
		
	elif source == 'pk_file':
		model_packs = pickle.load( open( '3_model_packs.pk', 'rb' ) )
		print('%sLoaded model data from pickle file.'%(spacer*in_i))
	
	return model_packs

def update_plot_style(plt, theme, mplib = None):
	# For more info see
	# https://matplotlib.org/stable/tutorials/introductory/customizing.html

	# Default updates
	plt.style.use('default')
	plt.rcParams.update({
		'savefig.dpi':300,
		'axes.linewidth': 0.5,
		'xtick.direction':'in',
		'ytick.direction':'in',
		'xtick.major.width':0.5,
		'ytick.major.width':0.5,
		'axes.grid':True,
		'grid.linestyle':'--',
		'grid.linewidth':'0.5',
		'grid.alpha':'0.5',
	})
	# Thematic updates
	if theme == 'paper':
		mplib.use('pgf')
		mplib.rcParams.update({
			'pgf.texsystem': 'pdflatex',
			'font.family': 'serif',
			'text.usetex': True,
			'pgf.rcfonts': False,
		})
		plt.rcParams.update({
			'figure.titlesize':11,
			'axes.titlesize':10,
			'axes.labelsize':9,
			'legend.fontsize':8,
			'xtick.labelsize':8,
			'ytick.labelsize':8,

		})
	elif theme == 'notebook':
		plt.rcParams.update({
			'font.family':'CMU Serif',
			'figure.titlesize':18,
			'figure.subplot.top':.85,
			'axes.titlesize':16,
			'axes.titlepad':10,
			'axes.labelsize':14,
			'xtick.labelsize':12,
			'ytick.labelsize':12,
			'legend.fontsize':12,
		})
	return plt

class TimeLog:
	def __init__(self, time_file_dir):
		self.time_file = time_file_dir	# Relative path of the time file
		self.time_df = pd.DataFrame(columns=['seconds','duration','message'])	# For writing a clean log
		self.in_s = STG['indent_string']	# String used to indent messages
		self.in_i = 0	# Level of indentation
		self.init_time = time.time()	# Timestamp for the beginning of the script
		self.start_times = [time.time()]	# List of timestamps at the beginning of code blocks
		self.first_log = True	# To track creation of log file

	def start(self):
		self.start_times.append(time.time())
		
	def in_left(self):
		self.in_i-=1
		
	def in_right(self):
		self.in_i+=1
		
	def in_str(self):
		return self.in_i*self.in_s
		
	def append(self, message):
		cur_time = self.start_times.pop(-1)
		cur_stamp = timeStamp(cur_time)
		df_new_row = pd.DataFrame.from_dict({'seconds':[time.time()-cur_time], 'duration':[cur_stamp], 'message':[message]})
		self.time_df = pd.concat([self.time_df, df_new_row])
		log_line = '%s%s - %s'%(self.in_s*self.in_i, message, cur_stamp)
		print(log_line)
		
		if self.first_log:
			with open(self.time_file, 'w') as file:
				file.write(log_line+'\n')
			self.first_log = False
		else:
			with open(self.time_file, 'a') as file:
				file.write(log_line+'\n')
				
	def out(self, message, indent=0):
		log_line = '%s%s'%(self.in_s*(self.in_i+indent), message)
		print(log_line)
		
		if self.first_log:
			with open(self.time_file, 'w') as file:
				file.write(log_line+'\n')
			self.first_log = False
		else:
			with open(self.time_file, 'a') as file:
				file.write(log_line+'\n')
	
	def closure(self):
		log_line = '%s %s'%('\nScript executed in', timeStamp(self.init_time))
		print(log_line)
		with open(self.time_file, 'a') as file:
			file.write(log_line+'\n')
		file_name_csv = '.'.join(self.time_file.split('.')[:-1])+'.csv'
		self.time_df.to_csv(file_name_csv, index=False)

def timeStamp(ref_time):
	m,s = divmod(time.time()-ref_time,60)
	h,m = divmod(m,60)
	return('{:.0f}:{:02.0f}:{:06.3f}'.format(h, m, s))

def write_overview(dict_list, overview_file):
	with open(overview_file,'w') as file:
		for dict in dict_list:
			file.write(dict['dict_name']+'\n')
			for key, value in dict.items():
				if key != 'dict_name':
					file.write('\t%s: %s\n'%(key,value.__str__()))
									
def match_query(query, coords, elcons, **kwargs):
	# Returns node labels and data corresponding to query
	# Inputs are
	#	coords:	Dataframe of nodes
	#	'query':	Dict containing query information
	#	'space_res':		Space resolution
	## Point type
	# Input 'query' has the following keys
	#	'type'
	#	'center'
	#	'output'
	q_x = round(query['center'][0],kwargs['space_res'])
	q_y = round(query['center'][1],kwargs['space_res'])
	q_z = round(query['center'][2],kwargs['space_res'])
	
	if query['type']=='point':
		query = {
			'type':'point',
			'center':query['center'],
		}
		# See if the query point is even in the model
		outline = {
			'x':(coords['x'].min(),coords['x'].max()),
			'y':(coords['y'].min(),coords['y'].max()),
			'z':(coords['z'].min(),coords['z'].max()),
		}
		if ( q_x >= outline['x'][0] and q_x <= outline['x'][1] and
			 q_y >= outline['y'][0] and q_y <= outline['y'][1] and
			 q_z >= outline['z'][0] and q_z <= outline['z'][1]):
			# Find distnace of all nodes from the query
			shifted_coords = coords-np.array([q_x,q_y,q_z])
			distance = np.sqrt(np.square(shifted_coords).sum(axis=1))
			if round(min(distance),kwargs['space_res']) == 0: # There is a node matching the query
				query['alignment'] = 'match'
				query['labels'] = distance.idxmin()
			else:	# The query is somewhere between model nodes
				query['alignment'] = 'offset'
				# Scaling factor used to search around the query
				# minimum distance is multiplied by this value
				search_scale = 2 
				scale_increase = 2
				search_counter = 1
				interp_labels = distance[distance<min(distance)*search_scale].index.tolist()
				# While there are less than 50 nodes, the search continues
				while len(interp_labels)<50:
					search_scale += search_counter*scale_increase
					search_counter += 1
					interp_labels = distance[distance<min(distance)*search_scale].index.tolist()
				query['labels'] = interp_labels
			query['coords'] = coords.loc[query['labels']]
		else:
			query['alignment'] = 'out'
	## Box type
	# kwargs
	#	'coords'
	#	'elcons'
	#	'space_res'
	# Input 'query' has the following keys
	#	'type'
	#	'crd_min'
	#	'crd_max'
	#	'output'
	elif query['type']=='box':
		query = {
			'type':'box',
			'labels':[],
		}
		
		# Find center of box
		max_ar = np.asarray(query['crd_max'])
		min_ar = np.asarray(query['crd_min'])
		mean_ar = np.mean( np.array([max_ar,min_ar]), axis=0 )
		# Find radius of shpere around box
		radius = np.sqrt(np.square(max_ar-mean_ar).sum(axis=0))*1.1 #TODO: ghetto fix for coordiates falling out of the edge of the box
		# Shift coords onto the center
		shifted_coords = coords-mean_ar
		# Find distance of all nodes from center
		distance = np.round(np.sqrt(np.square(shifted_coords).sum(axis=1)),kwargs['space_res'])
		# Find label of points inside sphere
		sphere_labels = distance[distance<radius].index.tolist()
		# Find labels that fall inside the box
		
		for label in sphere_labels:
			if (coords.loc[label]<max_ar).all and (coords.loc[label]>min_ar).all:
				query['labels'].append(label)

		
		# Find surrounding elements
		surrounding_elements = []
		for index, row in elcons.iterrows():
			el_nodes = [int(x) for x in row['connection'][1:-1].split(',')]
			for nd_label in query['labels']:
				if nd_label in el_nodes:
					if index not in surrounding_elements:
						surrounding_elements.append(index)
		
		# Check to see if they are in the box
		box_elements = []
		for el_label in surrounding_elements:
			el_nodes_str = elcons.loc[el_label]['connection'][1:-1].split(',')
			el_nodes_int = [int(x) for x in el_nodes_str]
			node_sum = np.array([0.0,0.0,0.0])
			counter=0
			for node_lb in el_nodes_int:
				node_sum+=coords.loc[node_lb]
				counter+=1
			node_mean = np.array(node_sum/counter)
			if (node_mean<max_ar).all() and (node_mean>min_ar).all():
				box_elements.append(el_label)
		query['elcons'] = elcons.loc[box_elements]
		
		# Redefine node labels based on elements
		query['labels'] = []
		for el_label in box_elements:
			el_nodes_str = elcons.loc[el_label]['connection'][1:-1].split(',')
			el_nodes_int = [int(x) for x in el_nodes_str]
			for node_lb in el_nodes_int:
				if node_lb not in query['labels']:
					query['labels'].append(node_lb)
		
		# Added corresponding coordinates
		query['coords'] = coords.loc[query['labels']]
	
	return query

def match_history(query, nt_df, **kwargs):
	# Returns the time history of the query as a dataframe
	## Point type
	# kwargs
	#	'nt_df'
	#	'mat'
	#	'matEng'
	#	'coords'
	#	'time_res'
	#	'query'
	#		'type'
	#		'center'
	if query['type']=='point':
		if query['alignment'] == 'match': # If there was a perfect match
			if isinstance(nt_df,pd.DataFrame):
				export_df = nt_df.loc[query['labels']].to_frame('nt11')
				export_df.index.name = 'time'
				export_df.index = export_df.index.map(float)
				export_df.index = export_df.index.to_series().apply(lambda x: np.round(x,kwargs['time_res']))
			else:
				export_df = nt_df.loc[[query['labels']]]
		elif query['alignment'] == 'offset': # If interpolation is needed
			export_values = []
			for cur_time in nt_df.columns:
				# print('Interpolating frame_time %s'%(cur_time))
				values_given = nt_df[cur_time].loc[query['labels']].values.tolist()
				coords_given = query['coords'].values.tolist()
				
				# Matlab interpolation
				coords_given = kwargs['mat'].double(coords_given)
				values_given = kwargs['mat'].double(values_given)
				coords_query = kwargs['mat'].double(list(query['center']))
				# 'linear' (default) | 'nearest' | 'natural'
				value_query = kwargs['matEng']._mlab_scatteredInterpolant(
					coords_given, values_given, coords_query, 'nearest')
				export_values.append(value_query)
			export_df = pd.DataFrame(data=export_values, columns=['nt11'], index=nt_df.columns)
			export_df.index.name = 'time'
			export_df.index = export_df.index.map(float)
	## Box type
	# kwargs
	#	'nt_df'
	#	'query'
	#	'time_res'
	#	'matlab'
	# Input 'query' has the following keys
	#	'type'
	#	'crd_min'
	#	'crd_max'
	elif query['type']=='box':
		# export_df = nt_df.loc[query['labels']]
		export_df = nt_df.loc[query['labels']].transpose()
		export_df.index.name = 'time'
		export_df.index = export_df.index.map(float)
		export_df.index = export_df.index.to_series().apply(lambda x: np.round(x,kwargs['time_res']))
		export_df = export_df.transpose()
	return export_df

def shift_index(df, shift_value):
    df_new = df.copy()
    df_new.index = df.index-shift_value
    return df_new
	
def scale_index(df, scale_value):
    df_new = df.copy()
    df_new.index = df.index*scale_value
    return df_new
	
def shift_scale_index(df, shift_value, scale_value):
    df_new = df.copy()
    df_new.index = (df.index-shift_value)*scale_value
    return df_new
	
def scale_shift_index(df, scale_value, shift_value):
    df_new = df.copy()
    df_new.index = df.index*scale_value-shift_value
    return df_new
	
def interp_values_at_index(given_df, target_index):
	# Interpolate given_df for new index values in target_index
	# Returns only the new interpolated values
	empty_df = pd.DataFrame(index=target_index, columns = given_df.columns)
	diff_index = empty_df.index.difference(given_df.index)
	empty_df = empty_df.loc[diff_index]
	combine_df = pd.concat([empty_df,given_df],axis=0)
	combine_df = combine_df.sort_index()
	for col in combine_df:
		combine_df[col] = pd.to_numeric(combine_df[col], errors='coerce')
	combine_df = combine_df.interpolate('index')
	combine_df = combine_df.dropna()
	return combine_df.loc[target_index,:]
	
def interp_value_at_index(given_df, target_index):
	# Interpolate given_df for new index values in target_index
	# Returns only the new interpolated values
	empty_df = pd.DataFrame(index=[target_index], columns = given_df.columns)
	diff_index = empty_df.index.difference(given_df.index)
	empty_df = empty_df.loc[diff_index]
	combine_df = pd.concat([empty_df,given_df],axis=0)
	combine_df = combine_df.sort_index()
	for col in combine_df:
		combine_df[col] = pd.to_numeric(combine_df[col], errors='coerce')
	combine_df = combine_df.interpolate('index')
	combine_df = combine_df.dropna()
	return combine_df.loc[[target_index],:].values[0][0]

def interp_index_at_value(given_df, target_value, target_column):
	# Reverse interpolate the index for a given column's value
	flip_df = given_df.set_index(target_column)
	flip_df[0] = given_df.index
	return interp_values_at_index(flip_df,target_value).loc[target_value,:].values[0][0]

def interp_reverse(given_df, target_value, target_column):
	# Reverse interpolate the index for a given column's value
	flip_df = given_df.set_index(target_column)
	flip_df[0] = given_df.index
	return interp_values_at_index(flip_df,target_value).loc[target_value,:]

def interp_append_index(given_df, added_index):
	# Interpolate given_df for new index values in added_index
	# Returns the combined dataframe with added_index
	empty_df = pd.DataFrame(index=added_index, columns = given_df.columns)
	diff_index = empty_df.index.difference(given_df.index)
	empty_df = empty_df.loc[diff_index]
	combine_df = pd.concat([empty_df,given_df],axis=0)
	combine_df = combine_df.sort_index()
	for col in combine_df:
		combine_df[col] = pd.to_numeric(combine_df[col], errors='coerce')
	combine_df = combine_df.interpolate('index')
	combine_df = combine_df.dropna()
	return combine_df

def appcat(given_df, idx_value, row_elements):
	row_df = pd.DataFrame(row_elements, index=[idx_value], columns = given_df.columns)
	row_df.index.name = given_df.index.name
	return pd.concat([given_df, row_df])

def R2_measure_df(given_index, ref_df, eval_df):
	ref_values = interp_values_at_index(ref_df, given_index).values.flatten()
	eval_values = interp_values_at_index(eval_df, given_index).values.flatten()

	residuals = ref_values - eval_values

	ss_res = np.sum(residuals**2)
	ss_tot = np.sum((ref_values-np.mean(ref_values))**2)
	r_squared = 1 - (ss_res / ss_tot)
	return r_squared
	
def RMSE_df(given_index, ref_df, eval_df):
	ref_values = interp_values_at_index(ref_df, given_index).values.flatten()
	eval_values = interp_values_at_index(eval_df, given_index).values.flatten()

	residuals = ref_values - eval_values
	residuals = residuals[~np.isnan(residuals)]
	rmse = np.sqrt(np.mean((residuals)**2))
	
	return rmse
	
def shape_function(corners_df, coords_tuple):
	# This function gets a dataframe of 4 coordinates in 3d or 2d and gives nodal 
	# weights corresponding to x and y in the coords tuple
	# The wights are based on linear shape functions from FE
	# weight_df can be used through T_df.dot(weight_df) to computed a weighted average response
	# at the point of interest

    # sort the corners
    sorted_corners = corners_df.sort_values(by=['y','x'])
    
    # get corner coordinates
    xx = []
    yy = []
    for label, row in sorted_corners.iterrows():
        xx.append(row['x'])
        yy.append(row['y'])
    a = xx[1]-xx[0]
    b = yy[2]-yy[0]
    
    # calculate weights
    x_qry = coords_tuple[0]
    y_qry = coords_tuple[1]
    w0 = (x_qry-xx[1])*(y_qry-yy[2])/(a*b)
    w1 =-(x_qry-xx[0])*(y_qry-yy[3])/(a*b)
    w2 =-(x_qry-xx[3])*(y_qry-yy[0])/(a*b)
    w3 = (x_qry-xx[2])*(y_qry-yy[1])/(a*b)

    weight_df = pd.DataFrame([w0,w1,w2,w3], columns=['w'], index=sorted_corners.index)
    weight_df = weight_df.reindex(corners_df.index)
    return weight_df
	
class Errors:
	def __init__(self, y_ref, y_eval):
		self.ref = y_ref
		self.average = np.mean(y_ref)
		self.range = np.max(y_ref)-np.min(y_ref)
		self.variance = np.var(y_ref)
		self.STD = np.std(y_ref)
		
		self.eval = y_eval
		self.residual = y_eval - y_ref
		
	def SE(self):
		return np.sum(np.square(self.residual))
		
	def MSE(self):
		return np.mean(np.square(self.residual))
		
	def RMSE(self):
		return np.sqrt(np.mean(np.square(self.residual)))
		
	def rMSE(self):
		return np.mean(np.square(self.residual))/self.variance
		
	def R2(self):
		return 1-np.mean(np.square(self.residual))/self.variance
		
	def MAE(self):
		return np.mean(np.abs(self.residual))
		
	def MAPE(self):
		return np.mean(np.abs(self.residual/self.ref))
		
	def NMAE(self, norm='mean'):
		if norm == 'mean':
			nmae = np.mean(np.abs(self.residual))/self.average
		elif norm == 'range':
			nmae = np.mean(np.abs(self.residual))/self.range
		else:
			raise ValueError('Unknown normalization type')
		return nmae

	def NRMSE(self, norm='mean'):
		if norm == 'mean':
			nrmse = np.sqrt(np.mean(np.square(self.residual)))/self.average
		elif norm == 'range':
			nrmse = np.sqrt(np.mean(np.square(self.residual)))/self.range
		else:
			raise ValueError('Unknown normalization type')
		return nrmse
		
	def overview(self, col_name = 'error', unit='', norm='mean'):
		error_dict = {
			'NMAE [pc]': self.NMAE(norm)*100,
			'NRMSE [pc]':self.NRMSE(norm)*100,
			'MAPE [pc]': self.MAPE()*100,
			'R2 [pc]':   self.R2()*100,
			'MAE [%s]'%(unit): self.MAE(),
			'RMSE [%s]'%(unit):self.RMSE(),
		}
		error_df = pd.DataFrame.from_dict(error_dict, orient='index', columns=[col_name])
		return error_df





































