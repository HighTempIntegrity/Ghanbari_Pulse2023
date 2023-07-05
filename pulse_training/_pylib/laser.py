import numpy as np
import pandas as pd
import math
import os

def vector_angle(v1,v2):
	return math.atan2(v1[1],v1[0]) - math.atan2(v2[1],v2[0])

def rot_mat(angle):
	return np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])


class coord3D:
	# A small custom class for processing 3D coordinates
	def __init__(self, *args):
		PRC = 9
		if isinstance(args[0],float):
			self.xx = round(args[0],PRC)
			self.yy = round(args[1],PRC)
			self.zz = round(args[2],PRC)
			self.ar = np.array([self.xx,self.yy,self.zz])
		elif isinstance(args[0],np.ndarray) or isinstance(args,tuple):
			self.xx = round(args[0][0],PRC)
			self.yy = round(args[0][1],PRC)
			self.zz = round(args[0][2],PRC)
			self.ar = np.array([self.xx,self.yy,self.zz])

	def __str__(self):
		return '(%s,%s,%s)'%(self.xx, self.yy, self.zz)
	
	def distance(self, other):
		delta_x = self.xx - other.xx
		delta_y = self.yy - other.yy
		delta_z = self.zz - other.zz
		return (delta_x**2+delta_y**2+delta_z)**0.5
	
	def length(self):
		return (self.xx**2+self.yy**2+self.zz)**0.5
		
	def normal(self):
		length = (self.xx**2+self.yy**2+self.zz)**0.5
		norm_x = self.xx/length
		norm_y = self.yy/length
		norm_z = self.zz/length
		return(coord3D(norm_x,norm_y,norm_z).ar)


class LaserTracker:
	def __init__(self, laser_file):
		# Reading the laser file into 'contents'
		f = open(laser_file, 'r')
		contents = f.readlines()
		f.close()
		
		tracks_NO  = int(len(contents)/2)	# Number of trackes that laser is active
		layer_counter = 0
		self.vectors = []	# List of dictionaries with laser track information
		self.layers = []	# List of dictionaries with layer information
		for track_id in range(tracks_NO):	# Processing each track separately
			start_line_segments = contents[track_id*2].split(',')   # Activation line
			end_line_segments = contents[track_id*2+1].split(',') # Deactivation line
			
			track_p1 = coord3D(\
				float(start_line_segments[1]),\
				float(start_line_segments[2]),\
				float(start_line_segments[3]))
			track_p2 = coord3D(\
				float(end_line_segments[1]),\
				float(end_line_segments[2]),\
				float(end_line_segments[3]))
			track_vector = coord3D(track_p2.ar - track_p1.ar)
			
			# Checking for layer change
			fresh_layer = False # Set to True if the track belongs to a new lauer
			if self.vectors:	# If the vector list is not empty i.e. not first track
				prev_height = self.vectors[-1]['start'][2]
				cur_height = track_p1.ar[2]
				if cur_height > prev_height: # If the new track is at a higher z than last one
					layer_counter+=1
					self.layers.append({})
					fresh_layer = True
			else:	# If first track then it's new layer
				self.layers.append({})
				fresh_layer = True
			
			# vectors: 		laser track with direction
			# 'dir':	normal vector in direction of track
			# 'length':		length of track
			# 'start': 		spatial coordinates of track start
			# 'end': 		spatial coordinates of track end
			# 'on': 		total time that the track starts
			# 'off': 		total time that the track ends
			# 'duration':	how long the track takes
			# 'power':		laser power for the current vector
			# 'track_id':	index of trackes in total
			# 'layer_id':	index of layers in total
			# 'fresh_layer': True if it's the first track of current layer
			self.vectors.append({})
			self.vectors[-1]['dir'] = track_vector.normal()
			self.vectors[-1]['length'] = track_vector.length()
			self.vectors[-1]['start'] = track_p1.ar
			self.vectors[-1]['end'] = track_p2.ar
			self.vectors[-1]['on'] = float(start_line_segments[0])
			self.vectors[-1]['duration'] = float(end_line_segments[0]) - self.vectors[-1]['on']
			self.vectors[-1]['off'] = self.vectors[-1]['on']+self.vectors[-1]['duration']
			self.vectors[-1]['power'] = float(start_line_segments[4][:-1])
			self.vectors[-1]['track_id'] = track_id
			self.vectors[-1]['layer_id'] = layer_counter
			self.vectors[-1]['fresh_layer'] = fresh_layer
			if fresh_layer:
				self.layers[-1]['layer_id'] = layer_counter
				self.layers[-1]['on'] = self.vectors[-1]['on']
				self.layers[-1]['first_track_id'] = track_id
				if len(self.layers) > 1: # From 2nd onwards
					self.layers[-2]['last_track_id'] = track_id-1
			
		# Last layer
		self.layers[-1]['last_track_id'] = track_id
		
		for layer in self.layers:
			layer['track_count'] = layer['last_track_id']-layer['first_track_id']+1
				
	def laser_status(self, cur_time):
		# Returns the current position and state of laser
		# If the laser is off, it is stationary at the last track
		last_position = self.vectors[0]['start']
		last_track_time = self.vectors[0]['on']
		last_track_id = 0
		# Find which track we are at
		for vector in self.vectors:
			if cur_time<=vector['on']:
				break
			if cur_time<=vector['off']:
				# Find the time fraction
				track_time = cur_time-vector['on']
				cur_fraction = track_time/vector['duration']
				# Calculate the position
				cur_position = vector['start']+vector['dir']*cur_fraction*vector['length']
				# Return position and on/off state
				return cur_position,True,vector['track_id']
			else:
				last_position = vector['end']
				last_track_time = vector['off']
				last_track_id = vector['track_id']
		# If the time is outside of the laser event series
		# OR in the off state
		return last_position,False,last_track_id
		
	def laser_status_dict(self, cur_time):
		# Returns the current position and state of laser
		# If the laser is off, it is stationary at the last track
		status = {
			'position':self.vectors[0]['start'],
			'active':False,
			'track_id':0,
			'dir':np.array([0,0,0]),
		}
		last_track_time = self.vectors[0]['on']
		# Find which track we are at
		for vector in self.vectors:
			if cur_time<=vector['on']:
				break
			if cur_time<=vector['off']:
				# Find the time fraction
				track_time = cur_time-vector['on']
				cur_fraction = track_time/vector['duration']
				# Calculate the position
				status['position'] = vector['start']+vector['dir']*cur_fraction*vector['length']
				status['active'] = True
				status['track_id'] = vector['track_id']
				status['dir'] = vector['dir']
				# Return position and on/off state
				return status
			else:
				last_track_time = vector['off']
				status['position'] = vector['end']
				status['track_id'] = vector['track_id']
		# If the time is outside of the laser event series
		# OR in the off state
		return status
		
	def laser_positions(self, length_inc, rounding_digits):

		time_res = rounding_digits['time']
		space_res = rounding_digits['space']

		points = []	
		
		layer_start_time = 0
		for vector in self.vectors:
			duration = round((length_inc/vector['length'])*vector['duration'],time_res)
			seg_count = int(vector['length']/length_inc)
			for segment_id in range(seg_count+1):
				# a segment is defined wrt to current track/vector
				points.append({}) # add empty dictionary
				start = vector['start']+vector['dir']*length_inc*segment_id
				time2 = round(vector['on']+segment_id*duration,time_res)
				
				points[-1]['time2'] = time2
				points[-1]['x'] = round(start[0],space_res)
				points[-1]['y'] = round(start[1],space_res)
				points[-1]['z'] = round(start[2],space_res)
				
		return pd.DataFrame(points)

	def laser_time(self, L=0, T=0, r=0):
		track_id = L*self.layers[0]['track_count']+T
		for vector in self.vectors:
			if track_id == vector['track_id']:
				return vector['on']+r*vector['duration']
	
	def LT_track_id(self, L=0, T=0, r=0):
		return L*self.layers[0]['track_count']+T
		
	def LTr_to_position(self, L=0, T=0, r=0):
		cur_time = self.laser_time(L,T,r)
		cur_stat = self.laser_status(cur_time)
		return cur_stat[0]
		


class LocalEvents: #TODO: Update all function as a separate class from LaserEvents
	def __init__(self, LEO, MS_LCL, GLOBAL_TYPE, ROUND_DIGITS):
		# Dicts containing local model info
		# 'line1': line 1 of laser input file
		# 'line2': line 2 of laser input file
		# 'start': coordinates of where laser starts in local model
		# 'end': coordinates of where laser ends in local model
		# 'shift': translation vector for mesh coordinates
		# 'angle': angle of rotation wrt global coordinate system
		# 'time1': start time of local model wrt current layer
		# 'time2': total start time of local model 
		# 'duration': the run time of the local model 
		# 'timepoints0': timepoints for saving data in local model
		# 'timepoints1': timepoints consistent with the heating step of global model
		# 'timepoints2': timepoints wrt total time
		# 'id': index of local model
		# 'track_id': index of corresponding laser track
		# 'layer_id': index of corresponding layer
		# 'whole_step': step number in the global model
		# 'track_step': step number in the current track
		
		self.locals = []	
		
		local_length = MS_LCL['local_length']
		x_min = MS_LCL['xlim'][0]
		x_max = MS_LCL['xlim'][1]
		mid_points = MS_LCL['mid_points']
		time_res  = ROUND_DIGITS['time']
		space_res = ROUND_DIGITS['space']
		
		local_ID = 0
		
		layer_start_time = LEO.vectors[0]['on']
		for vector in LEO.vectors:
			ratios = []	# List of length ratios that the vectors passes the x limits
			ratios.append((x_min - vector['start'][0])/(vector['dir'][0]*vector['length']))
			ratios.append((x_max - vector['start'][0])/(vector['dir'][0]*vector['length']))
			ratio_min = min(ratios)	# the least ratio that corresponds to entry into probe region
			
			# time_vector: total time of where locals start in current vector
			# local_duration: duration of local model for current vector
			# x_length: length of track in probe region
			# local_number: number of local models in current track
			time_vector = ratio_min*vector['duration']+vector['on']
			local_duration = (local_length/vector['length'])*vector['duration']
			x_length = abs(x_max-x_min)
			local_number = int(x_length/local_length)
			
			for segment_id in range(local_number):
				self.locals.append({}) # add empty dictionary
				start = vector['start']+vector['dir']*vector['length']*ratio_min+\
					vector['dir']*x_length*segment_id/local_number
				end = start+vector['dir']*local_length
				time1 = round(time_vector+segment_id*local_duration-layer_start_time,time_res)
				time2 = round(time_vector+segment_id*local_duration,time_res)
				tps0 = np.linspace(0,local_duration,mid_points+2).tolist()
				tps1 = np.linspace(time1,time1+local_duration,mid_points+2).tolist()
				tps2 = np.linspace(time2,time2+local_duration,mid_points+2).tolist()
				format1 = {'t':0.0,'x':start[0],'y':start[1],'z':start[2],'pow':vector['power'],'sp':space_res,'ts':time_res}
				format2 = {'t':local_duration,'x':end[0],'y':end[1],'z':end[2],'pow':0.0,'sp':space_res,'ts':time_res}
				self.locals[-1]['line1'] = "{t:12.{ts}},{x:6.{sp}},{y:6.{sp}},{z:6.{sp}},{pow}\n".format(**format1)
				self.locals[-1]['line2'] = "{t:12.{ts}},{x:6.{sp}},{y:6.{sp}},{z:6.{sp}},{pow}\n".format(**format2)
				self.locals[-1]['start'] = start
				self.locals[-1]['end'] = end
				self.locals[-1]['shift'] = start - LEO.vectors[0]['start']
				self.locals[-1]['angle'] = math.degrees(math.acos(np.dot(vector['dir'],LEO.vectors[0]['dir'])))
				self.locals[-1]['time1'] = time1
				self.locals[-1]['time2'] = time2
				self.locals[-1]['duration'] = round(local_duration,time_res)
				self.locals[-1]['timepoints0'] = [round(x,time_res) for x in tps0]
				self.locals[-1]['timepoints1'] = [round(x,time_res) for x in tps1]
				self.locals[-1]['timepoints2'] = [round(x,time_res) for x in tps2]
				self.locals[-1]['id'] = local_ID
				self.locals[-1]['track_id'] = vector['track_id']
				self.locals[-1]['layer_id'] = vector['layer_id']
				if GLOBAL_TYPE == 'probe':
					self.locals[-1]['whole_step'] = (local_ID+1)+(vector['layer_id']*2+1)+(vector['track_id']+1)
				else:
					self.locals[-1]['whole_step'] = (vector['layer_id']+1)+(local_ID+1)
				if vector['fresh_layer']:
					self.locals[-1]['track_step'] = segment_id+2
				else:
					self.locals[-1]['track_step'] = segment_id+1
				local_ID+=1
	
	def find_tracks(self):
		self.tracks = []
		# Dicts containing local model info
		# 'id': index of laser track
		# 'local_first_id': index of first local model in track
		# 'local_last_id': index of last local model in track
		
		
		track_ID = 0
		for local in self.locals:
			if self.check_trackchange(local['id']):
				self.tracks.append({}) # add empty dictionary
				self.tracks[-1]['id'] = track_ID
				self.tracks[-1]['local_first_id'] = local['id']
				track_ID+=1
			if self.check_trackchange(local['id']+1):
				self.tracks[-1]['local_last_id'] = local['id']	
			
	def get_shift(self,local_id):
		return self.locals[local_id]['start'].ar-self.locals[0]['start'].ar
		
	def get_rotation(self,local_id):
		return self.locals[local_id]['angle']
	
	def check_trackchange(self,local_id):
		## Function to determine if laser has moved to a new track to handle exceptions
		
		# If the local is not in the list then it's for a new track
		try: self.locals[local_id]
		except IndexError:
			return True
			
		if local_id == 0:
			return True	# The track is new at the very beginning
		elif self.locals[local_id]['track_id'] == self.locals[local_id-1]['track_id']:
			return False
		else:
			return True

	def check_layerchange(self,local_id):
		## Function to determine if laser has moved to a new layer
		
		# If the local is not in the list then it's for a new layer
		try: self.locals[local_id]
		except IndexError:
			return True
			
		if local_id == 0:
			return True	# The layer is new at the very beginning
		elif self.locals[local_id]['layer_id'] == self.locals[local_id-1]['layer_id']:
			return False
		else:
			return True

	def get_timepoints1(self):
		tps = []
		for local in self.locals:
			tps.extend(local['timepoints1'])
		tps = list(set(tps))
		tps.sort()
		self.timepoints1 = tps
		
	def get_tps2(self, lcl_ids = []):
		tps = []

		for local in self.locals:
			if lcl_ids: # List is not empty
				if local['id'] in lcl_ids:
					tps.extend(local['timepoints2'])
			else: # List is empty
				tps.extend(local['timepoints2'])
				
		tps = list(set(tps))
		tps.sort()
		return tps
		
	def get_starts(self):
		tps = []
		for local in self.locals:
			tps.append(local['time1'])
		tps = list(set(tps))
		tps.sort()
		self.startpoints = tps
		
	def write_timepoints_global(self, FILE_NAME):
		with open(FILE_NAME,'w+') as file:
			file.write('*Time Points, name=TimePoints\n')
			ii = 0
			for point in self.timepoints:
				file.write(str(point)+', ')
				if ii % 8 == 7: # Maximum 8 data points per line
					file.write('\n')
				ii+=1
				
	def write_timepoints_local(self, FILE_NAME):
		with open(FILE_NAME,'w+') as file:
			file.write('*Time Points, name=TimePoints-local\n')
			ii = 0
			for point in self.locals[0]['timepoints0']:
				file.write(str(point)+', ')
				if ii % 8 == 7: # Maximum 8 data points per line
					file.write('\n')
				ii+=1

	def write_file(self, TAG, LOCAL_ID, ZF):
		filename = TAG + str(LOCAL_ID).zfill(ZF) + '_AM_laser.inp'
		with open(filename,'w+') as file:
			file.write(self.locals[LOCAL_ID]['line1'])
			file.write(self.locals[LOCAL_ID]['line2'])
	
	def exportLocals(self,FILE_NAME):
		with open(FILE_NAME,'w+') as file:
			export_keys = ['id','timepoints2']
			separator = '\t,\t'
			
			header = separator.join(export_keys)
			file.write(header+'\n')
			for local in self.locals:
				line = separator.join([str(local[x]) for x in export_keys])
				file.write(line+'\n')
	
	def time_wrt_global(self, local, relative_df):
		index_glb = relative_df.index + local['time2']
		relative_df.index = index_glb
		return relative_df
	
	
class BrickEvents:
	def __init__(self, LEO, LCL_CRD_MAX, LCL_CRD_MIN, PRECISION):
		# It is assumed that the laser moves in the positive x direction.
		# All coordinates are in Cartesian (x,y,z) system.
		#	LCL_CRD_MAX	tuple containing max positive local coordinates
		#	LCL_CRD_MIN	tuple containing min positive local coordinates
		#	LEO is the Laser Events Object
		
		brick_length = LCL_CRD_MAX[0]-LCL_CRD_MIN[0]
		template_max = coord3D(LCL_CRD_MAX)
		template_min = coord3D(LCL_CRD_MIN)
		time_res = PRECISION['time']
		space_res = PRECISION['space']
		
		# Dictonaries containing brick metadata
		# 'start'		coordinates of laser start
		# 'end'			coordinates of laser end
		# 'shift'		translation vector for mesh coordinates
		# 'angle'		angle of rotation wrt global coordinate system
		# 'time1'		start time of brick model wrt current layer
		# 'time2'		total start time of brick model 
		# 'laser_duration'	the duration that laser is active in brick
		# 'id'			index of brick model in total
		# 'segment_id'	index of brick model in its layers
		# 'track_id'	index of corresponding laser track
		# 'layer_id'	index of corresponding layer
		# 'line1': line 1 of laser input file
		# 'line2': line 2 of laser input file
		# 'total_step': step number in the global model
		
		self.bricks = []	
		
		brick_ID = 0
		
		layer_start_time = 0
		for vector in LEO.vectors:
			
			brick_duration = (brick_length/vector['length'])*vector['duration']
			bricksNtrack_NO = int(vector['length']/brick_length)
			if vector['fresh_layer']:
				layer_start_time = vector['on']
			
			for segment_id in range(bricksNtrack_NO):
				# a segment is defined wrt to current track/vector
				self.bricks.append({}) # add empty dictionary
				start = coord3D(vector['start'].ar+\
					vector['dir'].ar*brick_length*segment_id)
				end = coord3D(start.ar+\
					vector['dir'].ar*brick_length)
				time2 = round(vector['on']+segment_id*brick_duration,time_res)
				time1 = round(time2-layer_start_time,time_res)
				shift = coord3D(start.ar - LEO.vectors[0]['start'].ar)
				brick_duration = round(brick_duration,time_res)
				
				self.bricks[-1]['start'] = start.ar
				self.bricks[-1]['end'] = end.ar
				self.bricks[-1]['shift'] = shift.ar
				self.bricks[-1]['angle'] = math.degrees(math.acos(np.dot(vector['dir'].ar,LEO.vectors[0]['dir'].ar)))
				self.bricks[-1]['time2'] = time2
				self.bricks[-1]['time1'] = time1
				self.bricks[-1]['laser_duration'] = brick_duration
				self.bricks[-1]['id'] = brick_ID
				self.bricks[-1]['segment_id'] = segment_id
				self.bricks[-1]['track_id'] = vector['track_id']
				self.bricks[-1]['layer_id'] = vector['layer_id']
				self.bricks[-1]['crd_max_ar'] = template_max.ar + shift.ar
				self.bricks[-1]['crd_min_ar'] = template_min.ar + shift.ar
				format1 = {'t':0.0,'x':start.xx,'y':start.yy,'z':start.zz,'pow':vector['power'],'sp':space_res,'ts':time_res}
				format2 = {'t':brick_duration,'x':end.xx,'y':end.yy,'z':end.zz,'pow':0.0,'sp':space_res,'ts':time_res}
				self.bricks[-1]['line1'] = "{t:12.{ts}},{x:6.{sp}},{y:6.{sp}},{z:6.{sp}},{pow}\n".format(**format1)
				self.bricks[-1]['line2'] = "{t:12.{ts}},{x:6.{sp}},{y:6.{sp}},{z:6.{sp}},{pow}\n".format(**format2)
				self.bricks[-1]['total_step'] = (vector['layer_id']+1)*2
				brick_ID+=1

	def match_bricks(self, query):
		# query is a coordinate tuple (x,y,z)
		
		for brick in self.bricks:
			if self.point_in_cuboid(query, brick['crd_max_ar'], brick['crd_min_ar']):
				print(brick['id'])
			
	def point_in_cuboid(self, point, max_corner, min_corner):
		if  point[0]>=min_corner[0] and point[0]<=max_corner[0] and\
			point[1]>=min_corner[1] and point[1]<=max_corner[1] and\
			point[2]>=min_corner[2] and point[2]<=max_corner[2]:
			return True
		else:
			return False
	
	def point_in_brick(self, brick, query):
		return self.point_in_cuboid(query, brick['crd_max_ar'], brick['crd_min_ar'])
	
	def move_wrt_brick(self, brick, query):
		query_ar = coord3D(query).ar
		local_query_ar = query_ar - brick['shift']
		return local_query_ar
		
	def time_wrt_brick(self, brick, relative_df):
		index_glb = relative_df.index + brick['time2']
		relative_df.index = index_glb
		return relative_df

	def find_at_time(self, given_time):
		# Find and return the brick that has the given time
		brick_list = []
		for brick in self.bricks:
			if 'nt11' in brick['query']:
				start_time = brick['time2']
				end_time = max(brick['query']['nt11'].columns)
				if given_time>=start_time and given_time<=end_time:
					brick_list.append(brick)
		return brick_list
		

	def write_file(self, TAG, ID, ZF):
		filename = TAG + str(ID).zfill(ZF) + '_AM_laser.inp'
		with open(filename,'w+') as file:
			file.write(self.bricks[ID]['line1'])
			file.write(self.bricks[ID]['line2'])


class PulseTracker:
	def __init__(self, LEO, rounding_digits, pulse_duration=0, pulse_length=0 ):
		# It is assumed that the laser moves in the positive x direction.
		# All coordinates are in Cartesian (x,y,z) system.
		#	LEO is the Laser Events Object
		
		time_res = rounding_digits['time']
		space_res = rounding_digits['space']
		
		# Dictonaries containing pulse metadata
		# 'id'			index of pulse in total
		# 'time2'		total start time of pulse
		# 'time1'		start time wrt current layer
		# 'start'		coordinates of laser start
		# 'end'			coordinates of laser end
		# 'coords'		coordinates of pulse
		# 'shift'		translation vector for mesh coordinates
		# 'angle'		angle of rotation wrt global coordinate system
		# 'pulse_duration'	the duration that laser is active as a pulse
		# 'segment_id'	index of brick model in its layers
		# 'track_id'	index of corresponding laser track
		# 'layer_id'	index of corresponding layer
		
		self.events = []
		self.laser_tracker = LEO
		pulse_ID = 0
		pulse_ID_per_layer = 0
		
		layer_start_time = 0
		for vector in LEO.vectors:
			# Set length or duration based on what is given
			if pulse_length == 0:
				pulse_length = round((pulse_duration/vector['duration'])*vector['length'],space_res)
			elif pulse_duration == 0:
				pulse_duration = round((pulse_length/vector['length'])*vector['duration'],time_res)
			else:
				ValueError('Not enough information about pulse given.')

			if vector['fresh_layer']:
				layer_start_time = vector['on']
				pulse_ID_per_layer = 0
			
			pulse_in_track_count = int(vector['length']/pulse_length)
			
			for segment_id in range(pulse_in_track_count):
				# a segment is defined wrt to current track/vector
				self.events.append({}) # add empty dictionary
				start = vector['start']+vector['dir']*pulse_length*segment_id
				end = start+vector['dir']*pulse_length
				mid = (start+end)/2
				time2 = round(vector['on']+segment_id*pulse_duration,time_res)
				time1 = round(time2-layer_start_time,time_res)
				shift = mid - LEO.vectors[0]['start']
				
				self.events[-1]['id'] = pulse_ID
				self.events[-1]['id_L'] = pulse_ID_per_layer
				self.events[-1]['coords'] = mid
				self.events[-1]['dir'] = vector['dir']
				self.events[-1]['shift'] = shift
				self.events[-1]['angle'] = vector_angle(vector['dir'],LEO.vectors[0]['dir'])
				self.events[-1]['time2'] = time2
				self.events[-1]['time1'] = time1
				self.events[-1]['time_act'] = time2+pulse_duration/2
				self.events[-1]['segment_id'] = segment_id
				self.events[-1]['track_id'] = vector['track_id']
				self.events[-1]['layer_id'] = vector['layer_id']
				self.events[-1]['dist_scan'] = abs(mid[0])
				self.events[-1]['dist_hatch'] = abs(mid[1])
				self.events[-1]['dist_build'] = abs(mid[2])
				# self.events[-1]['start'] = start
				# self.events[-1]['end'] = end
				# self.events[-1]['angle_degrees'] = math.degrees(math.acos(np.dot(vector['dir'],LEO.vectors[0]['dir'])))
				# self.events[-1]['pulse_duration'] = pulse_duration

				pulse_ID+=1
				pulse_ID_per_layer+=1
		
	def at_layer(self, L_id):
		event_subset = []
		
		for event in self.events:
			if event['layer_id'] == L_id:
				event_subset.append(event)
			elif event['layer_id'] > L_id:
				break
		
		return event_subset
	
	def LTr_event_id(self, L_id, T_id, ratio):
		target_time = self.laser_tracker.laser_time(L_id, T_id, ratio)
		for event in self.events:
			if event['time_act'] > target_time:
				return event['id']
	
	def at_layer_upto_LTr(self, L, T, r):
		event_subset = []
		event_id = self.LTr_event_id(L, T, r)
		
		for event in self.events:
			if event['layer_id'] == L:
				if event['id'] < event_id:
					event_subset.append(event)
			elif event['layer_id'] > L:
				break
		
		return event_subset


class GhostTracker:
	def __init__(self, pulse_events, boundary_planes, padding):
		all_ghosts = []
		
		# Mirror the pulses wrt all boundary planes
		for boundary in boundary_planes:
			boundary_type = list(boundary.keys())[0]
			for pulse in pulse_events:
				if boundary_type == 'y_bot' or boundary_type == 'y_top':
					y_p = pulse['coords'][1]
					mirror_vector = np.array([0,2*(boundary[boundary_type]-y_p),0])
					angle_addon = 0
					
					# x_vec = pulse['dir']
					# y_vec = np.matmul(rot_mat(np.pi/2),x_vec)
					

					# if round(abs(vector_angle(y_vec, mirror_vector)),1)>0:
						# y_factor = 1
					# else:
						# y_factor = -1
				elif boundary_type == 'x_bot' or boundary_type == 'x_top':
					x_p = pulse['coords'][0]
					mirror_vector = np.array([2*(boundary[boundary_type]-x_p),0,0])
					angle_addon = np.pi
					# y_factor = 1
					
				all_ghosts.append({
					'coords':pulse['coords']+mirror_vector,
					'angle':pulse['angle']+angle_addon,
					'time_act':pulse['time_act'],
					'boundary_type':boundary_type,
					# 'y_factor':y_factor,
				})
		
		cross_section = {
			'y_min':-0.56,
			'y_max': 0.56,
			'x_min':-4.04,
			'x_max': 4.04,
		}
		
		# Filter the ghost events
		ghost_subset = []
		for ghost in all_ghosts:
			if ghost['coords'][0]>cross_section['x_min']-padding['x_pad'] and\
			   ghost['coords'][0]<cross_section['x_max']+padding['x_pad'] and\
			   ghost['coords'][1]>cross_section['y_min']-padding['y_pad'] and\
			   ghost['coords'][1]<cross_section['y_max']+padding['y_pad']:
			   ghost_subset.append(ghost)
		
		ghost_subset = sorted(ghost_subset, key=lambda g: g['time_act']) 
		
		dist_min = {'x':0.035,'y':0.035}
		for id, ghost in enumerate(ghost_subset):
			# add an index to each event; *_L for Layer
			ghost['id_L'] = id
			if 'y' in ghost['boundary_type']:
				type = 'y'
			else:
				type = 'x'
			
			# Find the minimum distance to boundary
			dist_cur = box_dist(cross_section, ghost['coords'], type)
			dist_min[type] = min(dist_cur,dist_min[type])
			
			# Check if it is at minimum distance
			# if not then move the point
			if dist_cur-dist_min[type] >1e-6:
				if ghost['boundary_type'] == 'y_bot':
					ghost['coords'][1] = -0.595
				elif ghost['boundary_type'] == 'y_top':
					ghost['coords'][1] = 0.595
				elif ghost['boundary_type'] == 'x_bot':
					ghost['coords'][0] = -4.075
				elif ghost['boundary_type'] == 'x_top':
					ghost['coords'][0] = 4.075
			
			# update its distance value
			ghost['dist_border'] = round(dist_cur,6)

		self.events = ghost_subset


def box_dist(box, point, type):
	if type == 'x':
		return min(abs(point[0]-box['x_min']),
			abs(point[0]-box['x_max']))
	elif type == 'y':
		return min(abs(point[1]-box['y_min']),
			abs(point[1]-box['y_max']))

