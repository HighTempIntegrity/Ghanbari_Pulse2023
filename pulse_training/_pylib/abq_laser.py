import numpy as np

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
				
		return points

	def LTr_to_time2(self, L=0, T=0, r=0):
		# in: layer_id, relative track_id, ratio of track
		# out: total time (time2)
		track_id = L*self.layers[0]['track_count']+T
		for vector in self.vectors:
			if track_id == vector['track_id']:
				return vector['on']+r*vector['duration']
	
	def LTr_to_trackID(self, L=0, T=0, r=0):
		# Returns total track id given the layer id and relative track id
		return L*self.layers[0]['track_count']+T
		

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
