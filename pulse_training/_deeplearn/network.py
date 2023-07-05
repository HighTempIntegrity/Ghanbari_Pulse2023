import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# import gc


class Swish(nn.Module):
	def __init__(self, ):
		super().__init__()

	def forward(self, x):
		return x * torch.sigmoid(x)

class Sin(nn.Module):
	def __init__(self, ):
		super().__init__()

	def forward(self, x):
		return torch.sin(x)

# Activation function translator
def activation_translate(name):
	if name in ['tanh', 'Tanh']:
		return nn.Tanh()
	elif name in ['relu', 'ReLU']:
		return nn.ReLU(inplace=True)
	elif name in ['lrelu', 'LReLU']:
		return nn.LeakyReLU(inplace=True)
	elif name in ['sigmoid', 'Sigmoid']:
		return nn.Sigmoid()
	elif name in ['softplus', 'Softplus']:
		return nn.Softplus(beta=1)
	elif name in ['celu', 'CeLU']:
		return nn.CELU()
	elif name in ['swish']:
		return Swish()
	elif name in ['sin']:
		return Sin()
	else:
		raise ValueError('Unknown activation function.')


class Exponential(nn.Module):
	def __init__(self, ):
		super().__init__()

	def forward(self, x):
		return torch.exp(x)
		
class Square(nn.Module):
	def __init__(self, ):
		super().__init__()

	def forward(self, x):
		return torch.square(x)
		
class Identity(nn.Module):
	def __init__(self, ):
		super().__init__()

	def forward(self, x):
		return x

def lastivation_translate(name):
	if name in ['softplus', 'Softplus']:
		return nn.Softplus(beta=1)
	elif name in ['exp']:
		return Exponential()
	elif name in ['square']:
		return Square()
	elif name in ['identity']:
		return Identity()
	else:
		raise ValueError('Unknown activation function.')
			
# Inherit a network class from nn.Module
class NeuralNetPulse(nn.Module):
	def __init__(self, Din, Dout, device, net_arch, events):
		super(NeuralNetPulse, self).__init__()

		self.init_pars = { # For saving model data and loading templates
			'Din':Din,
			'Dout':Dout,
			'device':device,
			'net_arch':net_arch,
			'events':events,
		}
		self.device = device
		self.events = events
		self.tf = net_arch['time_filter']
		self.sf = net_arch['space_filter']
		if 'multi_model' in net_arch:
			self.multi = net_arch['multi_model']
		else:
			self.multi = False
		
		layers = net_arch['hidden_layers']
		neurons = net_arch['neurons']
		
		# Network architecture
		self.activation = activation_translate(net_arch['activation'])
		self.input_layer = nn.Linear(Din, neurons)
		self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(layers)])
		self.output_layer = nn.Linear(neurons, Dout)
		self.lastivation = lastivation_translate(net_arch['last_act'])

	def forward(self, x):
		x = self.activation(self.input_layer(x))
		for k, l in enumerate(self.hidden_layers):
			x = self.activation(l(x))
		x = self.lastivation(self.output_layer(x))
		return x


	def save(self, epoch_history, file_name, fe_packs):
		save_data = {
			'init_pars':self.init_pars,
			'model_state_dict':self.state_dict(),
			'optimizer_state_dict':self.optimizer.state_dict(), # optimizer has to be defined later
			'history':epoch_history,
			'fe_model':fe_packs,
		}
		torch.save(save_data, file_name)

	# Sum model output based on pulse events
	def pulsum(self, x_tns):
		self.pulsum_init(x_tns)
		if self.multi:
			u_return = []
			for ii, cur_x_rep in enumerate(self.x_rep):
				cur_u_rep = torch.zeros((len(cur_x_rep), 1),device=self.device)
				cur_u_rep[self.active_points[ii]] = self(cur_x_rep[self.active_points[ii]].float())
				u_return.append(sum(torch.split(cur_u_rep,self.len_x[ii])))
			return u_return
		else: # old style
			u_rep = torch.zeros((len(self.x_rep), 1),device=self.device)
			u_rep[self.active_points] = self(self.x_rep[self.active_points].float())
			return sum(torch.split(u_rep,self.len_x))
		
		
	# Define the pulse based input vector and save it in the model
		# In combination with pulsum_internal() acts as pulsum()
	def pulsum_init(self, x_tns):
		if self.multi:
			self.len_x = []
			self.x_rep = []
			self.active_points = []
			for ii, cur_x in enumerate(x_tns):
				self.len_x.append(len(cur_x))
				self.x_rep.append(cur_x.repeat(len(self.events[ii]),1))
				for event in self.events[ii]:
					x_sub = slice(event['id_L']*self.len_x[-1],(event['id_L']+1)*self.len_x[-1],1)
					self.x_rep[-1][x_sub] = torch.matmul((self.x_rep[-1][x_sub]-event['shift_tns']).double(), event['rotate_tns'])
					self.x_rep[-1][x_sub] = self.space_transform(self.x_rep[-1][x_sub])
				self.active_points.append(self.filter_active(self.x_rep[-1]))
		else: # old style
			self.len_x = len(x_tns)
			self.x_rep = x_tns.repeat(len(self.events),1)
			for event in self.events:
				x_sub = slice(event['id_L']*self.len_x,(event['id_L']+1)*self.len_x,1)
				self.x_rep[x_sub] = torch.matmul((self.x_rep[x_sub]-event['shift_tns']).double(), event['rotate_tns'])
				self.x_rep[x_sub] = self.space_transform(self.x_rep[x_sub])
			self.active_points = self.filter_active()
				
	# Evaluate the function based on previously defined input
		# In combination with pulsum_init() acts as pulsum()
	def pulsum_internal(self):
		if self.multi:
			u_return = []
			for ii, cur_x_rep in enumerate(self.x_rep):
				cur_u_rep = torch.zeros((len(cur_x_rep), 1),device=self.device)
				cur_u_rep[self.active_points[ii]] = self(cur_x_rep[self.active_points[ii]].float())
				u_return.append(sum(torch.split(cur_u_rep,self.len_x[ii])))
			return u_return
		else: # old style
			u_rep = torch.zeros((len(self.x_rep), 1),device=self.device)
			u_rep[self.active_points] = self(self.x_rep[self.active_points].float())
			return sum(torch.split(u_rep,self.len_x))

	# Create a filter for points that need to be evaluated
	def filter_active(self, points_pls = None):
		if points_pls is None:
			points_pls = self.x_rep
		active_nodes = points_pls[:,1]**2>-1 # Init. with True
		if self.tf['cutoff']:
			active_time  = points_pls[:,3] > self.tf['cutoff_start']
			active_nodes = active_nodes & active_time
		if self.sf['cutoff']:
			active_space = (points_pls[:,0]/self.sf['x_dis'])**2+\
				(points_pls[:,1]/self.sf['y_dis'])**2+\
				(points_pls[:,2]/self.sf['z_dis'])**2-1 < self.sf['cutoff_value'] 
			active_nodes = active_nodes & active_space
		return active_nodes

	def pulsum_memory(self, x_tns, grad=False):
		self.len_x = len(x_tns)
		if grad:
			u_pred = self.pulsum_memory_innerworking(x_tns)
		else:
			with torch.no_grad():
				u_pred = self.pulsum_memory_innerworking(x_tns)
		return u_pred

	def pulsum_memory_innerworking(self, x_tns):
		u_pred = torch.zeros((len(x_tns), 1),device=self.device)
		for event in self.events:
			x_pulse = torch.clone(x_tns)
			x_pulse = torch.matmul((x_pulse-event['shift_tns']).float(), event['rotate_tns'].float())
			x_pulse = self.space_transform(x_pulse)
			active_points = self.filter_active(x_pulse)
			u_pred[active_points] += self(x_pulse[active_points].float())
		return u_pred

	def pulsum_memory_benchmark(self, x_tns, time_log):
		self.len_x = len(x_tns)
		with torch.no_grad():
			u_pred = torch.zeros((len(x_tns), 1),device=self.device)
			time_log.accumulate()
			for event in self.events:
				x_pulse = torch.clone(x_tns)
				x_pulse = torch.matmul((x_pulse-event['shift_tns']).float(), event['rotate_tns'].float())
				x_pulse = self.space_transform(x_pulse)
				active_points = self.filter_active(x_pulse)
				time_log.accum_begin()
				u_pred[active_points] += self(x_pulse[active_points].float())
				time_log.accum_end()
			time_log.append('Raw eval time.',time_log.accum_sum)
		return u_pred


	def pulsum_unique(self, x_tns):
		with torch.no_grad():
			# Prepare
			len_x = len(x_tns)
			x_rep = x_tns.repeat(len(self.events),1)
			for event in self.events:
				x_sub = slice(event['id_L']*len_x,(event['id_L']+1)*len_x,1)
				x_rep[x_sub] = torch.matmul((x_rep[x_sub]-event['shift_tns']).double(), event['rotate_tns'])
				x_rep[x_sub] = self.space_transform(x_rep[x_sub])
			
			# Find uniques in current evaluation
			x_unq_cur_, indx_cur_ = torch.unique(x_rep, return_inverse=True, dim=0)
			
			# if hasattr(self, 'x_unq_old'):
				# combined = torch.concat([self.x_unq_old,x_unq_cur_],dim=0)
				# x_unq_all_, indx_all_, counts_all_ = torch.unique(combined, return_inverse=True, return_counts=True, dim=0)
				# x_unq_new_ = x_unq_all_[counts_all_ == 1]
				# self.x_unq_old = x_unq_all_
				# print(len(self.x_unq_old))
			# else:
				# self.x_unq_old = x_unq_cur_

			
			active_points_cur_ = self.filter_active(x_unq_cur_)

			u_unique_cur_ = torch.zeros((len(x_unq_cur_), 1),device=self.device)
			u_unique_cur_[active_points_cur_] = self(x_unq_cur_[active_points_cur_].float())
			
			# Return u_rep for current evaluation
			u_rep = u_unique_cur_[indx_cur_]
			return sum(torch.split(u_rep,len_x))


	def pulsum_df(self, coords_df, time_cols, lbl_trns):
		response_df = pd.DataFrame(index=coords_df.index)
		coords_ar = coords_df.to_numpy()
		if not hasattr(self,'gpu_batch_limit'):
			self.gpu_batch_limit = 1
		while True:
			try:
				time_batches = np.array_split(time_cols,self.gpu_batch_limit)
				for time_batch in time_batches:
					cur_batch_input_tns = torch.empty(0,self.init_pars['Din'],device=self.device)
					for cur_time in time_batch:
						input_ar_ = np.pad(coords_ar,((0,0),(0,1)),'constant',constant_values=cur_time)
						cur_input_tns_ = torch.from_numpy(input_ar_).to(self.device)
						cur_batch_input_tns = torch.cat((cur_batch_input_tns,cur_input_tns_),0)
					with torch.no_grad():
						u_pred = self.pulsum(cur_batch_input_tns)
					pulse_ar = lbl_trns.u2T(u_pred.cpu().detach().numpy())
					pulse_ar_2d = np.reshape(pulse_ar,(-1,len(coords_df)))
					pulse_df = pd.DataFrame(pulse_ar_2d.T, index=coords_df.index, columns = time_batch)
					response_df = pd.concat([response_df,pulse_df],axis=1)
				break
			except RuntimeError as e:
				print(e)
				# print('Cuda ran out of memory with %i batches'%(self.gpu_batch_limit))
				self.gpu_batch_limit+=1
				torch.cuda.empty_cache()
		return response_df

	def pulsum_separate(self, x_tns):
		self.pulsum_init(x_tns)
		u_rep = torch.zeros((len(self.x_rep), 1),device=self.device)
		u_rep[self.active_points] = self(self.x_rep[self.active_points].float())
		return torch.split(u_rep,self.len_x)
	
	
	# Sum model output based on pulse events
	def pulsum_debug(self, x_tns):
		with torch.no_grad():
			self.pulsum_init(x_tns)
			u_rep = torch.zeros((len(self.x_rep), 1),device=self.device)
			u_rep[self.active_points] = self(self.x_rep[self.active_points].float())
			return torch.split(u_rep,self.len_x)


	def space_transform(self, points_pls):
		points_pls[:,1] = abs(points_pls[:,1]) # Mirror y values
		return points_pls

	# Free the internal varaibles saved through pulsum_init
	def free_internal(self):
		if hasattr(self, 'len_x'):
			del self.len_x
		if hasattr(self, 'x_rep'):
			del self.x_rep
		if hasattr(self, 'active_points'):
			del self.active_points
		torch.cuda.empty_cache()
		
	# Update model pulse events
	def update_events(self, new_events):
		self.events = new_events


class NeuralNetGeometry(NeuralNetPulse):
	def __init__(self, Din, Dout, device, net_arch, events):
		super().__init__(Din, Dout, device, net_arch, events)
		self.Dxtra = 3
		# cutoff = 3.52
		self.cutoff = 1.52
		
		self.input_layer = nn.Linear(Din+self.Dxtra, net_arch['neurons']) # for the distance info as input


	def pulsum_init(self, x_tns):
		if self.multi:
			self.len_x = []
			self.x_rep = []
			self.active_points = []
			for ii, cur_x in enumerate(x_tns):
				self.len_x.append(len(cur_x))
				self.x_rep.append(cur_x.repeat(len(self.events[ii]),1))
				for event in self.events[ii]:
					x_sub = slice(event['id_L']*self.len_x[-1],(event['id_L']+1)*self.len_x[-1],1)
					self.x_rep[-1][x_sub] = torch.matmul((self.x_rep[-1][x_sub]-event['shift_tns']).double(), event['rotate_tns'])
					self.x_rep[-1][x_sub] = self.space_transform(self.x_rep[-1][x_sub], event)
				# ##### New part for geomtery function
				self.x_rep[-1] = torch.nn.functional.pad(self.x_rep[-1],(0,self.Dxtra))
				for event in self.events[ii]:
					x_sub = slice(event['id_L']*self.len_x[-1],(event['id_L']+1)*self.len_x[-1],1)
					self.x_rep[-1][x_sub][:,4] = self.input_dist_scan(event)
					self.x_rep[-1][x_sub][:,5] = self.input_dist_hatch(event)
					self.x_rep[-1][x_sub][:,6] = self.input_dist_build(event)
				# ##### 
				self.active_points.append(self.filter_active(self.x_rep[-1]))
		else: # old style
			self.len_x = len(x_tns)
			self.x_rep = x_tns.repeat(len(self.events),1)
			for event in self.events:
				x_sub = slice(event['id_L']*self.len_x,(event['id_L']+1)*self.len_x,1)
				self.x_rep[x_sub] = torch.matmul((self.x_rep[x_sub]-event['shift_tns']).double(), event['rotate_tns'])
				self.x_rep[x_sub] = self.space_transform(self.x_rep[x_sub], event)
			# ##### New part for geomtery function
			self.x_rep = torch.nn.functional.pad(self.x_rep,(0,self.Dxtra))
			for event in self.events:
				x_sub = slice(event['id_L']*self.len_x,(event['id_L']+1)*self.len_x,1)
				self.x_rep[x_sub][:,4] = self.input_dist_scan(event)
				self.x_rep[x_sub][:,5] = self.input_dist_hatch(event)
				self.x_rep[x_sub][:,6] = self.input_dist_build(event)
			# ##### 
			self.active_points = self.filter_active()
			
	def pulsum_memory_innerworking(self, x_tns):
		u_pred = torch.zeros((len(x_tns), 1),device=self.device)
		for event in self.events:
			x_pulse = torch.clone(x_tns)
			x_pulse = torch.matmul((x_pulse-event['shift_tns']).float(), event['rotate_tns'].float())
			x_pulse = self.space_transform(x_pulse, event)
			# ##### New part for geomtery function
			x_pulse = torch.nn.functional.pad(x_pulse,(0,self.Dxtra))
			x_pulse[:,4] = self.input_dist_scan(event)
			x_pulse[:,5] = self.input_dist_hatch(event)
			x_pulse[:,6] = self.input_dist_build(event)
			# ##### 
			active_points = self.filter_active(x_pulse)
			u_pred[active_points] += self(x_pulse[active_points].float())
		return u_pred
	
	def pulsum_memory_benchmark(self, x_tns, time_log):
		self.len_x = len(x_tns)
		with torch.no_grad():
			u_pred = torch.zeros((len(x_tns), 1),device=self.device)
			time_log.accumulate()
			for event in self.events:
				x_pulse = torch.clone(x_tns)
				x_pulse = torch.matmul((x_pulse-event['shift_tns']).float(), event['rotate_tns'].float())
				x_pulse = self.space_transform(x_pulse, event)
				# ##### New part for geomtery function
				x_pulse = torch.nn.functional.pad(x_pulse,(0,self.Dxtra))
				x_pulse[:,4] = self.input_dist_scan(event)
				x_pulse[:,5] = self.input_dist_hatch(event)
				x_pulse[:,6] = self.input_dist_build(event)
				# ##### 
				active_points = self.filter_active(x_pulse)
				time_log.accum_begin()
				u_pred[active_points] += self(x_pulse[active_points].float())
				time_log.accum_end()
			time_log.append('Raw eval time.',time_log.accum_sum)
		return u_pred

	
	
	def space_transform(self, points_pls, event):
		if event['coords'][1] > 1e-6: # positive y
			if round(event['angle'])==0: # + x dir
				points_pls[:,1] = -1*points_pls[:,1]
		else: # negative y
			if round(event['angle'])!=0: # - x dir
				points_pls[:,1] = -1*points_pls[:,1]
				
		return points_pls
	
	def input_dist_scan(self, event):
		beta=100
		max=0.52
		return max-np.log(1+np.exp(beta*(event['dist_scan']-self.cutoff)))/beta
	
	def input_dist_hatch(self, event):
		max = 0.525
		return max-event['dist_hatch']

	def input_dist_build(self, event):
		return event['dist_build']
		
	def update_geom_cutoff(self,track_len):
		if track_len == 4:
			self.cutoff = 1.52
		elif track_len == 8:
			self.cutoff = 3.52
	
	
# Pre-determine the tranformation associated with each pulse/ghost event for 3D models
def append_transform(events, device):
	# for event in events.pulses:
	for event in events:
		event['shift_tns'] = torch.tensor([event['coords'][0], event['coords'][1], event['coords'][2], event['time_act']], device=device)
		angle_tns = torch.tensor(event['angle'], device=device)
		s = torch.sin(angle_tns)
		c = torch.cos(angle_tns)
		rotate_tns = torch.eye(4)
		rotate_tns[0][0:2] = torch.stack([c, s])
		rotate_tns[1][0:2] = torch.stack([-s, c])
		event['rotate_tns'] = rotate_tns.double().to(device)


class LabelTransform:
	# A class for transforming temperatures and energy in different ways
	def __init__(self, scale_type, ref_temp=25):
		self.type = scale_type
		self.a = [1/178, 5/78, 231/35260]
		self.b = [-25/178, -6215/78, 29431/35260]
		self.T = [1360,1399]
		self.rho = 8351.91e-9
		self.u = [749999*self.rho,1000000*self.rho]
		self.rT = ref_temp
		
		if self.type == 'linear':
			self.ls = 0.18
			self.li = 364.5
			self.rho = 8240e-9
		elif self.type == 'HX':
			self.ls = 0.2181
			self.li = 432.33
			self.rho = 8240e-9
			
	def pw_forward(self, x):
		return np.piecewise(x,
			condlist = [x < self.T[0],
						((x >= self.T[0]) & (x < self.T[1])),
						x >= self.T[1]],
            funclist = [lambda x:(self.a[0]*x+self.b[0])*1e5*self.rho,
                        lambda x:(self.a[1]*x+self.b[1])*1e5*self.rho,
                        lambda x:(self.a[2]*x+self.b[2])*1e5*self.rho])
		
	def pw_reverse(self, x):
		return np.piecewise(x,
            condlist = [x < self.u[0],
                        ((x >= self.u[0]) & (x < self.u[1])),
                        x >= self.u[1]],
            funclist = [lambda x:(x/(1e5*self.rho)-self.b[0])/self.a[0],
                        lambda x:(x/(1e5*self.rho)-self.b[1])/self.a[1],
                        lambda x:(x/(1e5*self.rho)-self.b[2])/self.a[2]])
	
	def T2u(self, temperatures):
		if self.type == 'constant':
			energies = (temperatures-self.rT)*(8220e-9*486)
		elif self.type == 'linear' or self.type == 'HX':
			energies = self.rho*(self.ls/2*(temperatures**2-self.rT**2) + self.li*(temperatures-self.rT))
		elif self.type == 'piecewise':
			energies = self.pw_forward(temperatures)
		return energies

	def u2T(self, energies):
		if self.type == 'constant':
			temperatures = energies/(8220e-9*486)+self.rT
		elif self.type == 'linear' or self.type == 'HX':
			temperatures = -self.li/self.ls+(2*energies/(self.rho*self.ls)+(self.rT+self.li/self.ls)**2)**0.5
		elif self.type == 'piecewise':
			temperatures = self.pw_reverse(energies)
		return temperatures
		
		
		
		
		
		
		
		
		