# Pytorch stuff
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import pickle
import blosc

# Weight initializer
def init_xavier(model, retrain_seed):
	torch.manual_seed(retrain_seed)
	def init_weights(m):
		if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
			g = nn.init.calculate_gain('tanh')
			torch.nn.init.xavier_uniform_(m.weight, gain=g)
			#torch.nn.init.xavier_normal_(m.weight, gain=g)
			m.bias.data.fill_(0)
	model.apply(init_weights)

def define_levels(nt_df, batch_num, batch_size, aliasing_factor):
	Tbins = round(nt_df.to_numpy().max() - nt_df.to_numpy().min())
	counts = nt_df.stack().value_counts(bins=Tbins, sort=False)
	lefts = [ci.left for ci in counts.index]
	lefts.reverse()
	cumsum_counts = np.flip(counts.values).cumsum()
	cumsum_df = pd.DataFrame(lefts,cumsum_counts)
	cumsum_df=cumsum_df.sort_index()

	if 0 in aliasing_factor:
		sample_size = []
		for af, bs in zip([aliasing_factor[0],aliasing_factor[2]], [batch_size[0],batch_size[2]]):
			sample_size.append(af*bs*(batch_num+1))
		levels = []
		levels.append(cumsum_df.loc[sample_size[0]:,:].iloc[0,0])
		levels.append(cumsum_df.loc[cumsum_df.index[-1]-sample_size[1]:,:].iloc[0,0])
	else:
		sample_size = [zzip[0]*zzip[1]*(batch_num+1) for zzip in zip(aliasing_factor, batch_size)]
		levels = [cumsum_df.loc[sscs:,:].iloc[0,0] for sscs in np.cumsum(sample_size)]
	return levels
	
def simple_sampler(nt_df, coords_df, time_range, box_limits, frequency):
	time_vector = nt_df.columns.to_numpy()
	if time_range:
		start = time_range[0][0]
		end = time_range[0][1]
		time_vector = time_vector[time_vector>=start]
		time_vector = time_vector[time_vector<=end]
	
	coords_subset = coords_df
	if box_limits:
		a = coords_df>=box_limits[0]
		b = coords_df<=box_limits[1]
		c = a & b
		d = c['x']&c['y']&c['z']
		coords_subset = coords_df[d]

	coords_subset = coords_subset[::frequency]
	node_labels = coords_subset.index
	coords_vector = coords_subset.to_numpy()
	
	# Repeat the coords and times
	x_rep = np.repeat(coords_vector, repeats=len(time_vector), axis=0)
	t_rep = np.tile(time_vector,(len(coords_vector)))
	t_rep = t_rep.reshape(len(t_rep),1)
	samples = np.concatenate((x_rep,t_rep),axis=1)
	labels = nt_df.loc[node_labels,time_vector].values.reshape(-1,1)

	return samples, labels
		
def contour_sampler(nt_df, coords_df, time_range, **kwargs):
	time_vector = nt_df.columns
	if time_range: # In case a limited time range is defined
		start = time_range[0][0]
		end = time_range[0][1]
		time_vector = time_vector[time_vector>=start]
		time_vector = time_vector[time_vector<=end]
	box_limits = kwargs['space_subset']
	if box_limits:
		a = coords_df>=box_limits[0]
		b = coords_df<=box_limits[1]
		c = a & b
		box_flags = c['x']&c['y']&c['z']
	else:
		box_flags = pd.Series(data=True, index=coords_df.index)
	
	T_list = kwargs['levels']
	# Initialize empty arrays
	sample_batch = []
	label_batch = []
	for ii in range(len(T_list)+1):
		sample_batch.append(np.array([]).reshape(0,4))
		label_batch.append(np.array([]).reshape(0,1))
	
	for time_frame in time_vector:
		node_flags = coords_df['x']**2 < 0 # init boolean series with False
		for batch_id, T_level in enumerate(T_list):
			contour_flags = nt_df.loc[:,time_frame] > T_level
			node_flags = -node_flags & contour_flags
			cur_samples, cur_labels = define_sets(nt_df, coords_df, box_flags & node_flags, time_frame)
			sample_batch[batch_id] = np.concatenate((sample_batch[batch_id],cur_samples),axis=0)
			label_batch[batch_id] = np.concatenate((label_batch[batch_id],cur_labels),axis=0)
			node_flags = contour_flags

		last_samples, last_labels = define_sets(nt_df, coords_df, box_flags & -node_flags, time_frame)
		sample_batch[-1] = np.concatenate((sample_batch[-1],last_samples),axis=0)
		label_batch[-1] = np.concatenate((label_batch[-1],last_labels),axis=0)
		
	return sample_batch, label_batch

def contour_sampler_new(model_pack, levels, time_range, space_limits):
	nt_df = model_pack['nt11']
	coords_df = model_pack['coords']

	time_vector = nt_df.columns
	if time_range: # In case a limited time range is defined
		start = time_range[0][0]
		end = time_range[0][1]
		time_vector = time_vector[time_vector>=start]
		time_vector = time_vector[time_vector<=end]
	if space_limits:
		a = coords_df>=space_limits[0]
		b = coords_df<=space_limits[1]
		c = a & b
		box_flags = c['x']&c['y']&c['z']
	else:
		box_flags = pd.Series(data=True, index=coords_df.index)
	T_list = levels
	# Initialize empty arrays
	sample_batch = []
	label_batch = []
	for ii in range(len(T_list)+1):
		sample_batch.append(np.array([]).reshape(0,4))
		label_batch.append(np.array([]).reshape(0,1))
	
	for time_frame in time_vector:
		node_flags = coords_df['x']**2 < 0 # init boolean series with False
		for batch_id, T_level in enumerate(T_list):
			contour_flags = nt_df.loc[:,time_frame] > T_level
			node_flags = -node_flags & contour_flags
			cur_samples, cur_labels = define_sets(nt_df, coords_df, box_flags & node_flags, time_frame)
			sample_batch[batch_id] = np.concatenate((sample_batch[batch_id],cur_samples),axis=0)
			label_batch[batch_id] = np.concatenate((label_batch[batch_id],cur_labels),axis=0)
			node_flags = contour_flags

		last_samples, last_labels = define_sets(nt_df, coords_df, box_flags & -node_flags, time_frame)
		sample_batch[-1] = np.concatenate((sample_batch[-1],last_samples),axis=0)
		label_batch[-1] = np.concatenate((label_batch[-1],last_labels),axis=0)
	
	contour_keys = ['>%i'%(Tlvl) for Tlvl in levels]
	contour_keys.append('<%i'%(levels[-1]))
	
	model_pack['data'] = {}
	for ii, key in enumerate(contour_keys):
		model_pack['data'][key] = {
			'sampl_ar':sample_batch[ii],
			'label_ar':label_batch[ii],
			}

def full_sampler(model_pack, pickle_root, source='fe_data', save=False):
	# Initialize empty arrays
	data_batch = np.array([]).reshape(0,5)

	pickle_base_name = 'smp_%s'%(model_pack['name'])
	pickle_files = os.listdir(pickle_root)
	# Check if samples have been pickled
	if source == 'pk_file' and any([pickle_base_name in file for file in pickle_files]):
		# iterate over sampled pickles in order
		for file in sorted([file for file in pickle_files if pickle_base_name in file]):
			cur_rel_name = os.path.join(pickle_root, file)
			with open(cur_rel_name, "rb") as f:
				compressed_pickle = f.read()
			depressed_pickle = blosc.decompress(compressed_pickle)
			data_chunk = pickle.loads(depressed_pickle)  # turn bytes object back into data
			data_batch = np.concatenate((data_batch,data_chunk),axis=0)
	else:	
		nt_df = model_pack['nt11']
		coords_df = model_pack['coords']
		box_flags = pd.Series(data=True, index=coords_df.index)
		for time_frame in nt_df.columns:
			cur_data = define_sets_combined(nt_df, coords_df, box_flags, time_frame)
			data_batch = np.concatenate((data_batch,cur_data),axis=0)

		if save:
			chunksize = 2**25
			for ii, data_chunk in enumerate(np.array_split(data_batch,len(data_batch)//chunksize+1)):
				pickle_name = os.path.join(pickle_root, 'smp_%s-%05d.pk'%(model_pack['name'],ii))
				pickled_data = pickle.dumps(data_chunk)  # returns data as a bytes object
				compressed_pickle = blosc.compress(pickled_data)
				with open(pickle_name, "wb") as f:
					f.write(compressed_pickle)
	model_pack['data'] = data_batch
	return model_pack

def contour_divider(model_pack, levels, space_limits, time_range):
	data_full = model_pack['data']
	T_list = levels
	data_batch = []
	for ii in range(len(T_list)+1):
		data_batch.append(np.array([]).reshape(0,5))
	
	
	subset_flags = data_full[:,-1]**2 > -1 # init boolean series with True
	if space_limits:
		greater_flags = data_full[:,0:3]>=space_limits[0]
		lesser_flags = data_full[:,0:3]<=space_limits[1]
		combined_flags = greater_flags & lesser_flags
		space_flags = np.all(combined_flags,axis=1)
		subset_flags = subset_flags & space_flags
	
	if time_range: # In case a limited time range is defined
		start = time_range[0][0]
		end = time_range[0][1]
		later_flags = data_full[:,3]>=start
		early_flags = data_full[:,3]<=end
		time_flags = later_flags & early_flags
		subset_flags = subset_flags & time_flags
	
	memory_flags = data_full[:,-1]**2 < 0 # init boolean series with False
	memory_flags = memory_flags.flatten()
	for batch_id, T_level in enumerate(T_list):
		contour_flags = data_full[:,-1]>T_level
		contour_flags = contour_flags.flatten()
		memory_flags = ~memory_flags & contour_flags
		data_batch[batch_id] = data_full[memory_flags&subset_flags,:]
		memory_flags = contour_flags

	# sample_batch[-1] = sample_full[~memory_flags]
	# label_batch[-1] = label_full[~memory_flags]
	data_batch[-1] = data_full[~memory_flags&subset_flags]
	
	contour_keys = ['>%i'%(Tlvl) for Tlvl in levels]
	contour_keys.append('<%i'%(levels[-1]))
	
	model_pack['data'] = {}
	for ii, key in enumerate(contour_keys):
		model_pack['data'][key] = {
			'sampl_ar':data_batch[ii][:,:-1],
			'label_ar':data_batch[ii][:,-1].reshape(-1,1),
			'total_len':len(data_batch[ii]),
			}
	return model_pack

def lighten_samples(model_pack, rng, max_len):
	for lvl in model_pack['data']:
		cur_len = model_pack['data'][lvl]['total_len']
		if cur_len > max_len:
			data_batch = np.concatenate((model_pack['data'][lvl]['sampl_ar'],
				model_pack['data'][lvl]['label_ar'])
				,axis=1)
			idx = rng.integers(int(cur_len), size=int(max_len))
			data_batch = data_batch[idx,:]
			model_pack['data'][lvl]['sampl_ar'] = data_batch[:,:-1]
			model_pack['data'][lvl]['label_ar'] = data_batch[:,-1].reshape(-1,1)
	return model_pack
	
def define_sets(nt_df, coords_df, node_flags, time):
	coords_subset = coords_df[node_flags]
	coords_vector = coords_subset.to_numpy()
	node_labels = coords_subset.index
	new_samples = np.pad(coords_vector,((0,0),(0,1)),'constant',constant_values=time)
	new_labels = nt_df.loc[node_labels,time].values.reshape(-1,1)
	return new_samples, new_labels

def define_sets_combined(nt_df, coords_df, node_flags, time):
	coords_subset = coords_df[node_flags]
	coords_vector = coords_subset.to_numpy()
	node_labels = coords_subset.index
	new_samples = np.pad(coords_vector,((0,0),(0,1)),'constant',constant_values=time)
	new_labels = nt_df.loc[node_labels,time].values.reshape(-1,1)
	return np.concatenate((new_samples,new_labels),axis=1)


class PointGenerator:
	def __init__(self, sampling_type='sobol', seed=1, dimensions=1):
		self.type = sampling_type
		self.seed = seed
		self.dim = dimensions
		
		if self.type.lower() == 'sobol':
			from scipy.stats import qmc
			self.qmc = qmc
	
	def generate(self, num):
		if self.type.lower() == 'sobol':
			# Get the first power of 2 that is larger than sample size
			p2 = 1
			num2 = 2**p2
			while (num2<num):
				p2+=1
				num2=2**p2

			sampler = self.qmc.Sobol(d=self.dim , scramble=True, seed=self.seed)
			return sampler.random_base2(m=p2)
	