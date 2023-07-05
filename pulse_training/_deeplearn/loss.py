import torch
import torch.nn as nn
import torch.optim as optim
from sys import stdout
import os
import copy
import sys
import numpy as np
import pandas as pd

from . import utility
 
def loss_translate(loss_prop, new_batch_lens=None):
	if loss_prop['loss_type'] == 'L1':
		loss_fe = lambda u_t, u_p : nn.functional.l1_loss(u_t, u_p)
	elif (loss_prop['loss_type'] == 'MSE') or (loss_prop['loss_type'] == 'L2'):
		loss_fe = lambda u_t, u_p : torch.mean(torch.square(u_p.reshape(-1,) - u_t.reshape(-1,)))
	elif (loss_prop['loss_type'] == 'aymMSE'):
		loss_fe = lambda u_t, u_p : torch.mean(2**(torch.sign(u_p.reshape(-1,) - u_t.reshape(-1,)))*\
			torch.square(u_p.reshape(-1,) - u_t.reshape(-1,)))
	elif loss_prop['loss_type'] == 'RMSE':
		loss_fe = lambda u_t, u_p : torch.sqrt(torch.mean((u_t.reshape(-1, ) - u_p.reshape(-1, ))**2))
	elif loss_prop['loss_type'] == 'L3':
		loss_fe = lambda u_t, u_p : torch.mean((abs(u_t.reshape(-1, ) - u_p.reshape(-1, )))**3)
	elif loss_prop['loss_type'] == 'L4':
		loss_fe = lambda u_t, u_p : torch.mean((u_t.reshape(-1, ) - u_p.reshape(-1, ))**4)
	
	# if loss_prop['reg_factor']>0:
	loss_reg = lambda m: regularization(m, loss_prop['reg_power'])*loss_prop['reg_factor']

	if loss_prop['ysymm']['factor']>0:
		loss_ysymm = lambda m:ysymm_res(m, **loss_prop['ysymm'])*loss_prop['ysymm']['factor']

	if new_batch_lens is None:
		subs = loss_prop['subsets']
	else:
		subs = new_batch_lens
	wgts = loss_prop['weights']
	
	if loss_prop['model_type'] == 'pulse':
		if len(subs) == 1:
			subs = subs[0]
			wgts = wgts[0]
			loss_function = lambda m, u_t, u_p: {
				**{'fe_c%i'%(sub_id):\
					wgts[sub_id]*\
					loss_fe(u_t[subs[sub_id]:subs[sub_id+1]],
							u_p[subs[sub_id]:subs[sub_id+1]])\
					for sub_id in range(len(subs)-1)},
				**{'reg':  loss_reg(m),'ysymm':loss_ysymm(m),}
				}
		else:#MultiModel
			loss_function = lambda m, u_t, u_p: {
				**{'fe_m%ic%i'%(model_id,sub_id):\
					wgts[model_id][sub_id]*\
					loss_fe(u_t[model_id][subs[model_id][sub_id]:subs[model_id][sub_id+1]],
							u_p[model_id][subs[model_id][sub_id]:subs[model_id][sub_id+1]])\
					for model_id in range(len(subs)) for sub_id in range(len(subs[model_id])-1)},
				**{'reg':  loss_reg(m),'ysymm':loss_ysymm(m),}
				}
			
	elif loss_prop['model_type'] == 'geomtery':
		if len(subs) == 1:
			subs = subs[0]
			loss_function = lambda m, u_t, u_p: {
				**{'fe_c%i'%(sub_id):\
					loss_fe(u_t[subs[sub_id]:subs[sub_id+1]],
							u_p[subs[sub_id]:subs[sub_id+1]])\
					for sub_id in range(len(subs)-1)},
				**{'reg':  loss_reg(m),}
				}
		else:#MultiModel
			loss_function = lambda m, u_t, u_p: {
				**{'fe_m%ic%i'%(model_id,sub_id):\
					loss_fe(u_t[model_id][subs[model_id][sub_id]:subs[model_id][sub_id+1]],
							u_p[model_id][subs[model_id][sub_id]:subs[model_id][sub_id+1]])\
					for model_id in range(len(subs)) for sub_id in range(len(subs[model_id])-1)},
				**{'reg':  loss_reg(m),}
				}
	else:
		raise ValueError('Unknown model type.')
	
	return loss_function
	
# Custom pulse loss
class PulseLoss(torch.nn.Module):

	def __init__(self, loss_prop):
		super(PulseLoss, self).__init__()
		self.loss_fn = loss_translate(loss_prop)
		self.eval_type = loss_prop['model_evaluation_type']

	def forward(self, model, x_trains_, u_trains_):
		if self.eval_type == 'internal':
			u_preds_ = model.pulsum_internal()
		else: # memory
			u_preds_ = model.pulsum_memory(x_trains_, True)
		
		loss_terms = self.loss_fn(model, u_trains_, u_preds_ )
		loss_terms['total'] = sum(loss_terms.values())
		# del u_pred_
		return loss_terms


# Regularization function
def regularization(model, power):
    reg_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss = reg_loss + torch.norm(param, power)
    return reg_loss

def ysymm_res(model, input_tns=None, **kwargs):
	if input_tns == None:
		pgen = utility.PointGenerator('sobol', kwargs['seed'], 3)
	
		# points = pgen.generate(kwargs['points_num'])
		# points[:,0] = points[:,0]*(kwargs['x_bounds'][1]-kwargs['x_bounds'][0])+kwargs['x_bounds'][0]
		# points[:,1] = points[:,1]*(kwargs['z_bounds'][1]-kwargs['z_bounds'][0])+kwargs['z_bounds'][0]
		# points[:,2] = points[:,2]*(kwargs['t_bounds'][1]-kwargs['t_bounds'][0])+kwargs['t_bounds'][0]
		# points = np.insert(points, 1, 0, axis=1)
		# input_tns = torch.tensor(points, device=model.device)
		
		points_redist = pgen.generate(kwargs['points_num'])
		points_redist[:,0] = 1/4*np.arctanh(2*(points_redist[:,0]-0.5))
		points_redist[:,[1,2]] = 1/4*np.arctanh(points_redist[:,[1,2]])
		points_redist[:,1] = -points_redist[:,1]
		points_redist = np.insert(points_redist, 1, 0, axis=1)
		redist_tns = torch.tensor(points_redist, device=model.device)
		input_tns = redist_tns
		
		# input_tns = torch.cat((input_tns,extra_tns),0)
		
	du = grad_spacetime(model, input_tns)

	residual = du['y']
	return torch.mean(residual**2)
	# return torch.mean(abs(residual))

def grad_spacetime(model, input_tns):
    input_tns.requires_grad = True
    u = model(input_tns.float())
    du = {}
    u_grad, = torch.autograd.grad(u, input_tns, grad_outputs=torch.ones_like(u), create_graph=True)
    # du['x'] = u_grad[:,0]
    du['y'] = u_grad[:,1]
    # du['z'] = u_grad[:,2]
    # du['t'] = u_grad[:,3]
    # du['xx'] = torch.autograd.grad(du['x'], input_tns, grad_outputs=torch.ones_like(du['x']), create_graph=True)[0][:,0]
    du['yy'] = torch.autograd.grad(du['y'], input_tns, grad_outputs=torch.ones_like(du['y']), create_graph=True)[0][:,1]
    # du['zz'] = torch.autograd.grad(du['z'], input_tns, grad_outputs=torch.ones_like(du['z']), create_graph=True)[0][:,2]
    return du


# Training the pulse model
def train(model, training_packs, test_dict, opt_prop, loss_prop, res_prop, time_log, save_folder, fe_packs):
	optimizer = model.optimizer
	
	fe_model_packs = [{'name':mp['name'],'directory':mp['directory'],'laser':mp['laser']} for mp in fe_packs]
	
	max_loss_jump = opt_prop['max_loss_jump']
	switch_high_loss_jump = False
	# Loading data to resume training
	if res_prop['checkpoint']:
		load = res_prop['checkpoint']
		epoch_history = load['history']
		epoch_range = range(len(epoch_history),len(epoch_history)+opt_prop['epochs'])
		if res_prop['load_model']:
			model.load_state_dict(load['model_state_dict'])
		prev_batch_loss = None
		prev_peak_loss = None
		if res_prop['load_optimizer']:
			optimizer.load_state_dict(load['optimizer_state_dict'])
			for epoch in reversed(epoch_history):
				for batch in reversed(epoch['batches']):
					if batch['converged']:
						prev_batch_loss = batch['loss']	# Gets last good batch loss for divergence check
						prev_peak_loss = pd.DataFrame(batch['iterations'])['loss'].max()
						break
				if prev_batch_loss is not None:
					break
		else:
			# prev_batch_loss = None
			max_loss_jump = opt_prop['max_loss_jump']*10
			switch_high_loss_jump = True
		time_log.out('Training was loaded.')
	else:	# Training from scratch
		epoch_history = []
		epoch_range = range(opt_prop['epochs'])
	
	scheduler_down = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt_prop['scheduler_gamma'],verbose=True)
	scheduler_up = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt_prop['scheduler_gamma']**-0.25,verbose=True)
	patience_counter = 0

	loss_obj = PulseLoss(loss_prop)
	
	# List of loss values through training
	# This is returned in the end
	model_state_dict = copy.deepcopy(model.state_dict()) # To retry in case of divergence
	optimizer_state_dict = copy.deepcopy(optimizer.state_dict())# To retry in case of divergence
	
	
	# Loop over epochs
	time_log.in_right()
	for epoch_id in epoch_range:
		time_log.start()
		model.train()	# Set model to training mode
		epoch_nfo = {'batches':[]}	# Items in epoch_history
		time_log.out('='*40)	# Divider in cmd window
		
		# Create iterators for extra training sets
		if len(training_packs[0])>1:
			dl_iterators0 = [iter(dl) for dl in training_packs[0]['data_loaders'][1:]]
			
		# Iterators for multi model training
		if len(training_packs)>1:
			for tp in training_packs[1:]:
				tp['dl_iterators'] = [iter(dl) for dl in tp['data_loaders']]

		# Loop over batches of training data
		# For each new epoch, the batches are newly randomized
		time_log.in_right()
		# https://stackoverflow.com/questions/51444059/how-to-iterate-over-two-dataloaders-simultaneously-using-pytorch
		for batch_id, (x_train_0,u_train_0) in enumerate(training_packs[0]['data_loaders'][0]):
			if len(training_packs[0])>1:
				for dl_iter0 in dl_iterators0:
					(x_train_addon_, u_train_addon_) = next(dl_iter0)
					x_train_0 = torch.cat((x_train_0,x_train_addon_),0)
					u_train_0 = torch.cat((u_train_0,u_train_addon_),0)
			x_trains_ = [x_train_0]
			u_trains_ = [u_train_0]
			if len(training_packs)>1: #MultiModel
				for tp in training_packs[1:]:
					for ii, dl_iter in enumerate(tp['dl_iterators']):
						if ii == 0:
							(cur_x_train_, cur_u_train_) = next(dl_iter)
						else:
							(x_train_addon_, u_train_addon_) = next(dl_iter)
							cur_x_train_ = torch.cat((cur_x_train_,x_train_addon_),0)
							cur_u_train_ = torch.cat((cur_u_train_,u_train_addon_),0)
					x_trains_.append(cur_x_train_)
					u_trains_.append(cur_u_train_)
			else:
				u_trains_ = u_trains_[0]
				
			# x_train_	batch subset of training data input
			# u_train_	batch subset of training data output
			# (x_train_, u_train_) = dataloaders[0]
			
			# Limiting the number of batches
			if epoch_id == 0:
				if batch_id > 0: # Do only 1 batch in the beginning
					break
			else:
				if batch_id == opt_prop['batch_num']:
					break
			
			# Batch timing is only relevant when there is more than 1 batch
			# Otherwise an epoch is the same as full batch
			if opt_prop['batch_num'] > 1:
				time_log.start()
			
			# Preparation for the current batch
			batch_nfo = {'iterations':[]}
			it_id = [0]
			if loss_prop['model_evaluation_type'] == 'internal':
				if len(training_packs)==1:
					model.pulsum_init(x_trains_[0]) # Define pulse input once for batch
				else: #MultiModel
					model.pulsum_init(x_trains_) # Define pulse input once for batch

			def closure():
				# zero/none the parameter gradients
				optimizer.zero_grad(set_to_none=True)
				# Compute the loss function over the batch
				loss_dict = loss_obj.forward(model, x_trains_, u_trains_)
				# Compute the gradient with respect to the network parameters:
				loss_dict['total'].backward()
				
				batch_nfo['iterations'].append({'loss':loss_dict['total'].item()})
				if opt_prop['algorithm']=='LBFGS' and sys.platform == 'win32':
					loss_report = '  '.join(['%s: %.3e(%.2f)'%(k, loss_dict[k], loss_dict[k]/loss_dict['total']) for k in loss_dict.keys()])
					stdout.write('\r%sEpoch %i Batch %i Iter %03d - %s'%(time_log.in_str(),
						epoch_id, batch_id, it_id[0], loss_report))
					stdout.flush()
				it_id[0]+=1
				return loss_dict['total']
			
			# Update the parameters according to the chosen optimizer
			# The training data remains the same but network is updated
			# in every iteration
			optimizer.step(closure)
			
			# Code snippet to free up gpu memory
			if torch.cuda.is_available():
				del x_trains_
				del u_trains_
				model.free_internal()
				torch.cuda.empty_cache()
			
			print() # Newline after iterations
			iteration_losses = [i['loss'] for i in batch_nfo['iterations']]
			batch_nfo['loss'] = sum(iteration_losses)/len(iteration_losses)
			epoch_nfo['batches'].append(batch_nfo)
			# Show batch info if there is more than one
			if opt_prop['batch_num'] > 1:
				time_log.append('Epoch %i Batch %i - loss: %.3e'%(epoch_id, batch_id,batch_nfo['loss']))
				time_log.out('-'*20)
			
			## Divergence check
			# Define current batch loss
			cur_batch_loss = batch_nfo['loss']
			cur_peak_loss = pd.DataFrame(batch_nfo['iterations'])['loss'].max()
			# Define last batch loss
			if (epoch_id == 0 and batch_id == 0) or\
				(prev_batch_loss == None): # At the very beginning
				prev_batch_loss = cur_batch_loss
				prev_peak_loss = cur_peak_loss
				
			# The check
			if cur_batch_loss/prev_batch_loss > max_loss_jump or\
				cur_peak_loss/prev_peak_loss > max_loss_jump*10 or\
				torch.isnan(torch.tensor(cur_batch_loss)):
				time_log.out('Optimization diverged.')
				time_log.out('   Loss ratio:     %.2f (%.2f)'%(cur_batch_loss/prev_batch_loss, max_loss_jump))
				time_log.out('   Peak ratio:     %.2f (%.2f)'%(cur_peak_loss/prev_peak_loss, 10*max_loss_jump))
				time_log.out('Aborting current epoch.')
				epoch_nfo['batches'][-1]['converged']=False
				model.load_state_dict(model_state_dict)
				optimizer.load_state_dict(optimizer_state_dict)
				# Adjust the learning rate based on scheduler
				scheduler_down.step()
				time_log.out('Adjusted learning rate to %.3e'%(scheduler_down.get_last_lr()[0]))
				patience_counter = -1
				break
			else:	# If it's converged
				prev_batch_loss = cur_batch_loss
				prev_peak_loss = cur_peak_loss
				epoch_nfo['batches'][-1]['converged']=True
				if switch_high_loss_jump: # In case of restart
					switch_high_loss_jump=False
					max_loss_jump = opt_prop['max_loss_jump']
		time_log.in_left()
		
		# Save the state for loading in the next epoch
		model_state_dict = copy.deepcopy(model.state_dict())
		optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
		
		# Code snippet from main PINN repo to free up gpu memory
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
				
		# Calculate loss and test for epoch
		batch_losses = [b['loss'] for b in epoch_nfo['batches']]
		epoch_nfo['loss'] = sum(batch_losses)/len(batch_losses)
		epoch_nfo['test'] = test_function(model, test_dict, loss_prop)

		if opt_prop['LBFGS']['max_iter'] > 1:
			time_log.append('Epoch %5i - train loss: %.3e'%(epoch_id,epoch_nfo['loss']))
			time_log.out('%s test   MSE: %.3e - NMAE: %.2f%% - R2: %.2f%%'%(' '*13,
				epoch_nfo['test']['MSE'],
				epoch_nfo['test']['NMAE'],
				epoch_nfo['test']['R2']))
		epoch_history.append(epoch_nfo)
		epoch_file_name = '%s_e%0.4i.pth'%(loss_prop['model_type'], epoch_id)
		model.save(epoch_history,os.path.join(save_folder,epoch_file_name), fe_model_packs)
	
		patience_counter+=1
		if patience_counter == opt_prop['patience']:
			scheduler_up.step()
			time_log.out('Adjusted learning rate to %.3e'%(scheduler_up.get_last_lr()[0]))
			patience_counter = 0

			
		## Early stopping
		if epoch_id > 1:
			if abs(epoch_nfo['loss']-epoch_history[-2]['loss'])<opt_prop['min_loss_change']:
				time_log.out('%sLittle improvement in loss observed. Stopping optimization.'%(time_log.in_str()))
				break
		
	time_log.in_left()
	
	return epoch_history


def test_function(model, test_dict, loss_prop):
	model.eval() # Ready model for evaluation
	if len(test_dict['x_test'])==1:#SingleModel
		x_test = test_dict['x_test'][0]
		y_test = test_dict['y_test'][0]
		# subsets = test_dict['subsets'][0]
	else:#MultiModel
		x_test = test_dict['x_test']
		y_test = test_dict['y_test']
		# subsets = test_dict['subsets']
		
	
	y_test_pred = model.pulsum(x_test)
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	
	loss_fn = loss_translate(loss_prop, test_dict['subsets'])
	loss_terms = loss_fn(model, y_test_pred, y_test)
	loss_terms['total'] = sum(loss_terms.values())
	loss_terms = {key:float(loss_terms[key].cpu().detach().numpy()) for key in loss_terms}
	
	# if len(test_dict['x_test'])>1:#MultiModel
		# y_test = y_test[0]
		# y_test_pred = y_test_pred[0]
		
	if len(test_dict['x_test']) == 1:#SingleModel
		y_test = [y_test]
		y_test_pred = [y_test_pred]
	
	
	# test_range = torch.max(tset['y_test'])-torch.min(tset['y_test'])
	test_metrics = {
		'MSE':[],
		'NMAE':[],
		'NRMSE':[],
		'R2':[],
	}
	for mii, y_test_ in enumerate(y_test):
		test_mean = torch.mean(torch.abs(y_test_))
		MSE = torch.mean(torch.square(y_test_ - y_test_pred[mii]))
		test_metrics['MSE'].append(MSE)
		test_metrics['R2'].append(1-MSE/torch.var(y_test_, unbiased=False))
		test_metrics['NRMSE'].append(torch.sqrt(MSE)/test_mean)
		test_metrics['NMAE'].append(torch.mean(torch.abs(y_test_-y_test_pred[mii]))/test_mean)
	for key in test_metrics:
		test_metrics[key] = [x.cpu().detach().numpy() for x in test_metrics[key]]
		test_metrics[key] = sum(test_metrics[key])/len(test_metrics[key])
		if key in ['NMAE','NRMSE','R2']:
			test_metrics[key]*=100
	test_metrics['loss'] = loss_terms

	if torch.cuda.is_available():
		del y_test_pred
		del test_mean
		del MSE
		torch.cuda.empty_cache()
	return test_metrics

