
import pandas as pd
import numpy as np
import random


from sklearn import model_selection
from sklearn.metrics import mean_squared_error


def create_folds(data, num_splits):
	data["kfold"] = -1
	kf = model_selection.KFold(n_splits=num_splits, shuffle=True, random_state=2021)
	for f, (t_, v_) in enumerate(kf.split(X=data)):
		data.loc[v_, 'kfold'] = f
	return data

####    Import Dependencies - Augmentation    #######################
from augmentation import Augmenter


####    Import Dependencies - Modelling    #######################


from glob import glob
import os
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import gc
gc.enable()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import (
	Dataset, DataLoader, 
	SequentialSampler, RandomSampler
)
from transformers import AutoConfig
from transformers import (
	get_cosine_schedule_with_warmup, 
	get_cosine_with_hard_restarts_schedule_with_warmup,
	get_linear_schedule_with_warmup
)
from transformers import AdamW
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
from IPython.display import clear_output
from tqdm import tqdm, trange


####    CONVERT EXAMPLES TO FEATURES    ############################


def convert_examples_to_features(data, tokenizer, max_len, is_test=False):
	data = data.replace('\n', '')
	tok = tokenizer.encode_plus(
		data, 
		max_length=max_len, 
		truncation=True,
		return_attention_mask=True,
		return_token_type_ids=True
	)
	curr_sent = {}
	padding_length = max_len - len(tok['input_ids'])
	curr_sent['input_ids'] = tok['input_ids'] + ([tokenizer.pad_token_id] * padding_length)
	curr_sent['token_type_ids'] = tok['token_type_ids'] + \
		([0] * padding_length)
	curr_sent['attention_mask'] = tok['attention_mask'] + \
		([0] * padding_length)
	return curr_sent



####    DATA SET RETRIEVER    ######################################



class DatasetRetriever(Dataset):
	def __init__(self, data, tokenizer, max_len, is_test=False, is_weighted= False):
		self.data = data
		if 'excerpt' in self.data.columns:
			self.excerpts = self.data.excerpt.values.tolist()
		else:
			self.excerpts = self.data.text.values.tolist()
		self.targets = self.data.target.values.tolist()
		self.tokenizer = tokenizer
		self.is_test = is_test
		self.max_len = max_len
		self.is_weighted = is_weighted
		if self.is_weighted:
			self.weights = self.data.weights.values.tolist()
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, item):
		excerpt, label = self.excerpts[item], self.targets[item]
		features = convert_examples_to_features(
			excerpt, self.tokenizer, 
			self.max_len, self.is_test
		)

		if self.is_weighted:
			weights = self.weights[item]
			return {
				'input_ids':torch.tensor(features['input_ids'], dtype=torch.long),
				'token_type_ids':torch.tensor(features['token_type_ids'], dtype=torch.long),
				'attention_mask':torch.tensor(features['attention_mask'], dtype=torch.long),
				'label':torch.tensor(label, dtype=torch.double),
				'weights':torch.tensor(weights, dtype=torch.double),

			}

		else:
			return {
				'input_ids':torch.tensor(features['input_ids'], dtype=torch.long),
				'token_type_ids':torch.tensor(features['token_type_ids'], dtype=torch.long),
				'attention_mask':torch.tensor(features['attention_mask'], dtype=torch.long),
				'label':torch.tensor(label, dtype=torch.double),
			}



####    MODEL    ####################################################



class CommonLitModel(nn.Module):
	def __init__(
		self, 
		model_name, 
		config,  
		multisample_dropout=False,
		output_hidden_states=False
	):
		super(CommonLitModel, self).__init__()
		self.config = config
		self.roberta = AutoModel.from_pretrained(
			model_name, 
			output_hidden_states=output_hidden_states
		)
		self.layer_norm = nn.LayerNorm(config.hidden_size)
		if multisample_dropout:
			self.dropouts = nn.ModuleList([
				nn.Dropout(0.5) for _ in range(5)
			])
		else:
			self.dropouts = nn.ModuleList([nn.Dropout(0.3)])
		#self.regressor = nn.Linear(config.hidden_size*2, 1)
		self.regressor = nn.Linear(config.hidden_size, 1)
		self._init_weights(self.layer_norm)
		self._init_weights(self.regressor)
 
	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
			if module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.Embedding):
			module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
			if module.padding_idx is not None:
				module.weight.data[module.padding_idx].zero_()
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)
 
	def forward(
		self, 
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		labels=None
	):
		outputs = self.roberta(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
		)
		sequence_output = outputs[1]
		sequence_output = self.layer_norm(sequence_output)
 
		# max-avg head
		# average_pool = torch.mean(sequence_output, 1)
		# max_pool, _ = torch.max(sequence_output, 1)
		# concat_sequence_output = torch.cat((average_pool, max_pool), 1)
 
		# multi-sample dropout
		for i, dropout in enumerate(self.dropouts):
			if i == 0:
				logits = self.regressor(dropout(sequence_output))
			else:
				logits += self.regressor(dropout(sequence_output))
		
		logits /= len(self.dropouts)
 
		# calculate loss
		loss = None
		if labels is not None:
			# regression task
			loss_fn = torch.nn.MSELoss()
			logits = logits.view(-1).to(labels.dtype)
			loss = torch.sqrt(loss_fn(logits, labels.view(-1)))
		
		output = (logits,) + outputs[2:]
		return ((loss,) + output) if loss is not None else output


####    LAMB OPTIMIZER    #######################################################



class Lamb(Optimizer):
	# Reference code: https://github.com/cybertronai/pytorch-lamb

	def __init__(
		self,
		params,
		lr: float = 1e-3,
		betas = (0.9, 0.999),
		eps: float = 1e-6,
		weight_decay: float = 0,
		clamp_value: float = 10,
		adam: bool = False,
		debias: bool = False,
	):
		if lr <= 0.0:
			raise ValueError('Invalid learning rate: {}'.format(lr))
		if eps < 0.0:
			raise ValueError('Invalid epsilon value: {}'.format(eps))
		if not 0.0 <= betas[0] < 1.0:
			raise ValueError(
				'Invalid beta parameter at index 0: {}'.format(betas[0])
			)
		if not 0.0 <= betas[1] < 1.0:
			raise ValueError(
				'Invalid beta parameter at index 1: {}'.format(betas[1])
			)
		if weight_decay < 0:
			raise ValueError(
				'Invalid weight_decay value: {}'.format(weight_decay)
			)
		if clamp_value < 0.0:
			raise ValueError('Invalid clamp value: {}'.format(clamp_value))

		defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
		self.clamp_value = clamp_value
		self.adam = adam
		self.debias = debias

		super(Lamb, self).__init__(params, defaults)

	def step(self, closure = None):
		loss = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				grad = p.grad.data
				if grad.is_sparse:
					msg = (
						'Lamb does not support sparse gradients, '
						'please consider SparseAdam instead'
					)
					raise RuntimeError(msg)

				state = self.state[p]

				# State initialization
				if len(state) == 0:
					state['step'] = 0
					# Exponential moving average of gradient values
					state['exp_avg'] = torch.zeros_like(
						p, memory_format=torch.preserve_format
					)
					# Exponential moving average of squared gradient values
					state['exp_avg_sq'] = torch.zeros_like(
						p, memory_format=torch.preserve_format
					)

				exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
				beta1, beta2 = group['betas']

				state['step'] += 1

				# Decay the first and second moment running average coefficient
				# m_t
				exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
				# v_t
				exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

				# Paper v3 does not use debiasing.
				if self.debias:
					bias_correction = math.sqrt(1 - beta2 ** state['step'])
					bias_correction /= 1 - beta1 ** state['step']
				else:
					bias_correction = 1

				# Apply bias to lr to avoid broadcast.
				step_size = group['lr'] * bias_correction

				weight_norm = torch.norm(p.data).clamp(0, self.clamp_value)

				adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
				if group['weight_decay'] != 0:
					adam_step.add_(p.data, alpha=group['weight_decay'])

				adam_norm = torch.norm(adam_step)
				if weight_norm == 0 or adam_norm == 0:
					trust_ratio = 1
				else:
					trust_ratio = weight_norm / adam_norm
				state['weight_norm'] = weight_norm
				state['adam_norm'] = adam_norm
				state['trust_ratio'] = trust_ratio
				if self.adam:
					trust_ratio = 1

				p.data.add_(adam_step, alpha=-step_size * trust_ratio)

		return loss



#### DIFFERENTIAL LEARNING RATE AND WEIGHT DECAY    ################################

def get_optimizer_params(model, base_model=''):
	# differential learning rate and weight decay
	param_optimizer = list(model.named_parameters())
	learning_rate = 2e-5
	no_decay = ['bias', 'gamma', 'beta']
	if base_model == 'roberta-base':
		group1=['layer.0.','layer.1.','layer.2.','layer.3.']
		group2=['layer.4.','layer.5.','layer.6.','layer.7.']    
		group3=['layer.8.','layer.9.','layer.10.','layer.11.']
		group_all=['layer.0.','layer.1.','layer.2.','layer.3.','layer.4.','layer.5.','layer.6.','layer.7.','layer.8.','layer.9.','layer.10.','layer.11.']
	elif base_model == 'roberta-large':
		group1=['layer.0.','layer.1.','layer.2.','layer.3.', 'layer.4.','layer.5.','layer.6.','layer.7.'] 
		group2=['layer.8.','layer.9.','layer.10.','layer.11.','layer.12.','layer.13.','layer.14.','layer.15.']
		group3=['layer.16.','layer.17.','layer.18.','layer.19.','layer.20.','layer.21.','layer.22.','layer.23.']
		group_all=group1+group2+group3

	optimizer_parameters = [
		{'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': 0.01},
		{'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': 0.01, 'lr': learning_rate/2.6},
		{'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': 0.01, 'lr': learning_rate},
		{'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': 0.01, 'lr': learning_rate*2.6},
		{'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': 0.0},
		{'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': 0.0, 'lr': learning_rate/2.6},
		{'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': 0.0, 'lr': learning_rate},
		{'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': 0.0, 'lr': learning_rate*2.6},
		{'params': [p for n, p in model.named_parameters() if "roberta" not in n], 'lr':1e-3, "momentum" : 0.99},
	]
	return optimizer_parameters



####    UTILITIES     ################################################################



def make_model(model_name='../content/roberta-base-5-epochs/', tokenizer_name= '', num_labels=1):
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
	config = AutoConfig.from_pretrained(model_name)
	config.update({'num_labels':num_labels})
	model = CommonLitModel(model_name, config=config)
	return model, tokenizer

def make_optimizer(model, optimizer_name="AdamW", base_model=''):
	optimizer_grouped_parameters = get_optimizer_params(model, base_model)
	kwargs = {
			'lr':2e-5,
			'weight_decay':0.01,
			'betas': (0.9, 0.98),
			'eps': 1e-06
	}
	if optimizer_name == "LAMB":
		optimizer = Lamb(optimizer_grouped_parameters, **kwargs)
		return optimizer
	elif optimizer_name == "Adam":
		from torch.optim import Adam
		optimizer = Adam(optimizer_grouped_parameters, **kwargs)
		return optimizer
	elif optimizer_name == "AdamW":
		optimizer = AdamW(optimizer_grouped_parameters, **kwargs)
		return optimizer
	else:
		raise Exception('Unknown optimizer: {}'.format(optimizer_name))

def make_scheduler(optimizer, decay_name='linear', t_max=None, warmup_steps=None):
	if decay_name == 'step':
		scheduler = optim.lr_scheduler.MultiStepLR(
			optimizer,
			milestones=[30, 60, 90],
			gamma=0.1
		)
	elif decay_name == 'cosine':
		scheduler = lrs.CosineAnnealingLR(
			optimizer,
			T_max=t_max
		)
	elif decay_name == "cosine_warmup":
		scheduler = get_cosine_schedule_with_warmup(
			optimizer,
			num_warmup_steps=warmup_steps,
			num_training_steps=t_max
		)
	elif decay_name == "linear":
		scheduler = get_linear_schedule_with_warmup(
			optimizer, 
			num_warmup_steps=warmup_steps, 
			num_training_steps=t_max
		)
	else:
		raise Exception('Unknown lr scheduler: {}'.format(decay_type))    
	return scheduler    

def make_loader(
	data, 
	tokenizer, 
	max_len,
	batch_size,
	fold=0,
	augmentation=False,
	augmentation_config_location=None,
	is_weighted = False
):
	train_set, valid_set = data[data['kfold']!=fold], data[data['kfold']==fold]

	if augmentation:
		augmenter = Augmenter(augmentation_config_location)
		aug_data = augmenter.generate(train_set)
		aug_data['kfold'] = fold
		train_set = pd.concat([train_set, aug_data])


	if is_weighted:
		pow_fn = 2
		weight_range = [0.7,1.3]
		train_set.loc[:,'weights'] = 1 - ((train_set['standard_error']**pow_fn) - (train_set['standard_error'].min()**pow_fn)) / (train_set['standard_error'].max()**pow_fn)
		train_set.loc[:,'weights'] = weight_range[0] + (weight_range[1] - weight_range[0])*train_set['weights']

		valid_set.loc[:,'weights'] = 1 - ((valid_set['standard_error']**pow_fn) - (valid_set['standard_error'].min()**pow_fn)) / (valid_set['standard_error'].max()**pow_fn)
		valid_set.loc[:,'weights'] = weight_range[0] + (weight_range[1] - weight_range[0])*valid_set['weights']




	train_dataset = DatasetRetriever(train_set, tokenizer, max_len, is_weighted=is_weighted)
	valid_dataset = DatasetRetriever(valid_set, tokenizer, max_len, is_weighted=is_weighted)

	train_sampler = RandomSampler(train_dataset)
	train_loader = DataLoader(
		train_dataset, 
		batch_size=batch_size, 
		sampler=train_sampler, 
		pin_memory=True, 
		drop_last=False, 
		num_workers=4
	)

	valid_sampler = SequentialSampler(valid_dataset)
	valid_loader = DataLoader(
		valid_dataset, 
		batch_size=batch_size // 2, 
		sampler=valid_sampler, 
		pin_memory=True, 
		drop_last=False, 
		num_workers=4
	)

	return train_loader, valid_loader

####    METRICS    ###########################################################


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
		self.max = 0
		self.min = 1e5

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
		if val > self.max:
			self.max = val
		if val < self.min:
			self.min = val


####    TRAINER    ############################################################


class Trainer:
	def __init__(self, model, optimizer, scheduler, model_output_location, scalar=None, log_interval=1, evaluate_interval=10, is_weighted=False):
		self.model = model
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.scalar = scalar
		self.log_interval = log_interval
		self.evaluate_interval = evaluate_interval
		self.evaluator = Evaluator(self.model, self.scalar)
		self.model_ouput_location = model_output_location
		self.is_weighted = is_weighted

	def train(self, train_loader, valid_loader, epoch, 
			  result_dict, tokenizer, fold):
		count = 0
		losses = AverageMeter()
		weighted_losses = AverageMeter()
		self.model.train()
		
		for batch_idx, batch_data in enumerate(train_loader):
			input_ids, attention_mask, token_type_ids, labels = batch_data['input_ids'], \
				batch_data['attention_mask'], batch_data['token_type_ids'], batch_data['label']
			input_ids, attention_mask, token_type_ids, labels = \
				input_ids.cuda(), attention_mask.cuda(), token_type_ids.cuda(), labels.cuda()

			if self.is_weighted:
				weights =  batch_data['weights']
				weights = weights.cuda()

			
			if self.scalar is not None:
				with torch.cuda.amp.autocast():
					outputs = self.model(
						input_ids=input_ids,
						attention_mask=attention_mask,
						token_type_ids=token_type_ids,
						labels=labels
					)
			else:
				outputs = self.model(
					input_ids=input_ids,
					attention_mask=attention_mask,
					token_type_ids=token_type_ids,
					labels=labels
				)

			loss, logits = outputs[:2]
			count += labels.size(0)
			losses.update(loss.item(), input_ids.size(0))
			
			if self.scalar is not None:
				self.scalar.scale(loss).backward()
				self.scalar.step(self.optimizer)
				self.scalar.update()
			
			elif self.is_weighted:
				weighted_loss = torch.sqrt(torch.mean(weights.view(-1)*torch.square(logits-labels.view(-1))))
				weighted_losses.update(weighted_loss.item(), input_ids.size(0))
				weighted_loss.backward() 
				self.optimizer.step()

			else:
				loss.backward()
				self.optimizer.step()

			self.scheduler.step()
			self.optimizer.zero_grad()

			if batch_idx % self.log_interval == 0:
				_s = str(len(str(len(train_loader.sampler))))
				if self.is_weighted:
					ret = [
						('epoch: {:0>3} [{: >' + _s + '}/{} ({: >3.0f}%)]').format(epoch, count, len(train_loader.sampler), 100 * count / len(train_loader.sampler)),
						'train_loss: {: >4.5f}'.format(losses.avg),'weighted_loss: {: >4.5f}'.format(weighted_losses.avg),
					]
				else:
					ret = [
						('epoch: {:0>3} [{: >' + _s + '}/{} ({: >3.0f}%)]').format(epoch, count, len(train_loader.sampler), 100 * count / len(train_loader.sampler)),
						'train_loss: {: >4.5f}'.format(losses.avg),
						]

				print(', '.join(ret))
			
			if batch_idx % self.evaluate_interval == 0:
				result_dict = self.evaluator.evaluate(
					valid_loader, 
					epoch, 
					result_dict, 
					tokenizer
				)
				if result_dict['val_loss'][-1] < result_dict['best_val_loss']:
					print("{} epoch, best epoch was updated! valid_loss: {: >4.5f}".format(epoch, result_dict['val_loss'][-1]))
					result_dict["best_val_loss"] = result_dict['val_loss'][-1]
					torch.save(self.model.state_dict(), self.model_ouput_location + f"model{fold}.bin")

		result_dict['train_loss'].append(losses.avg)
		return result_dict



####    EVALUATOR    ###########################################################################


class Evaluator:
	def __init__(self, model, scalar=None):
		self.model = model
		self.scalar = scalar
	
	def worst_result(self):
		ret = {
			'loss':float('inf'),
			'accuracy':0.0
		}
		return ret

	def result_to_str(self, result):
		ret = [
			'epoch: {epoch:0>3}',
			'loss: {loss: >4.2e}'
		]
		for metric in self.evaluation_metrics:
			ret.append('{}: {}'.format(metric.name, metric.fmtstr))
		return ', '.join(ret).format(**result)

	def save(self, result):
		with open('result_dict.json', 'w') as f:
			f.write(json.dumps(result, sort_keys=True, indent=4, ensure_ascii=False))
	
	def load(self):
		result = self.worst_result
		if os.path.exists('result_dict.json'):
			with open('result_dict.json', 'r') as f:
				try:
					result = json.loads(f.read())
				except:
					pass
		return result

	def evaluate(self, data_loader, epoch, result_dict, tokenizer):
		losses = AverageMeter()

		self.model.eval()
		total_loss = 0
		with torch.no_grad():
			for batch_idx, batch_data in enumerate(data_loader):
				input_ids, attention_mask, token_type_ids, labels = batch_data['input_ids'], \
					batch_data['attention_mask'], batch_data['token_type_ids'], batch_data['label']
				input_ids, attention_mask, token_type_ids, labels = input_ids.cuda(), \
					attention_mask.cuda(), token_type_ids.cuda(), labels.cuda()
				
				if self.scalar is not None:
					with torch.cuda.amp.autocast():
						outputs = self.model(
							input_ids=input_ids,
							attention_mask=attention_mask,
							token_type_ids=token_type_ids,
							labels=labels
						)
				else:
					outputs = self.model(
						input_ids=input_ids,
						attention_mask=attention_mask,
						token_type_ids=token_type_ids,
						labels=labels
					)
				
				loss, logits = outputs[:2]
				losses.update(loss.item(), input_ids.size(0))

		print('----Validation Results Summary----')
		print('Epoch: [{}] valid_loss: {: >4.5f}'.format(epoch, losses.avg))

		result_dict['val_loss'].append(losses.avg)        
		return result_dict


####    CONFIG    #########################################################


def config(train, fold=0, model_weight_location = 'model_output/mlm/', augmentation=True, augmentation_config_location='augmentation_config.json', tokenizer_name='', base_model='', is_weighted=False):
	random.seed(2021)
	np.random.seed(2021)
	torch.manual_seed(2021)
	torch.cuda.manual_seed(2021)
	torch.cuda.manual_seed_all(2021)
	epochs = 5
	max_len = 250
	batch_size = 8

	model, tokenizer = make_model(model_name=model_weight_location, tokenizer_name=tokenizer_name, num_labels=1)
	train_loader, valid_loader = make_loader(
		train, tokenizer, max_len=max_len,
		batch_size=batch_size, fold=fold, 
		augmentation=augmentation, augmentation_config_location=augmentation_config_location, is_weighted=is_weighted
	)

	import math
	num_update_steps_per_epoch = len(train_loader)
	max_train_steps = epochs * num_update_steps_per_epoch
	warmup_proportion = 0.06
	if warmup_proportion != 0:
		warmup_steps = math.ceil((max_train_steps * 6) / 100)
	else:
		warmup_steps = 0

	optimizer = make_optimizer(model, "AdamW", base_model)
	scheduler = make_scheduler(
		optimizer, decay_name='cosine_warmup', 
		t_max=max_train_steps, 
		warmup_steps=warmup_steps
	)    

	if torch.cuda.device_count() >= 1:
		print('Model pushed to {} GPU(s), type {}.'.format(
			torch.cuda.device_count(), 
			torch.cuda.get_device_name(0))
		)
		model = model.cuda() 
	else:
		raise ValueError('CPU training is not supported')

	# scaler = torch.cuda.amp.GradScaler()
	scaler = None

	result_dict = {
		'epoch':[], 
		'train_loss': [], 
		'val_loss' : [], 
		'best_val_loss': np.inf
	}
	return (
		model, tokenizer, 
		optimizer, scheduler, 
		scaler, train_loader, 
		valid_loader, result_dict, 
		epochs
	)



####    RUN    ###########################################################



def run(train, fold=0, model_weight_location = 'model_output/mlm/', model_ouput_location = 'model_output/finetuning/', augmentation=True, augmentation_config_location='augmentation_config.json', tokenizer_name='', base_model='', is_weighted=False):
	model, tokenizer, optimizer, scheduler, scaler, \
		train_loader, valid_loader, result_dict, epochs = config(train, fold, model_weight_location, augmentation, augmentation_config_location, tokenizer_name, base_model, is_weighted)
	
	import time
	trainer = Trainer(model, optimizer, scheduler, model_ouput_location, scaler, is_weighted=is_weighted)
	train_time_list = []

	for epoch in range(epochs):
		result_dict['epoch'] = epoch

		torch.cuda.synchronize()
		tic1 = time.time()

		result_dict = trainer.train(train_loader, valid_loader, epoch, 
									result_dict, tokenizer, fold)

		torch.cuda.synchronize()
		tic2 = time.time() 
		train_time_list.append(tic2 - tic1)

	torch.cuda.empty_cache()
	del model, tokenizer, optimizer, scheduler, \
		scaler, train_loader, valid_loader,
	gc.collect()
	return result_dict



####    MAIN    ########################################################


def main(args):

	train = pd.read_csv(args.train_data)
	if args.aug_data != '' and args.aug_data is not None:
		train_size = len(train)
		aug = pd.read_csv(args.aug_data)
		train = pd.concat([train, aug]).reset_index(drop=True)
		print('Used aug data. Increased size from {} to {}'.format(train_size, len(train)))
	test = pd.read_csv(args.test_data)

	is_weighted = str(args.is_weighted).lower() != 'false'
	
	if is_weighted:
		print('WARNING !! WE ARE USING  STANDARD ERROR WEIGHTS FOR LOSS')
		# dropping one row which had 0 standard error 
		train = train[train['standard_error']!=0].reset_index(drop=True)

	train = create_folds(train, num_splits=5)

	model_weight_location = args.model_weight_location
	model_ouput_location = args.model_output_location
	base_model = args.base_model
	tokenizer_name = args.base_model

	augmentation = str(args.augmentation).lower() == 'true'
	augmentation_config_location = args.augmentation_config_location

	

	result_list = []
	for fold in range(5):
		print('----')
		print(f'FOLD: {fold}')
		result_dict = run(train, fold, model_weight_location, model_ouput_location, augmentation, augmentation_config_location, tokenizer_name, base_model, is_weighted)
		result_list.append(result_dict)
		print('----')

	[print("FOLD::", i, "Loss:: ", fold['best_val_loss']) for i, fold in enumerate(result_list)]

	oof = np.zeros(len(train))
	for fold in tqdm(range(5), total=5):
		model, tokenizer = make_model(model_weight_location, tokenizer_name)
		model.load_state_dict(
			torch.load(model_ouput_location + f'model{fold}.bin')
		)
		model.cuda()
		model.eval()
		val_index = train[train.kfold==fold].index.tolist()
		train_loader, val_loader = make_loader(train, tokenizer, 250, 16, fold=fold, augmentation=False, augmentation_config_location=None, is_weighted=False)
		# scalar = torch.cuda.amp.GradScaler()
		scalar = None
		preds = []
		for index, data in enumerate(val_loader):
			input_ids, attention_mask, token_type_ids, labels = data['input_ids'], \
				data['attention_mask'], data['token_type_ids'], data['label']
			input_ids, attention_mask, token_type_ids, labels = input_ids.cuda(), \
				attention_mask.cuda(), token_type_ids.cuda(), labels.cuda()
			if scalar is not None:
				with torch.cuda.amp.autocast():
					outputs = model(
						input_ids=input_ids,
						attention_mask=attention_mask,
						token_type_ids=token_type_ids,
						labels=labels
					)
			else:
				outputs = model(
					input_ids=input_ids,
					attention_mask=attention_mask,
					token_type_ids=token_type_ids,
					labels=labels
				)
			
			loss, logits = outputs[:2]
			preds += logits.cpu().detach().numpy().tolist()
		oof[val_index] = preds


	print("cv", round(np.sqrt(mean_squared_error(train.target.values, oof)), 4))




####    FOOTER    #################################################
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='parameters for data and weights location',
								 prefix_chars='-',
								 )

	parser.add_argument('-train_data',  default='data/raw/train.csv', help='Train Data Location')
	parser.add_argument('-aug_data',  default='', help='aug Data Location')
	parser.add_argument('-test_data',  default='data/raw/test.csv', help='Test Data Location')

	parser.add_argument('-model_weight_location',  default='model_output/mlm/', help='Pretrained model weight location (mlm) Location')

	parser.add_argument('-model_output_location',  default='model_output/finetuning/', help='Model output weight location')

	parser.add_argument('-augmentation', default='false', help='Data augmentation')

	parser.add_argument('-augmentation_config_location', default='augmentation_config.json', help='Data augmentation config')

	parser.add_argument('-base_model',  default='roberta-base', help='base model name')

	parser.add_argument('-is_weighted',  default='false', help='add weights for loss function')


	args = parser.parse_args()
	
	main(args)
