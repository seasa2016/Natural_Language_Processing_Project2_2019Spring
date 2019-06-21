from time import gmtime, strftime
import os
import argparse

from data.dataloader import itemDataset,collate_fn

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils

from models.Siamese import siamese
from models.QaLSTM import qalstm
from models.AttnLSTM import attnlstm
from models.BiMPM import bimpm

def get_data(train_file,eval_file,batch_size,pred,maxlen):
	train_dataset = itemDataset( file_name=train_file,mode='train',pred=pred,maxlen=maxlen)
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=16,collate_fn=collate_fn)
	
	eval_dataset = itemDataset( file_name=eval_file,mode='eval',pred=pred,maxlen=maxlen)
	eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size,shuffle=True, num_workers=16,collate_fn=collate_fn)
	
	return {
		'train':train_dataloader,
		'eval':eval_dataloader
	}

def convert(data,device):
	for name in data:
		if(type(data[name])==list):
			for i in range(len(data[name])):
				data[name][i] = data[name][i].to(device)
		else:
			data[name] = data[name].to(device)
	return data

def process(args):
	print("check device")
	if(torch.cuda.is_available() and args.gpu>=0):
		device = torch.device('cuda')
		print('the device is in cuda')
	else:
		device = torch.device('cpu')
		print('the device is in cpu')

	print("loading data")
	dataloader = get_data(os.path.join(args.data,'total.csv'),os.path.join(args.data,'eval.csv'),args.batch_size,args.pred,args.maxlen)

	print("setting model")
	if(args.model=='siamese'):
		model = siamese(args)
	elif(args.model=='qalstm'):
		model = qalstm(args)
	elif(args.model=='attnlstm'):
		model = attnlstm(args)
	elif(args.model=='bimpm'):
		model = bimpm(args)

	model = model.to(device=device)

	print(model)
	optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
	scheduler = optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.5)	


	acc_best = 000
	print("start training")

	model.zero_grad()
	for now in range(args.epoch):
		model.train()
		train(model,dataloader['train'],optimizer,device)
		model.eval()
		acc_best = eval(model,dataloader['eval'],device,acc_best,now,args)
		scheduler.step()

def train(model,data_set,optimizer,device):
	w = [1.0/16,1.0/15,1.0/5]
	total={}
	for i,data in enumerate(data_set):
		data = convert(data,device)

		#deal with the classfication part
		loss,out = model(data['query'],data['length'],data['label'])
		loss.backward()

		for cla in out:
			if(cla not in total):
				total[cla]={}
			for t in out[cla]:
				try:
					total[cla][t] += out[cla][t]
				except:
					total[cla][t] = out[cla][t]

		if(i%1==0):
			optimizer.step()
			model.zero_grad()

		if(i%160==0):
			print(i,'train loss:{0}  acc:{1}/{2}, weighted:{3}'.format(total['loss'],total['count']['correct'],total['count']['num'],total['count']['weighted']/160))
			for cla in total:
				for t in total[cla]:
					total[cla][t] = 0

	

def eval(model,data_set,device,acc_best,now,args):
	total={}
	
	for i,data in enumerate(data_set):
		with torch.no_grad():
			#
			data = convert(data,device)
			loss,out = model(data['query'],data['length'],data['label'])
			
			for cla in out:
				if(cla not in total):
					total[cla]={}
				for t in out[cla]:
					try:
						total[cla][t] += out[cla][t]
					except:
						total[cla][t] = out[cla][t]
	
	print(i,'test loss:{0}  acc:{1}/{2}, weighted:{3}'.format(total['loss'],total['count']['correct'],total['count']['num'],total['count']['weighted']/len(data_set)))	
	print('-'*10)
	
	check = {
			'args':args,
			'model':model.state_dict()
			}
	torch.save(check, './saved_models/{0}/step_{1}.pkl'.format(args.save,now))

	if(total['count']['weighted']>acc_best):
		torch.save(check, './saved_models/{0}/best.pkl'.format(args.save))
		acc_best = total['count']['weighted']
	
	return acc_best
		
def main():
	parser = argparse.ArgumentParser()

	
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--dropout', default=0, type=int)
	parser.add_argument('--gpu', default=0, type=int)
	parser.add_argument('--embeds_dim', default=256, type=int)
	parser.add_argument('--hidden_dim', default=256, type=int)
	parser.add_argument('--num_layer', default=2, type=int)
	parser.add_argument('--learning_rate', default=0.0005, type=float)
	
	parser.add_argument('--print_freq', default=1, type=int)
	parser.add_argument('--input_size', default=49527, type=int)
	parser.add_argument('--batch_first', default=True, type=bool)
	parser.add_argument('--mode' , default= 'train', type=str)
	parser.add_argument('--epoch', default= 5, type=int)


	parser.add_argument('--data', default='./data/all_no_embedding/', type=str)
	parser.add_argument('--maxlen', default= 128, type=int)
	parser.add_argument('--attention', default='luong',type=str)

	parser.add_argument('--model', required=True)
	parser.add_argument('--pred', required=True)
	parser.add_argument('--save', required=True)
	
	args = parser.parse_args()
	with open('{0}/vocab'.format(args.data)) as f:
		args.word_num = len(f.readlines())
	
	
	if not os.path.exists('saved_models'):
		os.makedirs('saved_models')

	if not os.path.exists('./saved_models/{0}'.format(args.save)):
		os.makedirs('./saved_models/{0}'.format(args.save))

	print('training start!')
	process(args)
	print('training finished!')
	



if(__name__ == '__main__'):
	main()
