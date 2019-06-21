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

def get_data(test_file,batch_size,pred,maxlen):
	test_dataset = itemDataset( file_name=test_file,mode='test',pred=pred,maxlen=maxlen)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=16,collate_fn=collate_fn)
	
	return test_dataloader

def convert(data,device):
	for name in data:
		if(type(data[name])==list):
			for i in range(len(data[name])):
				data[name][i] = data[name][i].to(device)
		else:
			data[name] = data[name].to(device)
	return data

def process(args,checkpoint):
	print("check device")
	if(torch.cuda.is_available() and args.gpu>=0):
		device = torch.device('cuda')
		print('the device is in cuda')
	else:
		device = torch.device('cpu')
		print('the device is in cpu')

	print("loading data")
	try:
		dataloader = get_data(args.data,args.batch_size,checkpoint['args'].pred,checkpoint['args'].maxlen)
	except:
		dataloader = get_data(args.data,args.batch_size,checkpoint['args'].pred,128)

	print("setting model and load from pretrain")
	if(checkpoint['args'].model=='siamese'):
		model = siamese(checkpoint['args'])
	elif(checkpoint['args'].model=='qalstm'):
		model = qalstm(checkpoint['args'])
	elif(checkpoint['args'].model=='attnlstm'):
		model = attnlstm(checkpoint['args'])
	elif(checkpoint['args'].model=='bimpm'):
		model = bimpm(checkpoint['args'])
	
	model.load_state_dict(checkpoint['model'])
	model = model.to(device=device)

	print("start testing")

	model.eval()
	out = test(model,dataloader,device)
	with open(args.out,'w') as f:
		for i in range(len(out[0])):
			f.write('{0}'.format(out[0][i]))
			for j in range(len(out[1])):
				f.write(',{0}'.format(out[1][j][i]))
			f.write('\n')


def test(model,data_set,device):
	def append(total,out):
		if(len(total)==0):
			for i in range(len(out)):
				total.append(out[i].tolist())
		else:
			for i in range(len(out)):
				total[i].extend(out[i].tolist())
		
	total=[[],[]]
	for i,data in enumerate(data_set):
		with torch.no_grad():
			data = convert(data,device)
			out = model(data['query'],data['length'])
			
			total[0].extend(out[0].tolist())
			append(total[1],out[1])

	return total
		
def main():
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--gpu', default=0, type=int)
	
	parser.add_argument('--mode' , default= 'test', type=str)
	parser.add_argument('--data', default='./data/all_no_embedding/test.csv', type=str)
	parser.add_argument('--save', required=True)
	
	parser.add_argument('--out', required=True)
	
	args = parser.parse_args()

	if(os.path.exists('./saved_models/{0}/best.pkl'.format(args.save))):
		checkpoint = torch.load( './saved_models/{0}/best.pkl'.format(args.save) )
	else:
		raise ValueError('no this model')

	print('testing start!')
	process(args,checkpoint)
	print('training finished!')
	



if(__name__ == '__main__'):
	main()
