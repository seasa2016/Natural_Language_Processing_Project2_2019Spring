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
from gensim.models import KeyedVectors
from models.AttnLSTM import attnlstm


def get_data(train_file,eval_file,batch_size,maxlen,vocab,embedding):
	train_dataset = itemDataset( file_name=train_file,mode='train',vocab=vocab,embedding=embedding,maxlen=maxlen)
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=16,collate_fn=collate_fn)
	
	eval_dataset = itemDataset( file_name=eval_file,mode='eval',vocab=vocab,embedding=embedding,maxlen=maxlen)
	eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size,shuffle=True, num_workers=16,collate_fn=collate_fn)
	
	return {
		'train':train_dataloader,
		'eval':eval_dataloader
	}

def convert(data,device):
	for name in data:
		if(type(data[name])==list):
			pass
		else:
			data[name] = data[name].to(device)
	return data

def process(args,vocab):
	print("check device")
	if(torch.cuda.is_available() and args.gpu>=0):
		device = torch.device('cuda')
		print('the device is in cuda')
	else:
		device = torch.device('cpu')
		print('the device is in cpu')

	print("loading data")
	dataloader = get_data(os.path.join(args.data,'train.tsv'),os.path.join(args.data,'eval.tsv'),args.batch_size,args.maxlen,vocab,args.embedding)

	print("setting model")
	if(args.model=='attnlstm'):
		model = attnlstm(args,vocab)

	model = model.to(device=device)

	print(model)
	para = []
	for w in model.parameters():
		if(w.requires_grad ):
			para.append(w)
	optimizer = optim.Adam(para,lr=args.learning_rate)
	scheduler = optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.5)	


	acc_best = 0
	print("start training")

	model.zero_grad()
	for now in range(args.epoch):
		model.train()
		train(model,dataloader['train'],optimizer,device)
		model.eval()
		acc_best = eval(model,dataloader['eval'],device,acc_best,now,args)
		scheduler.step()

def train(model,data_set,optimizer,device):
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
		#print(out['num'])
		if(i%1==0):
			optimizer.step()
			model.zero_grad()

	print(i,'train loss:{0}  correct:{1} num:{2}'.format(total['loss'],total['correct'],total['num']))
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
	
	print(i,'test loss:{0}  correct:{1} num:{2}'.format(total['loss'],total['correct'],total['num']))
	print('-'*10)
	
	check = {
			'args':args,
			'model':model.state_dict()
			}
	torch.save(check, './saved_models/{0}/step_{1}.pkl'.format(args.save,now))

	if(total['correct'][0]>acc_best):
		torch.save(check, './saved_models/{0}/best.pkl'.format(args.save))
		acc_best = total['correct'][0]
	
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
	
	parser.add_argument('--embedding', default=False, type=bool)
	parser.add_argument('--batch_first', default=True, type=bool)
	parser.add_argument('--mode' , default= 'train', type=str)
	parser.add_argument('--epoch', default= 10, type=int)

	parser.add_argument('--data', default='./data/', type=str)
	parser.add_argument('--maxlen', default= 128, type=int)
	parser.add_argument('--attention', default='luong',type=str)

	parser.add_argument('--model', required=True)
	parser.add_argument('--save', required=True)
	
	args = parser.parse_args()
	if(args.embedding==False):
		vocab = {}
		with open('{0}/vocab'.format(args.data)) as f:
			for i,word in enumerate(f):
				word = word.strip().split()
				vocab[ word[0] ] = i
	else:
		#news_path = './data/embedding/GoogleNews-vectors-negative300.bin'
		#vocab = KeyedVectors.load_word2vec_format(news_path, binary=True)

		news_path = './data/embedding/gensim_glove_vectors.txt'
		vocab = KeyedVectors.load_word2vec_format(news_path, binary=False)
	
	if not os.path.exists('saved_models'):
		os.makedirs('saved_models')

	if not os.path.exists('./saved_models/{0}'.format(args.save)):
		os.makedirs('./saved_models/{0}'.format(args.save))

	print('training start!')
	process(args,vocab)
	print('training finished!')
	



if(__name__ == '__main__'):
	main()
