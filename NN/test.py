from time import gmtime, strftime
import os
import argparse

from data.dataloader import itemDataset,collate_fn
from gensim.models import KeyedVectors
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils

from models.AttnLSTM import attnlstm

label = {'taska':['NOT','OFF'],'taskb': ['UNT','TIN'],'taskc':['OTH', 'GRP', 'IND']}

def get_data(test_file,batch_size,vocab,embedding,maxlen):
	test_dataset = itemDataset( file_name=test_file,mode='test',vocab=vocab,embedding=embedding,maxlen=maxlen)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=16,collate_fn=collate_fn)
	
	return test_dataloader

def convert(data,device):
	for name in data:
		if(type(data[name])==list):
			pass
		else:
			data[name] = data[name].to(device)
	return data

def process(args,checkpoint,vocab):
	print("check device")
	if(torch.cuda.is_available() and args.gpu>=0):
		device = torch.device('cuda')
		print('the device is in cuda')
	else:
		device = torch.device('cpu')
		print('the device is in cpu')

	print("loading data")
	try:
		dataloader = get_data(args.data,args.batch_size,vocab,checkpoint['args'].embedding,checkpoint['args'].maxlen)
	except:
		dataloader = get_data(args.data,args.batch_size,vocab,checkpoint['args'].embedding,128)

	print("setting model and load from pretrain")
	
	if(checkpoint['args'].model=='attnlstm'):
		model = attnlstm(checkpoint['args'],vocab)
	
	model.load_state_dict(checkpoint['model'])
	model = model.to(device=device)

	print("start testing")

	model.eval()
	out = test(model,args,dataloader,device)
	with open(args.out,'w') as f:
		f.write("Id,Category\n")
		for i in range(len(out['id'])):
			f.write('{0},{1}\n'.format(out['id'][i],label[args.task][out['ans'][i]]))


def test(model, args, data_set, device):
	def append(total,out):
		if(len(total)==0):
			total.append(out)
		else:
			total.extend(out)
		
	total={'id':[],'ans':[]}
	for i,data in enumerate(data_set):
		with torch.no_grad():
			data = convert(data,device)
			temp = model(data['query'],data['length'])
			print(temp[1])
			out = temp[0]
			total['id'].extend(data['id'])
			if(args.task=='taska'):
				total['ans'].extend(out[0])
			elif(args.task=='taskb'):
				total['ans'].extend(out[1])
			elif(args.task=='taskc'):
				total['ans'].extend(out[2])
			
	return total
		
def main():
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--gpu', default=0, type=int)
	
	parser.add_argument('--mode' , default= 'test', type=str)
	parser.add_argument('--data', default='./data/testset-levela.tsv', type=str)
	
	parser.add_argument('--save', required=True)
	parser.add_argument('--out', required=True)
	parser.add_argument('--task', required=True)
	
	args = parser.parse_args()
	args.data = './data/testset-level{0}.tsv'.format(args.task[-1])

	if(os.path.exists(args.save)):
		checkpoint = torch.load(args.save)
	else:
		raise ValueError('no this path')
	
	if(checkpoint['args'].embedding==False):
		vocab = {}
		with open('./data/vocab') as f:
			for i,word in enumerate(f):
				word = word.strip().split()
				vocab[ word[0] ] = i
			args.word_num = len(vocab)
	else:
		news_path = './data/embedding/GoogleNews-vectors-negative300.bin'
		vocab = KeyedVectors.load_word2vec_format(news_path, binary=True)

	print('testing start!')
	process(args,checkpoint,vocab)
	print('training finished!')
	



if(__name__ == '__main__'):
	main()
