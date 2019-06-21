import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics
import numpy as np
from .Focal_loss import FocalLoss
w = [1.0/16,1.0/15,1.0/5]
def count(pred,label):
	total = {}
	
	total['num'] = pred.shape[0]
	total['correct'] = (pred == label).sum().item()
	sample_weight = [w[i] for i in pred]
	total['weighted'] = metrics.accuracy_score(label.tolist(), pred.tolist(), normalize=True, sample_weight=sample_weight)

	return total

class Focal_Two_class(nn.Module):
	def __init__(self,lin_dim1,lin_dim2):
		super(Focal_Two_class,self).__init__()
		self.lin_dim1 = lin_dim1
		self.lin_dim2 = lin_dim2
		

		self.linear1 = nn.Linear(self.lin_dim1,self.lin_dim2)
		self.linear2_1 = nn.Linear(self.lin_dim2,2)
		self.linear2_2 = nn.Linear(self.lin_dim2,2)
		self.dropout = nn.Dropout()

		self.criterion = FocalLoss(gamma=3)
		self.w = [1.0/16,1.0/15,1.0/5]
	def forward(self,x,label=None):
		out = self.linear1(x)
		out	= self.dropout(out)
		
		out_1 = self.linear2_1(F.relu(out))
		out_2 = self.linear2_2(F.relu(out))
		
		pred = (out_1.topk(1)[1]*(1+out_2.topk(1)[1])).view(-1)
		if(label is None):
			#return predict output
			return pred,[out_1,out_2]

		else:
			#return loss and acc
			total = {'loss':{},'count':{}}
			
			loss = self.criterion(out_1,label[0]) 
			total['loss']['relation'] = loss.cpu().detach().item()
			total_loss = loss

			loss = self.criterion(out_2,label[1]) 
			total['loss']['type'] = loss.cpu().detach().item()
			total_loss += loss
			
			total['count'] = count(pred,label[-1])
			
			return total_loss,total

class Two_class(nn.Module):
	def __init__(self,lin_dim1,lin_dim2):
		super(Two_class,self).__init__()
		self.lin_dim1 = lin_dim1
		self.lin_dim2 = lin_dim2
		

		self.linear1 = nn.Linear(self.lin_dim1,self.lin_dim2)
		self.linear2_1 = nn.Linear(self.lin_dim2,2)
		self.linear2_2 = nn.Linear(self.lin_dim2,2)
		self.dropout = nn.Dropout()

		self.criterion = nn.KLDivLoss()
		self.w = [1.0/16,1.0/15,1.0/5]
	def forward(self,x,label=None):
		out = self.linear1(x)
		out	= self.dropout(out)
		
		out_1 = self.linear2_1(F.relu(out))
		out_2 = self.linear2_2(F.relu(out))
		
		pred = (out_1.topk(1)[1]*(1+out_2.topk(1)[1])).view(-1)
		if(label is None):
			#return predict output
			return pred,[out_1,out_2]

		else:
			#return loss and acc
			total = {'loss':{},'count':{}}
			
			loss = self.criterion(F.log_softmax(out_1,dim=1),label[0]) 
			total['loss']['relation'] = loss.cpu().detach().item()
			total_loss = loss

			loss = self.criterion(F.log_softmax(out_2,dim=1),label[1]) 
			total['loss']['type'] = loss.cpu().detach().item()
			total_loss += loss
			
			total['count'] = count(pred,label[-1])
			
			return total_loss,total

class Two_regression(nn.Module):
	def __init__(self,lin_dim1,lin_dim2):
		super(Two_regression,self).__init__()
		self.lin_dim1 = lin_dim1
		self.lin_dim2 = lin_dim2

		self.linear1 = nn.Linear(self.lin_dim1,self.lin_dim2)
		self.linear2_1 = nn.Linear(self.lin_dim2,1)
		self.linear2_2 = nn.Linear(self.lin_dim2,1)
		self.dropout = nn.Dropout()

		self.criterion = nn.BCEWithLogitsLoss(reduction='none')
		
		self.threshold1 = 0.7
		self.threshold2 = 0.5

	def forward(self,x,label=None):
		out = self.linear1(x)
		out	= self.dropout(out)
		out_1 = self.linear2_1(F.relu(out))
		out_2 = self.linear2_2(F.relu(out))
				
		pred = ( (out_1.sigmoid()>self.threshold1).long()*(1+(out_2.sigmoid()>self.threshold2).long()) ).view(-1)
		if(label is None):
			#return predict output
			return pred,[out_1,out_2]
		else:
			#return loss and acc
			total = {'loss':{},'count':{}}
			loss = self.criterion(out_1,label[0]).mean()
			total['loss']['relation'] = loss.cpu().detach().item()
			total_loss = loss

			#drop out for unrelated data
			loss = (label[0].float()*self.criterion(out_2,label[1])).mean()
			total['loss']['type'] = loss.cpu().detach().item()
			total_loss += loss
			
			total['count'] = count(pred,label[-1])
			
			return total_loss,total

class Three_class(nn.Module):
	def __init__(self,lin_dim1,lin_dim2):
		super(Three_class,self).__init__()
		self.lin_dim1 = lin_dim1
		self.lin_dim2 = lin_dim2

		self.linear1 = nn.Linear(self.lin_dim1,self.lin_dim2)
		self.linear2 = nn.Linear(self.lin_dim2,3)
		self.dropout = nn.Dropout()

		self.criterion = nn.CrossEntropyLoss()
	def forward(self,x,label=None):
		out = self.linear1(x)
		out	= self.dropout(out)
		out = self.linear2(F.relu(out))
		
		pred = out.topk(1)[1].view(-1)
		if(label is None):
			#return predict output
			return pred,[out]
		else:
			#return loss and acc
			total = {'loss':{},'count':{}}
			
			loss = self.criterion(out,label[-1]) 
			total['loss']['total'] = loss.cpu().detach().item()
			total_loss = loss
			
			total['count'] = count(pred,label[-1])
			
			return total_loss,total

class Base(nn.Module):
	def __init__(self,args):
		super(Base, self).__init__()
		self.args = args

		self.word_emb =nn.Embedding(args.word_num,args.embeds_dim,padding_idx=0)

		if(args.mode == 'pretrain'):
			self.load()
			self.word_emb.weight.requires_grad = False
			print("here",self.word_emb.weight.requires_grad)

		if(args.pred=='two_class'):
			self.linear = Two_class(args.lin_dim1,args.lin_dim2)
		elif(args.pred=='two_regression'):
			self.linear = Two_regression(args.lin_dim1,args.lin_dim2)
		elif(args.pred=='three_class'):
			self.linear = Three_class(args.lin_dim1,args.lin_dim2)
		elif(args.pred=='focal_two_class'):
			self.linear = Focal_Two_class(args.lin_dim1,args.lin_dim2)


	def load(self):
		if(self.args.embed_type == 'glove'):
			pass
		elif(self.args.embed_type == 'fasttext'):
			with open('./data/embedding/cc.zh.300.vec') as f:
				f.readline()
				arr = np.zeros((self.word_emb.weight.shape[0],self.word_emb.weight.shape[1]),dtype=np.float32)
				for i,line in enumerate(f):
					for j,num in enumerate(line.strip().split()[1:]):
						arr[i+1,j] = float(num)
						
				self.word_emb.weight = nn.Parameter(torch.tensor(arr))
