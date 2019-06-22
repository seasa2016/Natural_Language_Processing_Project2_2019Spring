import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn import metrics
import numpy as np
from .Focal_loss import FocalLoss

def count(pred,label):
	total = {}
	
	total['num'] = f1_score(y_true=label.view(-1), y_pred=pred.view(-1),average='macro')
	total['correct'] = (pred == label).sum().item()
	
	return total

class Linear(nn.Module):
	def __init__(self,lin_dim1,lin_dim2):
		super(Linear,self).__init__()
		self.lin_dim1 = lin_dim1
		self.lin_dim2 = lin_dim2

		self.linear1 = nn.Linear(self.lin_dim1,self.lin_dim2)
		
		self.linear2_1 = nn.Linear(self.lin_dim2,2)
		self.linear2_2 = nn.Linear(self.lin_dim2,2)
		self.linear2_3 = nn.Linear(self.lin_dim2,3)
		self.dropout = nn.Dropout()

		self.criterion = nn.CrossEntropyLoss(size_average=False)
	def forward(self,x,labels=None):
		out = self.linear1(x)
		out	= self.dropout(out)

		out = [self.linear2_1(F.relu(out)),self.linear2_2(F.relu(out)),self.linear2_3(F.relu(out))]
		preds = [out[0].topk(1)[1].view(-1),out[1].topk(1)[1].view(-1),out[2].topk(1)[1].view(-1)]
		if(labels is None):
			#return predict output
			return preds,out
		else:
			#return loss and acc
			total = {'loss':{},'correct':{},'num':{}}
			
			loss = self.criterion( out[0].view(-1, 2), labels[:,0].view(-1) ).mean()
			total['loss']['a'] = loss.cpu().detach().item()
			total_loss = loss

			loss = self.criterion( out[1].view(-1, 2), labels[:,1].view(-1) ) * (labels[:,0].float().view(-1)).mean()
			total['loss']['b'] = loss.cpu().detach().item()
			total_loss += 16*loss
			
			loss = self.criterion( out[2].view(-1, 3), labels[:,2].view(-1) ) * (labels[:,1].float().view(-1)).mean()
			total['loss']['c'] = loss.cpu().detach().item()
			total_loss += 16*loss
			
			for i,pred in enumerate(preds):
				temp = count(pred,labels[:,i])
				total['num'][i] = temp['num']
				total['correct'][i] = temp['correct']
			
			return total_loss,total

class Base(nn.Module):
	def __init__(self,args,vocab):
		super(Base, self).__init__()
		self.args = args
		if(args.embedding == True):
			args.word_num = 30001
			args.embeds_dim = 300
		else:
			args.word_num = len(vocab)
		self.word_emb =nn.Embedding(args.word_num,args.embeds_dim,padding_idx=0)

		if(args.embedding == True):
			self.word_emb.load_state_dict({'weight': torch.tensor( vocab.vectors[:args.word_num]) } )
			self.word_emb.weight.requires_grad = True
			print("here",self.word_emb.weight.requires_grad)

		self.linear = Linear(args.lin_dim1,args.lin_dim2)

		
