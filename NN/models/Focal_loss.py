import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import os,sys,random,time
import argparse


class FocalLoss(nn.Module):
	def __init__(self, gamma=0, alpha=None, size_average=True):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
		
		if isinstance(alpha,(float,int)): 
			self.alpha = torch.Tensor([alpha,1-alpha])
		if isinstance(alpha,list): 
			self.alpha = torch.Tensor(alpha)
		
		self.size_average = size_average

	def forward(self, input, target):
		logpt = F.log_softmax(input,dim=-1)*target
		pt = Variable(logpt.data.exp())
		
		if(self.alpha is not None):
			if(self.alpha.type()!=input.data.type()):
				self.alpha = self.alpha.type_as(input.data)
			at = self.alpha.gather(0,target.data.view(-1))
			logpt = logpt * Variable(at)
		
		loss = (-1 * (1-pt)**self.gamma * logpt).sum(dim=-1)
		
		if(self.size_average): 
			return loss.mean()
		else: 
			return loss.sum()
	
def main():
	start_time = time.time()
	maxe = 0
	for i in range(1):
		x = torch.rand(4,3)*random.randint(1,4)
		x = Variable(x.cuda())
		#print(x.shape)
		l = torch.tensor([[0,0,1],[0,1,0],[0,0,0],[0,0,1]]).float()
		l = Variable(l.cuda())
		
		output0 = FocalLoss(gamma=0)(x,l)
		a = output0.item()
	print('time:',time.time()-start_time,'max_error:',maxe)


if(__name__=='__main__'):
	main()
