import torch
import torch.nn as nn
import torch.nn.functional as f

from .base import Base
from .Attention import Luong,Bahdanau

import math

class attnlstm(Base):
	def __init__(self,args,vocab):
		if(not hasattr(args,'lin_dim1')):
			args.lin_dim1 = args.hidden_dim * 2
			args.lin_dim2 = args.hidden_dim
			
		super(attnlstm,self).__init__(args,vocab)

		self.embeds_dim = args.embeds_dim
		self.hidden_dim = args.hidden_dim
		self.num_layer = args.num_layer
		self.batch_first = args.batch_first
		
		self.rnn = nn.GRU(self.embeds_dim, self.hidden_dim, batch_first=self.batch_first , bidirectional=True, num_layers=self.num_layer)

		if(args.attention == 'luong'):
			self.attention = Luong(args.hidden_dim,args.hidden_dim)

		elif(args.attention == 'bahdanau'):
			self.attention = Bahdanau(2*args.hidden_dim,args.hidden_dim)
		else:
			raise ValueError('no this attention')

	def forward(self,querys,length,labels=None):
		def pack(seq,seq_length):
			sorted_seq_lengths, indices = torch.sort(seq_length, descending=True)
			_, desorted_indices = torch.sort(indices, descending=False)
			if self.batch_first:
				seq = seq[indices]
			else:
				seq = seq[:, indices]
			packed_inputs = nn.utils.rnn.pack_padded_sequence(seq,
															sorted_seq_lengths.cpu().numpy(),
															batch_first=self.batch_first)
			return packed_inputs,desorted_indices

		def unpack(res, state,desorted_indices):
			padded_res,_ = nn.utils.rnn.pad_packed_sequence(res, batch_first=self.batch_first)
			state = [ _ for _ in state]

			for i in range(len(state)):
				state[i] = state[i][:,desorted_indices]
			
			if(self.batch_first):
				desorted_res = padded_res[desorted_indices]
			else:
				desorted_res = padded_res[:, desorted_indices]

			return desorted_res,state
		def feat_extract(output,lengths,mask):
			"""
			here we check for several output variant
			1.largest
			2.last
			3.mean
			"""
			if( self.batch_first == False ):
				output = output.transpose(0,1) 
			result = []
			for i in range(length.shape[0]):
				result.append( torch.cat([ output[i][ length[i]-1 ][:self.hidden_dim],output[i][0][self.hidden_dim:]], dim=-1) )
			

			return torch.stack( result , dim=0 ) 
		
		emb = self.word_emb(querys)
		query_embs = [emb,emb]
		mask = [querys.eq(0),querys.eq(0)]
		lengths = [length,length]
		att_emb = self.attention(query_embs,lengths,mask)[0]

		packed_inputs,desorted_indices = pack(att_emb,length)
		res, state = self.rnn(packed_inputs)
		query_res,_ = unpack(res, state,desorted_indices)
		query_result = feat_extract(query_res,length.int(),mask)
		
		out = self.linear(query_result,labels=labels)

		return out
