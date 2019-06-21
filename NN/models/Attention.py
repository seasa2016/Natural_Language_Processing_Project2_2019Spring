import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Bahdanau(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(Bahdanau,self).__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim

        self.linear1 = nn.Linear(self.in_dim,self.out_dim)
        self.linear2 = nn.Linear(self.out_dim,1)

    def forward(self,querys,lengths,masks):
        shapes = [querys[0].shape,querys[1].shape]

        query_expand = torch.cat([
            querys[0].repeat(1,shapes[1][1],1),
            querys[1].repeat(1,1,shapes[0][1]).view(shapes[1][0],shapes[1][1]*shapes[0][1],shapes[1][2])
        ],dim=-1)

        attn_weight = self.linear2( self.linear1( query_expand ).tanh() ).view(-1,shapes[0][1],shapes[1][1])
        weight_mask = masks[0].float().unsqueeze(2).bmm( masks[1].float().unsqueeze(1) )		 #batch*length1*length2

        #perform mask for the padding data
        attn_weight += -1e8*weight_mask.float()
        attn_weights = [F.softmax(attn_weight,dim=-2), F.softmax(attn_weight,dim=-1)]
        
        return [
            attn_weights[1].bmm(querys[1]),
            attn_weights[0].transpose(1,2).bmm(querys[0])
        ]


class Luong(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(Luong,self).__init__()
        self.hidden_dim = in_dim

    def forward(self,querys,lengths,masks):
        lengths = [lengths[0].float(),lengths[1].float()]

        query_normals = []
        for query,length in zip(querys,lengths):
            query_normals.append( query.div( math.sqrt( float(self.hidden_dim) ) ) )

        #get the attention
        attn_weight = query_normals[0].bmm( query_normals[1].transpose(1,2) )		 #batch*length1*length2
        weight_mask = masks[0].float().unsqueeze(2).bmm( masks[1].float().unsqueeze(1) )		 #batch*length1*length2

        #perform mask for the padding data
        attn_weight += -1e8*weight_mask.float()
        attn_weights = [F.softmax(attn_weight,dim=-2), F.softmax(attn_weight, dim=-1)]
        
        return [
            attn_weights[1].bmm(query_normals[1]),
            attn_weights[0].transpose(1,2).bmm(query_normals[0])
        ]

def main():
    model = Bahdanau(5)

    a = torch.rand(3,2,10)
    b = torch.rand(3,3,10)

    masks = [
                torch.tensor([[0,1],[0,0],[0,1]]),
                torch.tensor([[0,1,1],[0,0,0],[0,0,1]])
            ]

    temp = model( [a,b] , [torch.tensor([1,2,1]),torch.tensor([1,3,2])],masks )
    print(temp[0].shape)
    print(temp[1].shape)

if(__name__=='__main__'):
    main()
