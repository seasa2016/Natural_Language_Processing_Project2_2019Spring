import torch

a = torch.rand(2,3,2)
print(a)
print(a.transpose(0,1))
print('-'*10)
print(a.transpose(0,1).expand(2,3,2,2).transpose(0,1).contiguous().view(2,6,2))