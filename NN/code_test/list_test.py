import torch

def check():
    a = torch.zeros(3,3)
    b = torch.zeros(3,3)
    c = torch.ones(3,3)
    return a,[b,c]

qq = []

def append(qq):
    t = check()
    if(len(qq)==0):
        for i in range(len(t[1])):
            qq.append(t[1][i].tolist())
    else:
        for i in range(len(t[1])):
            qq[i].extend(t[1][i].tolist())
    print(qq[0],len(qq[0]))
    

append(qq)
append(qq)
append(qq)