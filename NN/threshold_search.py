import sys
import numpy as np
import math
import pandas as pd
from sklearn import metrics

def sigmoid(x):
	return 1/(1+np.exp(-x))

def count(pred,label):
	w = [1.0/16,1.0/15,1.0/5]
	sample_weight = [w[i] for i in pred]
	total = metrics.accuracy_score(label, pred, normalize=True, sample_weight=sample_weight)

	return total


datas = [[],[]]
with open(sys.argv[1]) as f:
	for line in f:
		line = line.strip().split(',')

		datas[0].append(float(line[1][1:-1]))
		datas[1].append(float(line[2][1:-1]))


datas[0] = sigmoid(np.array(datas[0]))
datas[1] = sigmoid(np.array(datas[1]))


labels =pd.read_csv('./data/all_no_embedding/eval.csv')['label'].tolist()
c = {
	"unrelated":0,
	"agreed":1,
	"disagreed":2
}
labels = [ c[_] for _ in labels]

now = [0,0,0]
outputs = []
for i in range(10):
	i = i/10
	for j in range(10):
		j = j/10
		
		pred = (datas[0]>i).astype(int)* ( 1 + (datas[1]>j).astype(int) )
		

		acc = count(pred,labels)
		
		print(i,j,acc)

		if(acc>now[2]):
			now = [i,j,acc]
		outputs.append([i,j,acc])

#for output in outputs:
#	print(output)
print(now)

