from sklearn.metrics import matthews_corrcoef, f1_score
import sys
import pandas as pd
import numpy as np

def simple_accuracy(preds, labels):
	return (preds == labels).astype(int).mean()

def multi_acc_and_f1(pred, label):
	# convert pred to label
	acc = simple_accuracy(pred,label)
	f1 = f1_score(y_true=label, y_pred=pred,average='macro')
	print("acc",acc,"f1", f1)

if(__name__=='__main__'):
	data = pd.read_csv(sys.argv[1])

	temp={}
	s = set()
	for gid,cate in zip(data['Id'],data['Category']):
		temp[gid]=cate
		s.add(cate)

	pred=[]
	gold=[]

	mapping={key:i for i,key in enumerate(s)}

	with open(sys.argv[2]) as f:
		for line in f:
			line = line.strip().split(',')
			pred.append( mapping[ temp[int(line[0])] ])
			try:
				gold.append( mapping[ line[1] ])
			except:
				gold.append( len(mapping))

	multi_acc_and_f1(np.array(pred),np.array(gold))
