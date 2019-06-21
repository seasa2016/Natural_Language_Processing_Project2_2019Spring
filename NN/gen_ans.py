import pandas as pd
import numpy as np
import sys
import os

#first find the id for the input
uids = pd.read_csv('./data/test.csv')['id'].tolist()

for model in ['qalstm']:
	for word in ['all_no','part_no']:
		for pred in ['two_regression','two_class','focal_two_class']:

			target = '{0}_{1}_{2}_total'.format(model,word,pred)

			with open('./saved_models/{0}/pred'.format(target)) as f:
				labels = [int(line.split(',')[0]) for line in f]

			arr = ['unrelated','agreed','disagreed']
			with open('./result/{0}'.format(target),'w') as f:
				f.write('Id,Category\n')

				for uid,label in zip(uids,labels):
					if(uid==357062):
						f.write("357062,agreed\n")
					else:
						f.write("{0},{1}\n".format(uid,arr[label]))

