import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys

data = pd.read_csv(sys.argv[1], sep='\t')

train, valid = train_test_split( data, test_size=0.2, random_state=42)

train.to_csv('train.tsv', sep = '\t')
valid.to_csv('valid.tsv', sep = '\t')