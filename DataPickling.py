import gzip, cPickle
import numpy as np
import pandas as pd

data = []
labels = []

read1 = pd.read_csv('train_preprocessed.csv')
data = np.array(read1)
df = pd.read_csv('train_labels.csv')
labels = np.array(df)

# Data and labels are read
train_set_x = data[:600000]
val_set_x = data[600001:700000]
test_set_x = data[700001:]
train_set_y = labels[:600000]
val_set_y = labels[600001:700000]
test_set_y = labels[700001:]
# Divided dataset into 3 parts. I had 6281 images.

print test_set_y.shape[0]
print test_set_y.shape[1]

print val_set_y.shape[0]
print val_set_y.shape[1]
print train_set_y.shape[0]
print train_set_y.shape[1]

train_set = train_set_x, train_set_y
val_set = val_set_x, val_set_y
test_set = test_set_x, test_set_y

dataset = [train_set, val_set, test_set]

f = gzip.open('california-crime.pkl.gz','wb')
cPickle.dump(dataset, f, protocol=2)
f.close()