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
train_set_x = data[:2093]
val_set_x = data[2094:4187]
test_set_x = data[4188:6281]
train_set_y = labels[:2093]
val_set_y = labels[2094:4187]
test_set_y = labels[4188:6281]
# Divided dataset into 3 parts. I had 6281 images.

train_set = train_set_x, train_set_y
val_set = val_set_x, val_set_y
test_set = test_set_x, val_set_y

dataset = [train_set, val_set, test_set]

f = gzip.open('california-crime.pkl.gz','wb')
cPickle.dump(dataset, f, protocol=2)
f.close()