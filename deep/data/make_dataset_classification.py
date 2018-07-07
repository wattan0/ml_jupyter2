import pickle
import numpy as np
import os


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# cifar10 from https://www.cs.toronto.edu/~kriz/cifar.html
dic_data = unpickle('cifar-10-batches-py/data_batch_1')
dic_meta = unpickle('cifar-10-batches-py/batches.meta')
data = dic_data[b'data']
labels = dic_data[b'labels']
label_names = dic_meta[b'label_names']

labels = np.array(labels)
target_labels = [0, 1, 2]  # 抽出するクラス
n_train_images = 100  # 画像数

label = 0
train_data = []
train_labels = []
train_label_names = []
for target in target_labels:
    train_data.append(data[labels==target][:n_train_images])
    train_labels.append([label]*n_train_images)
    label += 1
    train_label_names.append(label_names[target])

train_data = np.concatenate(train_data, axis=0)
train_labels = np.concatenate(train_labels)
dic = {'data': train_data,
       'labels': train_labels,
       'label_names': train_label_names}

with open('classification_data.pkl', 'wb') as f:
    pickle.dump(dic, f)
