import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

def _one_hot_encode(x):
    n_class = len(set(x))
    return np.array(list(map(lambda i: np.eye(n_class)[i], x)))

def _build_data_loader(batch_size, X, y=None):
    size = X.shape[0]
    idx_array = np.arange(size)
    n_batch = int(np.ceil(size / float(batch_size)))
    batches = [(int(i * batch_size), int(min(size, (i + 1) * batch_size))) 
               for i in range(n_batch)]
    for batch_index, (start, end) in enumerate(batches):
        batch_ids = idx_array[start:end]
        x = Variable(torch.from_numpy(X[batch_ids]))
        if y is not None:
            yield x, Variable(torch.from_numpy(_one_hot_encode(y[batch_ids])))
        else:
            yield x
            
def _build_architecture(in_size, out_size, hidden_layer_sizes, activation):
    layers = [nn.Linear(in_size, hidden_layer_sizes[0]), activation]
    if len(hidden_layer_sizes) > 1:
        for size in hidden_layer_sizes[1:]:
            layers += [nn.Linear(size, size), activation]
    if out_size == 1:
        layers += [nn.Linear(hidden_layer_sizes[-1], out_size)]        
    else:
        layers += [nn.Linear(hidden_layer_sizes[-1], out_size), nn.Softmax(dim=out_size-1)]
    return nn.Sequential(*layers)
