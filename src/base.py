import torch.nn as nn
import torch.optim as optim

ACTIVATIONS = {
    'relu': nn.ReLU(inplace=True),
    'leaky_relu': nn.LeakyReLU(inplace=True),
    'elu': nn.ELU(inplace=True),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
}

OPTIMIZERS = {
    'rmsprop': optim.RMSprop,
    'adam': optim.Adam,
    'sgd': optim.SGD,
}

LOSS_FUNCS = {
    'mse': nn.MSELoss(),
    'crossentropy': nn.CrossEntropyLoss(),
    'bce': nn.BCELoss(),
}
