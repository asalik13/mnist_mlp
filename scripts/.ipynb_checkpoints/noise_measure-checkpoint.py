import torch
from torch._C import dtype
import torch.nn as nn
import h5py

from torch.nn import NLLLoss
from torch.optim import Adam



# MLP used to measure noise

class Model(nn.Module):

    # Structure of network
    def __init__(self):
        super(Model, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=10),
            nn.Softmax(dim=1)
        )

    # Foward pass
    def forward(self, x):
        return self.network(x)

    # To train model
    def train_model(self, inputs, target, epochs, silent = False):
        # define the optimization
        criterion = NLLLoss()
        optimizer = Adam(self.parameters())

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, target)
            _, labels = output.max(dim = 1)
            accuracy = (target == labels).sum() / labels.size()[0]*100

            if not silent:
                print(f"Epoch {epoch}:")
                print(f"Accuracy: {accuracy}")
            loss.backward()
            optimizer.step()
            
    def eval_model(self, inputs, targets, silent = False):






if __name__ == '__main__':

    # Loading scaled images into tensors...
    f = h5py.File('../mnist/mnist-scaled.hdf5', 'r')

    test_images = torch.tensor(f.get('test_images'))
    test_labels = torch.tensor(f.get('test_labels'))


    train_images = torch.tensor(f.get('train_images'), dtype = torch.float).reshape(-1, 1, 32, 32)
    train_labels = torch.tensor(f.get('train_labels'), dtype = torch.int64)


    

    model = Model()
    model.train_model(train_images, train_labels, epochs=100)

    train_metrics = []
    test_metrics = []

    targets = train_labels

    criterion = NLLLoss()
    optimizer = Adam(self.parameters())

    for epoch in range(100):
        optimizer.zero_grad()
        train_output = model(train_images)
        loss = criterion(train_output, targets)
        _, train_labels_output = train_output.max(dim = 1)
        train_accuracy = (train_labels_output == train_labels).sum() / train_labels.size()[0]*100

        test_output = model(test_images)
        _, test_labels_output = test_output.max(dim = 1)
        test_accuracy = (test_labels_output == test_labels).sum() / test_labels.size()[0]*100

        train_metrics.append(train_accuracy)
        test_metrics.append(test_accuracy)

        loss.backward()
        optimizer.step()

