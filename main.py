import torch
from torchvision import datasets
from torchvision import transforms
from mdrnn import MDRNN

from pdb import set_trace as bp

class ToMDRNNFormat(object):
    def __init__(self):
        pass
    def __call__(self, img):
        """ Re-order dimensions of a tensor to fit a MDRNN format. """
        # img, lbl = x
        img = torch.unsqueeze(img, -1) # Add batch_size dimension at the end.
        img = torch.movedim(img, 0, -1) # Move channel info at the end
        return img

class MyNet(torch.nn.Module):
    def __init__(self, size_in, dim_in, hidden_size):
        super().__init__()
        self.mdrnn1  = MDRNN(size_in, dim_in, hidden_size)
        self.flatten = torch.nn.Flatten()
        self.lin1    = torch.nn.Linear(hidden_size*28*28,10)
    def forward(self, x):
        x = self.mdrnn1(x)
        x = torch.movedim(x, -2, 0) # Move batch_size at the start
        x = self.flatten(x)
        x = self.lin1(x)
        return x

if __name__=="__main__":
    # Load data
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            ToMDRNNFormat()])
        )
    x_train, y_train = training_data[0]
    size_in = x_train.shape[-1]
    dim_in  = len(x_train.shape)-2
    # Define network
    hidden_size = 5
    net = torch.nn.Sequential(
        MyNet(size_in, dim_in, hidden_size),
        torch.nn.LogSoftmax()
    )
    # Test network
    a = net(x_train)
    bp()
