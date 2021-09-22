import numpy as np
import torch

from functools import reduce
from numpy import unravel_index

from pdb import set_trace as bp

class MDRNN(torch.nn.Module):
    """ [todo] """
    def __init__(self, size_in, dim_in, size_out):
        super().__init__()
        self.size_in  = size_in # Expected number of of features
        self.dim_in   = dim_in # Dimensionality of the input (2 for images for instance).
        self.size_out = size_out # Number of output size
        # Parameters
        self.w = torch.nn.Parameter(torch.Tensor(self.size_in, self.size_out))
        self.u = torch.nn.Parameter(torch.Tensor(self.dim_in, self.size_out))
        self.bias = torch.nn.Parameter(torch.Tensor(self.size_out))
        # Initialize weights
        k = np.sqrt(1/size_out)
        torch.nn.init.uniform_(self.w, a=-k, b=k)
        torch.nn.init.uniform_(self.u, a=-k, b=k)
        torch.nn.init.uniform_(self.bias, a=-k, b=k)
    # Iterate on all points of the sequence
    def iter_idx(self, dimensions, rev=[]):
        idx_total = reduce(lambda a,b: a*b, dimensions)
        x = 0
        while x < idx_total:
            idx = unravel_index(x, dimensions)
            # idx[rev] = self.size_in[rev] = idx[rev]
            yield idx
            x += 1
    def forward(self, x):
        """ Note: x is of shape (d1, ..., dn, batch_size, input_size). """
        dimensions = x.shape[:-2]
        batch_size = x.shape[-2]
        h = torch.Tensor(*dimensions, batch_size, self.size_out)
        for idx in self.iter_idx(x.shape[:-2]):
            a = torch.mm(x[idx], self.w)
            for i in range(len(idx)):
                if idx[i] > 0:
                    idx_bis = list(idx)
                    idx_bis[i] -= 1
                    idx_bis = tuple(idx_bis)
                    a += torch.multiply(h[idx_bis], self.u[i])
            h[idx] = torch.tanh(a + self.bias)
        return h
