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

class MDLSTM(torch.nn.Module):
    """ [todo] """
    def __init__(self, size_in, dim_in, size_out):
        super().__init__()
        self.size_in  = size_in # Expected number of of features
        self.dim_in   = dim_in # Dimensionality of the input (2 for images for instance).
        self.size_out = size_out # Number of output size
        # Parameters:
        # - Forget gates
        self.wf = torch.nn.Parameter(torch.Tensor(self.dim_in, self.size_in, self.size_out))
        self.uf = torch.nn.Parameter(torch.Tensor(self.dim_in, self.dim_in, self.size_out))
        self.biasf = torch.nn.Parameter(torch.Tensor(self.dim_in, self.size_out))
        # - Input gate
        self.wi = torch.nn.Parameter(torch.Tensor(self.size_in, self.size_out))
        self.ui = torch.nn.Parameter(torch.Tensor(self.dim_in, self.size_out))
        self.biasi = torch.nn.Parameter(torch.Tensor(self.size_out))
        # - Output gate
        self.wo = torch.nn.Parameter(torch.Tensor(self.size_in, self.size_out))
        self.uo = torch.nn.Parameter(torch.Tensor(self.dim_in, self.size_out))
        self.biaso = torch.nn.Parameter(torch.Tensor(self.size_out))
        # - Cell input gate
        self.wc = torch.nn.Parameter(torch.Tensor(self.size_in, self.size_out))
        self.uc = torch.nn.Parameter(torch.Tensor(self.dim_in, self.size_out))
        self.biasc = torch.nn.Parameter(torch.Tensor(self.size_out))
        # - Cell state
        self.s = None
        # Initialize weights
        k = np.sqrt(1/size_out)
        torch.nn.init.uniform_(self.wf, a=-k, b=k)
        torch.nn.init.uniform_(self.uf, a=-k, b=k)
        torch.nn.init.uniform_(self.biasf, a=-k, b=k)
        torch.nn.init.uniform_(self.wi, a=-k, b=k)
        torch.nn.init.uniform_(self.ui, a=-k, b=k)
        torch.nn.init.uniform_(self.biasi, a=-k, b=k)
        torch.nn.init.uniform_(self.wo, a=-k, b=k)
        torch.nn.init.uniform_(self.uo, a=-k, b=k)
        torch.nn.init.uniform_(self.biaso, a=-k, b=k)
        torch.nn.init.uniform_(self.wc, a=-k, b=k)
        torch.nn.init.uniform_(self.uc, a=-k, b=k)
        torch.nn.init.uniform_(self.biasc, a=-k, b=k)
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
        f = torch.Tensor(self.dim_in, *dimensions, batch_size, self.size_out)
        i = torch.Tensor(*dimensions, batch_size, self.size_out)
        o = torch.Tensor(*dimensions, batch_size, self.size_out)
        c = torch.Tensor(*dimensions, batch_size, self.size_out)
        s = torch.Tensor(*dimensions, batch_size, self.size_out)
        h = torch.Tensor(*dimensions, batch_size, self.size_out)
        for idx in self.iter_idx(dimensions):
            for l in range(self.dim_in): f[l][idx] = torch.mm(x[idx], self.wf[l]) + self.biasf[l]
            i[idx] = torch.mm(x[idx], self.wi) + self.biasi
            o[idx] = torch.mm(x[idx], self.wo) + self.biaso
            c[idx] = torch.mm(x[idx], self.wc) + self.biasc
            for k in range(self.dim_in):
                if idx[k] > 0:
                    idx_bis = list(idx)
                    idx_bis[k] -= 1
                    idx_bis = tuple(idx_bis)
                    for l in range(self.dim_in): f[l][idx] += torch.multiply(h[idx_bis], self.uf[l,k])
                    i[idx] += torch.multiply(h[idx_bis], self.ui[k])
                    o[idx] += torch.multiply(h[idx_bis], self.uo[k])
                    c[idx] += torch.multiply(h[idx_bis], self.uc[k])
        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        c = torch.sigmoid(c)
        # Second pass: compute cell state (we had to apply sigmoid activation to the gates)
        for idx in self.iter_idx(dimensions):
            s[idx] = i[idx] * c[idx]
            for k in range(self.dim_in):
                if idx[k] > 0:
                    idx_bis = list(idx)
                    idx_bis[k] -= 1
                    idx_bis = tuple(idx_bis)
                    s[idx] += f[k][idx] * s[idx_bis]
            s[idx] = torch.tanh(s[idx])
        # Third pass: compute final output (we had to compute the cell states)
        h = torch.multiply(o, torch.tanh(c))
        return h
