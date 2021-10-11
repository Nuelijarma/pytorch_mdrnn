import numpy as np
import torch

from functools import reduce
from numpy import unravel_index, ravel_multi_index

from pdb import set_trace as bp

def prev_seq(idx, k):
    """ Previous element of the sequence in a specific direction. """
    idx_bis = list(idx)
    idx_bis[k] -= 1
    idx_bis = tuple(idx_bis)
    return idx_bis

def prev_seq_flat(idx, k, shape):
    idx_bis = list(idx)
    idx_bis[k] -= 1
    idx_bis = ravel_multi_index(idx_bis, shape)
    return idx_bis

def prev_all(a, idx, shape):
    def default(): return torch.zeros(a.shape[1:])
    return [a[prev_seq_flat(idx, k, shape)] if idx[k] else default() for k in range(len(idx))]
    # return [a[prev_seq(idx,k)].detach() if idx[k] else 0 for k in range(len(idx))]

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
    """ Applies a single-layer LSTM to an input multi-dimensional sequence.

    Args:
        size_in: The number of expected features in the input `x`.
        dim_in: Dimensionality of the input (e.g. 2 for images).
        size_out: The number of features in the hidden state `h`.

    Input: x
        - **x** of shape `(d1, ..., dn, batch, size_in)`: tensor containing input features.

    Output: h
        - **h** of shape `(d1, ..., dn, batch, size_out)`: tensor containing output features.
    """
    def __init__(self, size_in, dim_in, size_out):
        super().__init__()
        self.mdlstmcell = MDLSTMCell(size_in, dim_in, size_out)
        self.size_out = size_out
    def iter_idx(self, dimensions):
        # Multi-dimensional range().
        # Simply uses NumPy's unravel_index() function to get from an integer
        # index to a tuple index.
        idx_total = reduce(lambda a,b: a*b, dimensions)
        x = 0
        while x < idx_total:
            yield unravel_index(x, dimensions)
            x += 1
    def forward(self, x):
        # **Note on states:**
        # States are stored as "1d"-tensors (bare for the batch size and output
        # dimension) and concatenated.
        # Uses unravel_index() and ravel_multi_index() to go from 1d indexing
        # to multi-dimensional indexing.
        # Ideally, s and h would be pre-allocated with shape `(d1, ..., dn,
        # batch_size, self.size_out)` and filled with a loop, but we CANNOT do
        # this. Filling the tensors with a loop would be doing in-place
        # operations, and PyTorch does not like in-place operations, even though
        # in this specific case this should cause no issue.
        dimensions = x.shape[:-2]
        batch_size = x.shape[-2]
        s = torch.empty((0, batch_size, self.size_out)) # Cell state
        h = torch.empty((0, batch_size, self.size_out)) # Hidden state
        for idx in self.iter_idx(dimensions):
            s_new, h_new = self.mdlstmcell(x[idx], (prev_all(s, idx, dimensions), prev_all(h, idx, dimensions)))
            s = torch.cat((s, s_new.reshape((1, batch_size, self.size_out))))
            h = torch.cat((h, h_new.reshape((1, batch_size, self.size_out))))
        h = torch.reshape(s, (*dimensions, batch_size, self.size_out))
        return h

class MDLSTMCell(torch.nn.Module):
    """ Single cell of a multi-dimensional LSTM.

    Args:
        size_in: The number of expected features in the input `x`.
        dim_in: Dimensionality of the input (e.g. 2 for images).
        size_out: The number of features in the hidden state `h`.

    Input: x, (s_0,h_0)
        - **x** of shape `(batch, size_in)`: tensor containing input features.
        - **s_0** of shape `(batch, size_in)`: tensor containing the initial cell state for each element in the batch.
        - **h_0** of shape `(batch, size_in)`: tensor containing the initial hidden state for each element in the batch.

    Outputs: (s, h)
        - **s** of shape `(batch, size_out)`: tensor containing the next cell state for each element in the batch.
        - **h** of shape `(batch, size_out)`: tensor containing the next hidden state for each element in the batch.
    """
    def __init__(self, size_in, dim_in, size_out):
        super().__init__()
        self.size_in  = size_in # Number of input features
        self.dim_in   = dim_in # Dimensionality of the input (2 for images for instance).
        self.size_out = size_out # Number of output features
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
    def forward(self, x, old):
        s_0, h_0 = old
        # Note on input dimension:
        # - x is of size (batch_size, self.size_in). It is the value of the
        #   input sequence at a given position.
        # - s_0 and h_0 are of size (self.dim_in,batch_size,self.size_out). They
        #   are the values of the cell state and hidden state at the "previous"
        #   positions, previous from every dimension (default should be 0).
        # 1/ Forget, input, output and cell activation gates.
        f = [torch.sigmoid(self.biasf[l] + torch.mm(x, self.wf[l]) + sum(torch.mul(h_0[k], self.uf[l][k]) for k in range(self.dim_in))) for l in range(self.dim_in)]
        i = torch.sigmoid(self.biasi + torch.mm(x, self.wi) + sum(torch.mul(h_0[k], self.ui[k]) for k in range(self.dim_in)))
        o = torch.sigmoid(self.biaso + torch.mm(x, self.wo) + sum(torch.mul(h_0[k], self.uo[k]) for k in range(self.dim_in)))
        c = torch.sigmoid(self.biasc + torch.mm(x, self.wc) + sum(torch.mul(h_0[k], self.uc[k]) for k in range(self.dim_in)))
        # 2/ Cell state
        s = torch.mul(i, c) + sum(torch.mul(f[k], s_0[k]) for k in range(self.dim_in))
        # 3/ Final output
        h = torch.mul(o, torch.tanh(s))
        return (s, h)
