import numpy as np
import torch

class Group():
    """
        Data structure for storing a representation of a group via a list of functions acting on the signals, and a Cayley table encoding the group structure. Note that the order of the list of functions must match the order in the Cayley table. Furthermore, the identity function must be the first function in the list.
    """
    def __init__(self, functions, cayley_table):
        self._functions = np.array(functions)
        self._cayley_table = np.array(cayley_table, dtype=np.int64)
        self._order = len(functions)
        # Indices of inverse functions
        self._inverse_indices = np.where(self.cayley_table == 0)[1]
        # Find and store an ordered list of inverses of the functions
        self._inverses = self._functions[self._inverse_indices]
        self._permuted_indices = self.cayley_table[self._inverse_indices]

    @property
    def permuted_indices(self):
        return self._permuted_indices

    @property
    def functions(self):
        return self._functions

    @property
    def cayley_table(self):
        return self._cayley_table

    @property
    def order(self):
        return self._order

    @property
    def inverses(self):
        return self._inverses

    @property
    def inverse_indices(self):
        return self._inverse_indices

"""
    Functions for the symmetric group D2 acting on a stereo image.
"""

def d2_mh(x: torch.Tensor) -> torch.Tensor:
    """
        Mirror image around horizontal axis by flipping in the height-dimension.
    """
    y = torch.flip(x, dims=[-2])
    return y

def d2_mv(x: torch.Tensor) -> torch.Tensor:
    """
        Mirror image tensor around vertical axis by flipping in the width dimension and swapping views.
    """
    y = torch.flip(x, dims=[-1, -3])
    return y

def d2_r(x: torch.Tensor) -> torch.Tensor:
    """
        Rotate image tensor 180 degrees CCW. Rotate both views by flipping in height and width dimensions. Then swap views to get the rotation of a stereo image.
    """
    return torch.flip(x, dims=[-1, -2, -3])

def d2_e(x: torch.Tensor) -> torch.Tensor:
    """
        Every group element deserves its own function! Just because you're an element of the trivial group doesn't mean you're not important. 
    """
    return x
