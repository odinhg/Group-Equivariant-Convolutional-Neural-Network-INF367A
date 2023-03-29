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
        # Find and store an ordered list of inverses of the functions
        self._inverses = self._functions[np.where(self.cayley_table == 0)[1]]

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

"""
    Functions for the symmetric group D2 of a rectangle
"""

def d2_mh(x: torch.Tensor) -> torch.Tensor:
    """
        Mirror image tensor around horizontal axis. Supports mini-batches.
    """
    return torch.flip(x, dims=[-2])

def d2_mv(x: torch.Tensor) -> torch.Tensor:
    """
        Mirror image tensor around vertical axis. Supports mini-batches.
    """
    return torch.flip(x, dims=[-1])

def d2_r(x: torch.Tensor) -> torch.Tensor:
    """
        Rotate image tensor 180 degrees CCW around center. Supports mini-batches.
    """
    return torch.flip(x, dims=[-1, -2])

def d2_e(x: torch.Tensor) -> torch.Tensor:
    """
        Identity.
    """
    return x
