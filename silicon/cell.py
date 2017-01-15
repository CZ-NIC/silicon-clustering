# -*- coding: utf-8 -*-

import numpy as np


class Cell(object):
    """
    A group of rows with specified projection coordinates in the same ranges.
    Divides data rows into groups such that only nearby (adjacent) groups need
    to be considered for similarity.

    First to be filled by `add_row` and then finalised by `compute_masked_m`.
    """

    def __init__(self, coords):

        self.coords = tuple(coords)
        # List of cell row numbers
        self.elements_idx = []
        # Array of cell row numbers after mask
        self.masked_idx = None
        # Feature sparse matrix corresponding to rows of self.masked_idx
        self.masked_mat = None

    def __str__(self):

        if self.masked_idx:
            return "<Cell %s, %d rows, %d with mask>" % (self.coords, len(self.elements_idx), len(self.masked_idx))
        else:
            return "<Cell %s, %d rows, no mask>" % (self.coords, len(self.elements_idx))

    def __repr__(self):

        return str(self)

    def add_row(self, row):
        """
        Add a row. Can be called repeatedly with the same row.
        Call only before calling `compute_masked_m`.
        """

        assert self.masked_idx is None
        self.elements_idx.append(row)

    def compute_masked_m(self, mask, ensemble):
        """
        Compute masked row index array and masked feature matrices for this cell.
        """

        idx = np.unique(self.elements_idx)
        self.elements_idx = list(idx)
        self.masked_idx = idx[mask[idx]]
        self.masked_mat = ensemble.data[self.masked_idx]
