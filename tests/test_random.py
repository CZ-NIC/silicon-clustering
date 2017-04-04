# -*- coding: utf-8 -*-

import numpy as np
import os
import scipy.sparse
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import silicon as sc


class BasicTestSuite(unittest.TestCase):
    """Basic test cases and unit test."""

    def gen_rand_data(self, N, dim, radius, density, rnd):
        data = np.zeros((N, dim))
        for i in range(N):
            for j in range(dim):
                if rnd.rand() < density:
                    data[i][j] = (rnd.rand() - 0.5) * 2.0 * radius
        return data

    def test_random_array(self):
        "Cluster a random 1000x100 array"
        rnd = np.random.RandomState(42)
        data = self.gen_rand_data(1000, 100, 1.0, 0.1, rnd)
        ens = sc.CosineClustering(
                data, sim_threshold=0.4, cell_dims=2, normalize=True, rnd=rnd)
        ens.run()
        assert (sorted(map(len, ens.clusters)))[-12:] == [4, 5, 5, 6, 7, 7, 8, 11, 11, 11, 14, 394]

    def test_random_array_plain(self):
        "Cluster a random 1000x100 array, no nibbles, dim=1"
        rnd = np.random.RandomState(42)
        data = self.gen_rand_data(1000, 100, 1.0, 0.1, rnd)
        ens = sc.CosineClustering(
                data, sim_threshold=0.4, cell_dims=1, normalize=True, rnd=rnd)
        ens.run(nibbles=0)
        assert (sorted(map(len, ens.clusters)))[-12:] == [4, 5, 5, 6, 7, 7, 8, 11, 11, 11, 14, 394]

    def test_random_sparse(self):
        "Cluster a random 1000x100 sparse matrix (10% full)"
        rnd = np.random.RandomState(42)
        data = scipy.sparse.csr_matrix(self.gen_rand_data(1000, 100, 1.0, 0.1, rnd))
        ens = sc.CosineClustering(
                data, sim_threshold=0.4, cell_dims=2, normalize=True, rnd=rnd)
        ens.run()
        assert (sorted(map(len, ens.clusters)))[-12:] == [4, 5, 5, 6, 7, 7, 8, 11, 11, 11, 14, 394]

