# -*- coding: utf-8 -*-

import numpy as np
import os
import scipy.sparse
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import silicon as sc


class CosineTestSuite(unittest.TestCase):
    """Basic test cases and unit test."""

    TOP_12_SIZES = [4, 5, 5, 6, 7, 7, 8, 11, 11, 11, 14, 394]

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
        ens = sc.CosineClustering(data, verbosity=2,
                                  sim_threshold=0.4, cell_dims=2, rnd=rnd)
        ens.run()
        #print(ens, (sorted(map(len, ens.clusters)))[-12:])
        assert (sorted(map(len, ens.clusters)))[-12:] == self.TOP_12_SIZES

    def test_random_array_no_nibble(self):
        "Cluster a random 1000x100 array, no nibbles, dim=2"
        rnd = np.random.RandomState(42)
        data = self.gen_rand_data(1000, 100, 1.0, 0.1, rnd)
        ens = sc.CosineClustering(data, verbosity=2,
                                  sim_threshold=0.4, cell_dims=2, rnd=rnd)
        ens.run(nibbles=0)
        #print(ens, (sorted(map(len, ens.clusters)))[-12:])
        assert (sorted(map(len, ens.clusters)))[-12:] == self.TOP_12_SIZES

    def test_random_array_full(self):
        "Cluster a random 1000x100 array, no nibbles, full product (dims=0)"
        rnd = np.random.RandomState(42)
        data = self.gen_rand_data(1000, 100, 1.0, 0.1, rnd)
        ens = sc.CosineClustering(data, verbosity=2,
                                  sim_threshold=0.4, cell_dims=0, rnd=rnd)
        ens.run(nibbles=0)
        #print(ens, (sorted(map(len, ens.clusters)))[-12:])
        assert (sorted(map(len, ens.clusters)))[-12:] == self.TOP_12_SIZES

    def test_random_sparse(self):
        "Cluster a random 1000x100 sparse matrix (10% full)"
        rnd = np.random.RandomState(42)
        data = scipy.sparse.csr_matrix(self.gen_rand_data(1000, 100, 1.0, 0.1, rnd))
        ens = sc.CosineClustering(data, verbosity=2,
                                  sim_threshold=0.4, cell_dims=2, rnd=rnd)
        ens.run()
        #print((sorted(map(len, ens.clusters)))[-12:])
        assert (sorted(map(len, ens.clusters)))[-12:] == self.TOP_12_SIZES

