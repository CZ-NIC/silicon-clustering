# -*- coding: utf-8 -*-

import numpy as np
import os
import scipy.sparse
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import silicon as sc

class BasicTestSuite(unittest.TestCase):
    """Basic test cases and unit test."""

    def test_cell(self):
        "Create a Cell"
        cell = sc.cell.Cell((-2, 1))
        assert 'no mask' in str(cell)

    def test_cluster(self):
        "Create a Cluster"
        cluster = sc.Cluster(42)
        assert 'Cluster no 42' in str(cluster)
        cluster.recompute_elems(np.array([2,3,42,0,42]))
        assert np.array_equal(cluster.elements_idx, [2, 4])


class CosineTestSuite(unittest.TestCase):
    """Basic test cases and unit test."""

    TEST_ARRAY = [
        [1.0, 1.0],
        [1.0, 1.01],
        [0.1, 3.0],
        [-0.1, 3.0],
        [-0.3, 3.0],
        [-0.5, 3.0],
        [-1.0, 2.0]]

    def test_ensemble_array(self):
        "Cluster a small array, check results"
        data = np.array(self.TEST_ARRAY)
        ens = sc.CosineClustering(
                data, sim_threshold=0.9942, cell_dims=2, rnd=np.random.RandomState(42), verbosity=2)
        ens.run()
        #print(ens.t_angle, ens.t_cos, ens)
        assert np.array_equal(ens.cluster_map, [0, 0, 1, 1, 1, 1, 2])
        assert len(ens.clusters) == 3
        assert len(ens.clusters_by_size()) == 3
        assert len(ens.clusters_by_size()[0]) == 4

    def test_ensemble_sparse(self):
        "Cluster a small sparse matrix, check results"
        data = scipy.sparse.csr_matrix(self.TEST_ARRAY)
        ens = sc.CosineClustering(
                data, sim_threshold=0.9942, cell_dims=2, rnd=42)
        ens.run()
        assert np.array_equal(ens.cluster_map, [0, 0, 1, 1, 1, 1, 2])

    def test_plot_array(self):
        "Cluster a small array, plot resulting clusters"

        try:
            import matplotlib as mpl
            import matplotlib.pyplot as plt
        except ImportError as exc:
            self.skipTest("Matplotlib/Pyplot not found")

        data = np.array(self.TEST_ARRAY)
        ens = sc.CosineClustering(
                data, sim_threshold=0.9942, cell_dims=2, rnd=42)
        ens.run()

        #plt.clf()
        ens.plot()
        #plt.savefig("test_plot_array-1.png")

        #plt.clf()
        ens.clusters[1].plot(ens)
        #plt.savefig("test_plot_array-2.png")

    def test_plot_sparse(self):
        "Cluster a small sparse matrix, plot resulting clusters"
        try:
            import matplotlib as mpl
            import matplotlib.pyplot as plt
        except ImportError as exc:
            self.skipTest("Matplotlib/Pyplot not found")

        data = scipy.sparse.csr_matrix(self.TEST_ARRAY)
        ens = sc.CosineClustering(
                data, sim_threshold=0.9942, cell_dims=2, rnd=42)
        ens.run()

        #plt.clf()
        ens.plot()
        #plt.savefig("test_plot_sparse-1.png")

        #plt.clf()
        ens.clusters[1].plot(ens)
        #plt.savefig("test_plot_sparse-2.png")

