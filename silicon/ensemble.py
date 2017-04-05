# -*- coding: utf-8 -*-

import itertools
import logging
import numpy as np
import pickle
import sklearn.decomposition
import scipy.sparse

from .cluster import Cluster
from .cell import Cell
from .utils import ProgressLogger, format_historgram, importPlt, toArray

logger = logging.getLogger('silicon')


class BaseClustering(object):
    """
    An optimized base for clustering computation.
    """

    NO_CLUSTER = -1
    DEAFULT_VERBOSITY = 1

    def __init__(self, data, distance, cell_dims=4, rnd=None, verbosity=None):
        """Common initialization (internal)"""

        # Numpy 2D array, Matrix or `scipy.sparse.csr_matrix` sparse matrix, columns are features
        self.data = data

        # The distance threshold for clustering
        self.distance = distance

        # Normalize and check types
        if isinstance(self.data, np.ndarray) and len(self.data.shape) == 2:
            self.sparse = False
        elif isinstance(self.data, scipy.sparse.csr.csr_matrix):
            self.sparse = True
        else:
            raise TypeError("%s needs data as 2D numpy.ndarray or scipy.sparse.csr_matrix" % self.__class__)

        # Number of points (rows)
        self.n_rows = self.data.shape[0]

        # Number of features (columns)
        self.n_features = self.data.shape[1]

        # Map row -> cluster_number
        self.cluster_map = np.repeat([self.NO_CLUSTER], self.n_rows)

        # List of Cluster instances (by number)
        self.clusters = []

        # Number of dimensions to split into cells
        self.cell_dims = cell_dims

        # PCA components of a row sample
        self.PCA_comps = None

        # Coordinates of the points
        self.PCA_coords = None

        # Dictionary of existing cells (int_coords) -> Cell
        self.cells = {}

        # Mask of rows to be included in the final adjacent-cell multiplication
        self.cell_mask = None

        # Random number generator to use (instance of numpy.random)
        if isinstance(rnd, int) or rnd is None:
            self.rnd = np.random.RandomState(rnd)
        elif isinstance(rnd, np.random.RandomState):
            self.rnd = rnd
        else:
            raise TypeError("%s needs np.random.RandomState, int or None as rnd" % self.__class__)

        # Vebosity level: 0, 1, 2
        if verbosity is None:
            self.verbosity = self.DEAFULT_VERBOSITY
        else:
            self.verbosity = verbosity

    def progress(self, msg, *a, **k):
        """Log progress if enabled."""
        if self.verbosity >= 1:
            logger.info(msg, *a, **k)

    def details(self, msg, *a, **k):
        """Log details/stats if enabled."""
        if self.verbosity >= 2:
            logger.info(msg, *a, **k)
            
    def new_progress_logger(self, *a, **k):
        """Create a new ProgressLogger if enabled."""
        return ProgressLogger(logger if self.verbosity >= 1 else None, *a, **k)

    def run_generic(self, nibbles=1000, PCA_samples=5000):
        """
        Run the generic dist-based clustering procedure on the data (after preprocessing).
        """

        # Compute PCA and coordinates
        if self.cell_dims > 0:
            self.progress("Computing feature PCA (%d dimensions) ..." % self.cell_dims)
            self.PCA_comps = self.compute_PCAs(samples=PCA_samples, components=self.cell_dims)
            self.PCA_coords = self.data.dot(self.PCA_comps.T)
        else:
            self.PCA_comps = []
            self.PCA_coords = np.zeros((self.n_rows, 1))


        nib_count = min(nibbles, self.n_rows // 2)
        self.progress("Nibble-clustering %d times ..." % nib_count)
        self.nibble_clusters(nib_count)

        cluster_hist_nibble = "    Unclustered rows: %d\n%s" % (
            sum(self.cluster_map == self.NO_CLUSTER),
            format_historgram([len(c) for c in self.clusters]))

        self.progress("Splitting rows into cells ...")
        self.split_to_cells()

        self.progress("Computing cluster rows to include in adjacent cell multiplication ...")
        self.compute_cell_mask()

        self.progress("Computing per-cell matrices ...")
        self.compute_cell_matrices()

        self.details("Cell full volume histogram:\n%s",
                     format_historgram([len(c.elements_idx) for c in self.cells.values()]))

        self.details("Cell masked volume histogram:\n%s",
                     format_historgram([len(c.masked_idx) for c in self.cells.values()]))

        self.progress("Pairwise multiplying adjacent cells ...")
        self.multiply_adjacent_masked_cells()

        # Cleanup
        self.clusters = [c for c in self.clusters if len(c) > 0]
        for i, c in enumerate(self.clusters):
            c.number = i
            for r in c.elements_idx:
                self.cluster_map[r] = i
        del self.cell_mask
        del self.cells

        self.details("Clustering control: %d nonempty, %d empty, total size %d (of %d rows), %d rows unclustered",
                     len([c for c in self.clusters if len(c) > 0]),
                     len([c for c in self.clusters if len(c) == 0]),
                     sum([len(c) for c in self.clusters]), self.n_rows,
                     sum(self.cluster_map == self.NO_CLUSTER))

        self.details("Cluster volumes histogram after nibble:\n%s", cluster_hist_nibble)

        self.details("Final cluster volumes histogram:\n%s", format_historgram([len(c) for c in self.clusters]))

    def __str__(self):
        return "<{}, dist={}, {} rows, {} features, {} clusters>".format(
            self.__class__.__name__, self.distance, self.n_rows, self.n_features, len(self.clusters))

    def __repr__(self):
        return str(self)

    def compute_PCAs(self, samples, components):
        """
        Compute approximates of principial components of every feature.
        Only a bounded number of vectors is sampled for speed.
        """

        self.PCA_comps = []

        if samples <= self.n_rows:
            PCA_sample_ix = self.rnd.choice(self.n_rows, size=samples, replace=False)
        else:
            PCA_sample_ix = np.arange(self.n_rows)

        logger.debug("Finding %d PCAs for %d samples ...", components, samples)
        # noinspection PyArgumentList
        P = sklearn.decomposition.PCA(components, random_state=self.rnd, svd_solver='randomized')
        m = toArray(self.data[PCA_sample_ix])
        P.fit(m)
        return np.array([comp / (comp.dot(comp) ** 0.5) for comp in P.components_])

    def plot(self, filter_=None):
        """
        Plot all the points (or those given by filter_) according to the PCA coordinates.
        """

        mpl, plt = importPlt()

        assert (self.PCA_coords is not None)
        coords = self.PCA_coords
        if filter_ is not None:
            coords = coords[filter_]

        side = self.distance
        cropside = max(side, 2. / 120)
        H_, x_, y_, im = plt.hist2d(coords[:, 0], coords[:, 1], (2 / cropside, 2 / cropside),
                                    range=((-1, 1), (-1, 1)), norm=mpl.colors.LogNorm())
        plt.plot((-0.9, -0.9 + side), (-0.9, -0.9), 'k-', lw=2)
        plt.colorbar(im)
        plt.title("PCA projection")
        plt.gcf().set_size_inches(6, 6)
        plt.tight_layout(rect=(0, 0, 1, 0.95))

    def new_cluster(self):

        c = Cluster(len(self.clusters))
        self.clusters.append(c)
        assert self.clusters[c.number] is c
        return c

    def split_to_cells(self):
        """
        Split rows into cells of side `t_dist` and coordinates given by `PCA_coords`.
        """
        assert (self.PCA_coords is not None)

        for row in range(self.n_rows):
            coords = tuple((self.PCA_coords[row] // self.distance).astype(int))
            if coords not in self.cells:
                self.cells[coords] = Cell(coords)
            self.cells[coords].add_row(row)

    def cluster_points_of_mask(self, mask):
        """
        Given a bool-mask or a row-array `mask`, join these points into a single cluster,
        possibly merging existing clusters or creating a new one.
        """

        clusters = [self.clusters[cn] for cn in set(self.cluster_map[mask]) if cn != self.NO_CLUSTER]
        clusters_str = '[' + ','.join([str(c) for c in clusters]) + ']'

        if clusters:
            res_cluster = clusters[0]
            for c in clusters[1:]:
                res_cluster = res_cluster.merge(c, self)
        else:
            res_cluster = self.new_cluster()

        old_count = len(res_cluster)
        self.cluster_map[mask] = res_cluster.number
        res_cluster.recompute_elems(self.cluster_map)  # Potentially slower

        logger.debug("Clustered %s plus %d points into %s",
                     clusters_str, len(res_cluster) - old_count, res_cluster)
        return res_cluster

    def cluster_two_rows(self, r1, r2):
        """
        Merge two given rows into a cluster. The rows may be the same or the same cluster.
        May merge two clusters or create a new cluster. Similar to `cluster_points_of_mask`.
        """

        cn1 = self.cluster_map[r1]
        cn2 = self.cluster_map[r2]

        if cn1 == self.NO_CLUSTER and cn2 == self.NO_CLUSTER:
            nc = self.new_cluster()
            nc.add(r1, self)
            if r1 != r2:
                nc.add(r2, self)

        elif cn1 == self.NO_CLUSTER:
            self.clusters[cn2].add(r1, self)

        elif cn2 == self.NO_CLUSTER:
            self.clusters[cn1].add(r2, self)

        else:
            if cn1 != cn2:
                self.clusters[cn1].merge(self.clusters[cn2], self)

    def nibble_clusters(self, nibbles):
        """
        Repeatedly call `cluster_around` for random center points, preferring points
        not in a cluster as a center.
        """

        pl = self.new_progress_logger(total=nibbles, msg="Nibbling progress")
        for i in range(nibbles):
            # Try to find index not in a cluster already, but not too hard
            ix = self.rnd.randint(0, self.n_rows)
            for j in range(5):
                if self.cluster_map[ix] == self.NO_CLUSTER:
                    break
                ix = self.rnd.randint(0, self.n_rows)
            self.cluster_around(ix)
            pl.set(i)

    def compute_cell_mask(self, minsize=100, maxsize=1000, fraction=0.1, cellminsize=200):
        """
        Compute a mask of points to include in adjacent cell multiplication.

        All nonclustered rows are included. Between `minsize` and `maxsize`
        points of every cluster are included (using `fraction` points in between).

        Then every cell mask is extended to have at least `cellminsize` points
        (or all cell rows if smaller). This is done to e.g. to avoid leaving out
        almost-empty but nibble-clustered cells entirely (as they may be a bridge
        to another cluster part).
        """

        self.cell_mask = (self.cluster_map == self.NO_CLUSTER)

        pl = self.new_progress_logger(total=len(self.clusters), msg="Cell mask from clusters")
        for c in self.clusters:
            pl.update(1)
            if len(c) > 0:
                n = int((len(c) - minsize) * fraction + minsize)
                n = min(n, maxsize, len(c))
                c_returns = self.rnd.choice(c.elements_idx, size=n, replace=False)
                self.cell_mask[c_returns] = True

        pl = self.new_progress_logger(total=len(self.cells), msg="Cell mask for small cells")
        for c in self.cells.values():
            pl.update(1)
            if len(c.elements_idx) <= cellminsize:
                # Include the entire cell
                self.cell_mask[np.array(c.elements_idx)] = True
            else:
                c_returns = self.rnd.choice(c.elements_idx, size=cellminsize, replace=False)
                self.cell_mask[c_returns] = True

    def compute_cell_matrices(self):
        """
        Compute masked feature matrices for all the cells.
        """

        pl = self.new_progress_logger(total=len(self.cells), msg="Cell matrix computation")
        for c in self.cells.values():
            c.compute_masked_m(self.cell_mask, self)
            pl.update(1)

    def multiply_adjacent_masked_cells(self):

        # special case for cell_dims == 0
        if self.cell_dims == 0:
            assert len(self.cells) == 1
            self.multiply_masked_cells(self.cells[(0,)], self.cells[(0,)])
            return

        deltas = np.array(list(itertools.product(*([(-1, 0, 1)] * self.cell_dims))))
        pl = self.new_progress_logger(total=len(self.cells), msg="Adjacent cell multiplication")

        for c1_coords, c1 in self.cells.items():
            pl.update(1)
            for delta in deltas:
                c2_coords = tuple(np.array(c1_coords) + delta)
                if c1_coords >= c2_coords and c2_coords in self.cells:
                    c2 = self.cells[c2_coords]
                    self.multiply_masked_cells(c1, c2)

    def clusters_by_size(self):
        """
        Return the clusters sorted largest-to-smallest. May include empty clusters.
        """

        return sorted(self.clusters, key=lambda c: len(c), reverse=True)

    def save_pickle(self, pickle_fname):
        """
        Save the clustering as a python pickle. The size is approximately 50% of a CSV.
        """

        logger.info("Saving pickled ensemble to %s ...", pickle_fname)
        with open(pickle_fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_pickle(cls, pickle_fname):
        """
        Load the clustering from a python pickle.
        """

        logger.info("Loading pickled ensemble from %s ...", pickle_fname)
        with open(pickle_fname, 'rb') as f:
            ce = pickle.load(f)
            if not isinstance(ce, cls):
                logger.error("Loaded class %s, expected %s", type(ce), cls)
                return None
            logger.info("Loaded %s", ce)
            return ce


class CosineClustering(BaseClustering):
    """
    An optimized cosine-similarity clustering computation.
    
    TODO: Update

    The features are kept as SparseFeatures - scipy sparse matrices with rows indexed
    by `row_cats` categories and columns indexed by per-feature categorical indexes.

    Uses several tricks to speed up the computation compared to all-pair scalar products via matrix multiplication:

    * The data is projected into `cell_dims` principal components by fearure. The projection divides
      the data into cells of size `self.t_dist` so only rows from adjacent cells (all cell coordinates +-1)
      have a chance to be similar enough. Then the vectors of the rows of the adjacent cells are multiplied.

      This would still mean that some cells are too big for matrix multiplication so a second trick is used
      before the cell multiplication.

    * Nibbling at cca 1000 random points of the dataset. For a random center row, the similarities to all other rows are
      computed and all similar points are clustered together (possibly merging existing clusters).
      This has the effect of pre-clustering most dense points of the dataset (esp. repeated values) - dense clusters
      have a good chance to be hit with a center, eliminating most of the mass of the cluster (as well as the
      respective cells).
      The number of nibble iterations should be tuned according to data (to reasonably decrease the cell size).

    * To combine the two tricks, a portion of the clustered points (e.g. 10%) together with all the unclustered
      points are considered for adjacent cell multiplication. The 10% "returned" points should ensure that the
      nibbled clusters are linked with any points not hit by the nibble but close to and in effect belonging
      to the clusters.
    """

    NO_CLUSTER = -1

    def __init__(self, data, sim_threshold=0.99, cell_dims=4, rnd=None, normalized=False, verbosity=None):
        """
        Create a clustering ensemble for given data.
        
        Since the algorithm needs normalized (length 1) vectors, a copy of the data is normalized.
        If your data is normalized, set normalized=False (then the data is not copied).
        Do not modify the data during the ensemble lifetime.
        
        See the class docstring for more parameters.
        """

        self.normalized = normalized

        # Cosine similarity (compared to feature similarities)
        self.t_cos = sim_threshold

        # Vector angle for all the features together
        self.t_angle = np.arccos(self.t_cos)

        # Euclidean distance corresponding to the similarity and angle when normalized
        t_dist = (2 * (1.0 - self.t_cos)) ** 0.5
        assert abs(t_dist - 2 * np.sin(self.t_angle / 2)) < 1e-6

        super(CosineClustering, self).__init__(data, t_dist, cell_dims=cell_dims, rnd=rnd, verbosity=verbosity)

    def run(self, nibbles=1000, PCA_samples=5000):
        """
        Run the clustering procedure on the data.
        """

        # Normalize a copy of the data if requested
        if not self.normalized:
            self.progress("Normalising features ...")
            if self.sparse:
                norm_rows = scipy.sparse.csr_matrix(1.0 / np.power(self.data.power(2.0).sum(axis=1), 0.5))
                self.data = self.data.multiply(norm_rows)
            else:
                norm_rows = 1.0 / (self.data ** 2.0).sum(axis=1) ** 0.5
                self.data = self.data * np.array([norm_rows]).T

        self.run_generic(nibbles=nibbles, PCA_samples=PCA_samples)

    def cluster_around(self, center_row):
        """
        Cluster points around a point given by `center_row`.
        Computes the distances of `center_row` with all other points.
        """

        center = self.data[center_row]

        sim = toArray(self.data.dot(center.T)).flatten()
        near_ix = (sim >= self.t_cos)

        return self.cluster_points_of_mask(near_ix)

    def multiply_masked_cells(self, c1, c2):

        product = toArray(c1.masked_mat.dot(c2.masked_mat.T))
        assert product.shape == (len(c1.masked_idx), len(c2.masked_idx))

        for i1, i2 in zip(*(product > self.t_cos).nonzero()):
            # translate matrix indices to row indices and cluster them
            self.cluster_two_rows(c1.masked_idx[i1], c2.masked_idx[i2])

    def __str__(self):
        return "<{}, cos={:f}, dist={:f}, {} rows, {} features, " \
               "{} clusters>".format(
            self.__class__.__name__, self.t_cos, self.distance, self.n_rows,
            self.n_features, len(self.clusters))

