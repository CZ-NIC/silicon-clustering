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


class ClusteringBase(object):
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
        self._clusters = []

        # Number of dimensions to split into cells
        self.cell_dims = min(cell_dims, self.n_features)

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

    def _progress(self, msg, *a, **k):
        """Log progress if enabled."""
        if self.verbosity >= 1:
            logger.info(msg, *a, **k)

    def _details(self, msg, *a, **k):
        """Log details/stats if enabled."""
        if self.verbosity >= 2:
            logger.info(msg, *a, **k)

    def _new_progress_logger(self, *a, **k):
        """Create a new ProgressLogger if enabled."""
        return ProgressLogger(logger if self.verbosity >= 1 else None, *a, **k)

    def _run_generic(self, nibbles=1000, PCA_samples=5000):
        """
        Run the generic dist-based clustering procedure on the data (after preprocessing).
        """

        # Compute PCA and coordinates
        if self.cell_dims > 0:
            self._progress("Computing feature PCA (%d dimensions) ..." % self.cell_dims)
            self.PCA_comps = self._compute_PCAs(samples=PCA_samples, components=self.cell_dims)
            self.PCA_coords = self.data.dot(self.PCA_comps.T)
        else:
            self.PCA_comps = []
            self.PCA_coords = np.zeros((self.n_rows, 1))


        nib_count = min(nibbles, self.n_rows // 2)
        self._progress("Nibble-clustering %d times ..." % nib_count)
        self._nibble_clusters(nib_count)

        cluster_hist_nibble = "    Unclustered rows: %d\n%s" % (
            sum(self.cluster_map == self.NO_CLUSTER),
            format_historgram([len(c) for c in self._clusters]))

        self._progress("Splitting rows into cells ...")
        self._split_to_cells()

        self._progress("Computing cluster rows to include in adjacent cell multiplication ...")
        self._compute_cell_mask()

        self._progress("Computing per-cell matrices ...")
        self._compute_cell_matrices()

        self._details("Cell full volume histogram:\n%s",
                     format_historgram([len(c.elements_idx) for c in self.cells.values()]))

        self._details("Cell masked volume histogram:\n%s",
                     format_historgram([len(c.masked_idx) for c in self.cells.values()]))

        self._progress("Pairwise multiplying adjacent cells ...")
        self._multiply_adjacent_masked_cells()

        # Cleanup
        self._clusters = [c for c in self._clusters if len(c) > 0]
        for i, c in enumerate(self._clusters):
            c.number = i
            for r in c.elements_idx:
                self.cluster_map[r] = i
        del self.cell_mask
        del self.cells

        self._details("Clustering control: %d nonempty, %d empty, total size %d (of %d rows), %d rows unclustered",
                     len([c for c in self._clusters if len(c) > 0]),
                     len([c for c in self._clusters if len(c) == 0]),
                     sum([len(c) for c in self._clusters]), self.n_rows,
                     sum(self.cluster_map == self.NO_CLUSTER))

        self._details("Cluster volumes histogram after nibble:\n%s", cluster_hist_nibble)

        self._details("Final cluster volumes histogram:\n%s", format_historgram([len(c) for c in self._clusters]))

    def __str__(self):
        return "<{}, dist={}, {} rows, {} features, {} clusters>".format(
            self.__class__.__name__, self.distance, self.n_rows, self.n_features, len(self._clusters))

    def __repr__(self):
        return str(self)

    def _compute_PCAs(self, samples, components):
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

    def plot(self, filt=None):
        """
        Plot all the points (or those given by filt) according to the PCA coordinates.
        """

        mpl, plt = importPlt()

        assert (self.PCA_coords is not None)
        coords = self.PCA_coords
        if filt is not None:
            coords = coords[filt]

        side = self.distance
        cropside = max(side, 2. / 120)
        H_, x_, y_, im = plt.hist2d(coords[:, 0], coords[:, 1], (2 / cropside, 2 / cropside),
                                    range=((-1, 1), (-1, 1)), norm=mpl.colors.LogNorm())
        plt.plot((-0.9, -0.9 + side), (-0.9, -0.9), 'k-', lw=2)
        plt.colorbar(im)
        plt.title("PCA projection")
        plt.gcf().set_size_inches(6, 6)
        plt.tight_layout(rect=(0, 0, 1, 0.95))

    def _new_cluster(self):

        c = Cluster(len(self._clusters), self)
        self._clusters.append(c)
        assert self._clusters[c.number] is c
        return c

    def _split_to_cells(self):
        """
        Split rows into cells of side `t_dist` and coordinates given by `PCA_coords`.
        """
        assert (self.PCA_coords is not None)

        for row in range(self.n_rows):
            coords = tuple((self.PCA_coords[row] // self.distance).astype(int))
            if coords not in self.cells:
                self.cells[coords] = Cell(coords)
            self.cells[coords].add_row(row)

    def _cluster_points_of_mask(self, mask):
        """
        Given a bool-mask or a row-array `mask`, join these points into a single cluster,
        possibly merging existing clusters or creating a new one.
        """

        clusters = [self._clusters[cn] for cn in set(self.cluster_map[mask]) if cn != self.NO_CLUSTER]
        clusters_str = '[' + ','.join([str(c) for c in clusters]) + ']'

        if clusters:
            res_cluster = clusters[0]
            for c in clusters[1:]:
                res_cluster = res_cluster._merge(c)
        else:
            res_cluster = self._new_cluster()

        old_count = len(res_cluster)
        self.cluster_map[mask] = res_cluster.number
        res_cluster._recompute_elems(self.cluster_map)  # Potentially slower

        logger.debug("Clustered %s plus %d points into %s",
                     clusters_str, len(res_cluster) - old_count, res_cluster)
        return res_cluster

    def _cluster_two_rows(self, r1, r2):
        """
        Merge two given rows into a cluster. The rows may be the same or the same cluster.
        May merge two clusters or create a new cluster. Similar to `cluster_points_of_mask`.
        """

        cn1 = self.cluster_map[r1]
        cn2 = self.cluster_map[r2]

        if cn1 == self.NO_CLUSTER and cn2 == self.NO_CLUSTER:
            nc = self._new_cluster()
            nc._add(r1)
            if r1 != r2:
                nc._add(r2)

        elif cn1 == self.NO_CLUSTER:
            self._clusters[cn2]._add(r1)

        elif cn2 == self.NO_CLUSTER:
            self._clusters[cn1]._add(r2)

        else:
            if cn1 != cn2:
                self._clusters[cn1]._merge(self._clusters[cn2])

    def _nibble_clusters(self, nibbles):
        """
        Repeatedly call `cluster_around` for random center points, preferring points
        not in a cluster as a center.
        """

        pl = self._new_progress_logger(total=nibbles, msg="Nibbling progress")
        for i in range(nibbles):
            # Try to find index not in a cluster already, but not too hard
            ix = self.rnd.randint(0, self.n_rows)
            for j in range(5):
                if self.cluster_map[ix] == self.NO_CLUSTER:
                    break
                ix = self.rnd.randint(0, self.n_rows)
            self._cluster_around(ix)
            pl.set(i)

    def _compute_cell_mask(self, minsize=100, maxsize=1000, fraction=0.1, cellminsize=200):
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

        pl = self._new_progress_logger(total=len(self._clusters), msg="Cell mask from clusters")
        for c in self._clusters:
            pl.update(1)
            if len(c) > 0:
                n = int((len(c) - minsize) * fraction + minsize)
                n = min(n, maxsize, len(c))
                c_returns = self.rnd.choice(c.elements_idx, size=n, replace=False)
                self.cell_mask[c_returns] = True

        pl = self._new_progress_logger(total=len(self.cells), msg="Cell mask for small cells")
        for c in self.cells.values():
            pl.update(1)
            if len(c.elements_idx) <= cellminsize:
                # Include the entire cell
                self.cell_mask[np.array(c.elements_idx)] = True
            else:
                c_returns = self.rnd.choice(c.elements_idx, size=cellminsize, replace=False)
                self.cell_mask[c_returns] = True

    def _compute_cell_matrices(self):
        """
        Compute masked feature matrices for all the cells.
        """

        pl = self._new_progress_logger(total=len(self.cells), msg="Cell matrix computation")
        for c in self.cells.values():
            c.compute_masked_m(self.cell_mask, self)
            pl.update(1)

    def _multiply_adjacent_masked_cells(self):

        # special case for cell_dims == 0
        if self.cell_dims == 0:
            assert len(self.cells) == 1
            self._multiply_masked_cells(self.cells[(0,)], self.cells[(0,)])
            return

        deltas = np.array(list(itertools.product(*([(-1, 0, 1)] * self.cell_dims))))
        pl = self._new_progress_logger(total=len(self.cells), msg="Adjacent cell multiplication")

        for c1_coords, c1 in self.cells.items():
            pl.update(1)
            for delta in deltas:
                c2_coords = tuple(np.array(c1_coords) + delta)
                if c1_coords >= c2_coords and c2_coords in self.cells:
                    c2 = self.cells[c2_coords]
                    self._multiply_masked_cells(c1, c2)

    def clusters_by_size(self):
        """
        Return the nonempty clusters sorted largest-to-smallest.
        """

        return sorted(self._clusters, key=lambda c: len(c), reverse=True)

    def clusters(self):
        """Return the list of nonempty found clusters sorted by number."""
        return self._clusters

    def _save_pickle(self, pickle_fname):
        """
        Save the clustering as a python pickle. The size is approximately 50% of a CSV.
        """

        logger.info("Saving pickled ensemble to %s ...", pickle_fname)
        with open(pickle_fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def _load_pickle(cls, pickle_fname):
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


class CosineClustering(ClusteringBase):
    """
    An optimized cosine-similarity single-linkage clustering algorithm.

    Supports both dense arrays (`numpy.ndarray`) and sparse matrices
    (`scipy.sparse.csr_matrix`).
    Allows a simple cluster visualisation with `matplotlib`.

    The algorithm is not executed by creation but by calling `ensemble.run()`.

    Uses several tricks to speed up the computation compared to all-pair scalar
    products via matrix multiplication. See the docs for details.

    * The data is projected into `cell_dims` principal components by fearure and
      split into cells of the same dimensionality. Only rows of adjacent cells
      have to be considered.

    * Dense areas are pre-clustred by guessing some potential dense ball centers
      and clustering points similar to them, possibly merging existing clusters.
      Only few (cca 10%) of these pre-clustered rows are considered in the pairwise
      similarity computation, potentially losing some cluster links in special
      configurations.
    """

    NO_CLUSTER = -1

    def __init__(self, data, sim_threshold=0.99, cell_dims=4, rnd=None, normalized=False, verbosity=None):
        """
        Create a clustering ensemble for given data.

        Since the algorithm needs normalized (length 1) vectors, a copy of the data is normalized.
        If your data is normalized, set `normalized=False` (then the data is not copied).
        Do not modify the data during the ensemble lifetime.

        Verbosity 0-2 (default 1) determines the progress reporting of `run`.
        `rnd` is an optional `numpy.random.RandomState` instance to ensure
        repeatability. See the class docstring for `cell_dims`.
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
        Run the clustering procedure as created.
        """

        # Normalize a copy of the data if requested
        if not self.normalized:
            self._progress("Normalising features ...")
            if self.sparse:
                norm_rows = scipy.sparse.csr_matrix(1.0 / np.power(self.data.power(2.0).sum(axis=1), 0.5))
                self.data = self.data.multiply(norm_rows)
            else:
                norm_rows = 1.0 / (self.data ** 2.0).sum(axis=1) ** 0.5
                self.data = self.data * np.array([norm_rows]).T

        self._run_generic(nibbles=nibbles, PCA_samples=PCA_samples)

    def _cluster_around(self, center_row):
        """
        Cluster points around a point given by `center_row`.
        Computes the distances of `center_row` with all other points.
        """

        center = self.data[center_row]

        sim = toArray(self.data.dot(center.T)).flatten()
        near_ix = (sim >= self.t_cos)

        return self._cluster_points_of_mask(near_ix)

    def _multiply_masked_cells(self, c1, c2):

        product = toArray(c1.masked_mat.dot(c2.masked_mat.T))
        assert product.shape == (len(c1.masked_idx), len(c2.masked_idx))

        for i1, i2 in zip(*(product > self.t_cos).nonzero()):
            # translate matrix indices to row indices and cluster them
            self._cluster_two_rows(c1.masked_idx[i1], c2.masked_idx[i2])

    def __str__(self):
        return "<{}, cos={:f}, dist={:f}, {} rows, {} features, " \
               "{} clusters>".format(
            self.__class__.__name__, self.t_cos, self.distance, self.n_rows,
            self.n_features, len(self._clusters))
