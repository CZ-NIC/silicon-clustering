import itertools
import logging
import numpy as np
import pickle
import sklearn

from .cluster import Cluster
from .cell import Cell
from .utils import ProgressLogger, format_historgram

logger = logging.getLogger('cosinesimilarity')


class ClusteringEnsemble(object):
    """
    An optimized cosine-similarity clustering computation.

    The features are kept as SparseFeatures - scipy sparse matrices with rows indexed
    by `row_cats` categories and columns indexed by per-feature categorical indexes.

    Uses several tricks to speed up the computation compared to all-pair scalar products via matrix multiplication:

    * The data is projected into `cell_dims` principal components by fearure. The projection divides
      the data into cells of size `self.t_dist_single` so only rows from adjacent cells (all cell coordinates +-1)
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

    def __init__(self, sim_threshold=0.99, cell_dims=None):
        "After creation, call load_csv() to load the data."

        self.features = None
        self.n_rows = 0
        self.row_cats = None

        # Cosine similarity (compared to average of feature similarities)
        self.t_cos = sim_threshold
        self.t_cos_single = self.t_cos  # Recomputed later according to #features
        # Vector angle for all the features together
        self.t_angle = np.arccos(self.t_cos)
        self.t_angle_single = self.t_angle  # Recomputed later according to #features
        # Euclidean distance corresponding to the similarity and angle
        self.t_dist = 2 * np.sin(self.t_angle / 2)
        self.t_dist_single = self.t_dist  # Recomputed later according to #features

        self.cluster_map = None
        self.clusters = []

        self.PCA_comps = None
        self.PCA_coords = None

        self.cell_dims = cell_dims
        self.cells = {}

        self.cell_mask = None

    def init_from_features(self):
        "Update the configuration after loading the features. Called internally by load_csv etc."

        self.n_rows = self.features[0].m.shape[0]
        self.row_cats = self.features[0].row_cats
        self.cluster_map = np.repeat(self.NO_CLUSTER, self.n_rows)

        # Necessary cosine similarity for any single feature (assuming the others could be 1)
        self.t_cos_single = 1.0 - len(self.features) * (1.0 - self.t_cos)
        # Necessary vector angle for any single feature (similarly as t_cos_single)
        self.t_angle_single = np.arccos(self.t_cos_single)
        # Necessary euclidean distance for any single feature (simiarly as t_cos_single)
        self.t_dist_single = 2 * np.sin(self.t_angle_single / 2)

        if self.cell_dims and not len(self.cell_dims) == len(self.features):
            raise ValueError("ClusteringEnsemble cell_dims and feature number mismatch")

    def __str__(self):

        return "<ClusteringEnsemble [sim={:.5f}] {} rows, features ({})>".format(
            self.t_cos, self.n_rows, ', '.join(f.name for f in self.features))

    def __repr__(self):
        return str(self)

    def load_csv(self, fname, **kwargs):
        "Load a CSV file into freshly initialised ClusteringEnsemble"
        # TODO: remove

        assert self.features is None
        self.features = dataread.read_csv_sparse_matrix(fname, **kwargs)
        self.init_from_features()

    def normalise_features(self, exponent=2.0):
        "Normalise features separatedly to have L_(exponent) length 1"

        for f in self.features:
            f.normalise(2.0)

    def new_cluster(self):

        c = Cluster(len(self.clusters))
        self.clusters.append(c)
        return c

    def compute_PCAs(self, samples=5000, components=2):
        """
        Compute approximates of principial components of every feature.
        Only a bounded number of vectors is sampled for speed.
        """

        self.PCA_comps = []

        if samples <= self.n_rows:
            PCA_sample_ix = np.random.choice(self.n_rows, size=samples, replace=False)
        else:
            PCA_sample_ix = np.arange(self.n_rows)

        for fi, f in enumerate(self.features):
            cs = max(components, self.cell_dims[fi])
            logger.info("Finding %d PCAs for %d samples, feature %s ...", cs, samples, f.name)
            P = sklearn.decomposition.RandomizedPCA(cs)
            m = f.m[PCA_sample_ix].toarray()
            P.fit(m)
            self.PCA_comps.append(np.array([ comp / (comp.dot(comp)**0.5) for comp in P.components_ ]))

    def compute_PCA_coords(self, components=None):
        """
        Compute the coordinates of the feature points according to the normalised PCA vectors.
        """

        assert(self.PCA_comps is not None)
        logger.debug("Computing feature coordinates ...")
        self.PCA_coords = []

        for f, comps in zip(self.features, self.PCA_comps):
            self.PCA_coords.append(f.m * comps[:components].T)

    def plot_PCA_coords(self, feature_i, filter_=None):
        """
        Plot all the points (or those given by filter_) according to
        the PCA coordinates of the specified feature.
        """

        try:
            import matplotlib as mpl
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("Matplotlib nad Pyplot required for plot functions of cosinesimilarity") from exc

        assert(self.PCA_coords is not None)

        coords = self.PCA_coords[feature_i]
        if filter_ is not None:
            coords = coords[filter_]

        side = self.t_dist_single
        cropside = max(side, 2. / 120)
        H_, x_, y_, im = plt.hist2d(coords[:, 0], coords[:, 1], (2 / cropside, 2 / cropside), range=((-1, 1), (-1, 1)), norm=mpl.colors.LogNorm())
        plt.plot((-0.9, -0.9 + side), (-0.9, -0.9), 'k-', lw=2)
        plt.colorbar(im)
        fname = self.features[feature_i].name
        plt.title("PCA projection of '%s'" % fname)
        plt.gcf().set_size_inches(6, 6)
        plt.tight_layout(rect=(0, 0, 1, 0.95))

    def cell_coords_by_components(self, row):
        """
        Compute the cell coordinates for a given row.
        """

        c = []
        for d, coords in zip(self.cell_dims, self.PCA_coords):
            if d > 0:
                c.extend(list((coords[row][:d] // self.t_dist_single).astype(int)))

        return tuple(c)

    def split_to_cells(self):
        """
        Split rows into cells of side `t_dist_single` (per feature) and coordinates given by `PCA_coords`.
        `dims` tells how many dimensions to take from every feature, e.g. `dims=[2, 0, 1]`
        creates 3D cells, using 2 components from the first feature and 1 component from the third.
        """

        assert(self.PCA_coords is not None)

        self.cells = {}

        for row in range(self.n_rows):
            coords = self.cell_coords_by_components(row)
            if coords not in self.cells:
                self.cells[coords] = Cell(coords)
            self.cells[coords].add_row(row)

    def cluster_points_of_mask(self, mask):
        """
        Given a bool-mask or a row-array `mask`, join these points into a single cluster,
        possibly merging existing clusters or creating a new one.
        """

        clusters = [ self.clusters[cn] for cn in set(self.cluster_map[mask]) if cn != self.NO_CLUSTER ]
        clusters_str = '[' + ','.join([ str(c) for c in clusters ]) + ']'

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

    def cluster_around(self, center_row):
        """
        Cluster points around a point given by `center_row`.
        Computes the distances of `center_row` with all other points.
        """

        center_features = [f.m[center_row] for f in self.features]

        sim_f = [ f.m.dot(cf.T).toarray().flatten() for f, cf in zip(self.features, center_features) ]
        sim_all = np.sum(sim_f, axis=0) / len(sim_f)
        near_ix = (sim_all > self.t_cos)

        return self.cluster_points_of_mask(near_ix)

    def nibble_clusters(self, nibbles):
        """
        Repeatedly call `cluster_around` for random center points, preffering points
        not in a cluster as a center.
        """

        pl = ProgressLogger(logger, total=nibbles, msg="Nibbling progress")
        for i in range(nibbles):
            # Try to find index not in a cluster already, but not too hard
            for j in range(5):
                ix = np.random.randint(0, self.n_rows)
                if self.cluster_map[ix] == self.NO_CLUSTER:
                    break
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

        pl = ProgressLogger(logger, total=len(self.clusters), msg="Cell mask from clusters")
        for c in self.clusters:
            pl.update(1)
            if len(c) > 0:
                n = int((len(c) - minsize) * fraction + minsize)
                n = min(n, maxsize, len(c))
                c_returns = np.random.choice(c.elements_idx, size=n, replace=False)
                self.cell_mask[c_returns] = True

        pl = ProgressLogger(logger, total=len(self.cells), msg="Cell mask for small cells")
        for c in self.cells.values():
            pl.update(1)
            if len(c.elements_idx) <= cellminsize:
                # Include the entire cell
                self.cell_mask[np.array(c.elements_idx)] = True
            else:
                c_returns = np.random.choice(c.elements_idx, size=cellminsize, replace=False)
                self.cell_mask[c_returns] = True

    def compute_cell_matrices(self):
        """
        Compute masked feature matrices for all the cells.
        """

        pl = ProgressLogger(logger, total=len(self.cells), msg="Cell matrix computation")
        for c in self.cells.values():
            c.compute_masked_m(self.cell_mask, self)

    def multiply_adjacent_masked_cells(self):

        deltas = np.array(list(itertools.product( *([(-1, 0, 1)] * sum(self.cell_dims)) )))
        pl = ProgressLogger(logger, total=len(self.cells), msg="Adjacent cell multiplication")

        for c1_coords, c1 in self.cells.items():
            pl.update(1)
            for delta in deltas:
                c2_coords = tuple(np.array(c1_coords) + delta)
                if c1_coords >= c2_coords and c2_coords in self.cells:
                    c2 = self.cells[c2_coords]
                    self.multiply_masked_cells(c1, c2)

    def multiply_masked_cells(self, c1, c2):

        products = [ (m1 * m2.T).toarray() for m1, m2 in zip(c1.masked_mats, c2.masked_mats) ]
        product_avg = np.mean(products, axis=0)

        assert product_avg.shape == (len(c1.masked_idx), len(c2.masked_idx))

        for i1, i2 in zip(* (product_avg > self.t_cos).nonzero() ):
            # translate matrix indices to row indices and cluster them
            self.cluster_two_rows(c1.masked_idx[i1], c2.masked_idx[i2])

    def clusters_by_size(self):
        """
        Return the clusters sorted largest-to-smallest
        """

        return sorted(self.clusters, key=lambda c: len(c), reverse=True)

    def run_clustering(self, nibbles=1000):
        """
        After loading, run the clustering procedure on the data.
        Displays a lot of progress info and other information.
        """

        logger.info("Normalising features ...")
        self.normalise_features()

        logger.info("Computing feature PCAs ...")
        self.compute_PCAs(components=2)

        logger.info("Computing feature PCA coordinates ...")
        self.compute_PCA_coords()

        logger.info("Nibble-clustering ...")
        self.nibble_clusters(nibbles)

        cluster_hist_nibble = "    Unclustered rows: %d\n%s" % (
            sum(self.cluster_map == self.NO_CLUSTER),
            format_historgram([ len(c) for c in self.clusters ]))

        logger.info("Splitting rows into cells ...")
        self.split_to_cells()

        logger.info("Computing cluster rows to include in adjacent cell multiplication ...")
        self.compute_cell_mask()

        logger.info("Computing per-cell sparse matrices ...")
        self.compute_cell_matrices()

        logger.info("Cell full volume histogram:\n%s", format_historgram([ len(c.elements_idx) for c in self.cells.values() ]))

        logger.info("Cell masked volume histogram:\n%s", format_historgram([ len(c.masked_idx) for c in self.cells.values() ]))

        logger.info("Pairwise multiplying adjacent cells ...")
        self.multiply_adjacent_masked_cells()

        logger.info("Clustering control: %d nonempty, %d empty, total size %d (of %d rows), %d rows unclustered",
                    len([c for c in self.clusters if len(c) > 0]), len([c for c in self.clusters if len(c) == 0]),
                    sum([ len(c) for c in self.clusters ]), self.n_rows,
                    sum(self.cluster_map == self.NO_CLUSTER))

        logger.info("Cluster volumes histogram after nibble:\n%s", cluster_hist_nibble)

        logger.info("Final cluster volumes histogram:\n%s", format_historgram( [ len(c) for c in self.clusters ]))

    def save_pickle(self, pickle_fname):
        """
        Save the cluster as a python pickle. The size is approximately 50% of input CSV.
        """

        logger.info("Saving pickled ensemble to %s ...", pickle_fname)
        with open(pickle_fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_pickle(cls, pickle_fname):
        """
        Load the cluster from a python pickle.
        """

        logger.info("Loading pickled ensemble from %s ...", pickle_fname)
        with open(pickle_fname, 'rb') as f:
            ce = pickle.load(f)
            if not isinstance(ce, cls):
                logger.error("Loaded class %s, expected %s", type(ce), cls)
                return None
            logger.info("Loaded %s", ce)
            return ce
