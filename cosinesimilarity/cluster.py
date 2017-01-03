import numpy as np
import sklearn
import sklearn.preprocessing
import logging

logger = logging.getLogger('cosinesimilarity')


class Cluster(object):
    """
    A cluster of rows in the ensemble.
    The ensemble is not a part of the class to avoid circular references.
    Assumes that the `ensemble.cluster_map` is up to date.
    """

    def __init__(self, number):

        self.number = number
        self.elements_idx = np.array([], dtype=int)


    def __len__(self):

        return len(self.elements_idx)


    def __str__(self):

        return "<Cluster no {}, size {}>".format(self.number, len(self))


    def __repr__(self):

        return str(self)


    def add(self, idx, ensemble):
        """
        Add a row that is not in any other cluster.
        """

        assert ensemble.cluster_map[idx] == ensemble.NO_CLUSTER
        self.elements_idx = np.resize(self.elements_idx, len(self.elements_idx) + 1) ## SLOW
        self.elements_idx[-1] = idx
        ensemble.cluster_map[idx] = self.number


    def recompute_elems(self, cluster_map):
        """
        Recompute the set of indices based on `cluster_map`. SLOW (linear in `cluster_map`).
        """

        self.elements_idx = np.nonzero(cluster_map == self.number)[0]


    def merge(self, other, ensemble):
        """
        Merge two clusters (smaller into larger).
        Return the remaining one. The other cluster is kept as empty.
        """

        if self.number == other.number or len(other) == 0:
            return self

        if len(other) > len(self):
            return other.merge(self, ensemble)

        logger.debug("Merging %s into %s", other, self)
        ensemble.cluster_map[ other.elements_idx ] = self.number
        self.elements_idx = np.unique(np.concatenate((self.elements_idx, other.elements_idx)))
        other.elements_idx = np.array([], dtype=int)

        return self


    def get_PCA_for_feature(self, feature_i, ensemble, comps=2, max_rows=1000):
        """
        Compute and return normalised principal components for a feature within the cluster.

        Assumes that that the data has enough variability for this number of components.
        If the cluster has more than `max_rows` rows, selects `max_rows` pseudo-randomly.
        The randomness is seeded by the cluster number, so recomputations should be the same.
        """

        rnd = np.random.RandomState(self.number)

        if len(self) > max_rows:
            rows = rnd.choice(self.elements_idx, max_rows, replace=False)
        else:
            rows = self.elements_idx

        m = ensemble.features[feature_i].m[rows].toarray()
        P = sklearn.decomposition.RandomizedPCA(comps, random_state=rnd)
        P.fit(m)
        norm = sklearn.preprocessing.normalize(P.components_)
        return norm


    def plot_cluster_location(self, feature, ensemble):
        """
        Plot the cluster density over the overall ensemble data density in the background (for the given feature).
        If `ax` is given, plot into that axis (for saving etc.), otherwise plot and show in a window.
        """

        try:
            import matplotlib as mpl
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("Matplotlib nad Pyplot required for plot functions of cosinesimilarity") from exc

        side = ensemble.t_dist_single
        coords = ensemble.PCA_coords[feature]
        coords_c = ensemble.PCA_coords[feature][self.elements_idx]
        norm = mpl.colors.LogNorm()

        plt.gca().set_aspect('equal', adjustable='datalim')
        cropside = max(side, 2. / 120)
        plt.hist2d(coords[:,0], coords[:,1], (2 / cropside, 2 / cropside), range=((-1, 1), (-1, 1)), norm=norm, alpha=0.15)
        H_, x_, y_, im = plt.hist2d(coords_c[:,0], coords_c[:,1], (2 / cropside, 2 / cropside), range=((-1, 1), (-1, 1)), norm=norm)
        plt.plot((-0.9, -0.9 + side), (-0.9, -0.9), 'k-', lw=2)
        plt.colorbar(im)

    def plot_cluster_zoomed(self, feature, ensemble, resolution=(50,50)):
        """
        Plot the cluster density with cluster-specific PCA-given coordinates (for the given feature).

        If `ax` is given, plot into that axis (for saving etc.), otherwise plot and show in a window.
        The aspect-ratio of the plot is 1, `resolution` determines the number of histogram cells.
        Also plots a line of length `t_dist_single`.
        Returns True in success and False when the cluster does not have enough internal variance for
        a meningful projection.
        """

        side = ensemble.t_dist_single
        f_m = ensemble.features[feature].m[self.elements_idx]
        if (f_m.max(axis=0).toarray() == f_m.min(axis=0).toarray()).all():
            return False

        comps = self.get_PCA_for_feature(feature, ensemble, comps=2)
        coords = f_m * comps.T
        diameter = max(coords[:,0].max() - coords[:,0].min(), coords[:,1].max() - coords[:,1].min()) + 2 * side
        c1_avg = np.mean([coords[:,0].min(), coords[:,0].max()])
        c2_avg = np.mean([coords[:,1].min(), coords[:,1].max()])
        c1_lo = c1_avg - diameter / 2
        c1_hi = c1_avg + diameter / 2
        c2_lo = c2_avg - diameter / 2
        c2_hi = c2_avg + diameter / 2

        plt.gca().set_aspect('equal', adjustable='datalim')
        H_, x_, y_, im = plt.hist2d(coords[:,0], coords[:,1], resolution, range=((c1_lo, c1_hi), (c2_lo, c2_hi)), norm=mpl.colors.LogNorm())
        plt.plot((c1_lo + side / 3, c1_lo + side / 3 + side), (c2_lo + side / 3, c2_lo + side / 3), 'k-', lw=2)
        plt.colorbar(im)

        return True


    def plot_cluster_location_and_zoomed(self, feature, ensemble, resolution=(50,50)):

        fname = ensemble.features[feature].name

        plt.clf()
        plt.suptitle("Cluster %d, size %d, edge density %.4f" % (
            self.number, len(self), self.estimate_edge_density(ensemble)))

        plt.subplot(1, 2, 1)
        self.plot_cluster_location(feature, ensemble)
        plt.title("Location by '%s'" % fname)

        plt.subplot(1, 2, 2)
        self.plot_cluster_zoomed(feature, ensemble, resolution=resolution)
        plt.title("Zoom by '%s'" % fname)

        plt.gcf().set_size_inches(12,6)
        plt.tight_layout(rect=(0,0,1,0.95));


    def estimate_edge_density(self, ensemble, max_rows=1000):
        """
        Return an estimate of the ratio of similar elements to all element pairs (0.0-1.0).

        If the cluster has more than `max_rows` rows, selects `max_rows` pseudo-randomly.
        The randomness is seeded by the cluster number, so recomputations should be the same.
        """

        if len(self) <= 1:
            return 1.0

        if len(self) > max_rows:
            rnd = np.random.RandomState(self.number)
            rows = rnd.choice(self.elements_idx, max_rows, replace=False)
        else:
            rows = self.elements_idx

        mats = [ f.m[rows] for f in ensemble.features ]
        products = [ (m * m.T).toarray() for m in mats ]
        product_avg = np.mean(products, axis=0)
        close = len( (product_avg > ensemble.t_cos).nonzero()[0] )
        
        return close / (len(rows) ** 2)


    def get_row_names(self, ensemble):
        """
        Return an array of cluster elements row names.
        """

        return np.array(ensemble.row_cats.vals)[self.elements_idx]


    def get_feature_values(self, feature_i, ensemble):
        """
        Return an array of cluster value names encountered in feature `feature_i`.
        """

        f = ensemble.features[feature_i]
        value_idx = f.m[self.elements_idx].sum(axis=0).nonzero()[1]
        return np.array(f.value_cats.vals)[value_idx]




