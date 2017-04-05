Single-Linkage connectivity clustering
======================================

A fast algorithm computing connectivity components (clusters) of elements based
on their feature cosine similarity. Two elements are connected by an edge if
their cosine similarity is above a given threshold. Accepts either a numpy
2D array or a scipy column-sparse (CSR) matrix.

Install from `PyPI <https://pypi.python.org/pypi/silicon-clustering>`_
with ``pip install silicon-clustering``

    Tomáš Gavenčiak ``gavento@ucw.cz``

Usage example
-------------

::

    import silicon, numpy
    # 1000 rows, 10 features
    data = numpy.random.rand(1000, 10)
    # The ensemble normalizes rows of the data by default
    # Choose high verbosity (via logging), cosine similarity >=0.97
    ens = silicon.CosineClustering(data, sim_threshold=0.97, verbosity=2)
    ens.run()
    # (... progress reports)
    ens.clusters[0]
    # <Cluster no 0, size 13>
    ens.clusters_by_size()[0]
    # <Cluster no 36, size 22>
    # With pyplot you can see the projected points
    import matplotlib.pyplot as plt
    ens.plot(); plt.show()
    # or the individual clusters
    ens.clusters_by_size()[0].plot(ens); plt.show()

Details
-------

The algorithm uses several tricks to speed up the computation compared to traditional
all-pair scalar products via matrix multiplication:

* The data is projected into ``cell_dims`` principal components (PCA) by feature. The
  projection divides the data into cells of size ``self.distance`` so only rows from
  adjacent cells have a chance to be similar enough. Then the vectors of the rows of
  the adjacent cells are multiplied.

* This would still mean that some cells are too big for matrix multiplication so a second
  trick is used before the cell multiplication: nibbling at cca 1000 random points of the
  dataset. For a random center row, the similarities to all other rows are computed and
  all similar points are clustered together (possibly merging existing clusters). This has
  the effect of pre-clustering most dense points of the dataset (esp. repeated values)
  - dense clusters have a good chance to be hit with a center, eliminating most of the
  mass of the cluster (as well as the respective cells). The number of nibble
  iterations should be tuned according to data (to reasonably decrease the cell size).

* To combine the two tricks, a portion of the clustered points (e.g. 10%) together with
  all the unclustered points are considered for adjacent cell multiplication. The 10%
  returned points should ensure that the nibbled clusters are linked with any points not
  hit by the nibble but close to and in effect belonging to the clusters.

Since not all nibbled rows are used in the adjacent cell scalar product, the algorithm
may miss few individual cluster connections at nibble ball boundaries, but we found it
unlikely in practice.

The algorithm clusters 10000 gaussian points in 30 dimensions in under a second.