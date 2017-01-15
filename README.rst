Clustering by cosine similarity
===============================

Connectivity
Clustering
Sparse
Nearest-?

(Distance, Cosine Similarity)



A fast algorithm computing connectivity components (clusters) of elements based 
on their cosine similarity. Two elements are connected by an edge if their cosine 
similarity is above a given threshold.

The features are kept as SparseFeatures - scipy sparse matrices with rows indexed
by `row_cats` categories and columns indexed by per-feature categorical indexes.

Uses several tricks to speed up the computation compared to all-pair scalar
products via matrix multiplication:

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

