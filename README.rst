SIngle-LInkage CONnectivity clustering
======================================

A Python library for a fast approximation of
`single-linkage clustering <https://en.wikipedia.org/wiki/Single-linkage_clustering>`_ with
given eclidean distance or cosine similarity threshold.

Supports both dense arrays (numpy) and sparse matrices (scipy) and visualisation via `matplotlib`.
The algorithm easily clusters 10000 points in 30 dimensions in under a second.

This module grew into a separate library strating as an data exploratory project
in `CZ.NIC labs <https://labs.nic.cz/en/>`_ to cluster captured honeypot records.

Install from `PyPI <https://pypi.python.org/pypi/silicon-clustering>`_
with ``pip install silicon-clustering``

Github: `CZ-NIC/silicon-clustering <https://github.com/CZ-NIC/silicon-clustering>`_

Docs: `silicon-clustering.readthedocs.io <http://silicon-clustering.readthedocs.io/>`_

Author: Tomáš Gavenčiak, gavento@ucw.cz

.. image:: https://travis-ci.org/CZ-NIC/silicon-clustering.svg?branch=master
    :target: https://travis-ci.org/CZ-NIC/silicon-clustering

Usage example
-------------

::

    import silicon, numpy
    # use a fixed seed to get the same data
    rnd = numpy.random.RandomState(42)
    # create some data
    data = rnd.uniform(-1.0, 1.0 ,(1000, 3))
    # create and run the clustering instance
    c = silicon.CosineClustering(data, rnd=rnd, sim_threshold=0.995)
    c.run()
    print(c.clusters())

    import matplotlib.pyplot as plt
    # plot the data overview
    c.plot()
    plt.show()
    # plot the largest cluster
    c.clusters_by_size()[0].plot()
    plt.show()
