# -*- coding: utf-8 -*-

import datetime
import io
import logging
import sys
import numpy as np
import scipy.sparse
from six import raise_from

def importPlt():
    "Import and return (matplotlib, pyplot) or raise an exception"

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        return mpl, plt
    except ImportError as exc:
        raise_from(ImportError("Matplotlib nad Pyplot required for plot functions of "
                               "silicon-clustering"), exc)


def toArray(a):
    "Converts a sparse matrix to ndarray, ndarrays pass through, other types raise TypeError."
    if isinstance(a, np.ndarray):
        return a
    elif scipy.sparse.isspmatrix(a):
        return a.toarray()
    else:
        raise TypeError("_toarray() needs scipy.sparse.csr.csr_matrix or nupmy.ndarray")


def format_historgram(values, bins_n=10, top_n=10, indent=''):
    """Format a histogram of non-negative values.

    Returns a `bins_n` bin histogram of `values` with bin borders
    scaled logrithmically, writing `top_n` values explicitely.
    The lines are indented with `indent` or returned separately when
    `indent=None`.
    """

    values = sorted(values)
    s = []
    if top_n:
        topvals = values[-top_n:]

    if values:
        bins = [0] * bins_n
        maxval = values[-1] + (1 if isinstance(values[0], int) else 0)
        upper_bounds = [type(values[0])(maxval ** ((i + 1) / bins_n) + 0.01) for i in range(bins_n)]
        for v in values:
            for b in range(bins_n):
                if upper_bounds[b] > v:
                    bins[b] += 1
                    break
        for b in range(bins_n):
            low = upper_bounds[b - 1] if b > 0 else 0
            rng = "[%s, %s)" % (low, upper_bounds[b])
            if (low < upper_bounds[b]):
                s.append("%-14s  %d" % (rng, bins[b]))

    if top_n:
        s.append("Top:            %s" % (' '.join([str(v) for v in topvals])))

    if indent is None:
        return s
    else:
        return indent + ('\n' + indent).join(s)


class ProgressLogger(object):
    """
    A helper to show operation progress and ETA.

    Ensures that at most cca `max_messages` (default 50) are written but not
    more often than every `min_delay_sec` (default 5) seconds. Estimates ETA
    after 5% done and at least 5s. When no upper limit is given, just reports
    the number. Use `.update(increment)` or `.set(value)` to update the
    progress. If you call `.done()` afterwards, reports the total time and
    speed. All messages start with `msg` (e.g. to identify the operation).

    Writes to a given logger or a given file (prepends current time), writes
    to stderr for `logger=None`.
    """

    FIRST_FULL = 3
    ETA_MIN_PERCENT = 5
    ETA_MIN_TIME = datetime.timedelta(0, 5)

    def __init__(self, logger=None, total=None, msg="Operation progress", min_delay_sec=5.0, max_messages=50):

        self.logger = logger or sys.stderr
        self.logger = sys.stderr if logger is None or logger == 'stderr' else logger
        self.total = total
        self.msg = msg
        self.delay = datetime.timedelta(0, min_delay_sec)
        self.max_messages = max_messages

        self.value = 0
        self.messages = 0
        self.is_done = False
        self.start_time = datetime.datetime.now()
        self.last_time = self.start_time
        self.last_value = self.value

    def update(self, added):
        "Increment the progress by the given amount."
        self.set(self.value + added)

    def set(self, value):
        "Set the progress to the given amount."
        if self.is_done:
            raise Exception("Updating ProgressBar after done")
        self.value = value
        self.maybe_report()

    def write(self, msg):
        if isinstance(self.logger, logging.Logger):
            self.logger.info(msg)
        elif isinstance(self.logger, io.IOBase):
            self.logger.write("[%s] %s\n" % (datetime.datetime.now(), msg))
            self.logger.flush()
        else:
            raise TypeError('File or Logger required as logger')

    def maybe_report(self):
        "Report the state if conditions are met. Internal."
        if datetime.datetime.now() < self.last_time + self.delay:
            return
        if self.total and self.messages > self.FIRST_FULL and self.value < self.last_value + (self.total / self.max_messages):
            return
        self.report()

    def eta_str(self, prefix="ETA "):
        "Compute the ETA or None if not known (yet). Internal."
        if (self.total is None or self.value <= self.total * (self.ETA_MIN_PERCENT / 100.) or
                datetime.datetime.now() < self.start_time + self.ETA_MIN_TIME):
            return None
        t_eta = self.start_time + ((datetime.datetime.now() - self.start_time) * (self.total / self.value))
        return prefix + str(t_eta)

    def report(self):
        if self.total is None:
            self.write("%s %s of ?" % (self.msg, self.value))
        else:
            eta = self.eta_str()
            self.write("%s %s of %s (%2.2f%%) %s" % (
                self.msg, self.value, self.total, 100 * self.value / self.total, eta or ""))

        self.last_time = datetime.datetime.now()
        self.last_value = self.value
        self.messages += 1

    def done(self):
        if self.is_done:
            return  # Do not report twice by mistake (e.g. done() call and context exit)

        dt = (datetime.datetime.now() - self.start_time).total_seconds()
        if dt > 0 and self.value > 1e-6:
            per_sec = self.value / dt
            timing = " (%.2f / s)" % (per_sec, )
        else:
            timing = ""
        self.write("%s done at %s, %.2f s%s" % (self.msg, self.value, dt, timing))
        self.is_done = True

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        if not exc_info[0]:
            self.done()
