# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# Licence: BSD 3 clause

cimport cython
from libc.limits cimport INT_MAX
cimport numpy as np
import numpy as np

np.import_array()

cdef class CSRDataset:
    """An sklearn ``SequentialDataset`` backed by a scipy sparse CSR matrix. """

    def __cinit__(self, np.ndarray[DOUBLE, ndim=1, mode='c'] X_data,
                  np.ndarray[INTEGER, ndim=1, mode='c'] X_indptr,
                  np.ndarray[INTEGER, ndim=1, mode='c'] X_indices,
                  np.ndarray[DOUBLE, ndim=1, mode='c'] Y,
                  np.ndarray[DOUBLE, ndim=1, mode='c'] sample_weight):
        """Dataset backed by a scipy sparse CSR matrix.

        The feature indices of ``x`` are given by map_feature_position[0:nnz].
        The corresponding feature values are given by
        x_data_ptr[0:nnz].

        Parameters
        ----------
        X_data : ndarray, dtype=np.float64, ndim=1, mode='c'
            The data array of the CSR matrix; a one-dimensional c-continuous
            numpy array of dtype np.float64.
        X_indptr : ndarray, dtype=np.int32, ndim=1, mode='c'
            The index pointer array of the CSR matrix; a one-dimensional
            c-continuous numpy array of dtype np.int32.
        X_indices : ndarray, dtype=np.int32, ndim=1, mode='c'
            The column indices array of the CSR matrix; a one-dimensional
            c-continuous numpy array of dtype np.int32.
        Y : ndarray, dtype=np.float64, ndim=1, mode='c'
            The target values; a one-dimensional c-continuous numpy array of
            dtype np.float64.
        sample_weights : ndarray, dtype=np.float64, ndim=1, mode='c'
            The weight of each sample; a one-dimensional c-continuous numpy
            array of dtype np.float64.
        """

        self.X_data = X_data
        self.X_indptr = X_indptr
        self.X_indices = X_indices
        self.sample_weight = sample_weight
        self.Y = Y
        
        self.n_samples = Y.shape[0]
        self.n_observations = X_data.shape[0]
        self.current_index = -1
        self.X_data_ptr = <DOUBLE *>X_data.data
        self.X_indptr_ptr = <INTEGER *>X_indptr.data
        self.X_indices_ptr = <INTEGER *>X_indices.data
        self.Y_data_ptr = <DOUBLE *>Y.data
        self.sample_weight_data = <DOUBLE *> sample_weight.data

        # Use index array for fast shuffling
        cdef np.ndarray[INTEGER, ndim=1,
                        mode='c'] index = np.arange(0, self.n_samples, dtype=np.int32)
        self.index = index
        self.index_data_ptr = <INTEGER *> index.data

        # print("-- Initialize: %d:%d:%d:%d::%d:%d:%d:%d " % (
        #     self.X_indptr_ptr[0], self.X_indptr_ptr[1],
        #     self.X_indptr_ptr[2], self.X_indptr_ptr[3],
        #     self.X_indptr_ptr[4], self.X_indptr_ptr[5],
        #     self.X_indptr_ptr[6], self.X_indptr_ptr[7]))

    cdef void next(self, DOUBLE **x_data_ptr, INTEGER **map_feature_position,
                   int *nnz, DOUBLE *y, DOUBLE *sample_weight) nogil:

        cdef int current_index = self.current_index
        if current_index >= (self.n_samples - 1):
            current_index = -1
        current_index += 1
        cdef int sample_idx = self.index_data_ptr[current_index]
        cdef int offset = self.X_indptr_ptr[sample_idx]
        y[0] = self.Y_data_ptr[sample_idx]
        x_data_ptr[0] = self.X_data_ptr + offset
        map_feature_position[0] = self.X_indices_ptr + offset
        nnz[0] = self.X_indptr_ptr[sample_idx + 1] - offset

        sample_weight[0] = self.sample_weight_data[sample_idx]
        self.current_index = current_index

    cdef void shuffle(self, int seed):
        np.random.RandomState(seed).shuffle(self.index)
