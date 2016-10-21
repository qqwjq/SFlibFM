# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import datetime
import numpy as np
import random
import sys
import math
from time import time

from libc.math cimport exp, log, pow

cimport numpy as np
cimport cython

np.import_array()

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INTEGER

# MODEL CONSTANTS
DEF REGRESSION = 0
DEF CLASSIFICATION = 1
DEF OPTIMAL = 0
DEF INVERSE_SCALING = 1

DEF LEARNING_RATE_MULTIPLIER = 5

cdef class CSRDataset:
    """An sklearn ``SequentialDataset`` backed by a scipy sparse CSR matrix. """

    cdef Py_ssize_t n_samples
    cdef int n_observations
    cdef int current_index
    cdef int stride
    cdef DOUBLE *X_data_ptr
    cdef INTEGER *X_indptr_ptr
    cdef INTEGER *X_indices_ptr
    cdef DOUBLE *Y_data_ptr
    cdef np.ndarray feature_indices
    cdef INTEGER *feature_indices_ptr
    cdef np.ndarray index
    cdef INTEGER *index_data_ptr
    cdef DOUBLE *sample_weight_data

    cdef np.ndarray X_data
    cdef np.ndarray X_indptr
    cdef np.ndarray X_indices
    cdef np.ndarray sample_weight
    cdef np.ndarray Y    

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
                   int *nnz, DOUBLE *y, DOUBLE *sample_weight):

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

    cdef void shuffle(self, seed):
        np.random.RandomState(seed).shuffle(self.index)

cdef class SGD_Parameters(object):

    cdef double w0
    cdef np.ndarray w
    cdef np.ndarray v
    cdef double reg_0
    cdef np.ndarray reg_w
    cdef np.ndarray reg_v
    
    cdef double learning_rate

    def __init__(self,
                  double w0,
                  np.ndarray[DOUBLE, ndim=1, mode='c'] w,
                  np.ndarray[DOUBLE, ndim=2, mode='c'] v,
                  double reg_0,
                  np.ndarray[DOUBLE, ndim=1, mode='c'] reg_w,
                  np.ndarray[DOUBLE, ndim=2, mode='c'] reg_v,                  
                  double learning_rate):

        self.w0 = w0
        self.w = w
        self.v = v

        self.reg_0 = reg_0
        self.reg_w = reg_w
        self.reg_v = reg_v
        
        self.learning_rate = learning_rate

cdef class FM_fast_parallel(object):
    """Factorization Machine fitted by minimizing a regularized empirical loss with adaptive SGD.
    
    Parameters
    ----------
    w : np.ndarray[DOUBLE, ndim=1, mode='c']
    v : ndarray[DOUBLE, ndim=2, mode='c'] 
    num_factors : int 
    num_features : int 
    n_iter : int 
    k0 : int 
    k1 : int
    k2 : int
    w0 : double 
    t : double 
    t0 : double 
    l : double 
    power_t : double 
    min_target : double 
    max_target : double 
    eta0 : double 
    learning_rate_schedule : int 
    shuffle_training : int 
    task : int 
    seed : int 
    reg_speedup : int 
    """

    cdef int num_factors
    cdef int num_features
    cdef int num_attributes
    cdef int num_processes
    cdef np.ndarray map_feature_attribute

    cdef int n_iter
    cdef int k0
    cdef int k1
    cdef int k2

    cdef DOUBLE t
    cdef DOUBLE t0
    cdef DOUBLE l
    cdef DOUBLE power_t
    cdef DOUBLE min_target
    cdef DOUBLE max_target
    cdef DOUBLE grad_threshold
    cdef DOUBLE v_threshold
    cdef np.ndarray sum
    cdef np.ndarray sum_sqr
    cdef int task
    cdef int learning_rate_schedule
    
    cdef int shuffle_training
    cdef int seed
    cdef int reg_speedup

    cdef SGD_Parameters params

    cdef DOUBLE grad_0
    cdef np.ndarray grad_w
    cdef np.ndarray grad_v

    # cdef np.ndarray penalty_total
    # cdef np.ndarray penalty_received
    
    def __init__(self,
                  np.ndarray[DOUBLE, ndim=1, mode='c'] w,
                  np.ndarray[DOUBLE, ndim=2, mode='c'] v,
                  int num_factors,
                  int num_features,
                  int num_attributes,
                  int num_processes,
                  np.ndarray[DOUBLE, ndim=1, mode='c'] map_feature_attribute,
                  int n_iter,
                  int k0,
                  int k1,
                  int k2,
                  double w0,
                  double t,
                  double t0,
                  double power_t,
                  double min_target,
                  double max_target,
                  double eta0,
                  int learning_rate_schedule,
                  int shuffle_training,
                  int task,
                  int seed,
                  int reg_speedup,
                  double grad_threshold,
                  double v_threshold):

        self.num_factors = num_factors
        self.num_features = num_features
        self.num_attributes = num_attributes
        self.num_processes = num_processes

        self.map_feature_attribute = map_feature_attribute
        self.n_iter = n_iter
        self.k0 = k0
        self.k1 = k1
        self.k2 = k2
        self.t = 1
        self.t0 = t0
        self.power_t = power_t
        self.min_target = min_target
        self.max_target = max_target
        self.sum = np.zeros(self.num_factors)
        self.sum_sqr = np.zeros(self.num_factors)
        self.task = task
        self.learning_rate_schedule = learning_rate_schedule
        self.shuffle_training = shuffle_training
        self.seed = seed
        self.reg_speedup = reg_speedup
        self.grad_threshold = grad_threshold
        self.v_threshold = v_threshold

        self.params = SGD_Parameters(w0, w, v, 0.0, np.zeros(self.num_attributes),
            np.zeros((self.num_factors, self.num_attributes)), eta0)

        self.grad_0 = 0.0
        self.grad_w = np.zeros(self.num_features)
        self.grad_v = np.zeros((self.num_factors, self.num_features))

        # self.penalty_total = np.zeros(self.num_features)
        # self.penalty_received = np.zeros(self.num_features)

    cdef _predict_current_parameter(self, DOUBLE * x_data_ptr, 
                           INTEGER * map_feature_position, 
                           int xnnz):

        # Helper variables
        cdef DOUBLE result = 0.0
        cdef int feature
        cdef unsigned int i = 0
        cdef unsigned int f = 0
        cdef DOUBLE d

        # map instance variables to local variables
        cdef DOUBLE w0 = self.params.w0
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] w = self.params.w
        cdef np.ndarray[DOUBLE, ndim=2, mode='c'] v = self.params.v
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] sum_ = np.zeros(self.num_factors)
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] sum_sqr_ = np.zeros(self.num_factors)

        if self.k0 > 0:
            result += w0
        if self.k1 > 0:
            for i in range(xnnz):
                feature = map_feature_position[i]
                result += w[feature] * x_data_ptr[i]
        if self.k2 > 0:
            for f in range(self.num_factors):
                sum_[f] = 0.0
                sum_sqr_[f] = 0.0
                for i in range(xnnz):
                    feature = map_feature_position[i]
                    d = v[f, feature] * x_data_ptr[i]
                    sum_[f] += d
                    sum_sqr_[f] += d*d
                result += 0.5 * (sum_[f] * sum_[f] - sum_sqr_[f])

        # pass sum to sgd_theta
        self.sum = sum_
        if math.isnan(result):
            return 0.0
        else:
            return result

    cdef _predict_future_theta(self, DOUBLE * x_data_ptr, 
                           INTEGER * map_feature_position, 
                           int xnnz):
        cdef DOUBLE result = 0.0
        cdef unsigned int i = 0
        cdef unsigned int f = 0
        cdef DOUBLE d
        cdef DOUBLE w_dash = 0.0
        cdef DOUBLE v_dash = 0.0

        # map instance variables to local variables
        cdef DOUBLE w0 = self.params.w0
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] w = self.params.w
        cdef np.ndarray[DOUBLE, ndim=2, mode='c'] v = self.params.v
        cdef DOUBLE grad_0 = self.grad_0
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] grad_w = self.grad_w
        cdef np.ndarray[DOUBLE, ndim=2, mode='c'] grad_v = self.grad_v
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] sum_ = np.zeros(self.num_factors)
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] sum_sqr_ = np.zeros(self.num_factors)
        cdef DOUBLE learning_rate = self.params.learning_rate
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] reg_w = self.params.reg_w
        cdef np.ndarray[DOUBLE, ndim=2, mode='c'] reg_v = self.params.reg_v

        if self.k0 > 0:
            result += w0 - learning_rate * grad_0

        if self.k1 > 0:
            for i in xrange(xnnz):
                feature = map_feature_position[i]
                attribute = self.map_feature_attribute[i]
                assert(feature < self.num_features)
                w_dash = w[feature] - learning_rate * (grad_w[feature] + 2 * reg_w[attribute] * w[feature])
                result += w_dash * x_data_ptr[i]

        if self.k2 > 0:
            for f in xrange(self.num_factors):
                sum_[f] = 0.0
                sum_sqr_[f] = 0.0
                for i in xrange(xnnz):
                    feature = map_feature_position[i]
                    attribute = self.map_feature_attribute[i]
                    v_dash = v[f,feature] - learning_rate * (grad_v[f,feature] + 2 * reg_v[f,attribute] * v[f,feature])
                    d = v_dash * x_data_ptr[i]
                    sum_[f] += d
                    sum_sqr_[f] += d*d
                result += 0.5 * (sum_[f]*sum_[f] - sum_sqr_[f])

        self.sum = sum_
        return result

    cdef _scale_prediction(self, DOUBLE p):

        if self.task == REGRESSION:
            p = min(self.max_target, p)
            p = max(self.min_target, p)
        elif self.task == CLASSIFICATION:
            p = min(1000000, p)
            p = max(-1000000, p)
            p = 1.0 / (1.0 + exp(-p))
        return p
    
    def _predict(self, CSRDataset dataset):
        
        # Helper access variables
        cdef unsigned int i = 0
        cdef Py_ssize_t n_samples = dataset.n_samples
        cdef DOUBLE * x_data_ptr = NULL
        cdef INTEGER * map_feature_position = NULL
        cdef int xnnz
        cdef DOUBLE sample_weight = 1.0
        cdef DOUBLE y_placeholder
        cdef DOUBLE p = 0.0
    
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] return_preds = np.zeros(n_samples)
    
        for i in range(n_samples):
            dataset.next(& x_data_ptr, & map_feature_position, & xnnz, & y_placeholder,
                         & sample_weight)
            p = self._scale_prediction(self._predict_current_parameter(x_data_ptr, map_feature_position, xnnz))
            return_preds[i] = p
        return return_preds
    
    cdef _sgd_theta_step(self, DOUBLE * x_data_ptr, 
                        INTEGER * map_feature_position, 
                        int xnnz,
                        DOUBLE y):

        cdef DOUBLE mult = 0.0
        cdef DOUBLE p
        cdef int feature
        cdef unsigned int i = 0
        cdef unsigned int f = 0
        cdef DOUBLE d
        cdef DOUBLE grad_0

        cdef DOUBLE w0 = self.params.w0
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] w = self.params.w
        cdef np.ndarray[DOUBLE, ndim=2, mode='c'] v = self.params.v

        cdef DOUBLE learning_rate = self.params.learning_rate
        cdef DOUBLE reg_0 = self.params.reg_0
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] reg_w = self.params.reg_w
        cdef np.ndarray[DOUBLE, ndim=2, mode='c'] reg_v = self.params.reg_v

        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] grad_w = self.grad_w
        cdef np.ndarray[DOUBLE, ndim=2, mode='c'] grad_v = self.grad_v
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] sum_ = self.sum
        
        p = self._predict_current_parameter(x_data_ptr, map_feature_position, xnnz)

        if self.task == REGRESSION:
            p = min(self.max_target, p)
            p = max(self.min_target, p)
            mult = 2 * (p - y);
        else:
            mult = y * ( (1.0 / (1.0+exp(-y*p))) - 1.0)
        
        # Set learning schedule
        if self.learning_rate_schedule == OPTIMAL:
            self.learning_rate = 1.0 / (self.t + self.t0)

        elif self.learning_rate_schedule == INVERSE_SCALING:
            self.params.learning_rate = self.params.learning_rate / pow(self.t, self.power_t)

        # Update global bias
        if self.k0 > 0:
            grad_0 = truncate(mult, self.grad_threshold)
            w0 -= learning_rate * (grad_0 + 2 * reg_0 * w0)

        # Update feature biases
        if self.k1 > 0:
            for i in range(xnnz):
                feature = map_feature_position[i]
                attribute = self.map_feature_attribute[i]
                grad_w[feature] = truncate(mult * x_data_ptr[i], self.grad_threshold)
                w[feature] -= learning_rate * (grad_w[feature]
                                   + 2 * reg_w[attribute] * w[feature])
                if abs(w[feature]) < (learning_rate * reg_w[attribute]) or math.isnan(w[feature]):
                    w[feature] = 0.0

        # Update feature factor vectors
        if self.k2 > 0:
            for f in range(self.num_factors):
                for i in range(xnnz):
                    feature = map_feature_position[i]
                    attribute = self.map_feature_attribute[i]
                    grad_v[f,feature] = truncate(mult * (x_data_ptr[i] * (sum_[f] - v[f,feature] * x_data_ptr[i])), self.grad_threshold)
                    # if abs(grad_v[f,feature]) > 1000:
                    #     print("-- Epoch %.3f::%.3f::%.3f::%.3f::%.3f::%.3f" % (grad_v[f,feature], (sum_[f] - v[f,feature] * x_data_ptr[i]),
                    #         mult, v[f,feature], sum_[f], x_data_ptr[i]))
                    v[f,feature] -= learning_rate * (grad_v[f,feature] + 2 * reg_v[f,attribute] * v[f,feature])
                    if abs(v[f,feature]) < (learning_rate * reg_v[f,attribute]) or math.isnan(v[f, feature]):
                        v[f,feature] = 0.0
                    v[f,feature] = truncate(v[f,feature], self.v_threshold)

        # Pass updated vars to other functions
        self.params.learning_rate = learning_rate
        self.params.w0 = w0
        self.params.w = w
        self.params.v = v

        self.grad_0 = grad_0
        self.grad_w = grad_w
        self.grad_v = grad_v

        self.t += 1

    cdef _sgd_lambda_step(self, DOUBLE * validation_x_data_ptr, 
                        INTEGER * validation_map_feature_position, 
                        int validation_xnnz,
                        DOUBLE validation_y,
                        DOUBLE reg_learning_rate_multiplier):

        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] sum_f = np.zeros(self.num_features)
        cdef DOUBLE sum_f_dash = 0.0
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] sum_f_dash_f = np.zeros(self.num_features)
        cdef DOUBLE p
        cdef DOUBLE grad_loss
        cdef int feature
        cdef unsigned int i
        cdef unsigned int f
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] lambda_w_grad = np.zeros(self.num_features)
        cdef DOUBLE lambda_v_grad = 0.0
        cdef DOUBLE v_dash = 0.0

        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] w = self.params.w
        cdef np.ndarray[DOUBLE, ndim=2, mode='c'] v = self.params.v

        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] grad_w = self.grad_w
        cdef np.ndarray[DOUBLE, ndim=2, mode='c'] grad_v = self.grad_v
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] sum_ = np.zeros(self.num_factors)
        cdef DOUBLE learning_rate = self.params.learning_rate * reg_learning_rate_multiplier
        
        cdef DOUBLE reg_0 = self.params.reg_0
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] reg_w = self.params.reg_w
        cdef np.ndarray[DOUBLE, ndim=2, mode='c'] reg_v = self.params.reg_v

        p = self._predict_future_theta(validation_x_data_ptr, validation_map_feature_position, validation_xnnz)

        if self.task == REGRESSION:
            p = min(self.max_target, p)
            p = max(self.min_target, p)
            grad_loss = 2 * (p - validation_y)
        else:
            grad_loss = validation_y * ( (1.0 / (1.0 + exp(-validation_y*p))) - 1.0)

        if self.k1 > 0:
            lambda_w_grad = np.zeros(self.num_features)
            for i in xrange(validation_xnnz):
                feature = validation_map_feature_position[i]
                attribute = self.map_feature_attribute[i]
                lambda_w_grad[attribute] += validation_x_data_ptr[i] * w[feature]

            for attr in range(self.num_attributes):
                lambda_w_grad[attr] = -2 * learning_rate * lambda_w_grad[attr]
                reg_w[attr] -= learning_rate * grad_loss * lambda_w_grad[attr]
                if math.isnan(reg_w[attr]):
                    reg_w[attr] = 0.0
                else:
                    reg_w[attr] = max(0.0, reg_w[attr])

        if self.k2 > 0:
            for f in xrange(self.num_factors):
                sum_f = np.zeros(self.num_features)
                sum_f_dash = 0.0
                sum_f_dash_f = np.zeros(self.num_features)
                lambda_w_grad = np.zeros(self.num_features)
                for i in xrange(validation_xnnz):
                    feature = validation_map_feature_position[i]
                    attribute = self.map_feature_attribute[i]
                    v_dash = v[f,feature] - learning_rate * (grad_v[f,feature] + 2 * reg_v[f,attribute] * v[f,feature])
                    sum_f_dash += v_dash * validation_x_data_ptr[i]
                    sum_f[attribute] += v[f,feature] * validation_x_data_ptr[i]
                    sum_f_dash_f[attribute] += v_dash * validation_x_data_ptr[i] * v[f,feature] * validation_x_data_ptr[i]
                for attr in range(self.num_attributes):
                    lambda_v_grad = -2 * learning_rate * (sum_f_dash * sum_f[attr] - sum_f_dash_f[attr])
                    reg_v[f,attr] -= learning_rate * grad_loss * lambda_v_grad
                    if math.isnan(reg_v[f,attr]):
                        reg_v[f,attr] = 0.0
                    else:
                        reg_v[f,attr] = max(0.0, reg_v[f,attr])

        self.params.reg_w = reg_w
        self.params.reg_v = reg_v

    def get_w0(self):
        return self.params.w0

    def get_w(self):
        return self.params.w

    def get_v(self):
        return self.params.v

    def get_reg_w(self):
        return self.params.reg_w

    def get_reg_v(self):
        return self.params.reg_v

    def fit(self, CSRDataset dataset, CSRDataset validation_dataset):
    
        # get the data information into easy vars
        cdef Py_ssize_t n_samples = dataset.n_samples
        cdef Py_ssize_t n_validation_samples = validation_dataset.n_samples
        
        cdef DOUBLE * x_data_ptr = NULL
        cdef INTEGER * map_feature_position = NULL
    
        cdef DOUBLE * validation_x_data_ptr = NULL
        cdef INTEGER * validation_map_feature_position = NULL
    
        # helper variables
        cdef int xnnz
        cdef DOUBLE y = 0.0
        cdef DOUBLE validation_y = 0.0
        cdef int validation_xnnz        
        cdef unsigned int epoch = 0
        cdef unsigned int i = 0
    
        cdef DOUBLE sample_weight = 1.0
        cdef DOUBLE validation_sample_weight = 1.0

        # with nogil:
        for epoch in range(self.n_iter):
            
            print("-- Epoch %d" % (epoch + 1))
            
            if self.shuffle_training:
                dataset.shuffle(self.seed)

            reg_learning_rate_multiplier = 1.0
            if self.reg_speedup > 0:
                reg_learning_rate_multiplier = float(LEARNING_RATE_MULTIPLIER)

            # self.penalty_total = np.zeros(self.num_features)
            # self.penalty_received = np.zeros(self.num_features)

            for i in range(n_samples):

                # print "Training %d: Before theta :::%.5f:%.5f:%.5f:%.5f" % (i, 
                #     np.max(self.params.reg_w), np.max(self.params.reg_v), np.max(np.abs(self.params.w)), 
                #     np.max(np.abs(self.params.v)))

                dataset.next( & x_data_ptr, & map_feature_position, & xnnz, & y,
                             & sample_weight)
                self._sgd_theta_step(x_data_ptr, map_feature_position, xnnz, y)

                # print "Training %d: After theta :::%.5f:%.5f:%.5f:%.5f" % (i, 
                #     np.max(self.params.reg_w), np.max(self.params.reg_v), np.max(np.abs(self.params.w)), 
                #     np.max(np.abs(self.params.v)))

                if math.isnan(np.max(np.abs(self.params.w))):
                    return 0

                if epoch > 0:
                    if(self.reg_speedup == 0) or (self.reg_speedup > 0 and i % LEARNING_RATE_MULTIPLIER == 0):
                        validation_dataset.next( & validation_x_data_ptr, & validation_map_feature_position,
                                                 & validation_xnnz, & validation_y, 
                                                 & validation_sample_weight)

                        # print "Training %d: Before lambda :::%.5f:%.5f:%.5f:%.5f" % (i, 
                            # np.max(self.params.reg_w), np.max(self.params.reg_v), np.max(np.abs(self.params.w)), 
                            # np.max(np.abs(self.params.v)))
                        self._sgd_lambda_step(validation_x_data_ptr, validation_map_feature_position,
                                              validation_xnnz, validation_y, reg_learning_rate_multiplier)

cdef inline double max(double a, double b):
    return a if a >= b else b

cdef inline double min(double a, double b):
    return a if a <= b else b

cdef double sign(double a):
    if a > 0.0:
        return 1.0
    if a < 0.0:
        return -1.0
    return 0.0

cdef double truncate(double a, double threshold):
    if a > threshold:
        return threshold
    if a < (-1.0 * threshold):
        return (-1.0 * threshold)
    return a

cdef double abs(double a):
    if a < 0.0:
        return (-1.0 * a)
    return a

cdef _log_loss(DOUBLE p, DOUBLE y):
    cdef DOUBLE z

    z = p * y
    # approximately equal and saves the computation of the log
    if z > 18:
        return exp(-z)
    if z < -18:
        return -z
    return log(1.0 + exp(-z))

cdef _squared_loss(DOUBLE p, DOUBLE y):
    return 0.5 * (p - y) * (p - y)
