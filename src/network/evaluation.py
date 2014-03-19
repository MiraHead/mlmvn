#!/usr/bin/env python
#encoding=utf8

from __future__ import division
import numpy as np
import string


def confusion_matrix(na_true, na_pred, num_classes):
    ''' Returns confusion matrix m, where m[i,j] is
    how many objects of class i were classified as class j.

    @param na_true Numpy array of integer class indices for true
                   classes of data. Class indexing starts at 0.
    @param na_true Numpy array of integer class indices for predicted
                   classes of data. Class indexing starts at 0.
    @param num_classes Number of classes

    \bExample:
@verbatim
    true_vals      = np.array([0,0,0,0,1,1,1,1,2,2,2,2])
    predicted_vals = np.array([0,1,0,2,1,1,2,1,2,0,2,0])
    matrix = confusion_matrix(true_vals, predicted_vals, 3)
@endverbatim
    '''
    n = num_classes
    return np.bincount(n*na_true + na_pred, minlength=n*n).reshape(n, n)


def overall_accuracy(na_conf_matrix):
    ''' Counts overall accuracy, given confusion matrix:
    accuracy = correctly_classified / num_samples

    @param na_conf_matrix Numpy array with confusion matrix
    '''
    return np.sum(na_conf_matrix.diagonal()) / np.sum(na_conf_matrix)


def overall_weighted_precision(na_conf_matrix, na_class_weights=None):
    ''' Counts weighted precision mean... if not specified, weights are
    computed as relative frequency for each class.

    @param na_conf_matrix Numpy array with confusion matrix.
    '''
    # if not specified... count weights
    if na_class_weights is None:
        na_class_weights = np.sum(na_conf_matrix, axis=0)
        na_class_weights /= np.sum(na_class_weights)

    # compute arithmetic mean
    weighted_precisions = class_precisions(na_conf_matrix) * na_class_weights
    return weighted_precisions / na_conf_matrix.shape[0]


def class_recalls(na_confusion_matrix):
    ''' Counts accuracy: (tpcX + tncX) / ncX for each cX.

    cX - class X (X is integer indexed from 0)\n
    tpcX - true positives for class X
    tncX - true negatives for class X
    ncX - number of samples of class X
    '''
    return na_confusion_matrix.diagonal() / np.sum(na_confusion_matrix, axis=1)


def class_precisions(na_confusion_matrix):
    ''' Counts precision: tpcX / npcX for each cX.

    cX - class X (X is integer indexed from 0)\n
    tpcX - true positives for class X
    npcX - number of samples predicted as class X
    '''
    return na_confusion_matrix.diagonal() / np.sum(na_confusion_matrix, axis=0)

######################## PRINTING ######################################


def pprint_confusion_matrix(conf_matrix, ls_labels, max_col_width=12):
    matrix_list = conf_matrix.tolist()
    # insert column labels ... as a new list we don't
    matrix_list.insert(0, list(ls_labels))
    # insert row labels
    matrix_list[0].insert(0, "True(v)/Predicted(>)")
    i = 1
    for label in ls_labels:
        matrix_list[i].insert(0, label)
        i += 1

    max_lens = [max([len(str(r[i])) for r in matrix_list])
                for i in range(len(matrix_list[0]))]

    print "\n".join(["".join([string.rjust(str(e), l + 2)
                              for e, l in zip(r, max_lens)])
                     for r in matrix_list])
