#!/usr/bin/env python
#encoding=utf8

from __future__ import division
import numpy as np
import string

from ..dataio import dataio_const

#TODO.... beware of bisectors!!!!
def eval_writer(out_stream, mlmvn, dataset, num_samples, settings=None):

    np.seterr(divide='ignore', invalid='ignore')
    num_outputs = len(dataset.outputs)
    na_predicted = mlmvn.count_outputs(dataset.data[-num_samples:, :-num_outputs])
    out_stream.write("Evaluating dataset: %s\n" % dataset.relation)
    all_metrics = {}

    for att_id in dataset.outputs:
        (att, att_type) = dataset.ls_atts[att_id]
        out_stream.write("\nFor attribute: %s\n" % att)
        column_tfm = find_transformation(att_id, dataset.tfms)

        if att_type == dataio_const.NUMERIC_ATT:
            out_stream.write("SUPPORT FOR EVALUATION OF NUMERIC ATTRIBUTES NOT "
                             "IMPLEMENTED YET SEE RMSE FOR LEARNING END\n")
        elif att_type == dataio_const.NOMINAL_ATT:

            na_true = np.array(
                column_tfm.decode(dataset.data[-num_samples:, att_id]).flatten(),
                dtype=int
            )
            # index of column between predicted columns
            out_col_id = att_id - (dataset.data.shape[1] - num_outputs)

            if column_tfm.get_name() == 'DiscreteBisectorTfm':
                sectors = column_tfm.get_sectors()
                na_pred = sectors.bisector_function(na_predicted[:, out_col_id])
                na_pred = np.array(column_tfm.decode(na_pred).flatten(), dtype=int)
            else:
                na_pred = np.array(column_tfm.decode(na_predicted[:, out_col_id]).flatten(), dtype=int)

            ls_labels = dataset.d_nom_vals[att_id]

            conf_matrix = confusion_matrix(na_true, na_pred, len(ls_labels))
            out_stream.write(pretty_confusion_matrix(conf_matrix, ls_labels))

            acc = overall_accuracy(conf_matrix)
            out_stream.write('\nAccuracy for \"%s\" on %d evaluation samples: %2.4f\n' % (att, num_samples, acc))

            prec = class_precisions(conf_matrix)
            out_stream.write('\nPrecisions for \"%s\" on %d evaluation samples:\n' % (att, num_samples))
            out_stream.write(
                pretty_metric_for_classes(prec, ls_labels, ' precision')
            )

            rec = class_recalls(conf_matrix)
            out_stream.write('\nRecalls for \"%s\" on %d evaluation samples:\n' % (att, num_samples))
            out_stream.write(
                pretty_metric_for_classes(rec, ls_labels, ' recall')
            )

            all_metrics[att_id] = [acc, prec, rec]

    return all_metrics

def write_all_metrics(all_metrics, out_stream, dataset, num_samples, settings=None):

    out_stream.write("Evaluating dataset: %s\n" % dataset.relation)
    for att_id in dataset.outputs:
        (att, att_type) = dataset.ls_atts[att_id]
        out_stream.write("\nFor attribute: %s\n" % att)

        if att_type == dataio_const.NUMERIC_ATT:
            out_stream.write("SUPPORT FOR EVALUATION OF NUMERIC ATTRIBUTES NOT "
                             "IMPLEMENTED YET\n")

        elif att_type == dataio_const.NOMINAL_ATT:
            ls_labels = dataset.d_nom_vals[att_id]
            acc, prec, rec = all_metrics[att_id]
            out_stream.write('\nAccuracy for \"%s\" on %d evaluation samples: %2.4f\n' % (att, num_samples, acc))

            out_stream.write('\nPrecisions for \"%s\" on %d evaluation samples:\n' % (att, num_samples))
            out_stream.write(
                pretty_metric_for_classes(prec, ls_labels, ' precision')
            )

            out_stream.write('\nRecalls for \"%s\" on %d evaluation samples:\n' % (att, num_samples))
            out_stream.write(
                pretty_metric_for_classes(rec, ls_labels, ' recall')
            )

def find_transformation(att_id, tfms):
    for (tfm, on_columns) in tfms:
        if att_id in on_columns:
            return tfm



############### METRICS #########################

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
        na_class_weights = np.sum(na_conf_matrix, axis=0, dtype=float)
        na_class_weights /= np.sum(na_class_weights)

    # compute arithmetic mean
    weighted_precisions = class_precisions(na_conf_matrix) * na_class_weights
    return np.sum(weighted_precisions)


def class_recalls(na_confusion_matrix):
    ''' Counts accuracy: (tpcX + tncX) / ncX for each cX.

    cX - class X (X is integer indexed from 0)\n
    tpcX - true positives for class X
    tncX - true negatives for class X
    ncX - number of samples of class X
    '''
    return na_confusion_matrix.diagonal() / np.sum(na_confusion_matrix, axis=1, dtype=float)


def class_precisions(na_confusion_matrix):
    ''' Counts precision: tpcX / npcX for each cX.

    cX - class X (X is integer indexed from 0)\n
    tpcX - true positives for class X
    npcX - number of samples predicted as class X
    '''
    return na_confusion_matrix.diagonal() / np.sum(na_confusion_matrix, axis=0, dtype=float)

######################## PRINTING ######################################

def pretty_metric_for_classes(na_metric, ls_labels, metric_name):
    result = ''
    idx = 0
    for label in ls_labels:
        metric_val = na_metric[idx]
        if np.isnan(metric_val):
            metric_val = "Not enough representants"
        else:
            metric_val = "%2.4f" % metric_val

        result += '%s for \"%s\": %s\n' % (metric_name, label, metric_val)
        idx += 1

    return result


def pretty_confusion_matrix(conf_matrix, ls_labels, max_col_width=12):
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

    return "\n".join(["".join([string.rjust(str(e), l + 2)
                               for e, l in zip(r, max_lens)])
                      for r in matrix_list]) + '\n'
