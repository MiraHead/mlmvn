#!/usr/bin/env python
#encoding=utf8

import cPickle as pickle
import numpy as np
import gzip

from dataio_const import DataIOError
from ..network.transformations import Transformation


class MvndIOError(DataIOError):
    pass


def loadmvnd(filename):
    try:
        in_file = gzip.open(filename, 'rb')
        return _loadmvnd(in_file)
    finally:
        in_file.close()


def savemvnd(filename, dataset):
    try:
        if dataset.tfms is None:
            raise DataIOError("Dataset not transformed/preprocessed yet!!!")
        out_file = gzip.open(filename, 'wb')
        return _savemvnd(out_file, dataset)
    finally:
        out_file.close()


def _savemvnd(out_file, dataset):
    try:
        pickle.dump(dataset.relation, out_file)
        pickle.dump(dataset.ls_atts, out_file)
        pickle.dump(dataset.d_nom_vals, out_file)
        pickle.dump(dataset.outputs, out_file)

        pickle.dump(dataset.data.shape, out_file)
        # because whole dataset might be too big
        # to save it with pickle
        for row in dataset.data:
            pickle.dump(row, out_file)

        save_transformations(out_file, dataset.tfms)
    except (pickle.PickleError, MvndIOError) as e:
        raise MvndIOError("Error while serializing dataset:\n" + str(e))


def _loadmvnd(in_file):
    try:
        relation = pickle.load(in_file)
        ls_atts = pickle.load(in_file)
        d_nom_vals = pickle.load(in_file)
        outputs = pickle.load(in_file)

        data_shape = pickle.load(in_file)
        data = np.empty(data_shape, dtype=complex)
        for row_index in range(data_shape[0]):
            data[row_index, :] = pickle.load(in_file)

        tfms = load_transformations(in_file)

        return relation, ls_atts, d_nom_vals, data, tfms, outputs
    except (pickle.PickleError, MvndIOError) as e:
        raise MvndIOError("Error while deserializing dataset:\n" + str(e))


def save_transformations(out_file, tfms):
    if tfms is None:
        raise MvndIOError("Transformations passed as argument are None")

    if len(tfms) < 1:
        raise MvndIOError("No transformations to be saved")
    pickle.dump(len(tfms), out_file)
    for (tfm, on_columns) in tfms:
        tfm.save_to_file(out_file)
        pickle.dump(on_columns, out_file)


def load_transformations(in_file):
    tfms = []
    num_tfms = pickle.load(in_file)
    if num_tfms < 1:
        raise MvndIOError("No transformation can be loaded")

    for i in range(num_tfms):
        tfms.append(
            (Transformation.create_from_file(in_file), pickle.load(in_file))
        )

    return tfms
