#!/usr/bin/env python
#encoding=utf8

import mvndio
import arffio
from os import path


def load_dataset(filename, as_dataset=True):

    extension = path.splitext(filename)[1][1:]
    if extension == 'arff':
        dataset = arffio.loadarff(filename)
    elif extension == 'mvnd':
        dataset = mvndio.loadmvnd(filename)

    if as_dataset:
        # return as class Dataset
        return Dataset(*dataset)
    else:
        # return just tuple of values
        return dataset


def save_dataset(filename, dataset):
    mvndio.savemvnd(filename, dataset)


class Dataset:

    def __init__(self, relation, ls_atts, d_nom_vals, data):
        ## string name of relation
        self.relation = relation

        ## ls_attributes is list with all attribute names and types
        # eg. [('petal-width', 'num'), ('class', 'nom') ..]
        self.ls_atts = ls_atts

        ## dictionary containing lists with all possible values for each
        # nominal attribute. Key to this list is integer - position of
        # attribute in ls_atts
        self.d_nom_vals = d_nom_vals

        ## 2D numpy array with data
        self.data = data
