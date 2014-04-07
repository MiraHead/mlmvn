#!/usr/bin/env python
#encoding=utf8

""" Constants which can be used across all
data loading/saving modules to avoid cyclic dependencies"""

NUMERIC_ATT = 'num'
NOMINAL_ATT = 'nom'


class DataIOError(IOError):
    """ Error while loading/parsing/saving dataset """
    pass
