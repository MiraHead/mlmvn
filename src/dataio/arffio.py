#! /usr/bin/env python
'''
Arff loader for categorical and numerical attributes, based
on scipy.io.arff.arffloader With minor changes for this
project (eg. categorical attributes are mapped onto integers
and whole dataset is returned as numpy array of floats)

If any unsupported data types appear or if arff is malformed,
ParseArffError with info about error is raised.

@author Miroslav Hlavacek <mira.hlavackuj@gmail.com>
'''
from __future__ import division, absolute_import

from functools import partial
import numpy as np
from ..dataio.dataio_const import DataIOError
from ..dataio.dataio_const import NUMERIC_ATT
from ..dataio.dataio_const import NOMINAL_ATT


class ParseArffError(DataIOError):
    """ Error while parsing arff file - either
    malformed arff or unsupported arff functionality
    """
    pass


def loadarff(f):
    """Read an arff file.

    Retrieves name of relation, attribute names and types, possible values
    of nominal attributes and data. The data is returned as a numpy array of
    floats.\n

    It can read files with numeric and nominal attributes. All nominal
    attribute values are converted to integers (but stored as floats -
    because of numpy).\n

    Not implemented functionality:\n
    * date type attributes\n
    * string type attributes\n
    * relational type attributes\n
    * sparse files reading\n
    * missing values handling\n

    @param f : file-like or str - object to read from, or filename to open.
    @returns Tuple (relation, ls_attributes, d_nominal_values, data)
             where:\n
             \b relation is string name of relation in arff\n
             \b ls_attributes is list with all attribute names\n
             \b d_nominal_values is dictionary containing lists with all
             possible values for each nominal attribute. Key to this
             list is integer - position of attribute in ls_attributes.
             \b data is numpy array of float type, where shape is
             (n_samples, n_attributes)
    @throws ParseArffError This is raised if the given file is not
                           ARFF-formatted or some values are missing
                           or some values are of bad type or if some
                           data type is unsupported.
    """
    if hasattr(f, 'read'):
        ofile = f
    else:
        ofile = open(f, 'rt')
    try:
        return _loadarff(ofile)
    finally:
        if ofile is not f:  # only close what we opened
            ofile.close()


def _loadarff(in_file):
    # Parse the header file
    try:
        relation, ls_atts, d_nom_vals = read_header(in_file)
    except ValueError as e:
        raise ParseArffError("Error while parsing header, error was: "
                             + str(e))

    #prepare convertors and parse data

    convertors = []
    idx = 0
    for name, att_type in ls_atts:
        if att_type == NUMERIC_ATT:
            convertors.append(safe_float)
        elif att_type == NOMINAL_ATT:
            convertors.append(partial(safe_nominal, ls_values=d_nom_vals[idx]))
        idx += 1

    n_columns = len(convertors)

    def generator(row_iter):
        # skip comments and empty lines
        raw = row_iter.next()
        while len(raw.strip()) == 0 or raw[0] == '%':
            raw = row_iter.next()

        try:
            # retrieve delimiter of data from first data field
            delim = get_delim(raw)
            rows = raw.split(delim)

            if len(rows) != n_columns:
                raise ParseArffError('Wrong number of attributes on line: '
                                     + raw.strip())
            # 'compiling' the range since it does not change
            elems = list(range(n_columns))

            for i in elems:
                yield convertors[i](rows[i])
        except ValueError as e:
            raise ParseArffError('Error while parsing data: "%s" on line "%s"'
                                 % (str(e), raw.strip()))

        for raw in row_iter:

            rows = raw.split(delim)
            while not rows or rows[0][0] == '%':
                raw = row_iter.next()
                rows = raw.split(delim)

            if len(rows) != n_columns:
                raise ParseArffError('Wrong number of attributes on line: '
                                     + raw)

            try:
                for i in elems:
                    yield convertors[i](rows[i])
            except ValueError as e:
                raise ParseArffError('Type error or missing value while '
                                     'parsing data: "%s" on line:"%s"'
                                     % (str(e), raw))

    gen = generator(in_file)
    data = np.fromiter(gen, complex)
    # reshape array appropriately
    data = data.reshape(data.shape[0] / n_columns, n_columns)

    return relation, ls_atts, d_nom_vals, data


def read_header(in_file):
    """Read the header of the iterable in_file.

    Parse all attribute names, types and store
    possible values for any encountered nominal attribute.

    @param in_file File opened for textual reading
    @returns Tuple (relation, ls_attributes, d_nominal_values)
             where:\n
             \b relation is string name of relation in arff\n
             \b ls_attributes is list with all attribute names\n
             \b d_nominal_values is dictionary containing lists with all
             possible values for each nominal attribute. Key to this
             list is integer - position of attribute in ls_attributes.
    """

    # Header is everything up to DATA attribute
    relation = "Unknown relation"
    ls_attributes = []
    d_nominal_vals = {}
    num_attributes = 0

    keyword = ''
    while keyword != '@data':

        line = next(in_file)
        chunks = line.rstrip('\n').split()

        # ignore blank lines and commments
        if not chunks or chunks[0][0] != '@':
            continue

        try:
            keyword = chunks[0].lower()
            if keyword == '@attribute':
                name = chunks[1]
                att_type = parse_type(chunks[2])
                val_names = None

                if att_type == NOMINAL_ATT:
                    val_names = chunks[2].strip('{}').split(',')

                ls_attributes.append((name, att_type))
                if not val_names is None:
                    d_nominal_vals[num_attributes] = val_names
                num_attributes += 1

            elif keyword == '@relation':
                relation = chunks[1]
            elif keyword != '@data':
                raise ParseArffError("Error parsing line %s" % line)
        except KeyError as e:
            raise ParseArffError('Malformed arff attribute: %s on line %s '
                                 % (str(e), line))

    return relation, ls_attributes, d_nominal_vals


def parse_type(attrtype):
    """Given an arff attribute type description returns
    whether is attribute nominal or numeric, for other
    data types, ParseArffError is raised.

    @param String representing value of attribute
    @return String with either for given type defined in dataio...
            either NUMERIC_ATT or NOMINAL_ATT
    @throw ParseArffError If the type is unknown or unsupported
    """
    atype = attrtype.lower().strip()
    if atype[0] == '{':
        return NOMINAL_ATT
    elif atype[:len('real')] == 'real':
        return NUMERIC_ATT
    elif atype[:len('integer')] == 'integer':
        return NUMERIC_ATT
    elif atype[:len('numeric')] == 'numeric':
        return NUMERIC_ATT
    else:
        raise ParseArffError("Unknown or unsupported attribute %s" % atype)


def safe_float(data):
    """ float convertor """
    if data.strip()[0] == '{':
        raise ValueError("This looks like a sparse ARFF: not supported yet")
    return np.float(data)


def safe_nominal(data, ls_values):
    """ nominal convertor """
    svalue = data.strip()
    if svalue[0] == '{':
        raise ValueError("This looks like a sparse ARFF: not supported yet")
    if svalue in ls_values:
        return ls_values.index(svalue)
    else:
        raise ValueError('Not defined value of nominal attribute')


def get_delim(line):
    """Given a string representing a line of data, check whether the
    delimiter is ',' or space.
    """

    if ',' in line:
        return ','
    if ' ' in line:
        return ' '
    raise ValueError("delimiter not understood: " + line)
