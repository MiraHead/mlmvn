#!/usr/bin/env python
#encoding=utf8

'''
Module describing various transformations used while preparing input data for
multilayer feedforward neural network with continuous neurons

Every new transformation must be located in this file!!! - due to factory
creation of transformations in Transformation.create()
@verbatim
tfm = apply(getattr(sys.modules[__name__], tfm_name))
@endverbatim

@author Miroslav Hlavacek <mira.hlavackuj@gmail.com>
'''

from __future__ import division
import sys
import numpy as np
from numpy import pi
import cPickle as pickle
# project modules
import sectors


class Transformation(object):

    @staticmethod
    def create(tfm_name, *args, **kwargs):
        ''' Factory method for creating transformations

        @param tfm_name String with name of class derived from Transformation
                        which should be created. If tfm_name is not in this
                        module, ValueError is raised.
        @param **kwargs Named arguments - transformation's parameters.\n
                        Mandatory parameters are designated for "setter
                        injection" instead of constructor.\n
                        If some mandatory parameter is missing, set_state()
                        should raise KeyError, which in turn causes that
                        ValueError is raised by this function.
        @param *args If kwargs is empty and args are not empty, args[0] must
                     contain dictionary with known parameters for
                     transformation.
        '''
        if not kwargs and args:
            kwargs = args[0]

        try:
            tfm = apply(getattr(sys.modules[__name__], tfm_name))
            # setter injection - sets all parameters that are known
            # if some mandatory arguments are missing set_state
            # should raise KeyError
            tfm.set_state(kwargs)

            return tfm

        except AttributeError:
            raise ValueError('Not supported transformation: ' + tfm_name)
        except KeyError as e:
            raise ValueError('Transformation %s can not be created without'
                             ' parameter %s' % (tfm_name, str(e.args[0])))

    @staticmethod
    def create_from_file(in_file):
        ''' Creates transformation from file.

        @param in_file File opened for binary reading.
        '''
        tfm_name = pickle.load(in_file)
        kwargs = pickle.load(in_file)
        return Transformation.create(tfm_name, kwargs)

    def save_to_file(self, out_file):
        ''' Saves transformation to file.

        Class representing transformation must provide method get_state.

        @param out_file File opened for binary writing.
        '''
        try:
            pickle.dump(self.__class__.__name__, out_file)
            pickle.dump(self.get_state(), out_file)
        except AttributeError as e:
            raise IOError("Transformation can't be saved uninitialized"
                          " - some parameter was not set!: " + str(e))

    def encode(self, na_input, count_params=True):
        ''' Encodes given data.

        Some transformations need to compute some statistic data parameters
        (mean, deviation and so on). If these parameters should be computed
        on given Numpy array na_input, count_params should be set to True.

        Raises AttributeError if some of its parameters were not set/computed
        and may raise ValueError in case of some other problems with encoding.

        @param na_input Input to be encoded.
        @param count_metrics True if parameters for transformation (which are
                             not specified by user) should be computed on
                             columns of na_input dataset.
        '''
        raise NotImplementedError("encode() not implemented in "
                                  + self.__class__.__name__)

    def decode(self, na_output):
        ''' Decodes given data by the transformation.

        If some of the parameters for transformation were not set,
        raises AttributeError.

        @param na_output Data to be decoded
        '''
        raise NotImplementedError("decode() not implemented in "
                                  + self.__class__.__name__)

    def get_state(self):
        ''' Returns all the attributes for given transformation.
        If some attributes were not yet computed, raises AttributeError

        @return Dictionary with class attributes. This dictionary can be
                used to create instance of class using Transformation.create
        '''
        raise NotImplementedError("get_state() not implemented in "
                                  + self.__class__.__name__)

    def set_state(self, kwargs):
        ''' Setter injection of class attributes.

        Set state tries to set all the class attributes which are provided
        in kwargs.

        In case that transformation is newly created, set_state()
        must get attributes/transformation parameters which can't be computed
        from data using encode(..., count_params=True). If such an attribute
        is missing, set_state() throws KeyError
        @param kwargs Dictionary of transformation parameters.
        '''
        raise NotImplementedError("get_state() not implemented in "
                                  + self.__class__.__name__)

    def get_name(self):
        return self.__class__.__name__


class DiscreteBisectorTfm(Transformation):

    def encode(self, na_input, count_params=True):
        ''' Encodes discrete values (sector indices) to complex numbers -
        bisectors of sectors specified by those indices.

        @param na_input Numpy array of integers - class indices from
                        enum set {0,1,..,<number_of_values - 1>}
        @return Numpy array of complex numbers.
        @see Transformation.encode
        '''
        return self.__to_bisectors(na_input)

    def decode(self, na_output):
        ''' Inverse function to DiscreteBisectorTfm.encode
        @see Transformation.decode
        '''
        return self.sects.get_idx_by_bisector(na_output)

    def get_state(self):
        ''' @see Transformation.get_state '''
        return {'ls_sector_phases': self.sects.get_phases()}

    def set_state(self, kwargs):
        '''
        Either 'ls_sector_phases' or 'number_of_values' must be specified
        while setting state of this class/creating this class.

        @see Transformation.get_state
        '''
        if 'ls_sector_phases' in kwargs:
            ## Sectors used to encode/decode data
            self.sects = sectors.Sectors(
                ls_phases=kwargs['ls_sector_phases']
            )
        elif 'number_of_values' in kwargs:
            self.sects = sectors.Sectors(
                num_sectors=kwargs['number_of_values']
            )
        else:
            # no one of mandatory arguments is present
            raise KeyError("ls_sector_phases or number_of_values")

    def __to_bisectors(self, na_input):
        ''' Returns bisectors of sectors in complex domain, for given
        sector indices.

        @param na_input Numpy matrix of integer inidices of sectors.
        @return Numpy matrix of complex bisectors.
        '''
        max_value = np.max(na_input)
        n_sects = self.sects.get_number_of_sectors()

        if max_value >= n_sects:
            raise ValueError("Can not encode %d discrete values having only"
                             "%d sectors" % (max_value + 1, n_sects))

        return self.sects.get_bisector_by_idx(na_input)


class MinMaxNormalizeTfm(Transformation):

    def encode(self, na_input, count_params=True):
        ''' Maps numeric attributes onto unit circle, leaving
        space defined by mapping_gap.

        When used on unseen data (with count_params=False), some
        data outliers (data bigger than maximum or lower than minimum on set
        for which encode with count_params=True was used last time) may get
        mapped incorrectly if mapping_gap is not big enough.\n
        On the other hand - big mapping gap leaves less space for mapping
        "original" data...

        @see Transformation.encode
        '''
        if count_params:
            self.shifts = np.min(na_input, axis=0)
            na_input -= self.shifts
            self.factors = np.max(na_input, axis=0)
        else:
            na_input -= self.shifts

        na_input /= self.factors

        # turn [0,1] to phases from [0,2*pi-eps]
        na_input *= (2 * pi - self.mapping_gap)

        # map phases to unit circle vectors
        return np.cos(na_input) + np.sin(na_input)*1j

    def decode(self, na_output):
        '''
        @see Transformation.decode
        '''
        pi2 = 2 * pi
        # count angle in [0,2pi] for outputs
        result = np.angle(na_output)
        result += pi2
        result %= pi2

        # back to interval [0,1]
        result /= (pi2 - self.mapping_gap)
        # unnormalize
        result *= self.factors
        # and shift back (so that min/max value is set properly)
        result += self.shifts

        return result

    def get_state(self):
        ''' @see Transformation.get_state '''
        return {'mapping_gap': self.mapping_gap,
                'shifts': self.shifts,
                'factors': self.factors}

    def set_state(self, kwargs):
        ''' Sets state mandatory argument when called while creating class
        is 'mapping_gap'.

        @see mapping_gap
        @see Transformation.set_state
        '''
        ## How big sector of circle should not be used for encoding of samples
        # (in radians). This sector serves as "reserve" for future outliers,
        # or to allow possibility of predicting data bigger than maximal
        # value in column. \n But not lesser values than minimum! - each
        # complex number from mapping_gap is decoded as bigger than original
        # maximum.
        gap = kwargs['mapping_gap']
        if gap < 0 or gap >= 2 * pi:
            raise ValueError("Mapping gap must be from interval [0,2pi) maybe"
                             " you should read documentation")
        else:
            self.mapping_gap = gap

        if 'shifts' in kwargs:
            ## Minimum for each column - how much should samples be shifted
            # to get them all positive
            self.shifts = kwargs['shifts']
        if 'factors' in kwargs:
            ## Maximum value for each column - dividing shifted values by
            # maximum normalizes them to [0,1]
            self.factors = kwargs['factors']
