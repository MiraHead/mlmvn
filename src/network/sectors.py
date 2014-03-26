#!/usr/bin/env python
#encoding=utf8

'''
Module describing creation of sectors for multi-valued neuron.

Various implementations can be useful (knowing probabilities of classess one
might want to have sectors varying in size).

@author Miroslav Hlavacek <mira.hlavackuj@gmail.com>
'''

import cmath
import numpy as np

S_BORDER = 0
S_PHASE = 1
S_BISECTOR = 2

TOLERANCE = 0.00001

# TODO ...study for maybe even more optimization
# http://docs.scipy.org/doc/numpy/user/c-info.ufunc-tutorial.html


class Sectors():
    ''' Sectors represent sectors of multi-valued neuron (mvn) with discrete
    activation function.

    Provides activation function for neurons of networks.\n
    Provides function for retrieval of sector borders and bisectors.\n
    '''

    def __init__(self, ls_phases=None, num_sectors=0):
        ''' Creates sectors instance.

        @param num_sectors Intended number of uniform sectors.
        @param ls_phases List of phases of sector borders - each phase is
                         number from interval (-pi, pi] - do not use -pi!
                         Eg. [0, 2.094, -2.094] for 3 uniform sectors.
                         If ls_phases is set, num_sectors is counted as length
                         of ls_phases (num_sectors parameter is ignored)
        '''
        if ls_phases is None:
            if num_sectors < 2:
                raise ValueError("Specified discrete sectors can't be created"
                                 " - Bad number of sectors (less than 2)")
            ls_phases = []
            for i in range(num_sectors):
                comp_border = np.exp((1j * i * 2 * np.pi) / num_sectors)
                ls_phases.append(cmath.phase(comp_border))
        else:
            num_sectors = len(ls_phases)

        ls_phases.sort()
        ls_phases.append(ls_phases[0])

        tmp_sectors = []
        phase = ls_phases[0]
        for phase_next in ls_phases[1:]:
            comp_border = cmath.rect(1, phase)
            comp_border_next = cmath.rect(1, phase_next)

            # compute bisector (even if sectors are not uniform)
            bisector = comp_border + comp_border_next
            # normalize bisector to unit circle
            bisector /= abs(bisector)

            tmp_sectors.append((comp_border, phase, bisector))
            phase = phase_next

        self.__sectors = tuple(tmp_sectors)

        self.__sector_half = np.pi / num_sectors

    def activation_function(self, ndarray):
        ''' Retrieving sector border of sector to which complex numbers in
        given matrix belong based on step activation function.

        @param ndarray Numpy matrix on which we want to apply activation
                       function.
        @returns Numpy matrix with complex sector borders to each given
                 complex number.

        '''

        phases = np.angle(ndarray)
        outputs = np.zeros(phases.shape, dtype=complex)
        (border, phase, bisector) = self.__sectors[0]

        for (nborder, nphase, nbisector) in self.__sectors[1:]:

            # if both these conditions are true, on positions defined
            # by truth values in comparison are complex numbers belonging to
            # sector with phase "phase"
            comparison = phases < nphase
            comparison *= phase <= phases

            # give them value of sector border and assign them into
            # correct places of output
            outputs += comparison * border
            border = nborder
            phase = nphase

        # check boundaries of sectors... either one condition can be true (+=)
        comparison = phases < self.__sectors[0][S_PHASE]
        comparison += phases >= self.__sectors[-1][S_PHASE]

        # conditions above imply that numbers should get assigned last sector
        outputs += comparison * self.__sectors[-1][S_BORDER]

        return outputs

    def bisector_function(self, ndarray):
        ''' Retrieving bisector of sector to which complex numbers in ndarray
        belong.

        @param ndarray Numpy matrix on which we want to apply bisector
                       function.
        @returns Numpy matrix with sector bisectors for each given complex
                 number.
        '''
        # code is explained in activation_function... almost the same
        phases = np.angle(ndarray)
        outputs = np.zeros(phases.shape, dtype=complex)
        (border, phase, bisector) = self.__sectors[0]
        for (nborder, nphase, nbisector) in self.__sectors[1:]:

            comparison = phases < nphase
            comparison *= phase <= phases

            outputs += comparison * bisector
            bisector = nbisector
            phase = nphase

        comparison = phases < self.__sectors[0][S_PHASE]
        comparison += phases >= self.__sectors[-1][S_PHASE]
        outputs += comparison * self.__sectors[-1][S_BISECTOR]

        return outputs

    def get_border_by_idx(self, ndarray):
        ''' Returns sector border with given index.

        @param ndarray Numpy matrix of integers - desired sector indices.
                       Indexing starts at 0.
                       If integer which should specify class is bigger than
                       number of sectors - it getts mapped to 1 + 0j which
                       results in phase 0!!! This samples will be afterwards
                       all incorrectly classified!
        @returns Numpy matrix of complex sector borders
        '''

        result = np.zeros(ndarray.shape, dtype=complex)
        for i in range(int(np.max(ndarray)) + 1):
            comparison = ndarray == i
            result += comparison * self.__sectors[i][S_BORDER]

        return result


    def get_idx_by_border(self, na_sector_borders):
        ''' Inverse function to Sectors.get_border_by_idx

        @param na_sector_borders Numpy matrix with sector borders.
        @returns Numpy matrix of indices of sectors, to which borders belonged
        '''

        result = np.zeros(na_sector_borders.shape)
        # we don't have to test first sector... it's index is 0 and is properly
        # set by np.zeros already
        i = 0
        for (border, phase, bisector) in self.__sectors:
            comparison = np.abs(na_sector_borders - border) < TOLERANCE
            result += comparison * i
            i += 1

        return result

    def get_bisector_by_idx(self, ndarray):
        ''' Returns bisector of sector specified by index.

        @param ndarray Numpy matrix of integers - desired bisector indices.
                       Indexing starts at 0.
                       If integer which should specify class is bigger than
                       number of sectors - it getts mapped to first bisector.
        @returns Numpy matrix of complex bisectors.
        '''

        result = np.zeros(ndarray.shape, dtype=complex)
        for i in range(int(np.max(ndarray)) + 1):
            comparison = ndarray == i
            result += comparison * self.__sectors[i][S_BISECTOR]

        return result

    def get_idx_by_bisector(self, na_bisectors):
        ''' Inverse function to Sectors.get_bisector_by_idx

        @param na_bisectors Numpy matrix with sector borders.
        @returns Numpy matrix of indices of sectors, to which borders belonged
        '''

        result = np.zeros(na_bisectors.shape)
        # we don't have to test first sector... it's index is 0 and is properly
        # set by np.zeros already
        i = 0
        for (border, phase, bisector) in self.__sectors:
            comparison = np.abs(na_bisectors - bisector) < TOLERANCE
            result += comparison * i
            i += 1

        return result

    def get_sector_half(self):
        ''' Half of one sector in radians. In case that sectors are not
        uniform, corresponds to average half of sector (=> still returns
        half of "abstract" unifrom sector)

        @returns Half of one sector in radians.
        '''
        return self.__sector_half

    def get_phases(self):
        ''' Returns list of phases which can be used to re-create this sectors
        of this type. Init call should be:\n
        sectors.Sectors(ignored_integer, ls_phases=list_of_phases)

        @return List of phases.
        '''
        ls_phases = []
        for sector in self.__sectors:
            ls_phases.append(sector[S_PHASE])
        return ls_phases

    def get_number_of_sectors(self):
        ''' Returns number of sectors '''
        return len(self.get_phases())

    def __str__(self):
        sss = ""
        i = 0
        for (border, phase, bisector) in self.__sectors:
            sss += "%d. sector: %s (phase %s; bisector %s)\n"\
                % (i, border, phase, bisector)
            i += 1

        return sss
