#!/usr/bin/env python
#encoding=utf8

from __future__ import division
import numpy as np
import math
from cmath import pi
import sys

SC_ERROR_RATE = 0.5

import matplotlib.pyplot as plt


def LearningMLMVN(na_train_set,
                  na_validate_set,
                  mlmvn,
                  desired_angle_precision=2 * pi,
                  ratio_of_train_set_to_learn_from=0.0,
                  sc_train_rmse=2 * pi,
                  sc_validation_rmse=2 * pi,
                  sc_train_accuracy=0.0,
                  sc_validation_accuracy=0.0,
                  sc_num_iterations=sys.maxint,
                  record_history=False):

    ''' Function for learning the MLMVN networks.
    Supports network.DiscreteMLMVN, network.DiscreteLastLayerMLMVN and
    network.ContinuousMLMVN

    @param na_train_set Numpy matrix with training set (both inputs and
            desired outputs - in case of discrete mlmvn desired outputs should
            be shifted to bisectors of sectors.\nNumber of outputs is counted
            as number of output neurons of provided mlmvn network)
    @param na_validate_set Same format as train_set, but this dataset is left
                           out of learning process and used only to detect
                           overfitting (network does not generalize well)
    @param mlmvn MLMVN network to be learned.
    @param desired_angle_precision Angle in radians. If the angle difference
            of mlmvn output and desired output for sample is bigger than this
            value sample is considered to be incorrectly classified. These
            samples are then used for learning.
    @param ratio_of_train_set_to_learn_from Float - fraction of incorrectly
           classified samples will be used for learning during one iteration
           (useful when pass through data is costly operation) but may lead
           to overfitting or missing. If ratio_of_train_set_to_learn_from *
           number_of_incorrectly_classified_samples == 0, 1 sample is used
           for learning.
    @param sc_train_rmse Stopping criterion value of RMSE on training set.
            RMSE is measured as angle (radians).\n
            If RMSE on training set is lower than this value, training set is
            considered as learned.\n If sc_validation_rmse criterion is not
            achieved, learning will reset weights and start all over again.
    @param sc_validation_rmse Stopping criterion value of RMSE on validation
            set. Used to prevent overfitting indicated by low RMSE on training
            set and high RMSE on validation set. Therefore sc_validation_rmse
            should be around the same value as sc_train_rmse.
    @param sc_train_accuracy Stopping criterion using rate of correctly
            classified samples. Values are from interval [0,1] (not in
            percents!)
    @param sc_validation_accuracy Stopping criterion like sc_train_accuracy,
            but on validation set. Can be used to prevent overfitting like
            sc_validation_rmse.
    @param sc_num_iterations Learning will finish after specified number of
            iterations (even though other stopping criteria are not met).\n
            Defaultly set to maximal integer value so that it does not afflict
            learning.
    @param record_history Boolean indicating whether history for RMSEs and
            accuracies should be stored and plotted after the learning process
            terminates. If recording is set values for all the stoping
            criteria are counted (learning w/o recording might be faster, but
            there are some negative consequences.
    '''

    n_samples_ts = na_train_set.shape[0]
    n_samples_vs = na_validate_set.shape[0]

    # for discrete output set desired sector precision to half of the sector
    # (we suppose that outputs are optimized to bisectors
    # initialize rmse stopping criterion to half of sector (should be even
    # lower)
    if mlmvn.__class__.__name__ in ['DiscreteMLMVN', 'DiscreteLastLayerMLMVN']:
        desired_angle_precision = mlmvn.sects.get_sector_half()

    n_it = 0

    #*********** HISTORY *******
    if record_history:
        history = []

    #while 1:
        # determine how to learn... function for stopping criterion
    try:
        sys.stdout.write('LEARNING %s:\nProgress: ' % mlmvn.__class__.__name__)
        while 1:
            stop_learning = False
            n_it += 1
            sys.stdout.write('.')

            # count angle difference between desired outputs for set
            # and weighted sums of last layer
            na_angle_deltas = count_angle_deltas(mlmvn, na_train_set)

            # if this difference is bigger (or equal) than half of sector,
            # sample is incorrectly classified. Note that this is true only
            # in case of bisector optimalization and when sectors are
            # uniform!!!
            incorrect = np.where(
                np.sum(na_angle_deltas > desired_angle_precision, axis=1) != 0
            )[0]

            accuracy_ts = 1 - (incorrect.size / n_samples_ts)

            if accuracy_ts < sc_train_accuracy and not record_history and accuracy_ts != 1:
                learn_from_mystakes(mlmvn, na_train_set, incorrect,
                                    ratio_of_train_set_to_learn_from)
                continue

            rmse_ts = count_rmse(na_angle_deltas)

            if rmse_ts > sc_train_rmse and not record_history and accuracy_ts != 1:
                learn_from_mystakes(mlmvn, na_train_set, incorrect,
                                    ratio_of_train_set_to_learn_from)
                continue

            # similar calculations for validation set ...
            na_angle_deltas = count_angle_deltas(mlmvn, na_validate_set)

            incorrect_num_vs = np.sum(
                np.sum(na_angle_deltas > desired_angle_precision, axis=1) != 0
            )

            accuracy_vs = 1 - (incorrect_num_vs / n_samples_vs)
            rmse_vs = count_rmse(na_angle_deltas)

            '''
            print "TRAIN RMSE:       %f" % rmse_ts
            print "VALIDATION  RMSE: %f" % rmse_vs
            print "no. incorrect:                    %d" % incorrect.size
            print "no. incorrect in validation set:  %d" % incorrect_num_vs
            print "Accuracy...  Training set, Validation set:         %f4, %f4\n   at iteration %d" % (accuracy_ts, accuracy_vs, n_it)
            '''

            if record_history:
                history.append((accuracy_ts, accuracy_vs, rmse_ts, rmse_vs))

            #********** STOP LEARNING ??? ******************


            if n_it > sc_num_iterations:
                break

            if accuracy_ts >= sc_train_accuracy:
                if accuracy_vs >= sc_validation_accuracy:
                    stop_learning = True

            if rmse_ts <= sc_train_rmse:
                if rmse_vs <= sc_validation_rmse:
                    stop_learning = stop_learning and True
            else:
                stop_learning = False

            if stop_learning:
                break
            else:
                if accuracy_ts == 1:
                    # if accuracy is 1.0 and it's not desired stopping criteria -
                    # we do not have any more samples to learn from => start again
                    sys.stdout.write("\nRe-setting random weights - accuracy on train set == 1, Progress:\n")
                    mlmvn.reset_random_weights()
                    continue

            learn_from_mystakes(mlmvn, na_train_set, incorrect,
                                ratio_of_train_set_to_learn_from)
    except KeyboardInterrupt:
        print "Learning stopped by user!"

    sys.stdout.write('\nLearning finished at iteration %d\n' % n_it)

    if record_history:
        plt.clf()
        plt.plot(range(len(history)), history)
        plt.legend(['TS_acc', 'VS_acc', 'TS_rmse', 'VS_rmse'])
        plt.xlabel('Iteration')
        plt.show()


def learn_from_mystakes(mlmvn, na_train_set, mystakes, learn_ratio):

        if mystakes.size == 0:
            raise StopIteration('Can not learn further - no incorrect samples')

        num_samples = int(mystakes.size * learn_ratio)
        if num_samples == 0:
            num_samples = 1

        for bad_sample in np.random.randint(0, mystakes.size, size=num_samples):
            sys.stdout.write('.')
            sample = na_train_set[mystakes[bad_sample], :]
            mlmvn.sample_learning(sample[:-1], sample[-1])

def count_angle_deltas(mlmvn, dataset):

    n_outs = mlmvn.get_number_of_outputs()

    # TODO less operations possible?
    deltas = np.angle(dataset[:, -n_outs:])
    deltas += (2 * np.pi)
    deltas %= (2 * np.pi)
    deltas2 = np.angle(mlmvn.count_zets_of_last_layer(dataset[:, :-n_outs]))
    deltas2 += (2 * np.pi)
    deltas2 %= (2 * np.pi)

    deltas -= deltas2

    return np.abs(deltas)


def count_rmse(na_angle_deltas):
    ''' Counts rooted mean square error in terms of angle
    difference between weighted sums and desired outputs of neurons
    in last layer of network

    @param na_angle_deltas Numpy matrix of phase differences (in radians).\n
                            Indexing: [sample, output_neuron_of_mlmvn]
    '''
    rmse = np.sum(na_angle_deltas * na_angle_deltas)
    # na_angle_deltas.shape = (n_samples, n_output_neurons)
    # therefore na_angle_deltas.size is correct factor for dividing
    rmse /= na_angle_deltas.size

    return math.sqrt(rmse)
