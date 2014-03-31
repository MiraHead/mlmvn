#!/usr/bin/env python
#encoding=utf8

from __future__ import division
from gi.repository import GLib
import sys
import threading
import numpy as np
import math

COMPATIBLE_NETWORKS = {
    "RMSEandAccLearning": ['DiscreteMLMVN',
                           'DiscreteLastLayerMLMVN'],

    "RMSEandAccLearningSM": ['DiscreteMLMVN',
                             'DiscreteLastLayerMLMVN',
                             'ContinuousMLMVN']
}


class MLMVNLearning(threading.Thread):

    ## Required arguments which should be in params
    requires = []

    @staticmethod
    def create(learning_name, mlmvn, na_train_set, na_validation_set, params, cleanup_func):
        try:
            learning = apply(getattr(sys.modules[__name__], learning_name),
                             (mlmvn,
                              na_train_set,
                              na_validation_set,
                              params,
                              cleanup_func))
            return learning
        except AttributeError as e:
            raise ValueError('Not supported learning: ' + learning_name)
        except Exception as e:
            print "Unexpected exception: " + str(e)

    def __init__(self, mlmvn, na_train_set, na_validation_set, params, cleanup_func):
        super(MLMVNLearning, self).__init__()

        ## Event - if set, learning process is running else learning is in
        # paused state
        self.run_request = threading.Event()
        self.run_request.set()
        ## Event - if set learning should be stopped next time that learning
        # is paused
        self.stop_request = threading.Event()
        self.stop_request.clear()

        ##function which will be called after this thread finishes
        self.cleanup_func = cleanup_func

        ## MLMVN to be learned
        self.mlmvn = mlmvn

        ## Numpy array with train set
        self.na_train = na_train_set
        ## Numpy array with validation set
        self.na_validate = na_validation_set

        ## Dictionary containing specific parameters for learning process
        self.params = params

        if not 'record_history' in self.params:
            ## If set to True, history is plotted and shown to user when
            # learning process is paused (defaultly False)
            self.params['record_history'] = False

        if not 'pause_at_iteration' in self.params:
            ## Int number of iteration - if specified, learning will be paused
            # at this iteration. Defaultly learning will not pause.
            self.params['pause_at_iteration'] = -1

    def run(self):
        n_iteration = 0

        if self.params['record_history']:
            history = []

        m_validation = None
        while not self.stop_request.is_set():
            print "Learning started"
            while self.run_request.is_set():
                n_iteration += 1

                (m_train, wrong_idx) = self.get_metrics_and_errs(self.na_train)

                if self.na_validate.size != 0:
                    m_validation = self.get_metrics(self.na_validate)

                if self.params['record_history']:
                    history.append((m_train, m_validation))

                if self.stopping_criteria(metrics_train=m_train,
                                          metrics_validation=m_validation):
                    self.stop_learning()
                    print "Stopping criteria reached"
                else:
                    self.learn_from_samples(self.na_train, wrong_idx)

                print "  Iteration:              " + str(n_iteration)
                print "  Metrics test set:       " + str(m_train)
                print "  Metrics validation set: " + str(m_validation)

                if n_iteration == self.params['pause_at_iteration']:
                    self.pause_learning()

            if self.params['record_history']:
                self.plot_history(history)

            if not self.stop_request.is_set():
                print "Learning paused"
                # now... thread is supressed in "paused state" waiting for
                # run_request
                self.run_request.wait()
        print "Learning stopped"

        GLib.idle_add(self.cleanup_func)

    def join(self, timeout=None):
        self.stop_learning()
        super(MLMVNLearning, self).join(timeout)

    def stop_learning(self):
        self.stop_request.set()
        if not self.run_request.is_set():
            self.resume_learning()
        else:
            self.pause_learning()

    def pause_learning(self):
        self.run_request.clear()

    def resume_learning(self, new_params=None):
        self.apply_settings(new_params)
        self.run_request.set()

    def stopping_criteria(self, metrics_train=None, metrics_validation=None):
        raise NotImplementedError("stopping_criteria() not implemented in "
                                  + self.__class__.__name__)

    def get_metrics(self, na_dataset):
        raise NotImplementedError("get_metrics() not implemented in "
                                  + self.__class__.__name__)

    def get_metrics_and_errs(self, na_dataset):
        raise NotImplementedError("get_metrics_and_errs() not implemented in "
                                  + self.__class__.__name__)

    def plot_history(self, history):
        raise NotImplementedError("plot_history() not implemented in "
                                  + self.__class__.__name__)

    def learn_from_samples(self, samples, incorrect_indices):
        raise NotImplementedError("learn_from_samples() not implemented in "
                                  + self.__class__.__name__)

    #TODO error types specific to learning?
    def check_requirements(self):
        if self.mlmvn is None:
            raise ValueError("MLMVN to be learned is not specified.")

        for parameter in self.requires:
            if not parameter in self.params:
                raise ValueError("Missing parameter '%s' for %s learning"
                                 % (parameter, self.__class__.__name__))

    def apply_settings(self, params):
        if not params is None:
            self.params.update(params)

        self.check_requirements()

    def _count_angle_deltas(self, na_dataset):
        """ Counts angle difference between outputs
        of mlmvn and desired outputs for each sample
        in na_dataset.

        @param na_dataset Numpy array whose last columns are desired outputs.
        @return Numpy array with angle differences for those outputs.
        """
        n_outs = self.mlmvn.get_number_of_outputs()

        deltas = np.angle(na_dataset[:, -n_outs:])
        deltas += (2 * np.pi)
        deltas %= (2 * np.pi)
        deltas2 = np.angle(
            self.mlmvn.count_zets_of_last_layer(na_dataset[:, :-n_outs])
        )
        deltas2 += (2 * np.pi)
        deltas2 %= (2 * np.pi)

        deltas -= deltas2

        return np.abs(deltas)

    @staticmethod
    def _count_rmse(na_angle_deltas):
        ''' Counts rooted mean square error in terms of angle
        difference between weighted sums and desired outputs of neurons
        in last layer of network

        @param na_angle_deltas Numpy array of phase differences (in radians).\n
                                Indexing: [sample, output_neuron_of_mlmvn]
        @return Root mean squared error - average angle distance/error in
                radians.
        '''
        rmse = np.sum(na_angle_deltas * na_angle_deltas)
        # na_angle_deltas.shape = (n_samples, n_output_neurons)
        # therefore na_angle_deltas.size is correct factor for dividing
        rmse /= na_angle_deltas.size

        return math.sqrt(rmse)


class RMSEandAccLearningSM(MLMVNLearning):
    """ The same as RMSEandAccLearning, but parameter
    'desired_angle_precision" must be provided in parameters by user so
    that detection of errorreous samples is correct.

    Can be used for soft margins learning by specifying parameter:\n
    'desired_angle_precision' = <sector_size_in_rad> - <soft_margin_in_rad>\n

    In other words - an output is classified as errorreous if its angle
    difference from desired_output is bigger than 'desired_angle_precision'
    """

    def __init__(self, mlmvn, na_train_set, na_validation_set,
                 params, cleanup_func):

        super(RMSEandAccLearningSM, self).__init__(mlmvn, na_train_set,
                                                   na_validation_set,
                                                   params,
                                                   cleanup_func)

        self.params['desired_angle_precision'] = params['desired_angle_precision']

        self.check_requirements()


    def stopping_criteria(self, metrics_train=None, metrics_validation=None):

        stop_learning = True

        if not metrics_train is None:
            stop_learning = stop_learning \
                and metrics_train['rmse'] <= self.params['sc_train_rmse'] \
                and metrics_train['acc'] >= self.params['sc_train_accuracy']
        if not metrics_validation is None:
            stop_learning = stop_learning \
                and metrics_validation['rmse'] <= self.params['sc_validation_rmse'] \
                and metrics_validation['acc'] >= self.params['sc_validation_accuracy']

        return stop_learning

    def get_metrics_and_errs(self, na_dataset):

        na_angle_deltas = self._count_angle_deltas(na_dataset)

        incorrect_row_indices = np.where(
            np.sum(na_angle_deltas > self.params['desired_angle_precision'],
                   axis=1) != 0
        )[0]

        rmse = self._count_rmse(na_angle_deltas)
        accuracy = 1 - (incorrect_row_indices.size / na_dataset.shape[0])

        return ({'rmse': rmse, 'acc': accuracy}, incorrect_row_indices)

    def get_metrics(self, na_dataset):

        na_angle_deltas = self._count_angle_deltas(na_dataset)

        incorrect_num = np.sum(
            np.sum(na_angle_deltas > self.params['desired_angle_precision'],
                   axis=1) != 0
        )

        accuracy = 1 - (incorrect_num / na_dataset.shape[0])
        rmse = self._count_rmse(na_angle_deltas)

        return {'rmse': rmse, 'acc': accuracy}

    def learn_from_samples(self, na_dataset, na_mystakes):

        n_outputs = self.mlmvn.get_number_of_outputs()

        if not bool(na_mystakes.size):
            self.mlmvn.reset_random_weights()
            return

        num_samples = int(na_mystakes.size * self.params['fraction_to_learn'])
        if num_samples <= 0:
            num_samples = 1

        for bad_sample in np.random.randint(0,
                                            na_mystakes.size,
                                            size=num_samples
                                            ):
            sample = na_dataset[na_mystakes[bad_sample], :]
            self.mlmvn.sample_learning(sample[:-n_outputs], sample[-n_outputs])


class RMSEandAccLearning(RMSEandAccLearningSM):

    requires = ['sc_train_rmse',
                'sc_validation_rmse',
                'sc_train_accuracy',
                'sc_validation_accuracy',
                'fraction_to_learn',
                'desired_angle_precision']

    def __init__(self, mlmvn, na_train_set, na_validation_set,
                 params, cleanup_func):

        # for discrete output set desired sector precision to half of the
        # sector (we suppose that outputs are optimized to bisectors
        # initialize rmse stopping criterion to half of sector (could be even
        # lower)
        params['desired_angle_precision'] = mlmvn.sects.get_sector_half()

        super(RMSEandAccLearning, self).__init__(mlmvn, na_train_set,
                                                 na_validation_set,
                                                 params,
                                                 cleanup_func)

        self.check_requirements()


class RMSEandAccLearnigWthMinError(MLMVNLearning):
    """ the samples with specified error range are used to learn """
    pass
