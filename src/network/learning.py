#!/usr/bin/env python
#encoding=utf8

from __future__ import division
from gi.repository import GLib
import sys
import threading
from time import sleep
import numpy as np
import math
try:
    import matplotlib.pyplot as plt
except ImportError as e:
    # if user does not have matplotlib... do not crash just
    # announce possible problems
    sys.stderr.write("WARNING: matplotlib.pyplot not found... "
                     "plotting of learning history will not work")

## Contains for each MLMVNLearning list of networks which can be learned with it
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
    def create(learning_name, mlmvn, na_train_set, na_validation_set,
               params, cleanup_func, out_stream):
        try:
            learning = apply(getattr(sys.modules[__name__], learning_name),
                             (mlmvn,
                              na_train_set,
                              na_validation_set,
                              params,
                              cleanup_func,
                              out_stream))
            return learning
        #except AttributeError as e:
        #    raise ValueError('Not supported learning: ' + learning_name + ', error: ' + str(e))
        except Exception as e:
            raise ValueError("Unexpected exception: " + str(e))

    def __init__(self, mlmvn, na_train_set, na_validation_set,
                 params, cleanup_func, out_stream=sys.stdout):
        super(MLMVNLearning, self).__init__()

        self.history = []

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

        ## current iteration number
        self.n_iteration = 0

        ## Sets the stream self.out_stream where textual output will be written
        self.set_out(out_stream)

        if not 'record_history' in self.params:
            ## If set to True, history is plotted and shown to user when
            # learning process is paused (defaultly False)
            self.params['record_history'] = False

        if not 'pause_at_iteration' in self.params:
            ## Int number of iteration - if specified, learning will be paused
            # at this iteration. Defaultly learning will not pause.
            self.params['pause_at_iteration'] = -1

    def run(self):

        if self.params['record_history']:
            self.history = []

        m_validation = None
        msg = None
        while not self.stop_request.is_set():

            while self.run_request.is_set():
                self.n_iteration += 1

                (m_train, incorrect_samples) = self.get_metrics_and_errs(self.na_train)

                if self.na_validate.size != 0:
                    m_validation = self.get_metrics(self.na_validate)

                if self.params['record_history']:
                    self.history.append(
                        self.construct_datapoint(m_train,
                                                 m_validation)
                    )

                if not self.out_stream is None:
                    train_metrics = self.str_metric(m_train)
                    validation_metrics = self.str_metric(m_validation)
                    msg = ("\nIteration:\t\t\t\t%s\n"
                           "Metrics on training set:\t%s\n"
                           "Metrics on validation set:\t%s\n"
                           % (self.n_iteration, train_metrics,
                              validation_metrics))

                    self.out_stream.write(msg)

                if self.stopping_criteria(metrics_train=m_train,
                                          metrics_validation=m_validation):
                    if self.params['record_history']:
                        self.plot_history()

                    self.stop_learning()
                else:
                    self.learn_from_samples(incorrect_samples)

                if self.n_iteration == self.params['pause_at_iteration']:
                    self.pause_learning()

            if not self.stop_request.is_set():
                # now... thread is supressed in "paused state" waiting for
                # run_request
                self.run_request.wait()

        GLib.idle_add(self.cleanup_func)
        if not msg is None:
            sleep(0.5)
            if not self.out_stream is None:
                self.out_stream.write("\nStopped with:" + msg)

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
        """ Checks whether learning should be stopped
        @return True if learning should be stopped.
        """
        raise NotImplementedError("stopping_criteria() not implemented in "
                                  + self.__class__.__name__)

    def get_metrics(self, na_dataset):
        """ Returns metrics for dataset
        @return Dictionary with learning specific metrics
        """
        raise NotImplementedError("get_metrics() not implemented in "
                                  + self.__class__.__name__)

    def get_metrics_and_errs(self, na_dataset):
        """ Returns metrics and info about incorrectly classified samples for
        dataset (info about incorrectly classified samples may be counted
        together with metrics therefore it may be better than subsequent
        get_metrics() and get_errs()).

        @return Tuple (metrics, incorrect_info)\n
                metrics - Dictionary with learning specific metrics\n
                incorrect_info - info about incorrectly classified samples\n
        """
        raise NotImplementedError("get_metrics_and_errs() not implemented in "
                                  + self.__class__.__name__)

    def get_errs(self, na_dataset):
        """ Returns info about incorrectly classified samples

        @return Object - info about incorrectly classified samples.\n
        """
        raise NotImplementedError("get_errs() not implemented in "
                                  + self.__class__.__name__)

    def plot_history(self):
        """
        Plots self.history of learning using matplotlib...
        (if it is accessible in the system)
        """
        raise NotImplementedError("plot_history() not implemented in "
                                  + self.__class__.__name__)

    @staticmethod
    def construct_datapoint(metrics_train_set, metrics_validation_set):
        """ Given metrics on train and validation set constructs datapoint
        for plotting of learning history.
        """
        raise NotImplementedError("construct_datapoint() not implemented")

    @staticmethod
    def str_metric(metrics):
        """For metric returned by get_metrics() or get_metrics_and_errs()
        returns its string representation without \\n.
        """
        raise NotImplementedError("str_metric() not implemented")

    def learn_from_samples(self, incorrect_info):
        """ Learns from incorrectly classified samples in
        self.na_train set.

        @param incorrect_info Info about incorrectly classified samples in
               self.na_train set as returned by get_errs() or
               get_metrics_and_errs()
        """
        raise NotImplementedError("learn_from_samples() not implemented in "
                                  + self.__class__.__name__)

    #TODO error types specific to learning?
    def check_requirements(self):
        """Checks whether parameters needed for learning are specified
        if not raises ValueError
        """
        if self.mlmvn is None:
            raise ValueError("MLMVN to be learned is not specified.")

        for parameter in self.requires:
            if not parameter in self.params:
                raise ValueError("Missing parameter '%s' for %s learning"
                                 % (parameter, self.__class__.__name__))

    def apply_settings(self, params):
        """ Applies settings specified in params - like "setter injection"
        @param params Dictionary with parameters for this learning.
        """
        if not params is None:
            self.params.update(params)

        self.check_requirements()

    def set_out(self, out_stream):
        """ Checks output stream for write(str) method"""
        if out_stream is None:
            self.out_stream = None
        else:
            if not hasattr(out_stream, 'write'):
                raise ValueError("Text output can't be written to object: "
                                + str(out_stream))
            try:
                out_stream.write('\n')
                self.out_stream = out_stream
            except TypeError as e:
                raise ValueError("Text output stream %s can't be written to!"
                                "- error: %s" % (str(out_stream), str(e)))



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
    """ Error-correction learning of MLMVN with one sample.

    Parameters needed are in RMSEandAccLearningSM.requires
    Pramaters starting with sc_ are stopping criteria for performance metrics
    which have to be all satisfied in order to stop learning:\n
        eg. sc_train_rmse > (rmse on train set)  => OK\n
        eg. sc_train_acc < (accuracy for train set)  => OK\n
    @see RMSEandAccLearningSM.stopping_criteria

    Parameter 'fraction_to_learn' determines part of erroreous samples to
    learn from\n
    eg. 0.5 => learn from half of the incorrectly classified samples
    without another computations of metrics\n

    Parameter 'desired_angle_precision' specifies that an output is classified
    as errorreous if its angle difference from desired_output is bigger than
    'desired_angle_precision'\n

    It can be used for soft margins learning by specifying parameter:\n
    'desired_angle_precision' = <sector_size_in_rad> - <soft_margin_in_rad>\n
    """

    requires = ['sc_train_rmse',
                'sc_validation_rmse',
                'sc_train_accuracy',
                'sc_validation_accuracy',
                'fraction_to_learn',
                'desired_angle_precision']

    def __init__(self, mlmvn, na_train_set, na_validation_set,
                 params, cleanup_func, out_stream=sys.stdout):
        """
        Initializes learning process.

        @param mlmvn Network which will be learned (modified!).
        @param na_train_set Numpy matrix with learning samples
                    Indexing [ [(inputs), (outputs)], .. ] (sample on row)
        @param na_validation_set Numpy matrix with samples left out of learning
                            used to prevent overfitting
                    Indexing [ [(inputs), (outputs)], .. ] (sample on row)
        @param params User supplied settings of learning
                       (which is in RMSEandAccLearningSM.requires)
        @param cleanup_func Func. which will be called when learning finishes
        @param out_stream Object which provides write(str) method.
                          Info about learning progress is written to
                          out_stream.\n
                          out_stream == None supressess output.
        """

        super(RMSEandAccLearningSM, self).__init__(mlmvn, na_train_set,
                                                   na_validation_set,
                                                   params,
                                                   cleanup_func,
                                                   out_stream)

        self.params['desired_angle_precision'] = params['desired_angle_precision']

        self.check_requirements()

    def stopping_criteria(self, metrics_train=None, metrics_validation=None):
        """ Checks whether all stopping criteria were reached
        @return Boolean value whether to stop learning
        """

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
        """
        @return Tuple (metrics, incorrect_row_indices)\n
                Where: metrics is dictionary with metrics accuracy and rmse
                and incorrect_row_indices is Numpy array specifying incorrectly
                classified samples.
        """

        na_angle_deltas = self._count_angle_deltas(na_dataset)

        incorrect_row_indices = np.where(
            np.sum(na_angle_deltas > self.params['desired_angle_precision'],
                   axis=1) != 0
        )[0]

        rmse = self._count_rmse(na_angle_deltas)
        accuracy = 1 - (incorrect_row_indices.size / na_dataset.shape[0])

        return ({'rmse': rmse, 'acc': accuracy}, incorrect_row_indices)

    def get_metrics(self, na_dataset):
        """
        @return Dictionary with metrics accuracy (key 'acc') and rmse
                (key 'rmse').
        """

        na_angle_deltas = self._count_angle_deltas(na_dataset)

        incorrect_num = np.sum(
            np.sum(na_angle_deltas > self.params['desired_angle_precision'],
                   axis=1) != 0
        )

        accuracy = 1 - (incorrect_num / na_dataset.shape[0])
        rmse = self._count_rmse(na_angle_deltas)

        return {'rmse': rmse, 'acc': accuracy}

    @staticmethod
    def str_metric(metric):
        """@see MLMVNLearning.str_metric"""
        if metric is None:
            return "Not used"
        else:
            return ("Accuracy: %5.4f; RMSE: %5.4f"
                    % (metric['acc'], metric['rmse']))

    def learn_from_samples(self, na_mystakes):
        """@see MLMVNLearning.learn_from_samples"""

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
            sample = self.na_train[na_mystakes[bad_sample], :]
            self.mlmvn.sample_learning(sample[:-n_outputs], sample[-n_outputs])

    @staticmethod
    def construct_datapoint(metrics_train, metrics_validation):
        """@see MLMVNLearning.construct_datapoint"""
        result = (None, None, None, None)
        if not metrics_train is None:
            result = (metrics_train['acc'], metrics_train['rmse'], None, None)
        if not metrics_train is None:
            result = (result[0],
                      result[1],
                      metrics_validation['acc'],
                      metrics_validation['rmse'])

        return result

    def plot_history(self):
        """@see MLMVNLearning.plot_history"""
        #TODO ... thread problem! Thread save window!
        history = np.array(self.history)
        if history.shape[0] < 2:
            # nothing interesting to plot
            return

        plt.clf()
        rec_start = self.n_iteration - history.shape[0] + 1

        plt.figure(1)
        plt.subplot(211)
        plt.plot(range(rec_start, self.n_iteration+1), history[:, 0], 'g')
        plt.plot(range(rec_start, self.n_iteration+1), history[:, 2], 'r')
        plt.ylabel("Accuracy")
        plt.legend(['train set', 'validation set'], loc=4)

        plt.subplot(212)
        plt.plot(range(rec_start, self.n_iteration+1), history[:, 1], 'k--')
        plt.plot(range(rec_start, self.n_iteration+1), history[:, 3], 'y--')
        plt.ylabel("RMSE")
        plt.legend(['train set', 'validation set'])
        plt.xlabel('Iteration')

        plt.suptitle("Learning history")
        plt.show()


class RMSEandAccLearning(RMSEandAccLearningSM):
    """ Similar to RMSEandAccLearningSM but desired_angle_precision
    is set to half of the sector which results in classical
    K-valued treshold activation function and error detection.
    """
    def __init__(self, mlmvn, na_train_set, na_validation_set,
                 params, cleanup_func, out_stream=sys.stdout):

        # for discrete output set desired sector precision to half of the
        # sector (we suppose that outputs are optimized to bisectors
        # initialize rmse stopping criterion to half of sector (could be even
        # lower)
        params['desired_angle_precision'] = mlmvn.sects.get_sector_half()

        super(RMSEandAccLearning, self).__init__(mlmvn, na_train_set,
                                                 na_validation_set,
                                                 params,
                                                 cleanup_func,
                                                 out_stream)

        self.check_requirements()
