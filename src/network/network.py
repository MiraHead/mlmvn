#!/usr/bin/env python
#encoding=utf8
'''
Module describing multilayer neural network's functionality.

@author Miroslav Hlavacek <mira.hlavackuj@gmail.com>
'''

from __future__ import division
import numpy as np
import cPickle as pickle
# project modules
import layer
import sectors


class MLMVN(object):
    ''' MLMVN Factory class.

    Ensures:\n
    Creating MLMVN networks.\n
    Saving/loading MLMVN network with no prior type knowledge.
    '''

    @staticmethod
    def create(class_name, *args, **kwargs):
        ''' Creates new mlmvn network.

        @param class_name String with name of class
        @param args[0] Dict containing arguments for mlmvn creation
                       entries should correspond with dictionary returned
                       by function get_kwargs_for_loading() which should
                       be implemented by every class inheriting from MLMVN.\n
                       This construction allow new classess to use as many
                       arguments they want, and name them appropriately. And
                       does not restrict the position of argument passed to
                       MLMVN.create\n
                       Any unnecessary arguments in args[0] will be ignored.
        @param **kwargs Same arguments as in args[0] can be passed like
                        keyword arguments. If kwargs are not empty, args[0]
                        dictionary is ignored)

        Example for both ways of usage (but do not mix them!):
@verbatim
 mlmvn = MLMVN.create('DiscreteMLMVN', {'ls_neurons_per_layer': [4,10,1],
                                        'number_of_sectors': 3}
                     )

 mlmvn = MLMVN.create('DiscreteMLMVN',
                       ls_neurons_per_layer=[4,10,1],
                       number_of_sectors=3
                     )
@endverbatim

        @see DiscreteMLMVN.get_kwargs_for_loading()
        @see ContinuousMLMVN.get_kwargs_for_loading()
        '''
        try:
            if not bool(kwargs):
                kwargs = args[0]

            # default parameter for all of the networks that use it
            if not 'learning_rate' in kwargs:
                kwargs['learning_rate'] = 1.0

            if class_name == 'ContinuousMLMVN':
                return ContinuousMLMVN(kwargs['ls_neurons_per_layer'],
                                       kwargs['learning_rate'])

            # Creation of uniform sectors for discrete mlmvns
            if 'number_of_sectors' in kwargs:
                sects = sectors.Sectors(
                    num_sectors=kwargs['number_of_sectors']
                )

            if class_name == 'DiscreteMLMVN':
                return DiscreteMLMVN(kwargs['ls_neurons_per_layer'],
                                     sects,
                                     kwargs['learning_rate'])

            if class_name == 'DiscreteLastLayerMLMVN':
                return DiscreteLastLayerMLMVN(kwargs['ls_neurons_per_layer'],
                                              sects,
                                              kwargs['learning_rate'])

            raise ValueError('Not supported MLMVN type' + class_name)

        except KeyError as e:
            raise NameError("Class %s can't be created without %s argument "
                            "which was not present in dict of arguments "
                            "for class creation!"
                            % (class_name, str(e.args[0])))

    def save_to_file(self, out_file):
        ''' Saves given mlmvn to out_file.

        MLMVN class to be stored must provide save_state() function.\n
        MLMVN class to be stored must provide get_kwargs_for_loading()
        function.\n

        @param mlmvn Network to be saved
        @param out_file File opened for binary write
        '''
        pickle.dump(self.__class__.__name__, out_file)
        # store arguments necessary for network's creation
        pickle.dump(self.get_kwargs_for_loading(), out_file)
        self.save_state(out_file)

    @staticmethod
    def create_from_file(in_file):
        ''' Creates/loads network from given file.
        Stored MLMVN class must provide load_state() function.

        @param in_file File opened for binary reading with stored mlmvn.
        '''
        class_name = pickle.load(in_file)
        kwargs = pickle.load(in_file)

        loaded_mlmvn = MLMVN.create(class_name, kwargs)
        loaded_mlmvn.load_state(in_file)

        return loaded_mlmvn

    def save_state(self):
        raise NotImplementedError("Saving of network's layers state not "
                                  "implemented.")

    def get_kwargs_for_loading(self):
        raise NotImplementedError("Parameters for creation of network could "
                                  "not be saved!")

    def get_number_of_inputs(self):
        raise NotImplementedError("get_number_of_inputs() not implemented")

    def get_number_of_outputs(self):
        raise NotImplementedError("get_number_of_outputs() not implemented")

    def get_name(self):
        return self.__class__.__name__


class ContinuousMLMVN(MLMVN):
    ''' Represents multi-layered network with uniform layers and
    with neurons with continuous activation function.

    Provides functions for learning network (if sample and desired output
    provided) and for batch processing of samples - outputs for each sample
    are computed. Correctness of output lies on used sectors
    - various error-detection techniques may be applied - therefore
    error-detection is implemented outside this class.
    '''

    def __init__(self, ls_neurons_per_layers,
                 learning_rate):
        ''' Initializes network

        @param ls_neurons_per_layers List specifying number of layers and
            neurons in them.\n Example: [<#inputs for first layer>,
            <#neurons fst layer>, <#neurons hidden layer>,
            <#neurons out layer>]\n
            Example 2:
@verbatim
     [20,8,4,2] therefore indicates:
            layer 1: 20 inputs, 8 neurons
            layer 2: 8 inputs, 4 outputs
            last layer: 4 inputs, 2 outputs (network outputs)
@endverbatim

        @param layers_list List of [(n_inputs,n_neurons) .. ]
        @param learning_rate Specifies speed of learning should be in
                            interval (0,1] (where 1 == full speed), but higher
                            speed is also possible.
        '''

        ## List of layers of given network
        self.ls_layers = []

        # create first and hidden layers with continuous neurons
        for (n_inputs, n_neurons) in zip(ls_neurons_per_layers[:-2],
                                         ls_neurons_per_layers[1:-1]):
            self.ls_layers.append(layer.MVNLayer(n_neurons, n_inputs))

        # create last layer with continuous neurons
        n_inputs = ls_neurons_per_layers[-2]
        n_neurons = ls_neurons_per_layers[-1]
        self.ls_layers.append(layer.MVNLastLayer(n_neurons, n_inputs))

        # set upper layers of self.ls_layers for backpropagation alg.
        for (this_l, upper_l) in zip(self.ls_layers[:-1], self.ls_layers[1:]):
            this_l.set_upper_layer(upper_l)

        # set learning rate for each layer
        self.set_learning_rate(learning_rate)

    def reset_random_weights(self, seed=None):
        ''' Sets random weights for whole network once again.

        @param seed Integer or numpy array which defines seed for pseudo-random
                    initialization of weights by numpy.

        @see layer.MLMVNLayer.set_random_weights
        '''
        if not seed is None:
            np.random.seed(seed)

        for this_l in self.ls_layers:
            this_l.set_random_weights()

    def get_number_of_outputs(self):
        ''' Returns number of outputs of network. '''
        return self.ls_layers[-1].get_number_of_outputs()

    def get_number_of_inputs(self):
        ''' Returns number of inputs for network. '''
        return self.ls_layers[0].get_number_of_inputs()

    def get_learning_rate(self):
        """ Returns learning rate for whole network
        (learning rate is the same for each layer.

        @return Float with learning rate set in network
        """
        return self.ls_layers[0].get_learning_rate()

    def set_learning_rate(self, new_rate):
        ''' Sets new learning speed throughout network.

        Learning rate should be in interval (0,1] (where 1 == full speed),
        but higher speed is also possible.

        @param new_rate Learning rate to be set.
        '''
        for this_l in self.ls_layers:
            this_l.set_learning_rate(new_rate)

    # **************** COUNTING OUTPUTS ***************

    def count_outputs(self, samples):
        ''' Batch counting of outputs. Output is simply weighted sum for
        neurons in last layer.

        @param samples  Numpy matrix of inputs for first layer.\n
                       => Batch of learning samples. Indexing: [sample,feature]
        @return Numpy matrix of counted outputs.
                Indexing: [outputs_for_sample, output_of_nth_neuron]
        '''
        # count weighted sums/output for last layer
        return self.count_zets_of_last_layer(samples)

    def count_zets_of_last_layer(self, samples):
        ''' Batch counting of weighted sums of last layer

        @param samples  Numpy matrix of inputs for first layer.\n
                       => Batch of learning samples. Indexing: [sample,feature]
        @return Numpy matrix of counted outputs.
                Indexing: [zets_for_sample, zets_of_nth_neuron]
        '''
        prev_outputs = samples

        for this_l in self.ls_layers[:-1]:
            prev_outputs = np.array(this_l.count_outputs(prev_outputs))

        return self.ls_layers[-1].count_zets(prev_outputs)

    def count_output_for_learning(self, sample):
        ''' Counts output for one sample and prepares layers for learning.

        @param sample Numpy vector - one learning sample (just inputs).
        @return Output of layer for one learning sample - Numpy vector.\n
                SIDE EFFECT: Prepares network for updating weights by error
                produced by sample!!!
        '''
        prev_outputs = sample

        for this_l in self.ls_layers:
            prev_outputs = this_l.count_outputs_for_update(prev_outputs)

        return prev_outputs

    # ********** LEARNING *****************

    def sample_learning(self, sample, desired_outputs):
        ''' Learns with one learning sample.

        Corrects weights so that desired_outputs are achieved (or at least
        approached in case that learning rate is lower than 1).

        @param sample   Numpy vector of inputs for neurons.\n
        @param desired_outputs Numpy vector of desired outputs used for
                               correction of weights.\n
                               In case that bisector opitmization is used,
                               bisectors should be passed as desired_outputs.
        '''
        # stores zets for whole network - used while backpropagating error
        self.count_output_for_learning(sample)

        self.__back_propagation(desired_outputs)
        self.__forward_propagation(sample)

    def __back_propagation(self, desired_outputs):
        ''' Back propagates the error through whole network.

        @param desired_outputs Numpy vector of desired outputs for each neuron
                               (sector borders or sector bisectors or just
                               complex numbers in continuous case).
        '''

        # errors for last layer
        self.ls_layers[-1].count_errors_last_layer(desired_outputs)

        # upper layers are set and remembered in each
        # layer => no parameters needed
        for this_l in self.ls_layers[-2::-1]:
            this_l.count_errors()

    def __forward_propagation(self, sample):
        ''' Forward propagates changes - updates weights of neurons.

        @param sample Inputs for first layer of network.
        '''
        inputs = sample
        for this_l in self.ls_layers:
            this_l.update_weights(inputs)
            inputs = this_l.count_outputs_for_update(inputs)

    # *******************  SAVING/LOADING *************

    def get_kwargs_for_loading(self):
        ''' Returns dictionary necessary for this initialization of this
        MLMVN.

        @returns Dictionary of keyword arguments which can be passed as first
                 positional argument to MLMVN.create

        @see MLMVN.create
        '''
        kwargs = {}
        kwargs['learning_rate'] = self.get_learning_rate()

        layers = []
        for this_l in self.ls_layers:
            layers.append(this_l.get_number_of_inputs())

        layers.append(this_l.get_number_of_outputs())

        kwargs['ls_neurons_per_layer'] = layers

        return kwargs

    def save_state(self, out_file):
        ''' Saves current state of network.

        @param out_file File opened for binary writing.
        '''
        # save weights and learning rate for each layer
        for this_l in self.ls_layers:
            this_l.save_layer(out_file)

    def load_state(self, in_file):
        ''' Loads and sets state of network from file.

        @param in_file File opened for binary reading.
        '''
        # load weights and learning rates of layers
        for this_l in self.ls_layers:
            this_l.load_layer(in_file)


class DiscreteMLMVN(ContinuousMLMVN):
    ''' Represents multi-layered network with uniform layers and with
    multi-valued neurons with sectors (for activation function)
    of uniform size.

    Provides functions for learning network (if sample and desired output
    provided) and for batch processing of samples - outputs for each sample
    are computed. Correctness of output lies on used sectors
    - various error-detection techniques may be applied - therefore
    error-detection is implemented outside this class.
    '''

    def __init__(self, ls_neurons_per_layers,
                 used_sectors,
                 learning_rate):
        ''' Initialization ... same parameters as for ContinuousMLMVN, but
        we do also sectors of neurons.

        @param used_sectors Sectors that will provide activation function
                            for neurons.\n
        @see ContinuousMLMVN.__init__
        '''
        ## Sectors used for counting activation function of neurons
        self.sects = used_sectors

        ## Learning rate defining speed of learning
        self.learning_rate = learning_rate

        ## List of layers of given network
        self.ls_layers = []

        # create first and hidden layers
        for (n_inputs, n_neurons) in zip(ls_neurons_per_layers[:-2],
                                         ls_neurons_per_layers[1:-1]):
            self.ls_layers.append(
                layer.MVNLayer(n_neurons, n_inputs,
                               activation_func=self.sects.activation_function,
                               learning_rate=self.learning_rate)
            )

        # create last layer
        n_inputs = ls_neurons_per_layers[-2]
        n_neurons = ls_neurons_per_layers[-1]
        self.ls_layers.append(
            layer.MVNLastLayer(n_neurons, n_inputs,
                               activation_func=self.sects.activation_function,
                               learning_rate=self.learning_rate)
        )

        # set upper layers of self.ls_layers for backpropagation alg.
        for (this_l, upper_l) in zip(self.ls_layers[:-1], self.ls_layers[1:]):
            this_l.set_upper_layer(upper_l)

    # **************** COUNTING OUTPUTS ***************

    def count_outputs(self, samples):
        ''' Batch counting of outputs. Output of neuron is sector border.

        @param samples  Numpy matrix of inputs for first layer.\n
                       => Batch of learning samples. Indexing: [sample,feature]
        @return Numpy matrix of counted outputs.
                Indexing: [outputs_for_sample, output_of_nth_neuron]
        '''
        # count weighted sums for last layer and apply activation
        # function of last layer onto them
        weighted_sums = self.count_zets_of_last_layer(samples)

        return self.ls_layers[-1].activation_func(weighted_sums)

    # *******************  SAVING/LOADING *************

    def get_kwargs_for_loading(self):
        kwargs = super(DiscreteMLMVN, self).get_kwargs_for_loading()
        kwargs['number_of_sectors'] = self.sects.get_number_of_sectors()

        return kwargs


class DiscreteLastLayerMLMVN(DiscreteMLMVN):
    ''' Represents multi-layered network with continuous neurons in hidden
    layers and with multi-valued neurons as the last layer. Neurons in the
    last layer have sectors of uniform size.

    Provides functions for learning network (if sample and desired output
    provided) and for batch processing of samples - outputs for each sample
    are computed. Correctness of output lies on used sectors
    - various error-detection techniques may be applied - therefore
    error-detection is implemented outside this class.

    In fact, except passing values through network (which is changed by
    initializing layers as continuous), all functions remain the
    same as for DiscreteMLMVN.

    @see DiscreteMLMVN
    '''

    def __init__(self, ls_neurons_per_layers,
                 used_sectors,
                 learning_rate):
        ''' Initialization ... same parameters as for DiscreteMLMVN

        @see DiscreteMLMVN.__init__
        '''
        ## Sectors used for counting activation function of neurons in last
        # layer
        self.sects = used_sectors

        ## Learning rate defining speed of learning
        self.learning_rate = learning_rate

        ## List of layers of given network
        self.ls_layers = []

        # create first and hidden layers
        for (n_inputs, n_neurons) in zip(ls_neurons_per_layers[:-2],
                                         ls_neurons_per_layers[1:-1]):
            self.ls_layers.append(
                layer.MVNLayer(n_neurons, n_inputs,
                               learning_rate=self.learning_rate)
            )

        # create last layer
        n_inputs = ls_neurons_per_layers[-2]
        n_neurons = ls_neurons_per_layers[-1]
        self.ls_layers.append(
            layer.MVNLastLayer(n_neurons, n_inputs,
                               activation_func=self.sects.activation_function,
                               learning_rate=self.learning_rate)
        )

        # set upper layers of self.ls_layers for backpropagation alg.
        for (this_l, upper_l) in zip(self.ls_layers[:-1], self.ls_layers[1:]):
            this_l.set_upper_layer(upper_l)
