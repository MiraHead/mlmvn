#!/usr/bin/env python
#encoding=utf8
'''
Module describing multi layered neural network's functionality.

@author Miroslav Hlavacek <mira.hlavackuj@gmail.com>
'''

import numpy as np
import cPickle as pickle
import layer
import sectors

# TODO ... maybe try different initialization of weights?
# TODO ... maybe try MLMVN with different layers?

## Used to indicate that sectors should be continuous
CONTINUOUS = None

# NETWORK TYPES ... here should be list with constants specifying mlmvn types
# (supposing existence of multiple implementations)
NT_SIMPLE_MLMVN = 0


def save_network_to_file(mlmvn, filename):
    ''' Saves MLMVN network to binary file with given filename.

    Should work with any MLMVN network... in case of specific needs of various
    implementations... (like saving extra features) use network types (defined
    above) to distinguish mlmvns and update this function.

    Do not change the ordering of operations - has to correspond with loading!

    @param mlmvn Network to be saved.
    @param filename Filename where to store the network
    '''
    with open(filename, "wb") as out_file:
        # save network type
        pickle.dump(mlmvn.get_type(), out_file)
        # save layers specification
        pickle.dump(mlmvn.get_ls_neurons_per_layer(), out_file)
        # save sectors specification
        pickle.dump(mlmvn.sects.get_phases(), out_file)
        # save learning rate
        pickle.dump(mlmvn.get_learning_rate(), out_file)
        # save weights and (possibly unique) learning rates for layers
        mlmvn.save_layers(out_file)


def load_network_from_file(filename):
    ''' Loads MLMVN network from binary file with given filename.

    Should work with any MLMVN network... in case of specific needs of various
    implementations... (like saving extra features) use network types (defined
    above) to distinguish mlmvns and update this function.

    @param filename Filename where to store the network
    @return MLMVN network loaded from file
    '''
    with open(filename, "rb") as in_file:
        mlmvn_type = pickle.load(in_file)
        mlmvn_layers = pickle.load(in_file)

        # create sectors according to given specification
        sector_phases = pickle.load(in_file)
        # if phases are empty list... sectors should be continuous
        if not sector_phases:
            sects = sectors.ContinuousSectors()
        else:
            sects = sectors.Sectors(0, sector_phases)

        # Create appropriate network
        if mlmvn_type == NT_SIMPLE_MLMVN:

            mlmvn_learning_rate = pickle.load(in_file)

            mlmvn = MLMVN(mlmvn_layers,
                          used_sectors=sects,
                          learning_rate=mlmvn_learning_rate)

        mlmvn.load_layers(in_file)

        return mlmvn


class MLMVN():
    ''' Represents multi-layered network with uniform layers and
    with multi-valued uniform neurons.

    Provides functions for learning network (if sample and desired output
    provided) and for batch processing of samples - outputs for each sample
    are computed. Correctness of output lies on used sectors
    - various error-detection techniques may be applied - therefore
    error-detection is implemented outside this class.
    '''

    __type = NT_SIMPLE_MLMVN

    def __init__(self, ls_neurons_per_layers,
                 used_sectors=CONTINUOUS,
                 learning_rate=1):
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
        @param used_sectors Sectors used to encode/transform data.\n
                            Sectors provide activation function for neurons.\n
                            If ommited, continuous sectors/activation function
                            is used through whole network.
        @param learning_rate Specifies speed of learning should be in
                            interval (0,1] (where 1 == full speed), but higher
                            speed is also possible.
        '''

        ## Sectors
        if used_sectors == CONTINUOUS:
            # should ensure continuous activation function
            self.sects = sectors.ContinuousSectors()
        else:
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

    def reset_random_weights(self):
        ''' Sets random weights for whole network once again.

        @see layer.MLMVNLayer.set_random_weights
        '''
        for this_l in self.ls_layers:
            this_l.set_random_weights()

    def get_number_of_outputs(self):
        ''' Returns number of outputs of network. '''
        return self.ls_layers[-1].get_number_of_outputs()

    def get_number_of_inputs(self):
        ''' Returns number of inputs for network. '''
        return self.ls_layers[0].get_number_of_inputs()

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
        ''' Batch counting of outputs

        @param samples  Numpy matrix of inputs for first layer.\n
                       => Batch of learning samples. Indexing: [sample,feature]
        @return Numpy matrix of counted outputs.
                Indexing: [outputs_for_sample, output_of_nth_neuron]
        '''
        # count weighted sums for last layer and apply activation
        # function of last layer onto them
        weighted_sums = self.count_zets_of_last_layer(samples)

        return self.ls_layers[-1].activation_func(weighted_sums)

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
        ''' Learns with one learning sample

        @param sample   Numpy vector of inputs for neurons.\n
        @param desired_outputs Numpy vector of desired outputs used for
                               correction of weights.\n
                               In case that bisector opitmization is used,
                               bisectors should be passed as desired_outputs.
        '''
        # stores zets for whole network - used while backpropagating error
        self.count_output_for_learning(sample)

        self.back_propagation(desired_outputs)
        self.forward_propagation(sample)

    def back_propagation(self, desired_outputs):
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

    def forward_propagation(self, sample):
        ''' Forward propagates changes - updates weights of neurons.

        @param sample Inputs for first layer of network.
        '''
        inputs = sample
        for this_l in self.ls_layers:
            this_l.update_weights(inputs)
            inputs = this_l.count_outputs_for_update(inputs)

    # *******************  SAVING/LOADING *************

    def get_type(self):
        ''' Returns type of network ... to be saved '''
        return self.__type

    def get_ls_neurons_per_layer(self):
        ''' Returns network neurons per layer list which is used
        to initialize network.

        @see MLMVN.__init__
        '''
        ls_neurons_per_layers = []
        for this_l in self.ls_layers:
            ls_neurons_per_layers.append(this_l.get_number_of_inputs())

        ls_neurons_per_layers.append(this_l.get_number_of_outputs())

        return ls_neurons_per_layers

    def get_learning_rate(self):
        ''' Returns learning_rate. '''
        return self.learning_rate

    def save_layers(self, out_file):
        ''' Saves whole network to binary file.

        Network can be loaded afterwards, preserving used sectors, layers and
        learning rate -> network can for example continue in learning with new
        samples.

        @param out_file File opened for binary writing.
        '''
        # save weights and learning rate for each layer
        for this_l in self.ls_layers:
            this_l.save_layer(out_file)

    def load_layers(self, in_file):
        ''' Loads weights for each layer and used sectors.

        @param in_file File opened for binary reading
        '''
        # load weights and learning rates of layers
        for this_l in self.ls_layers:
            this_l.load_layer(in_file)
