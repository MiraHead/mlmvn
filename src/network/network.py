#!/usr/bin/env python
#encoding=utf8
'''
Module describing multi layered neural network's functionality.
'''

import numpy as np

import layer

# TODO ... maybe try different initialization of weights?
# TODO ... maybe try MLMVN with different layers?


## Used to indicate that neurons should be continuous.
CONTINUOUS = None


class MLMVN():
    ''' Represents multi-layered network with uniform layers and
    with multi-valued uniform neurons.

    Provides functions for learning network (if sample and desired output
    provided) and for batch processing of samples - outputs for each sample
    are computed. Correctness of output lies on used sectors
    - various error-detection techniques may be applied - therefore
    error-detection is implemented outside this class.
    '''

    def __init__(self, ls_neurons_per_layers, used_sectors=CONTINUOUS):
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
        '''

        ## Sectors
        if used_sectors == CONTINUOUS:
            # ensures continuous activation function
            self.activation_function = lambda x: x
        else:
            self.activation_function = used_sectors.activation_function

        ## List of layers of given network
        self.ls_layers = []

        # create first and hidden layers
        for (n_inputs, n_neurons) in zip(ls_neurons_per_layers[:-2],
                                         ls_neurons_per_layers[1:-1]):
            self.ls_layers.append(layer.MVNLayer(n_neurons, n_inputs,
                                    activation_func=self.activation_function))

        # create last layer
        n_inputs = ls_neurons_per_layers[-2]
        n_neurons = ls_neurons_per_layers[-1]
        self.ls_layers.append(layer.MVNLastLayer(n_neurons, n_inputs,
                              activation_func=self.activation_function))

        # set upper layers of self.ls_layers for backpropagation alg.
        for (this_l, upper_l) in zip(self.ls_layers[:-1], self.ls_layers[1:]):
            this_l.set_upper_layer(upper_l)

    def reset_random_weights(self):
        ''' Sets random weights for whole network once again.

        @see layer.MLMVNLayer.set_random_weights
        '''
        for this_l in self.ls_layers:
            this_l.set_random_weights()

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
        return self.ls_layers[-1].activation_func(
                                    self.count_zets_of_last_layer(samples))

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

    def get_number_of_outputs(self):
        ''' Returns number of outputs of network. '''
        return self.ls_layers[-1].get_number_of_outputs()

    def get_number_of_inputs(self):
        ''' Returns number of inputs for network. '''
        return self.ls_layers[0].get_number_of_inputs()

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
